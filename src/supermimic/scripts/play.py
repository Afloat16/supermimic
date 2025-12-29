"""Script to visualize World Model prediction vs Ground Truth."""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import torch
import trimesh
import tyro
import viser.transforms as vtf
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_from_matrix
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.viser.conversions import (
  create_primitive_mesh,
  get_body_name,
  mujoco_mesh_to_trimesh,
)

import supermimic.tasks  # noqa: F401
from supermimic.models import Policy, WorldModel


@dataclass
class PlayConfig:
  wandb_path: str
  task_id: str = "SuperMimic-WorldModel-G1"
  device: str | None = None
  num_steps_rollout: int = 1000000
  viewer: Literal["auto", "native", "viser"] = "viser"


def create_ghost_model(base_model):
  model = copy.deepcopy(base_model)
  model.geom_rgba[:, 3] = 0.8
  model.geom_rgba[:, 0:3] = [0.5, 0.5, 0.8]
  return model


class SuperMimicViewerMixin:
  def setup_supermimic(self, wm, num_steps_rollout):
    self.wm = wm
    self.num_steps_rollout = num_steps_rollout

    mj_model = self.env.unwrapped.sim.mj_model
    joint_names = [mj_model.joint(i).name for i in range(mj_model.njnt)]
    self.joint_q_adr = np.array(
      [mj_model.joint(n).qposadr[0] for n in joint_names if mj_model.joint(n).type != 0]
    )
    # Find free joint
    free_joint_id = next(
      (i for i in range(mj_model.njnt) if mj_model.joint(i).type == 0), -1
    )
    self.free_joint_q_adr = None
    if free_joint_id >= 0:
      adr = mj_model.joint(free_joint_id).qposadr[0]
      self.free_joint_q_adr = np.arange(adr, adr + 7)

    self.step_in_rollout = 0
    self.curr_pred_state = None

  def reset_env_supermimic(self):
    self.step_in_rollout = 0
    self.curr_pred_state = None

  def _get_obs_vec(self, o):
    return o["policy"], o["full_state"], o["goal"]

  def step_and_predict(self) -> np.ndarray | None:
    if self._is_paused:
      return None

    with torch.no_grad():
      with self._sim_timer.measure_time():
        obs = self.env.get_observations()
        curr_policy_gt, curr_gt_state, curr_goal = self._get_obs_vec(obs)

        if self.curr_pred_state is None or self.step_in_rollout == 0:
          self.curr_pred_state = curr_gt_state.clone()

        # Policy Step
        policy_input = torch.cat([curr_gt_state, curr_goal], dim=-1)
        action = self.policy(policy_input)

        # Env Step
        _, _, dones, _ = self.env.step(action)
        self._step_count += 1

        # Predict Next State (Ghost)
        out = self.wm(self.curr_pred_state, action, curr_goal)
        delta = out[..., :-2]
        term_logit = out[..., -1:]
        self.curr_pred_state += delta

        pred_done = (torch.sigmoid(term_logit) > 0.5).any().item()

        # Visualization: Construct qpos
        state = self.curr_pred_state[0]
        root_pos = state[0:3].cpu().numpy()
        root_rot6d = state[3:9]

        # Reconstruct Quaternion from 6D (Rows 0,1)
        r1 = root_rot6d[:3]
        r2 = root_rot6d[3:]
        r3 = torch.linalg.cross(r1, r2)
        mat = torch.stack([r1, r2, r3], dim=0)
        root_quat = quat_from_matrix(mat).cpu().numpy()

        qpos = np.zeros(self.env.unwrapped.sim.mj_model.nq)
        if self.free_joint_q_adr is not None:
          qpos[self.free_joint_q_adr[0:3]] = root_pos
          qpos[self.free_joint_q_adr[3:7]] = root_quat
        qpos[self.joint_q_adr] = state[15:44].cpu().numpy()

        self.step_in_rollout += 1
        if dones.any() or self.step_in_rollout >= self.num_steps_rollout or pred_done:
          self.step_in_rollout = 0

        return qpos


class SuperMimicNativeViewer(SuperMimicViewerMixin, NativeMujocoViewer):
  def __init__(self, env, policy, wm, ghost_model, num_steps_rollout, **kwargs):
    super().__init__(env, policy, **kwargs)
    self.setup_supermimic(wm, num_steps_rollout)
    self.ghost_model = ghost_model
    self.ghost_qpos = None

    base_env = self.env.unwrapped
    original_vis_callback = base_env.update_visualizers

    def combined_vis_callback(viz):
      original_vis_callback(viz)
      if self.ghost_qpos is not None:
        viz.add_ghost_mesh(self.ghost_qpos, model=self.ghost_model)

    base_env.update_visualizers = combined_vis_callback

  def reset_environment(self) -> None:
    super().reset_environment()
    self.reset_env_supermimic()
    self.ghost_qpos = None

  def step_simulation(self) -> None:
    qpos = self.step_and_predict()
    if qpos is not None:
      self.ghost_qpos = qpos


class SuperMimicViserViewer(SuperMimicViewerMixin, ViserPlayViewer):
  def __init__(self, env, policy, wm, ghost_model, num_steps_rollout, **kwargs):
    super().__init__(env, policy, **kwargs)
    self.setup_supermimic(wm, num_steps_rollout)
    self.ghost_model = ghost_model

    # Custom ghost visualization state
    self._custom_ghost_handles = {}
    self._custom_ghost_meshes = {}
    self._ghost_model_data = mujoco.MjData(ghost_model)

  def setup(self) -> None:
    super().setup()
    # Enable camera tracking by default
    self._scene.camera_tracking_enabled = True

  def reset_environment(self) -> None:
    super().reset_environment()
    self.reset_env_supermimic()

  def _create_geom_mesh_from_model(self, mj_model, geom_id):
    geom_type = mj_model.geom_type[geom_id]
    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
      return mujoco_mesh_to_trimesh(mj_model, geom_id, verbose=False)
    else:
      return create_primitive_mesh(mj_model, geom_id)

  def add_custom_ghost_mesh(self, qpos, model, name_prefix="ghost", alpha=0.5) -> None:
    """Add a ghost mesh by rendering the robot at a different pose.

    This is a custom implementation for SuperMimic to avoid modifying mjlab code.
    """
    if not self._scene.debug_visualization_enabled:
      return

    if isinstance(qpos, torch.Tensor):
      qpos = qpos.cpu().numpy()

    # Use model hash to support meshes caching
    model_hash = hash((model.ngeom, model.nbody, model.nq))

    # Update kinematics for ghost model
    self._ghost_model_data.qpos[:] = qpos
    mujoco.mj_forward(model, self._ghost_model_data)

    # Use current scene offset (camera tracking)
    scene_offset = self._scene._scene_offset

    # Group geoms by body
    body_geoms = {}
    for i in range(model.ngeom):
      body_id = model.geom_bodyid[i]
      is_collision = model.geom_contype[i] != 0 or model.geom_conaffinity[i] != 0
      if is_collision:
        continue

      if model.body_dofnum[body_id] == 0 and model.body_parentid[body_id] == 0:
        continue

      if body_id not in body_geoms:
        body_geoms[body_id] = []
      body_geoms[body_id].append(i)

    # Update or create mesh for each body
    for body_id, geom_indices in body_geoms.items():
      body_pos = self._ghost_model_data.xpos[body_id] + scene_offset
      body_quat = vtf.SO3.from_matrix(
        self._ghost_model_data.xmat[body_id].reshape(3, 3)
      ).wxyz

      handle_key = f"{name_prefix}_{body_id}"

      # Check if we already have a handle for this body
      if handle_key in self._custom_ghost_handles:
        handle = self._custom_ghost_handles[handle_key]
        handle.wxyz = body_quat
        handle.position = body_pos
      else:
        # Create mesh if not cached
        if model_hash not in self._custom_ghost_meshes:
          self._custom_ghost_meshes[model_hash] = {}

        if body_id not in self._custom_ghost_meshes[model_hash]:
          meshes = []
          for geom_id in geom_indices:
            mesh = self._create_geom_mesh_from_model(model, geom_id)
            if mesh is not None:
              geom_pos = model.geom_pos[geom_id]
              geom_quat = model.geom_quat[geom_id]
              transform = np.eye(4)
              transform[:3, :3] = vtf.SO3(geom_quat).as_matrix()
              transform[:3, 3] = geom_pos
              mesh.apply_transform(transform)
              meshes.append(mesh)

          if not meshes:
            continue

          combined_mesh = (
            meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
          )

          self._custom_ghost_meshes[model_hash][body_id] = combined_mesh
        else:
          combined_mesh = self._custom_ghost_meshes[model_hash][body_id]

        body_name = get_body_name(model, body_id)
        # Use name_prefix in the path to allow multiple ghosts
        handle_name = f"/debug/custom/{name_prefix}/body_{body_name}"

        # Extract color from geom (convert RGBA 0-1 to RGB 0-255)
        rgba = model.geom_rgba[geom_indices[0]].copy()
        color_uint8 = (rgba[:3] * 255).astype(np.uint8)

        handle = self._server.scene.add_mesh_simple(
          handle_name,
          combined_mesh.vertices,
          combined_mesh.faces,
          color=tuple(color_uint8),
          opacity=alpha,
          wxyz=body_quat,
          position=body_pos,
          cast_shadow=False,
          receive_shadow=False,
        )
        self._custom_ghost_handles[handle_key] = handle

  def step_simulation(self) -> None:
    qpos = self.step_and_predict()
    if qpos is not None:
      self.add_custom_ghost_mesh(
        qpos, model=self.ghost_model, name_prefix="prediction_ghost"
      )


def download_model(run, filename, target_dir, device):
  run.file(filename).download(root=str(target_dir), replace=True)
  path = target_dir / filename
  print(f"[INFO] Loading {filename} from {path}")
  return torch.load(path, map_location=device)


def play(cfg: PlayConfig):
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Load Env
  print(f"[INFO] Loading environment: {cfg.task_id}")
  env_cfg = load_env_cfg(cfg.task_id, play=True)
  env_cfg.scene.num_envs = 1

  # Connect to WandB and check for motion artifact
  import wandb

  print(f"[INFO] Connecting to WandB run: {cfg.wandb_path}")
  api = wandb.Api()
  run = api.run(cfg.wandb_path)

  # Check for linked motion artifacts
  motion_artifact = next((a for a in run.used_artifacts() if a.type == "motions"), None)
  if motion_artifact:
    print(f"[INFO] Found linked motion artifact: {motion_artifact.name}")
    download_dir = Path("logs") / cfg.task_id / run.id / "wandb_download"
    download_dir.mkdir(parents=True, exist_ok=True)

    motion_file = str(
      Path(motion_artifact.download(root=str(download_dir))) / "motion.npz"
    )

    env_cfg.commands["motion"].motion_file = motion_file
    print(f"[INFO] Auto-configured motion file: {motion_file}")

  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(base_env)

  obs, _ = env.reset()
  state_obs = obs["full_state"] if hasattr(obs, "keys") else obs
  num_state = state_obs.shape[-1]
  goal_obs = obs["goal"] if hasattr(obs, "keys") else obs
  num_goals = goal_obs.shape[-1]
  num_actions = env.num_actions

  # Load World Model
  wm = WorldModel(num_state, num_actions, num_goals).to(device)

  # Download from WandB
  download_dir = Path("logs") / cfg.task_id / run.id / "wandb_download"
  download_dir.mkdir(parents=True, exist_ok=True)

  # Download World Model
  wm_state = download_model(run, "world_model.pt", download_dir, device)
  wm.load_state_dict(wm_state, strict=False)
  wm.eval()

  # Load Policy
  policy_input_dim = num_state + num_goals
  policy = Policy(policy_input_dim, num_actions, normalization=True).to(device)

  # Download Policy
  policy_state = download_model(run, "policy.pt", download_dir, device)
  policy.load_state_dict(policy_state, strict=False)
  policy.eval()

  # Setup Ghost Model
  ghost_model = create_ghost_model(base_env.sim.mj_model)

  # Viewer Selection
  import os

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = cfg.viewer

  print(f"[INFO] Starting viewer: {resolved_viewer}")

  if resolved_viewer == "native":
    viewer = SuperMimicNativeViewer(
      env=env,
      policy=policy,
      wm=wm,
      ghost_model=ghost_model,
      num_steps_rollout=cfg.num_steps_rollout,
    )
    viewer.run()

  elif resolved_viewer == "viser":
    viewer = SuperMimicViserViewer(
      env=env,
      policy=policy,
      wm=wm,
      ghost_model=ghost_model,
      num_steps_rollout=cfg.num_steps_rollout,
    )
    viewer.run()

  env.close()


def main():
  args = tyro.cli(PlayConfig)
  play(args)


if __name__ == "__main__":
  main()
