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
from mjlab.scripts.gcs import ensure_default_checkpoint
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.viser.conversions import (
  create_primitive_mesh,
  get_body_name,
  mujoco_mesh_to_trimesh,
)

import supermimic.tasks  # noqa: F401
from supermimic.models import Policy, WorldModel
from supermimic.utils.helpers import (
  YawAlignedStateLayout,
  apply_yaw_correction,
  compute_yaw_aligned_state,
  construct_policy_input,
  construct_policy_input_from_yaw_state,
  integrate_global_state,
)


@dataclass
class PlayConfig:
  wandb_path: str
  task_id: str = "SuperMimic-WorldModel-G1"
  device: str | None = None
  num_steps_rollout: int = 1000000
  viewer: Literal["auto", "native", "viser"] = "viser"
  history: int = 8

  demo_policy: bool = False


def create_wm_vis_model(base_model):
  model = copy.deepcopy(base_model)
  model.geom_rgba[:, 3] = 0.8
  model.geom_rgba[:, 0:3] = [0.5, 0.5, 0.8]
  return model


class SuperMimicViewerMixin:
  def setup_supermimic(
    self,
    wm,
    layout,
    num_steps_rollout,
    history,
    action_scale,
    action_offset,
    use_demo_policy=False,
  ):
    self.wm = wm
    self.layout = layout
    self.num_steps_rollout = num_steps_rollout
    self.history = history
    self.use_demo_policy = use_demo_policy
    self.action_scale = action_scale
    self.action_offset = action_offset

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
    self.curr_pred_yaw_state_hist = None
    self.curr_wm_state = None
    self.last_action = None

  def reset_env_supermimic(self):
    self.step_in_rollout = 0
    self.curr_pred_yaw_state_hist = None
    self.curr_wm_state = None
    self.last_action = None

  def _compute_wm_goal(self, curr_goal, wm_state, target_pos, target_quat):
    base_goal = curr_goal[..., :-9]
    root_pos = wm_state[..., 0:3]
    root_quat = wm_state[..., 3:7]
    rel_pos, rel_quat = subtract_frame_transforms(
      root_pos, root_quat, target_pos, target_quat
    )

    mat = matrix_from_quat(rel_quat)
    rel_ori = mat[..., :2].reshape(*mat.shape[:-2], -1)

    return torch.cat([base_goal, rel_pos, rel_ori], dim=-1)

  def _get_obs_vec(self, o):
    return o["policy"], o["full_state"], o["goal"]

  def step_and_predict(self) -> np.ndarray | None:
    if self._is_paused:
      return None

    with torch.no_grad():
      with self._sim_timer.measure_time():
        obs = self.env.get_observations()
        curr_policy_gt, curr_gt_state, curr_goal = self._get_obs_vec(obs)

        # Extract targets for wm goal computation
        motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
        target_pos = motion_cmd.body_pos_w[:, 0].clone()
        target_quat = motion_cmd.body_quat_w[:, 0].clone()

        # Initialize wm state at start of rollout (or if reset)
        if self.curr_wm_state is None or self.step_in_rollout == 0:
          # curr_gt_state is [pos(3), quat(4), q(N), lin(3), ang(3), dq(N)]
          self.curr_wm_state = curr_gt_state.clone()

          # Initialize history with current state repeated
          curr_yaw_state = compute_yaw_aligned_state(self.curr_wm_state, self.layout)
          # [Batch, History, Dim]
          self.curr_pred_yaw_state_hist = curr_yaw_state.unsqueeze(1).repeat(
            1, self.history, 1
          )

        # Initialize last_action if None (start of rollout)
        if self.last_action is None:
          self.last_action = torch.zeros(
            (1, self.wm.num_actions), device=curr_gt_state.device
          )

        # Policy Step (using Ground Truth for policy input as per original script)
        if self.use_demo_policy:
          action = self.policy(curr_policy_gt)
        else:
          policy_goal = torch.cat([curr_goal[..., :-9], curr_goal[..., -6:]], dim=-1)
          policy_input = construct_policy_input(curr_gt_state, policy_goal, self.layout)
          action = self.policy(policy_input)

        # Env Step (Ground Truth simulation)
        _, _, dones, _ = self.env.step(action)
        self._step_count += 1

        # Predict Next State (WM)
        wm_goal = self._compute_wm_goal(
          curr_goal, self.curr_wm_state, target_pos, target_quat
        )

        if self.use_demo_policy:
          wm_action = action
        else:
          # Closed loop: Compute action based on CURRENT WM STATE
          wm_yaw_state = self.curr_pred_yaw_state_hist[:, -1, :]
          wm_policy_goal = torch.cat([wm_goal[..., :-9], wm_goal[..., -6:]], dim=-1)
          wm_policy_input = construct_policy_input_from_yaw_state(
            wm_yaw_state, wm_policy_goal
          )
          wm_action = self.policy(wm_policy_input)

        # Process action for World Model
        if self.use_demo_policy:
          action_offset = self.action_offset
        else:
          action_offset = curr_goal[..., : self.wm.num_actions]
        processed_wm_action = wm_action * self.action_scale + action_offset

        # 1. Predict Delta using history
        out = self.wm(
          self.curr_pred_yaw_state_hist,
          processed_wm_action,
          self.last_action,
          wm_goal,
        )
        num_rewards = self.wm.num_rewards
        delta = out[..., : -num_rewards - 1]
        term_logit = out[..., -1:]
        self.last_action = processed_wm_action.clone().detach()

        # 2. Advance Yaw-state (Yaw-Aligned)
        # Use last state in history as current base
        curr_yaw_state = self.curr_pred_yaw_state_hist[:, -1, :]
        yaw_state_next = curr_yaw_state + delta
        yaw_state_next = apply_yaw_correction(
          yaw_state_next, self.env.unwrapped.step_dt, self.layout
        )

        # Update history
        self.curr_pred_yaw_state_hist = torch.cat(
          [self.curr_pred_yaw_state_hist[:, 1:, :], yaw_state_next.unsqueeze(1)], dim=1
        )

        # 3. Integrate Global WM State
        self.curr_wm_state = integrate_global_state(
          self.curr_wm_state,
          yaw_state_next,
          self.env.unwrapped.step_dt,
          self.layout,
        )

        pred_done = (torch.sigmoid(term_logit) > 0.5).any().item()

        # Visualization: Construct qpos from Global WM State
        state = self.curr_wm_state[0]  # Take 0-th env
        root_pos = state[0:3].cpu().numpy()
        root_quat = state[3:7].cpu().numpy()

        # qpos structure for G1: [root_pos(3), root_quat(4), joints...]
        qpos = np.zeros(self.env.unwrapped.sim.mj_model.nq)
        if self.free_joint_q_adr is not None:
          qpos[self.free_joint_q_adr[0:3]] = root_pos
          qpos[self.free_joint_q_adr[3:7]] = root_quat

        # Raw state q index starts at 7
        q_start = 7
        q_end = 7 + self.layout.num_joints
        qpos[self.joint_q_adr] = state[q_start:q_end].cpu().numpy()

        self.step_in_rollout += 1
        if dones.any() or self.step_in_rollout >= self.num_steps_rollout or pred_done:
          self.step_in_rollout = 0
          self.curr_wm_state = None
          self.last_action = None

        return qpos


class SuperMimicNativeViewer(SuperMimicViewerMixin, NativeMujocoViewer):
  def __init__(
    self,
    env,
    policy,
    wm,
    wm_vis_model,
    layout,
    num_steps_rollout,
    history,
    action_scale,
    action_offset,
    use_demo_policy=False,
    **kwargs,
  ):
    super().__init__(env, policy, **kwargs)
    self.setup_supermimic(
      wm,
      layout,
      num_steps_rollout,
      history,
      action_scale,
      action_offset,
      use_demo_policy,
    )
    self.wm_vis_model = wm_vis_model
    self.wm_vis_qpos = None

    base_env = self.env.unwrapped
    original_vis_callback = base_env.update_visualizers

    def combined_vis_callback(viz):
      original_vis_callback(viz)
      if self.wm_vis_qpos is not None:
        viz.add_ghost_mesh(self.wm_vis_qpos, model=self.wm_vis_model)

    base_env.update_visualizers = combined_vis_callback

  def reset_environment(self) -> None:
    super().reset_environment()
    self.reset_env_supermimic()
    self.wm_vis_qpos = None

  def step_simulation(self) -> None:
    qpos = self.step_and_predict()
    if qpos is not None:
      self.wm_vis_qpos = qpos


class SuperMimicViserViewer(SuperMimicViewerMixin, ViserPlayViewer):
  def __init__(
    self,
    env,
    policy,
    wm,
    wm_vis_model,
    layout,
    num_steps_rollout,
    history,
    action_scale,
    action_offset,
    use_demo_policy=False,
    **kwargs,
  ):
    super().__init__(env, policy, **kwargs)
    self.setup_supermimic(
      wm,
      layout,
      num_steps_rollout,
      history,
      action_scale,
      action_offset,
      use_demo_policy,
    )
    self.wm_vis_model = wm_vis_model

    # Custom world model visualization state
    self._custom_wm_handles = {}
    self._custom_wm_meshes = {}
    self._wm_vis_model_data = mujoco.MjData(wm_vis_model)

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

  def add_custom_wm_mesh(self, qpos, model, name_prefix="wm", alpha=0.5) -> None:
    """Add a wm mesh by rendering the robot at a different pose.

    This is a custom implementation for SuperMimic to avoid modifying mjlab code.
    """
    if not self._scene.debug_visualization_enabled:
      return

    if isinstance(qpos, torch.Tensor):
      qpos = qpos.cpu().numpy()

    # Use model hash to support meshes caching
    model_hash = hash((model.ngeom, model.nbody, model.nq))

    # Update kinematics for wm model
    self._wm_vis_model_data.qpos[:] = qpos
    mujoco.mj_forward(model, self._wm_vis_model_data)

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
      body_pos = self._wm_vis_model_data.xpos[body_id] + scene_offset
      body_quat = vtf.SO3.from_matrix(
        self._wm_vis_model_data.xmat[body_id].reshape(3, 3)
      ).wxyz

      handle_key = f"{name_prefix}_{body_id}"

      # Check if we already have a handle for this body
      if handle_key in self._custom_wm_handles:
        handle = self._custom_wm_handles[handle_key]
        handle.wxyz = body_quat
        handle.position = body_pos
      else:
        # Create mesh if not cached
        if model_hash not in self._custom_wm_meshes:
          self._custom_wm_meshes[model_hash] = {}

        if body_id not in self._custom_wm_meshes[model_hash]:
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

          self._custom_wm_meshes[model_hash][body_id] = combined_mesh
        else:
          combined_mesh = self._custom_wm_meshes[model_hash][body_id]

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
        self._custom_wm_handles[handle_key] = handle

  def step_simulation(self) -> None:
    qpos = self.step_and_predict()
    if qpos is not None:
      self.add_custom_wm_mesh(
        qpos, model=self.wm_vis_model, name_prefix="prediction_wm"
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

  if cfg.demo_policy:
    from mjlab.asset_zoo.robots import G1_ACTION_SCALE
    from mjlab.envs.mdp.actions import JointPositionActionCfg
    from mjlab.scripts.gcs import ensure_default_motion

    # Force JointPositionAction for demo policy compatibility
    env_cfg.actions["joint_pos"] = JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=G1_ACTION_SCALE,
      use_default_offset=True,
    )
    print(" [INFO] Reverted to JointPositionAction for demo policy compatibility.")

    motion_file = ensure_default_motion()
    env_cfg.commands["motion"].motion_file = motion_file
    print(f"[INFO] Using demo motion file: {motion_file}")

  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(base_env)

  obs, _ = env.reset()
  state_obs = obs["full_state"] if hasattr(obs, "keys") else obs
  num_state = state_obs.shape[-1]
  goal_obs = obs["goal"] if hasattr(obs, "keys") else obs
  num_goals = goal_obs.shape[-1]
  num_actions = env.num_actions

  # Check for reward terms
  reward_manager = env.unwrapped.reward_manager
  reward_names = getattr(reward_manager, "active_terms", ["reward"])
  num_rewards = len(reward_names)
  print(f"[INFO] Reward Dim: {num_rewards} ({reward_names})")

  # Extract Scale/Offset
  action_manager = env.unwrapped.action_manager
  # Assume only one action term
  term_name = action_manager.active_terms[0]
  action_term = action_manager.get_term(term_name)
  action_scale = action_term.scale.to(device)
  action_offset = action_term.offset.to(device)

  # Load World Model
  layout = YawAlignedStateLayout((num_state - 13) // 2)
  wm_state_dim = layout.dim
  wm = WorldModel(wm_state_dim, num_actions, num_goals, num_rewards, cfg.history).to(
    device
  )

  # Download from WandB
  download_dir = Path("logs") / cfg.task_id / run.id / "wandb_download"
  download_dir.mkdir(parents=True, exist_ok=True)

  # Download World Model
  wm_state = download_model(run, "world_model.pt", download_dir, device)
  wm.load_state_dict(wm_state, strict=False)
  wm.eval()

  # Load Policy
  if cfg.demo_policy:
    policy_path = ensure_default_checkpoint()
    # Use policy dim for pretrained
    policy_obs = obs["policy"] if hasattr(obs, "keys") else obs
    policy_input_dim = policy_obs.shape[-1]
    print(
      f"[INFO] Using Demo Policy Dimension: {policy_input_dim} (from 'policy' group)"
    )
    policy = Policy(policy_input_dim, num_actions, normalization=True).to(device)
    print(f"[INFO] Loading demo policy from {policy_path}")
    policy.load_state_dict_from_rsl_rl_ckpt(policy_path, device)
  else:
    policy_input_dim = layout.dim + num_goals - 3
    policy = Policy(policy_input_dim, num_actions, normalization=True).to(device)
    # Download Policy from WandB
    policy_state = download_model(run, "policy.pt", download_dir, device)
    policy.load_state_dict(policy_state, strict=False)

  policy.eval()

  # Setup WM Vis Model
  wm_vis_model = create_wm_vis_model(base_env.sim.mj_model)

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
      wm_vis_model=wm_vis_model,
      layout=layout,
      num_steps_rollout=cfg.num_steps_rollout,
      history=cfg.history,
      action_scale=action_scale,
      action_offset=action_offset,
      use_demo_policy=cfg.demo_policy,
    )
    viewer.run()

  elif resolved_viewer == "viser":
    viewer = SuperMimicViserViewer(
      env=env,
      policy=policy,
      wm=wm,
      wm_vis_model=wm_vis_model,
      layout=layout,
      num_steps_rollout=cfg.num_steps_rollout,
      history=cfg.history,
      action_scale=action_scale,
      action_offset=action_offset,
      use_demo_policy=cfg.demo_policy,
    )
    viewer.run()

  env.close()


def main():
  args = tyro.cli(PlayConfig)
  play(args)


if __name__ == "__main__":
  main()
