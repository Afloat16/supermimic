"""Script to visualize World Model prediction vs Ground Truth."""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mjlab.tasks  # noqa: F401
import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.lab_api.math import subtract_frame_transforms
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from rsl_rl.models import MLPModel
from tensordict import TensorDict

import supermimic.tasks  # noqa: F401
from supermimic.models import WorldModel
from supermimic.representations import (
  MaxCoordRepresentation,
  create_representation,
)
from supermimic.utils.helpers import quat_to_rot6d, unpack_obs_dims


@dataclass
class PlayConfig:
  wandb_path: str
  device: str | None = None
  num_steps_rollout: int = 1000000
  viewer: Literal["auto", "native", "viser"] = "viser"
  history: int = 1


def create_wm_vis_model(base_model):
  model = copy.deepcopy(base_model)
  model.geom_rgba[:, 3] = 0.8
  model.geom_rgba[:, 0:3] = [0.5, 0.5, 0.8]
  return model


class SuperMimicViewerMixin:
  def setup_supermimic(
    self,
    wm,
    num_steps_rollout,
    history,
    action_scale,
  ):
    self.wm = wm
    self.num_steps_rollout = num_steps_rollout
    self.history = history
    self.action_scale = action_scale

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
    self.curr_pred_body_states_hist = None
    self.curr_wm_states = None
    self.last_actions = None

    # For MaxCoord: map entity-local body indexes to MuJoCo body IDs
    if isinstance(self.wm.representation, MaxCoordRepresentation):
      motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
      entity_body_indexes = motion_cmd.body_indexes
      mj_body_ids = motion_cmd.robot.indexing.body_ids[entity_body_indexes]
      self.mj_body_ids = mj_body_ids.cpu().numpy()

  def reset_env_supermimic(self):
    self.step_in_rollout = 0
    self.curr_pred_body_states_hist = None
    self.curr_wm_states = None
    self.last_actions = None

  def setup_wm_visualizer(
    self, wm_vis_model, *, update_every_n_frames: int | None = None
  ):
    self.wm_vis_model = wm_vis_model
    self.wm_vis_qpos = None
    self.wm_vis_xpos = None
    self.wm_vis_xquat = None

    # For MaxCoord: hide geoms belonging to non-tracked bodies (needed for
    # native viewer's mjv_addGeoms which renders all visible geoms).
    if isinstance(self.wm.representation, MaxCoordRepresentation):
      tracked_set = set(self.mj_body_ids.tolist())
      for geom_id in range(wm_vis_model.ngeom):
        if wm_vis_model.geom_bodyid[geom_id] not in tracked_set:
          wm_vis_model.geom_rgba[geom_id, 3] = 0

    base_env = self.env.unwrapped
    original_vis_callback = base_env.update_visualizers

    def combined_vis_callback(viz):
      if (
        update_every_n_frames is not None and self._counter % update_every_n_frames != 0
      ):
        return
      original_vis_callback(viz)
      if isinstance(self.wm.representation, MaxCoordRepresentation):
        if self.wm_vis_xpos is not None:
          viz.add_ghost_bodies(
            model=self.wm_vis_model,
            body_ids=self.mj_body_ids,
            body_xpos=self.wm_vis_xpos,
            body_xquat=self.wm_vis_xquat,
          )
      else:
        if self.wm_vis_qpos is not None:
          viz.add_ghost_entity(
            self.wm_vis_qpos,
            model=self.wm_vis_model,
          )

    base_env.update_visualizers = combined_vis_callback

  def reset_wm_visualizer(self) -> None:
    self.reset_env_supermimic()
    self.wm_vis_qpos = None
    self.wm_vis_xpos = None
    self.wm_vis_xquat = None

  def step_wm_visualizer(self) -> None:
    self.step_and_predict()

  def _extract_root_pos_quat(self, full_state):
    """Extract root position and quaternion from full state."""
    return self.wm.representation.extract_root_pos_quat(full_state)

  def _compute_wm_goal(self, curr_goals, wm_states, target_pos, target_quat):
    goal_extra = self.wm.representation.goal_dim_split()
    base_goal = curr_goals[..., : -(goal_extra + 6)]
    root_pos, root_quat = self._extract_root_pos_quat(wm_states)
    rel_pos, rel_quat = subtract_frame_transforms(
      root_pos, root_quat, target_pos, target_quat
    )

    rel_ori = quat_to_rot6d(rel_quat)

    return torch.cat([base_goal, rel_pos, rel_ori], dim=-1)

  def _get_obs_vec(self, o):
    return o["actor"], o["full_state"], o["goal"]

  def step_and_predict(self) -> None:
    if self._is_paused:
      return

    with torch.no_grad():
      with self._sim_timer.measure_time():
        obs = self.env.get_observations()
        curr_policy_gt, curr_gt_states, curr_goals = self._get_obs_vec(obs)

        # Extract targets for wm goal computation
        motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
        target_pos = motion_cmd.body_pos_w[:, 0].clone()
        target_quat = motion_cmd.body_quat_w[:, 0].clone()

        # Extract action offset (target joint positions) from motion command
        action_offset = motion_cmd.joint_pos.clone()

        # Initialize wm state at start of rollout (or if reset)
        if self.curr_wm_states is None or self.step_in_rollout == 0:
          self.curr_wm_states = curr_gt_states.clone()

          curr_body_states = self.wm.representation.represent(self.curr_wm_states)
          # [Batch, History, Dim]
          self.curr_pred_body_states_hist = curr_body_states.unsqueeze(1).repeat(
            1, self.history, 1
          )

        # Initialize last_action if None (start of rollout)
        if self.last_actions is None:
          self.last_actions = torch.zeros(
            (1, self.wm.num_actions), device=curr_gt_states.device
          )

        # Actor Step (using Ground Truth for policy input)
        actor_goal = self.wm.representation.extract_actor_goal(curr_goals)
        actor_input = self.wm.representation.construct_policy_input(
          curr_gt_states, actor_goal
        )

        actor_obs = TensorDict(
          {"actor": actor_input},
          batch_size=actor_input.shape[:-1],
          device=actor_input.device,
        )
        actions = self.policy(actor_obs)

        # Env Step (Ground Truth simulation)
        _, _, dones, _ = self.env.step(actions)
        self._step_count += 1

        # Predict Next State (WM)
        wm_goal = self._compute_wm_goal(
          curr_goals, self.curr_wm_states, target_pos, target_quat
        )

        # Closed loop: Compute action based on CURRENT WM STATE
        wm_body_states = self.curr_pred_body_states_hist[:, -1, :]
        wm_actor_goal = self.wm.representation.extract_actor_goal(wm_goal)
        wm_actor_input = torch.cat([wm_body_states, wm_actor_goal], dim=-1)
        wm_actor_obs = TensorDict(
          {"actor": wm_actor_input},
          batch_size=wm_actor_input.shape[:-1],
          device=wm_actor_input.device,
        )
        wm_actions = self.policy(wm_actor_obs)

        # Process action for World Model
        proc_wm_actions = wm_actions * self.action_scale + action_offset

        # Predict next state using history
        next_full_state, next_rep_state, _, term_logit = self.wm.transition(
          self.curr_pred_body_states_hist,
          proc_wm_actions,
          self.last_actions,
          wm_goal,
          self.curr_wm_states,
        )
        self.last_actions = proc_wm_actions.clone().detach()

        # Update history
        self.curr_pred_body_states_hist = torch.cat(
          [self.curr_pred_body_states_hist[:, 1:, :], next_rep_state.unsqueeze(1)],
          dim=1,
        )

        # Update full state (already integrated by transition)
        self.curr_wm_states = next_full_state

        pred_done = (torch.sigmoid(term_logit) > 0.5).any().item()

        # Visualization: update state for ghost rendering
        state = self.curr_wm_states[0]  # Take 0-th env

        if isinstance(self.wm.representation, MaxCoordRepresentation):
          B = self.wm.representation.num_bodies
          self.wm_vis_xpos = state[: B * 3].reshape(B, 3).cpu().numpy()
          self.wm_vis_xquat = state[B * 3 : B * 7].reshape(B, 4).cpu().numpy()
        else:
          nq = self.env.unwrapped.sim.mj_model.nq
          qpos = np.zeros(nq)
          root_pos, root_quat = self._extract_root_pos_quat(state)
          if self.free_joint_q_adr is not None:
            qpos[self.free_joint_q_adr[0:3]] = root_pos.cpu().numpy()
            qpos[self.free_joint_q_adr[3:7]] = root_quat.cpu().numpy()
          qpos[self.joint_q_adr] = state[7:nq].cpu().numpy()
          self.wm_vis_qpos = qpos

        self.step_in_rollout += 1
        if dones.any() or self.step_in_rollout >= self.num_steps_rollout or pred_done:
          self.step_in_rollout = 0
          self.curr_wm_states = None
          self.last_actions = None


class SuperMimicNativeViewer(SuperMimicViewerMixin, NativeMujocoViewer):
  def __init__(
    self,
    env,
    actor,
    wm,
    wm_vis_model,
    num_steps_rollout,
    history,
    action_scale,
    **kwargs,
  ):
    super().__init__(env, actor, **kwargs)
    self.setup_supermimic(
      wm,
      num_steps_rollout,
      history,
      action_scale,
    )
    self.setup_wm_visualizer(wm_vis_model)

  def reset_environment(self) -> None:
    super().reset_environment()
    self.reset_wm_visualizer()

  def step_simulation(self) -> None:
    self.step_wm_visualizer()


class SuperMimicViserViewer(SuperMimicViewerMixin, ViserPlayViewer):
  def __init__(
    self,
    env,
    actor,
    wm,
    wm_vis_model,
    num_steps_rollout,
    history,
    action_scale,
    **kwargs,
  ):
    super().__init__(env, actor, **kwargs)
    self.setup_supermimic(
      wm,
      num_steps_rollout,
      history,
      action_scale,
    )
    self.setup_wm_visualizer(wm_vis_model, update_every_n_frames=2)

  def setup(self) -> None:
    super().setup()
    # Enable camera tracking by default
    self._scene.camera_tracking_enabled = True

  def reset_environment(self) -> None:
    super().reset_environment()
    self.reset_wm_visualizer()

  def step_simulation(self) -> None:
    self.step_wm_visualizer()


def download_model(run, filename, target_dir, device):
  run.file(filename).download(root=str(target_dir), replace=True)
  path = target_dir / filename
  print(f"[INFO] Loading {filename} from {path}")
  return torch.load(path, map_location=device)


def play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Load Env
  print(f"[INFO] Loading environment: {task_id}")
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1

  # Connect to WandB and check for motion artifact
  import wandb

  print(f"[INFO] Connecting to WandB run: {cfg.wandb_path}")
  api = wandb.Api()
  run = api.run(cfg.wandb_path)

  # Check for linked motion artifacts (supports multi-motion)
  motion_artifacts = [a for a in run.used_artifacts() if a.type == "motions"]
  if motion_artifacts:
    motion_files = []
    for artifact in motion_artifacts:
      print(f"[INFO] Found linked motion artifact: {artifact.name}")
      motion_files.append(str(Path(artifact.download()) / "motion.npz"))

    # Pass list if multiple, single string if one
    motion_file = motion_files if len(motion_files) > 1 else motion_files[0]
    command_term = env_cfg.commands["motion"]
    assert isinstance(command_term, MotionCommandCfg)
    command_term.motion_file = motion_file
    print(f"[INFO] Loaded {len(motion_files)} motion(s)")

  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(base_env)

  obs, _ = env.reset()
  num_state, num_goals = unpack_obs_dims(obs)
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

  # Create representation based on task_id
  repr_name = "maxcoord" if "MaxCoord" in task_id else "body_frame"
  representation = create_representation(repr_name, num_state)

  print(
    f"[INFO] Representation: full={representation.full_state_dim}, "
    f"rep={representation.rep_state_dim}, pred={representation.prediction_dim}"
  )

  # Load World Model
  wm = WorldModel(
    representation,
    num_actions,
    num_goals,
    num_rewards,
    dt=env.unwrapped.step_dt,
    history=cfg.history,
  ).to(device)

  # Download from WandB
  download_dir = Path("logs") / task_id / run.id / "wandb_download"
  download_dir.mkdir(parents=True, exist_ok=True)

  # Download World Model
  wm_state = download_model(run, "world_model.pt", download_dir, device)
  wm.load_state_dict(wm_state, strict=False)
  wm.eval()

  # Load Actor
  goal_extra = representation.goal_dim_split()
  actor_input_dim = wm.num_rep_state + num_goals - goal_extra

  actor_obs = TensorDict(
    {"actor": torch.zeros(1, actor_input_dim, device=device)},
    batch_size=[1],
    device=device,
  )
  actor = MLPModel(
    obs=actor_obs,
    obs_groups={"actor": ["actor"]},
    obs_set="actor",
    output_dim=num_actions,
    hidden_dims=[512, 256, 128],
    obs_normalization=True,
  ).to(device)

  # Download Actor from WandB
  try:
    actor_state = download_model(run, "actor.pt", download_dir, device)
    print("[INFO] Loaded actor.pt")
  except Exception:
    print("[WARN] actor.pt not found, trying policy.pt")
    actor_state = download_model(run, "policy.pt", download_dir, device)

  # Handle key mapping from old Policy to new MLPModel
  new_actor_state = {}
  for k, v in actor_state.items():
    if k.startswith("actor."):
      new_key = "mlp." + k[len("actor.") :]
      new_actor_state[new_key] = v
    elif k.startswith("actor_obs_normalizer."):
      new_key = "obs_normalizer." + k[len("actor_obs_normalizer.") :]
      new_actor_state[new_key] = v
    elif k.startswith("mlp.") or k.startswith("obs_normalizer."):
      # Already in new format
      new_actor_state[k] = v

  actor.load_state_dict(new_actor_state, strict=False)

  actor.eval()

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
      actor=actor,
      wm=wm,
      wm_vis_model=wm_vis_model,
      num_steps_rollout=cfg.num_steps_rollout,
      history=cfg.history,
      action_scale=action_scale,
    )
    viewer.run()

  elif resolved_viewer == "viser":
    viewer = SuperMimicViserViewer(
      env=env,
      actor=actor,
      wm=wm,
      wm_vis_model=wm_vis_model,
      num_steps_rollout=cfg.num_steps_rollout,
      history=cfg.history,
      action_scale=action_scale,
    )
    viewer.run()

  env.close()


def main():
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    config=mjlab.TYRO_FLAGS,
  )
  play(chosen_task, args)


if __name__ == "__main__":
  main()
