"""Script to visualize World Model prediction vs Ground Truth."""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

import supermimic.tasks  # noqa: F401
from supermimic.models import Policy, WorldModel
from supermimic.utils.helpers import (
  BodyFrameStateLayout,
  compute_body_frame_state,
  construct_policy_input,
  construct_policy_input_from_body_state,
  integrate_global_state,
)


@dataclass
class PlayConfig:
  wandb_path: str
  task_id: str = "SuperMimic-WorldModel-G1"
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
    layout,
    num_steps_rollout,
    history,
    action_scale,
    action_offset,
  ):
    self.wm = wm
    self.layout = layout
    self.num_steps_rollout = num_steps_rollout
    self.history = history
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
    self.curr_pred_body_states_hist = None
    self.curr_wm_states = None
    self.last_actions = None

  def reset_env_supermimic(self):
    self.step_in_rollout = 0
    self.curr_pred_body_states_hist = None
    self.curr_wm_states = None
    self.last_actions = None

  def _compute_wm_goal(self, curr_goals, wm_states, target_pos, target_quat):
    base_goal = curr_goals[..., :-9]
    root_pos = wm_states[..., 0:3]
    root_quat = wm_states[..., 3:7]
    rel_pos, rel_quat = subtract_frame_transforms(
      root_pos, root_quat, target_pos, target_quat
    )

    mat = matrix_from_quat(rel_quat)
    rel_ori = mat[..., :2].reshape(*mat.shape[:-2], -1)

    return torch.cat([base_goal, rel_pos, rel_ori], dim=-1)

  def _get_obs_vec(self, o):
    return o["actor"], o["full_state"], o["goal"]

  def step_and_predict(self) -> np.ndarray | None:
    if self._is_paused:
      return None

    with torch.no_grad():
      with self._sim_timer.measure_time():
        obs = self.env.get_observations()
        curr_policy_gt, curr_gt_states, curr_goals = self._get_obs_vec(obs)

        # Extract targets for wm goal computation
        motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
        target_pos = motion_cmd.body_pos_w[:, 0].clone()
        target_quat = motion_cmd.body_quat_w[:, 0].clone()

        # Initialize wm state at start of rollout (or if reset)
        if self.curr_wm_states is None or self.step_in_rollout == 0:
          # curr_gt_states is [pos(3), quat(4), q(N), lin(3), ang(3), dq(N)]
          self.curr_wm_states = curr_gt_states.clone()

          curr_body_states = compute_body_frame_state(self.curr_wm_states, self.layout)
          # [Batch, History, Dim]
          self.curr_pred_body_states_hist = curr_body_states.unsqueeze(1).repeat(
            1, self.history, 1
          )

        # Initialize last_action if None (start of rollout)
        if self.last_actions is None:
          self.last_actions = torch.zeros(
            (1, self.wm.num_actions), device=curr_gt_states.device
          )

        # Policy Step (using Ground Truth for policy input as per original script)
        policy_goal = torch.cat([curr_goals[..., :-9], curr_goals[..., -6:]], dim=-1)
        policy_input = construct_policy_input(curr_gt_states, policy_goal, self.layout)
        actions = self.policy(policy_input)

        # Env Step (Ground Truth simulation)
        _, _, dones, _ = self.env.step(actions)
        self._step_count += 1

        # Predict Next State (WM)
        wm_goal = self._compute_wm_goal(
          curr_goals, self.curr_wm_states, target_pos, target_quat
        )

        # Closed loop: Compute action based on CURRENT WM STATE
        wm_body_states = self.curr_pred_body_states_hist[:, -1, :]
        wm_policy_goal = torch.cat([wm_goal[..., :-9], wm_goal[..., -6:]], dim=-1)
        wm_policy_input = construct_policy_input_from_body_state(
          wm_body_states, wm_policy_goal
        )
        wm_actions = self.policy(wm_policy_input)

        # Process action for World Model
        action_offset = curr_goals[..., : self.wm.num_actions]
        proc_wm_actions = wm_actions * self.action_scale + action_offset

        # 1. Predict Delta using history
        out = self.wm(
          self.curr_pred_body_states_hist,
          proc_wm_actions,
          self.last_actions,
          wm_goal,
        )
        num_rewards = self.wm.num_rewards
        delta = out[..., : -num_rewards - 1]
        term_logit = out[..., -1:]
        self.last_actions = proc_wm_actions.clone().detach()

        # 2. Advance Yaw-state (Yaw-Aligned)
        # Use last state in history as current base
        curr_body_states = self.curr_pred_body_states_hist[:, -1, :]
        body_state_next = curr_body_states + delta

        # Update history
        self.curr_pred_body_states_hist = torch.cat(
          [self.curr_pred_body_states_hist[:, 1:, :], body_state_next.unsqueeze(1)],
          dim=1,
        )

        # 3. Integrate Global WM State
        self.curr_wm_states = integrate_global_state(
          self.curr_wm_states,
          body_state_next,
          self.env.unwrapped.step_dt,
          self.layout,
        )

        pred_done = (torch.sigmoid(term_logit) > 0.5).any().item()

        # Visualization: Construct qpos from Global WM State
        state = self.curr_wm_states[0]  # Take 0-th env
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
          self.curr_wm_states = None
          self.last_actions = None

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
    )
    self.wm_vis_model = wm_vis_model
    self.wm_vis_qpos = None

    base_env = self.env.unwrapped
    original_vis_callback = base_env.update_visualizers

    def combined_vis_callback(viz):
      # Only update ghost when robot mesh updates (every 2 frames, when counter is even)
      if self._counter % 2 != 0:
        return
      original_vis_callback(viz)
      if self.wm_vis_qpos is not None:
        viz.add_ghost_mesh(self.wm_vis_qpos, model=self.wm_vis_model)

    base_env.update_visualizers = combined_vis_callback

  def setup(self) -> None:
    super().setup()
    # Enable camera tracking by default
    self._scene.camera_tracking_enabled = True

  def reset_environment(self) -> None:
    super().reset_environment()
    self.reset_env_supermimic()
    self.wm_vis_qpos = None

  def step_simulation(self) -> None:
    qpos = self.step_and_predict()
    if qpos is not None:
      self.wm_vis_qpos = qpos


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
  layout = BodyFrameStateLayout((num_state - 13) // 2)
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
    )
    viewer.run()

  env.close()


def main():
  args = tyro.cli(PlayConfig)
  play(args)


if __name__ == "__main__":
  main()
