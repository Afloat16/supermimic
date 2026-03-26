from __future__ import annotations

from typing import Any

import torch
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from tensordict import TensorDict

from supermimic.models import WorldModel
from supermimic.replay_buffer import ReplayBuffer
from supermimic.utils.helpers import extract_action_scale_offset, shift_history


class WorldModelEnv:
  def __init__(
    self,
    world_model: WorldModel,
    motion_file: str | list[str],
    env: Any,
    batch_size: int,
    horizon: int,
    replay_buffer: ReplayBuffer,
    device: str = "cpu",
  ) -> None:
    self.wm = world_model
    self.device = device

    # Get body_indexes from the env's MotionCommand for consistent motion loading
    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    body_indexes = motion_cmd.body_indexes.to(self.device)

    self.motion = MotionLoader(
      motion_file,
      body_indexes,
      device=self.device,
    )
    self.cfg = env.cfg
    self.num_envs = batch_size
    self.horizon = horizon
    self.replay_buffer = replay_buffer

    # Shapes and Dimensions
    self.num_actions = env.num_actions
    self.dt = env.unwrapped.step_dt

    # Action Processing
    self.action_scale, self.action_offset = extract_action_scale_offset(env)

    # Persistent State — initialized properly in reset(), declared here for clarity.
    self.motion_times = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.aux_states = torch.zeros((self.num_envs, 7), device=self.device)
    self.hist = torch.zeros(
      (self.num_envs, self.wm.history, self.wm.num_rep_state), device=self.device
    )
    self.last_actions = torch.zeros(
      (self.num_envs, self.num_actions), device=self.device
    )
    self._first_step = True

  def _sample_initial_states(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample initial states from replay buffer."""
    batch = self.replay_buffer._generate_batch(
      self.replay_buffer.replay_buf,
      valid_indices=None,
      sequence_length=self.wm.history,
      mini_batch_size=self.num_envs,
    )
    # batch[i] has shape (num_envs, history, dim)

    # Get full_states and motion_times from last timestep.
    full_states = batch[0][:, -1, :]
    motion_times = batch[4][:, -1, :].squeeze(-1).long()

    # Extract aux_state from full_state.
    aux_states = self.wm.representation.extract_aux_state(full_states)

    # Compute rep states for the whole history using representation.
    all_states = batch[0]  # (num_envs, history, full_state_dim)
    body_states_hist = self.wm.representation.represent(all_states)

    return (
      motion_times,
      aux_states.detach(),
      body_states_hist.detach(),
    )

  def reset(self) -> TensorDict:
    self.motion_times, self.aux_states, self.hist = self._sample_initial_states()
    self._first_step = True

    self._update_goal()
    return self.get_observations()

  def _update_goal(self) -> None:
    target_joints, wm_goal_obs, actor_goal_obs = self.wm.representation.compute_wm_goal(
      self.motion_times, self.aux_states, self.motion
    )
    # Clone to detach from CUDA graph output buffers (reduce-overhead mode).
    self._target_joints = target_joints.clone()
    self._wm_goal_obs = wm_goal_obs.clone()
    self._actor_goal_obs = actor_goal_obs

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, Any]]:
    # On first step after reset, use current action as last_action
    # to avoid a spurious action rate penalty.
    if self._first_step:
      self.last_actions = actions
      self._first_step = False

    proc_actions = actions * self.action_scale + self.action_offset
    proc_last_actions = self.last_actions * self.action_scale + self.action_offset

    next_aux, next_rep_state, pred_r, pred_d = self.wm(
      self.hist, proc_actions, proc_last_actions, self._wm_goal_obs, self.aux_states
    )
    pred_d = pred_d.clone()
    rewards = self.wm.representation.compute_wm_reward(
      next_aux,
      next_rep_state,
      pred_r,
      actions,
      self.last_actions,
      self.motion,
      self.motion_times,
      self.dt,
    )

    dones = torch.sigmoid(pred_d).squeeze(-1)

    next_hist = shift_history(self.hist, next_rep_state)

    next_motion_times = self.motion_times + 1
    # Pad at clip boundaries (motion_ends are exclusive upper bounds).
    at_end = torch.isin(next_motion_times, self.motion.motion_ends)
    next_motion_times[at_end] -= 1

    self.aux_states = next_aux
    self.hist = next_hist
    self.last_actions = actions
    self.motion_times = next_motion_times

    self._update_goal()
    return self.get_observations(), rewards, dones, {}

  def get_observations(self) -> TensorDict:
    curr_body_states = self.hist[:, -1, :]
    return TensorDict(
      {
        "actor": torch.cat([curr_body_states, self._actor_goal_obs], dim=-1),
        "critic": torch.cat([curr_body_states, self._wm_goal_obs], dim=-1),
      },
      batch_size=[self.num_envs],
      device=self.device,
    )
