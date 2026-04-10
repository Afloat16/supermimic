"""Motion-tracking world model environment."""

from __future__ import annotations

from typing import Any

import torch
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from tensordict import TensorDict

from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.models import WorldModel
from supermimic.replay_buffer import ReplayBuffer
from supermimic.utils.helpers import extract_action_scale_offset


class MotionTrackingEnv(WorldModelEnv):
  """WM env for motion tracking. Goals from reference motion clips."""

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
    action_scale, action_offset = extract_action_scale_offset(env)
    super().__init__(
      world_model=world_model,
      action_scale=action_scale,
      action_offset=action_offset,
      batch_size=batch_size,
      horizon=horizon,
      replay_buffer=replay_buffer,
      device=device,
    )

    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    body_indexes = motion_cmd.body_indexes.to(self.device)
    self.motion = MotionLoader(motion_file, body_indexes, device=self.device)
    self.cfg = env.cfg
    self._num_actions = env.num_actions
    self.motion_times = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

  @property
  def num_actions(self) -> int:
    return self._num_actions

  def _on_reset(self, batch: list[torch.Tensor | None]) -> None:
    self.motion_times = batch[4].squeeze(-1).long()
    self._update_goal()

  def _get_wm_goal(self) -> torch.Tensor:
    return self._wm_goal_obs

  def _get_policy_obs(self) -> TensorDict:
    return self._build_obs_tensordict(self._actor_goal_obs, self._wm_goal_obs)

  def _process_actions(self, actions: torch.Tensor) -> torch.Tensor:
    super()._process_actions(actions)
    return actions * self.action_scale + self.action_offset

  def _compute_rewards(
    self,
    actions: torch.Tensor,
    next_rep_state: torch.Tensor,
    next_aux_state: torch.Tensor,
    pred_reward: torch.Tensor,
  ) -> torch.Tensor:
    return self.wm.representation.compute_wm_reward(
      next_aux_state,
      next_rep_state,
      pred_reward,
      actions,
      self.last_actions,
      self.motion,
      self.motion_times,
      self.dt,
    )

  def _compute_commands(self) -> None:
    next_motion_times = self.motion_times + 1
    at_end = torch.isin(next_motion_times, self.motion.motion_ends)
    self.motion_times = torch.where(at_end, self.motion_times, next_motion_times)
    self._update_goal()

  def _update_goal(self) -> None:
    target_joints, wm_goal_obs, actor_goal_obs = self.wm.representation.compute_wm_goal(
      self.motion_times, self.aux_states, self.motion
    )
    # Clone wm_goal/target_joints to detach from CUDA graph output buffers
    # (reduce-overhead mode). actor_goal_obs is safe without clone because
    # it's only used inside torch.cat in _build_obs_tensordict.
    self._target_joints = target_joints.clone()
    self._wm_goal_obs = wm_goal_obs.clone()
    self._actor_goal_obs = actor_goal_obs
