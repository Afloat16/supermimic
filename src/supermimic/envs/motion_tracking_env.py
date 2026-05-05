"""Motion-tracking world model environment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from tensordict import TensorDict

from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.replay_buffer import ReplayBuffer

if TYPE_CHECKING:
  from supermimic.models import Dynamics
  from supermimic.representations import StateRepresentation


class MotionTrackingEnv(WorldModelEnv):
  """WM env for motion tracking. Goals from reference motion clips."""

  def __init__(
    self,
    dynamics: Dynamics,
    representation: StateRepresentation,
    motion_file: str | list[str],
    env: Any,
    batch_size: int,
    horizon: int,
    replay_buffer: ReplayBuffer,
    device: str = "cpu",
  ) -> None:
    super().__init__(
      dynamics=dynamics,
      representation=representation,
      batch_size=batch_size,
      horizon=horizon,
      replay_buffer=replay_buffer,
      device=device,
    )

    self._num_actions = env.num_actions

    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    body_indexes = motion_cmd.body_indexes.to(self.device)
    self.motion = MotionLoader(motion_file, body_indexes, device=self.device)
    self.motion_times = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._update_goal()

  @property
  def num_actions(self) -> int:
    return self._num_actions

  def _on_reset(self, batch: TensorDict) -> None:
    self.motion_times = batch["motion_time"]
    self._update_goal()

  def _get_wm_goal(self) -> torch.Tensor:
    return self._wm_goal_obs

  def _get_policy_obs(self) -> TensorDict:
    # Detach to avoid a second autograd path through last_actions ("backward second time" in BPTT).
    actor_tensor = self.representation.build_actor_obs(
      self.rep_state,
      self._actor_goal_obs,
      self._num_actions,
      aux_state=self.aux_state,
      last_action=self.last_actions.detach(),
      noisy=True,
    )
    critic_tensor = torch.cat([self.rep_state, self._wm_goal_obs], dim=-1)
    return TensorDict(
      {"actor": actor_tensor, "critic": critic_tensor},
      batch_size=[self.num_envs],
      device=self.device,
    )

  def _wm_step(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    goal: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    next_aux, next_rep_state = self.dynamics(rep_state, action, aux_state)
    term_logit = self.representation.predict_termination(next_rep_state, action, goal)
    reward = self.representation.compute_wm_reward(
      next_rep_state,
      next_aux,
      action,
      last_action,
      goal,
      self.motion,
      self.motion_times,
    )
    return next_aux, next_rep_state, term_logit, reward

  def _compute_commands(self) -> None:
    next_motion_times = self.motion_times + 1
    at_end = torch.isin(next_motion_times, self.motion.motion_ends)
    self.motion_times = torch.where(at_end, self.motion_times, next_motion_times)
    self._update_goal()

  def _update_goal(self) -> None:
    _, wm_goal_obs, actor_goal_obs = self.representation.compute_wm_goal(
      self.motion_times, self.aux_state, self.motion
    )
    # Clone both goal tensors to detach from CUDA graph output buffers
    # (reduce-overhead mode). Each lives one step in obs TensorDicts that AC
    # captures into rollout storage; without the clone, the next compiled
    # goal-update overwrites the buffer they still point at.
    self._wm_goal_obs = wm_goal_obs.clone()
    self._actor_goal_obs = actor_goal_obs.clone()
