"""Multi-task differentiable world model env for concurrent encoder training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict

from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.tasks.task_spec import build_actor_obs_td

if TYPE_CHECKING:
  from supermimic.models import Dynamics
  from supermimic.replay_buffer import ReplayBuffer
  from supermimic.representations import StateRepresentation
  from supermimic.sim_context import SimContext
  from supermimic.tasks.task_spec import TaskSpec


class MultiTaskWorldModelEnv(WorldModelEnv):
  """WM env for motion tracking with per-slot goal modalities.

  Reward is motion-tracking for every slot; tasks differ only in what
  ``goal_<name>`` key the actor's encoder consumes. ``_get_policy_obs``
  emits all per-task goal keys full-batch and the actor slices its own
  slot range.
  """

  def __init__(
    self,
    dynamics: "Dynamics",
    representation: "StateRepresentation",
    env: Any,
    batch_size: int,
    horizon: int,
    replay_buffer: "ReplayBuffer",
    sim_context: "SimContext",
    task_specs: list["TaskSpec"],
    device: str = "cpu",
  ) -> None:
    self.task_specs = task_specs
    assert task_specs[0].name == "motion", "task_specs[0] must be the motion task"
    super().__init__(
      dynamics=dynamics,
      representation=representation,
      env=env,
      batch_size=batch_size,
      horizon=horizon,
      replay_buffer=replay_buffer,
      sim_context=sim_context,
      device=device,
    )

  def _build_task_goals(self) -> None:
    # The anchor task's goal is the actor goal already built in _update_goal; reuse it.
    anchor, *rest = self.task_specs
    self._task_goal_obs: dict[str, torch.Tensor] = {anchor.name: self._actor_goal_obs}
    for ts in rest:
      self._task_goal_obs[ts.name] = ts.extract_training_goal(
        self.motion, self.motion_times, self.aux_state
      ).clone()

  def _get_policy_obs(self) -> TensorDict:
    sensor = self.representation.compute_actor_sensor(
      self.rep_state,
      self.num_actions,
      self.sim_ctx,
      aux_state=self.aux_state,
      last_action=self.last_actions,
      noisy=True,
    )
    critic = torch.cat([self.rep_state, self._wm_goal_obs, self.sim_ctx], dim=-1)
    return build_actor_obs_td(
      sensor, self._task_goal_obs, self.num_envs, self.device, critic=critic
    )
