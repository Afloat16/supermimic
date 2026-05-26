"""Motion-tracking world model environment."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from supermimic.envs.world_model_env import WorldModelEnv


class MotionTrackingEnv(WorldModelEnv):
  """WM env for a single-actor policy. Actor obs comes from the reference clip."""

  def _get_policy_obs(self) -> TensorDict:
    actor = self.representation.build_actor_obs(
      self.rep_state,
      self._actor_goal_obs,
      self.num_actions,
      self.sim_ctx,
      aux_state=self.aux_state,
      last_action=self.last_actions,
      noisy=True,
    )
    critic = torch.cat([self.rep_state, self._wm_goal_obs, self.sim_ctx], dim=-1)
    return TensorDict(
      {"actor": actor, "critic": critic},
      batch_size=[self.num_envs],
      device=self.device,
    )
