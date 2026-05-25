import torch
import torch.nn as nn
from tensordict import TensorDict

from supermimic.learners.policy_learner import (
  BpttPolicyLearner,
  BpttPolicyLearnerCfg,
)


class _DummyActor(nn.Module):
  def __init__(self):
    super().__init__()
    self.weight = nn.Parameter(torch.zeros(()))

  def forward(self, obs: TensorDict) -> torch.Tensor:
    return obs["actor"].new_zeros(obs.batch_size[0], 1) + self.weight * 0.0


class _DummyRepresentation:
  dt = 1.0


class _TwoStepEnv:
  num_envs = 1
  num_actions = 1
  representation = _DummyRepresentation()

  def __init__(self):
    self.step_idx = 0

  def reset(self) -> TensorDict:
    self.step_idx = 0
    return self._obs()

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    del actions
    rewards = torch.ones(1)
    term_prob = torch.tensor([0.5 if self.step_idx == 0 else 0.0])
    self.step_idx += 1
    return self._obs(), rewards, term_prob, {}

  def _obs(self) -> TensorDict:
    return TensorDict({"actor": torch.zeros(1, 1)}, batch_size=[1])


class _InspectingLearner(BpttPolicyLearner):
  def _update_actor(self, obs: TensorDict) -> dict[str, float]:
    self._batch_return = self._compute_actor_return(obs)
    return {}


def test_survival_masking_counts_current_transition_reward_before_done():
  learner = _InspectingLearner(
    _DummyActor(),
    _TwoStepEnv(),
    BpttPolicyLearnerCfg(horizon=2, gamma=1.0, survival_masking=True),
  )

  learner._train_batch()

  assert torch.allclose(learner._batch_return, torch.tensor([1.5]))
