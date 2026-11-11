from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from supermimic.learners import policy_learner
from supermimic.learners.policy_learner import ActorCriticPolicyLearner


class RecordingCritic(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.zeros(1))
    self.seen = []

  def forward(self, x):
    if torch.is_grad_enabled():
      self.seen.append(x.detach().clone())
    return x * self.weight


@pytest.mark.parametrize("count", [1, 2, 3, 5, 7, 8])
def test_critic_uses_each_sample_once_per_epoch(count, monkeypatch):
  learner = ActorCriticPolicyLearner.__new__(ActorCriticPolicyLearner)
  learner.critic = RecordingCritic()
  learner.cfg = SimpleNamespace(gamma=0.9, lam=0.95)
  learner._v_max = 100.0
  learner.critic_optimizers = []
  monkeypatch.setattr(policy_learner, "_optimize_step", lambda *args, **kwargs: None)
  rollout = [
    TensorDict(
      {
        "critic": torch.arange(count).float().unsqueeze(1),
        "reward": torch.ones(count),
        "done": torch.zeros(count),
      },
      batch_size=[count],
    )
  ]
  stats = learner._fit_critic(rollout, rollout[0])
  assert torch.isfinite(stats["value_function"])
  seen = torch.cat(learner.critic.seen).reshape(5, count)
  for epoch in seen:
    torch.testing.assert_close(epoch.sort().values, torch.arange(count).float())
  assert all(batch.shape[0] > 0 for batch in learner.critic.seen)
