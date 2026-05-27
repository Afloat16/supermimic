import torch
import torch.nn as nn
from tensordict import TensorDict

from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
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


class _ActorCriticTwoStepEnv(_TwoStepEnv):
  def get_observations(self) -> TensorDict:
    return self._obs()

  def _obs(self) -> TensorDict:
    return TensorDict(
      {"actor": torch.zeros(1, 1), "critic": torch.zeros(1, 1)}, batch_size=[1]
    )


class _InspectingLearner(BpttPolicyLearner):
  def _update_actor(self, obs: TensorDict) -> dict[str, float]:
    self._batch_return = self._compute_actor_return(obs)
    return {}


class _ConstantCritic(nn.Module):
  def __init__(self):
    super().__init__()
    self.value = nn.Parameter(torch.zeros(()))

  def forward(self, obs: TensorDict) -> torch.Tensor:
    return obs["critic"].new_ones(obs.batch_size[0], 1) * self.value


def test_survival_masking_counts_current_transition_reward_before_done():
  learner = _InspectingLearner(
    _DummyActor(),
    _TwoStepEnv(),
    BpttPolicyLearnerCfg(horizon=2, gamma=1.0, survival_masking=True),
  )

  learner._train_batch()

  assert torch.allclose(learner._batch_return, torch.tensor([1.5]))


def test_actor_critic_gae_uses_soft_term_probs():
  env = _ActorCriticTwoStepEnv()
  learner = ActorCriticPolicyLearner(
    _DummyActor(),
    _ConstantCritic(),
    env,
    ActorCriticPolicyLearnerCfg(horizon=2, gamma=1.0, lam=1.0, survival_masking=True),
  )

  obs = env.reset()
  learner._pre_rollout()
  for _ in range(2):
    actions = torch.zeros(1, 1)
    next_obs, rewards, term_prob, _ = env.step(actions)
    learner._on_step(obs, actions, next_obs, rewards, term_prob)
    obs = next_obs

  learner._compute_gae_returns(obs)

  assert learner.rollout_storage.dones.dtype == torch.float32
  assert torch.allclose(
    learner.rollout_storage.dones[:, 0, 0], torch.tensor([0.5, 0.0])
  )
  assert torch.allclose(
    learner.rollout_storage.returns[:, 0, 0], torch.tensor([1.5, 1.0])
  )
