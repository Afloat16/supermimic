"""Policy learners using a differentiable world model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.learners.learner_utils import _optimize_step


@dataclass
class BpttPolicyLearnerCfg:
  lr: float = 1e-3
  num_batches: int = 1
  batch_size: int = 4096
  noise: float = 0.1
  horizon: int = 32
  gamma: float = 1.0
  survival_masking: bool = True
  beta2: float = 0.95  # faster Adam EMA recovery after gradient spikes
  grad_clip_value: float = 0.1  # per-parameter gradient element clipping


@dataclass
class ActorCriticPolicyLearnerCfg(BpttPolicyLearnerCfg):
  gamma: float = 0.99
  lam: float = 0.95


class SoftDoneRolloutStorage(RolloutStorage):
  """Rollout storage that preserves soft done probabilities for GAE."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.dones = torch.zeros_like(self.rewards)
    # mini_batch_generator flattens these though add_transition skips them.
    self.distribution_params = (
      torch.zeros_like(self.actions),
      torch.zeros_like(self.actions),
    )

  def add_transition(self, transition: RolloutStorage.Transition) -> None:
    # Copy only the fields the GAE value fit reads.
    if self.step >= self.num_transitions_per_env:
      raise OverflowError("Rollout buffer overflow! Call clear() before adding.")
    self.observations[self.step].copy_(transition.observations)
    self.values[self.step].copy_(transition.values)
    self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
    self.dones[self.step].copy_(transition.dones.view(-1, 1))
    self.step += 1


class BpttPolicyLearner:
  """Policy Learner using direct Backpropagation Through Time (BPTT)."""

  def __init__(
    self,
    policy: nn.Module,
    wm_env: WorldModelEnv,
    cfg: BpttPolicyLearnerCfg,
    device: str = "cpu",
  ):
    self.actor = policy
    self.wm_env = wm_env
    self.cfg = cfg
    self.device = device
    self.actor_params = list(self.actor.parameters())
    self.actor_optimizer = torch.optim.Adam(
      self.actor_params, lr=self.cfg.lr, betas=(0.9, self.cfg.beta2)
    )

  def _prepare_update(self) -> None:
    self.actor.train()
    self.wm_env.dynamics.eval()
    self.wm_env.representation.eval()
    for p in self.wm_env.dynamics.parameters():
      p.requires_grad_(False)
    for p in self.wm_env.representation.parameters():
      p.requires_grad_(False)

  def _optimize_step(
    self, loss: torch.Tensor, max_norm: float = 1.0
  ) -> torch.Tensor | None:
    return _optimize_step(
      self.actor_optimizer,
      loss,
      self.actor_params,
      max_norm,
      self.cfg.grad_clip_value,
    )

  def _pre_rollout(self) -> None:
    n = self.wm_env.num_envs
    self._batch_return = torch.zeros(n, device=self.device)
    self._batch_aux_loss = torch.zeros(n, device=self.device)
    self._survival_rate = torch.ones(n, device=self.device)
    self._current_steps = torch.zeros(n, device=self.device)
    for p in self.actor.parameters():
      p.requires_grad_(True)

  def _on_step(
    self,
    obs: TensorDict,
    actions: torch.Tensor,
    next_obs: TensorDict,
    rewards: torch.Tensor,
    term_prob: torch.Tensor,
  ) -> None:
    """Accumulate the discounted return and actor auxiliary loss."""
    del obs, actions, next_obs, term_prob  # only rewards used in base
    # Actor return uses accumulated survival_rate * gamma^t
    discount = self.cfg.gamma**self._current_steps
    if self.cfg.survival_masking:
      discount = discount * self._survival_rate
    self._batch_return = self._batch_return + discount * rewards
    self._current_steps = self._current_steps + 1

    aux = getattr(self.actor, "auxiliary_loss", lambda: 0.0)()
    if isinstance(aux, torch.Tensor):
      self._batch_aux_loss = self._batch_aux_loss + aux

  def _compute_actor_return(self, obs: TensorDict) -> torch.Tensor:
    del obs
    return self._batch_return

  def _update_extras(self, obs: TensorDict) -> dict[str, float]:
    del obs
    return {}

  def _update_actor(self, obs: TensorDict) -> dict[str, float]:
    self._batch_return = self._compute_actor_return(obs)
    loss = (-self._batch_return + self._batch_aux_loss).mean()
    grad_norm = self._optimize_step(loss)
    extra_stats = self._update_extras(obs)
    extra_stats.update(getattr(self.actor, "auxiliary_stats", lambda: {})())
    return {"grad_norm": grad_norm, **extra_stats}

  def _train_batch(self) -> dict[str, float]:
    obs = self.wm_env.reset()
    self._pre_rollout()

    for _ in range(self.cfg.horizon):
      actions = self.actor(obs)
      next_obs, rewards, term_prob, _ = self.wm_env.step(actions)
      self._on_step(obs, actions, next_obs, rewards, term_prob)
      if self.cfg.survival_masking:
        self._survival_rate = self._survival_rate * (1 - term_prob)
      obs = next_obs

    return self._update_actor(obs)

  def update(self) -> dict[str, float]:
    self._prepare_update()

    epoch_return = 0.0
    epoch_extra_stats: dict[str, float] = {}

    for _ in range(self.cfg.num_batches):
      extra_stats = self._train_batch()

      epoch_return += self._batch_return.mean().item()

      for k, v in extra_stats.items():
        if k not in epoch_extra_stats:
          epoch_extra_stats[k] = 0.0
        epoch_extra_stats[k] += v

    mean_reward = (
      epoch_return
      / self.cfg.horizon
      / self.cfg.num_batches
      / self.wm_env.representation.dt
    )

    stats = {
      "Policy/reward": mean_reward,
    }

    # Add extra stats
    for k, v in epoch_extra_stats.items():
      stats[f"Policy/{k}"] = v / self.cfg.num_batches

    return stats


class ActorCriticPolicyLearner(BpttPolicyLearner):
  """Actor-critic variant that adds a critic bootstrap and GAE critic update."""

  def __init__(
    self,
    policy: nn.Module,
    critic: nn.Module,
    wm_env: WorldModelEnv,
    cfg: ActorCriticPolicyLearnerCfg,
    device: str = "cpu",
  ):
    super().__init__(policy, wm_env, cfg, device)
    self.critic = critic
    self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self.cfg.lr)

    self.rollout_storage = SoftDoneRolloutStorage(
      training_type="rl",
      num_envs=self.wm_env.num_envs,
      num_transitions_per_env=self.cfg.horizon,
      obs=self.wm_env.get_observations().select("critic"),
      actions_shape=[self.wm_env.num_actions],
      device=self.device,
    )
    self.transition = RolloutStorage.Transition()

  def _pre_rollout(self) -> None:
    super()._pre_rollout()
    for p in self.critic.parameters():
      p.requires_grad_(False)

  def _on_step(
    self,
    obs: TensorDict,
    actions: torch.Tensor,
    next_obs: TensorDict,
    rewards: torch.Tensor,
    term_prob: torch.Tensor,
  ) -> None:
    # Record the pre-step transition into the rollout storage for GAE.
    values = self.critic(obs)
    self.transition.values = values.detach().clone()
    self.transition.observations = TensorDict(
      {"critic": obs["critic"].detach().clone()},
      batch_size=[actions.shape[0]],
      device=actions.device,
    )
    # GAE uses per-step term_prob as effective discount (1 - term_prob) * gamma
    self.transition.rewards = rewards.detach().clone()
    self.transition.dones = term_prob.detach().clone()
    self.rollout_storage.add_transition(self.transition)
    self.transition.clear()

    self._last_term_prob = term_prob

    # Shared discounted return + actor auxiliary loss accumulation.
    super()._on_step(obs, actions, next_obs, rewards, term_prob)

  def _compute_actor_return(self, obs: TensorDict) -> torch.Tensor:
    bootstrap = (self.cfg.gamma**self._current_steps).view(-1, 1) * self.critic(obs)
    if self.cfg.survival_masking:
      bootstrap = bootstrap * self._survival_rate.view(-1, 1)
    else:
      bootstrap = bootstrap * (1.0 - self._last_term_prob).view(-1, 1)
    return self._batch_return + bootstrap.squeeze(-1)

  def _update_extras(self, obs: TensorDict) -> dict[str, float]:
    # Critic Update
    self._compute_gae_returns(obs)
    for p in self.actor.parameters():
      p.requires_grad_(False)
    for p in self.critic.parameters():
      p.requires_grad_(True)

    total_value_loss = 0.0
    num_updates = 0
    for batch in self.rollout_storage.mini_batch_generator(4, 5):
      num_updates += 1
      value_batch = self.critic(batch.observations)

      value_loss = (batch.returns - value_batch).pow(2).mean()

      _optimize_step(
        self.critic_optimizer, value_loss, self.critic.parameters(), max_norm=1.0
      )
      total_value_loss += value_loss.item()
    self.rollout_storage.clear()

    return {"value_function": total_value_loss / num_updates}

  def _compute_gae_returns(self, critic_input: TensorDict) -> None:
    st = self.rollout_storage
    # Compute value for the last step
    last_values = self.critic(critic_input).detach()
    # Compute returns and advantages
    advantage = 0
    for step in reversed(range(st.num_transitions_per_env)):
      # If we are at the last step, bootstrap the return value
      next_values = (
        last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
      )
      # 1 if we are not in a terminal state, 0 otherwise
      next_is_not_terminal = 1.0 - st.dones[step]
      # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
      delta = (
        st.rewards[step]
        + next_is_not_terminal * self.cfg.gamma * next_values
        - st.values[step]
      )
      # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
      advantage = (
        delta + next_is_not_terminal * self.cfg.gamma * self.cfg.lam * advantage
      )
      # Return: R_t = A(s_t, a_t) + V(s_t)
      st.returns[step] = advantage + st.values[step]
