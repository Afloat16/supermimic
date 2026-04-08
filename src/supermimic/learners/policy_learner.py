"""Policy learners using a differentiable world model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.learners.learner_utils import _optimize_step


@dataclass
class PolicyLearnerCfg:
  lr: float = 1e-3
  num_batches: int = 1
  batch_size: int = 2048
  noise: float = 0.1
  horizon: int = 32
  motion_file: str | list[str] | None = None
  clip_param: float = 0.2
  gamma: float = 0.99
  lam: float = 0.95
  use_clipped_value_loss: bool = True
  survival_masking: bool = False


class BpttPolicyLearner:
  """Policy Learner using direct Backpropagation Through Time (BPTT)."""

  def __init__(
    self,
    policy: nn.Module,
    wm_env: WorldModelEnv,
    cfg: PolicyLearnerCfg,
    dt: float,
    device: str = "cpu",
  ):
    self.actor = policy
    self.wm_env = wm_env
    self.cfg = cfg
    self.device = device
    self.dt = dt
    self.actor_params = list(self.actor.parameters())
    self.actor_optimizer = torch.optim.Adam(self.actor_params, lr=self.cfg.lr)

  def _prepare_update(self) -> None:
    self.actor.train()
    self.wm_env.wm.eval()
    for p in self.wm_env.wm.parameters():
      p.requires_grad_(False)

  def _optimize_step(
    self, loss: torch.Tensor, max_norm: float = 1.0
  ) -> torch.Tensor | None:
    return _optimize_step(self.actor_optimizer, loss, self.actor_params, max_norm)

  def act(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor | None]:
    return self.actor(obs), None

  def _prepare_batch(self) -> None:
    self._batch_return = torch.zeros(self.wm_env.num_envs, device=self.device)
    self._batch_aux_loss = torch.zeros(self.wm_env.num_envs, device=self.device)
    self._survival_rate = torch.ones(self.wm_env.num_envs, device=self.device)
    for p in self.actor.parameters():
      p.requires_grad_(True)

  def _on_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, Any],
    actions: torch.Tensor,
    values: torch.Tensor | None,
  ) -> None:
    if self.cfg.survival_masking:
      self._batch_return += rewards * self._survival_rate
    else:
      self._batch_return += rewards

    aux = getattr(self.actor, "auxiliary_loss", lambda: 0.0)()
    if isinstance(aux, torch.Tensor):
      self._batch_aux_loss = self._batch_aux_loss + aux

  def _compute_return(self, obs: TensorDict) -> torch.Tensor:
    return self._batch_return

  def _update_extras(self, obs: TensorDict) -> dict[str, float]:
    return {}

  def _update_actor(self, obs: TensorDict) -> dict[str, float]:
    self._batch_return = self._compute_return(obs)
    loss = (-self._batch_return + self._batch_aux_loss).mean()
    grad_norm = self._optimize_step(loss)
    extra_stats = self._update_extras(obs)
    extra_stats.update(getattr(self.actor, "auxiliary_stats", lambda: {})())
    return {"grad_norm": grad_norm, **extra_stats}

  def train_batch(self) -> dict[str, float]:
    obs = self.wm_env.reset()
    self._prepare_batch()

    for _ in range(self.cfg.horizon):
      actions, values = self.act(obs)
      obs, rewards, dones, extras = self.wm_env.step(actions)

      if self.cfg.survival_masking:
        self._survival_rate = self._survival_rate * (1 - dones)

      self.actor.update_normalization(obs.detach())
      self._on_step(obs, rewards, dones, extras, actions, values)

    return self._update_actor(obs)

  def update(self) -> dict[str, float]:
    self._prepare_update()

    epoch_return = 0.0
    epoch_extra_stats = {}

    for _ in range(self.cfg.num_batches):
      extra_stats = self.train_batch()

      epoch_return += self._batch_return.mean().item()

      for k, v in extra_stats.items():
        if k not in epoch_extra_stats:
          epoch_extra_stats[k] = 0.0
        epoch_extra_stats[k] += v

    mean_reward = epoch_return / self.cfg.horizon / self.cfg.num_batches / self.dt

    stats = {
      "Policy/reward": mean_reward,
    }

    # Add extra stats
    for k, v in epoch_extra_stats.items():
      stats[f"Policy/{k}"] = v / self.cfg.num_batches

    return stats


class ActorCriticPolicyLearner(BpttPolicyLearner):
  """Policy Learner using Actor-Critic with RolloutStorage."""

  def __init__(
    self,
    policy: nn.Module,
    critic: nn.Module,
    wm_env: WorldModelEnv,
    cfg: PolicyLearnerCfg,
    dt: float,
    device: str = "cpu",
  ):
    super().__init__(policy, wm_env, cfg, dt, device)
    self.critic = critic
    self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self.cfg.lr)

    obs = self.wm_env.reset()
    obs_td = TensorDict(obs, batch_size=[self.wm_env.num_envs], device=self.device)
    self.rollout_storage = RolloutStorage(
      training_type="rl",
      num_envs=self.wm_env.num_envs,
      num_transitions_per_env=self.cfg.horizon,
      obs=obs_td,
      actions_shape=[self.wm_env.num_actions],
      device=self.device,
    )
    self.transition = RolloutStorage.Transition()

  def act(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor | None]:
    actions = self.actor(obs)
    values = self.critic(obs)

    self.transition.actions = actions.detach().clone()
    self.transition.values = values.detach().clone()
    self.transition.actions_log_prob = torch.zeros_like(self.transition.values)
    self.transition.distribution_params = (
      torch.zeros_like(actions),
      torch.zeros_like(actions),
    )
    self.transition.observations = TensorDict(
      {k: v.detach().clone() for k, v in obs.items()},
      batch_size=[actions.shape[0]],
      device=actions.device,
    )
    return actions, values

  def _prepare_batch(self) -> None:
    super()._prepare_batch()
    for p in self.critic.parameters():
      p.requires_grad_(False)
    self._current_steps = torch.zeros(self.wm_env.num_envs, device=self.device)

  def _on_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, Any],
    actions: torch.Tensor,
    values: torch.Tensor | None,
  ) -> None:
    self.critic.update_normalization(obs.detach())

    # GAE uses raw per-step dones for effective discount (1 - done) * gamma
    self.transition.rewards = rewards.detach().clone()
    self.transition.dones = dones.detach().clone()
    self.rollout_storage.add_transition(self.transition)
    self.transition.clear()

    # Actor return uses accumulated survival_rate * gamma^t
    discount = self.cfg.gamma**self._current_steps
    if self.cfg.survival_masking:
      discount = discount * self._survival_rate

    self._batch_return += discount * rewards

    self._current_steps += 1

    self._last_dones = dones

  def _compute_return(self, obs: TensorDict) -> torch.Tensor:
    bootstrap = (self.cfg.gamma**self._current_steps).view(-1, 1) * self.critic(obs)
    if self.cfg.survival_masking:
      bootstrap *= self._survival_rate.view(-1, 1)
    else:
      bootstrap *= (1.0 - self._last_dones).view(-1, 1)
    return self._batch_return + bootstrap.squeeze(-1)

  def _update_extras(self, obs: TensorDict) -> dict[str, float]:
    # Critic Update
    self.compute_returns(obs)
    for param in self.actor.parameters():
      param.requires_grad_(False)
    for param in self.critic.parameters():
      param.requires_grad_(True)
    batch_values_loss = 0.0
    num_value_updates = 0

    generator = self.rollout_storage.mini_batch_generator(4, 5)
    for batch in generator:
      num_value_updates += 1

      value_batch = self.critic(batch.observations)

      # Value function loss
      if self.cfg.use_clipped_value_loss:
        value_clipped = batch.values + (value_batch - batch.values).clamp(
          -self.cfg.clip_param, self.cfg.clip_param
        )
        value_losses = (value_batch - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - value_batch).pow(2).mean()

      _optimize_step(self.critic_optimizer, value_loss)
      batch_values_loss += value_loss.item()
    self.rollout_storage.clear()

    return {
      "value_function": batch_values_loss / num_value_updates,
    }

  def compute_returns(self, critic_input: TensorDict) -> None:
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
      next_is_not_terminal = 1.0 - st.dones[step].float()
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
