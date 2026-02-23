from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from supermimic.utils.helpers import shift_history

from .world_model_env import WorldModelEnv


def load_wandb_checkpoint(run_path: str, task_id: str, model, device):
  """Download and load a world model checkpoint from W&B."""
  print(f"[INFO] Loading World Model from W&B: {run_path}")

  run_id = run_path.split("/")[-1]
  download_dir = Path("logs") / task_id / "wandb_checkpoints" / run_id
  checkpoint_path = download_dir / "world_model.pt"

  if not checkpoint_path.exists():
    print(f"[INFO] Downloading world_model.pt from {run_path}...")
    api = wandb.Api()
    run = api.run(run_path)
    run.file("world_model.pt").download(root=str(download_dir), replace=True)

  state_dict = torch.load(checkpoint_path, map_location=device)
  model.load_state_dict(state_dict)
  print(f"[INFO] World Model loaded from {checkpoint_path}")


def _optimize_step(optimizer, loss, params=None, max_norm=None):
  """Run one optimizer step: zero_grad → backward → optional clip → step."""
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  grad_norm = None
  if params is not None and max_norm is not None:
    grad_norm = torch.nn.utils.clip_grad_norm_(
      params, max_norm=max_norm, error_if_nonfinite=True
    )
  optimizer.step()
  return grad_norm


@dataclass
class PolicyLearnerCfg:
  lr: float = 1e-3
  num_batches: int = 1
  batch_size: int = 2048
  noise: float = 0.1
  horizon: int = 64
  motion_file: str | list[str] | None = None
  clip_param: float = 0.2
  gamma: float = 0.99
  lam: float = 0.95
  use_clipped_value_loss: bool = True
  auto_reset: bool = True


class BasePolicyLearner:
  """Base class for Policy Learners using Differentiable World Model."""

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

    self._batch_return = None
    self._num_trajectories = None

  def _prepare_update(self):
    self.actor.train()
    self.wm_env.wm.eval()
    for p in self.wm_env.wm.parameters():
      p.requires_grad_(False)

  def _reset_and_detach_env(self):
    self.wm_env.detach()
    return self.wm_env.reset()

  def _optimize_step(self, loss, max_norm=10.0):
    return _optimize_step(self.actor_optimizer, loss, self.actor_params, max_norm)

  def act(self, obs):
    return self.actor(obs), None

  def _prepare_batch(self):
    self._batch_return = 0
    self._num_trajectories = self.cfg.batch_size
    for p in self.actor.parameters():
      p.requires_grad_(True)

  def _on_step(self, obs, rewards, dones, extras, actions, values):
    raise NotImplementedError

  def _compute_actor_loss(self, obs):
    raise NotImplementedError

  def _update_extras(self, obs):
    return {}

  def _update_actor(self, obs):
    loss = self._compute_actor_loss(obs)
    grad_norm = self._optimize_step(loss)
    extra_stats = self._update_extras(obs)
    return {"grad_norm": grad_norm, **extra_stats}

  def train_batch(self):
    obs = self._reset_and_detach_env()
    self._prepare_batch()

    for _ in range(self.cfg.horizon):
      actions, values = self.act(obs)
      obs, rewards, dones, extras = self.wm_env.step(actions, self.cfg.auto_reset)

      self._on_step(obs, rewards, dones, extras, actions, values)

      if self.cfg.auto_reset:
        self._num_trajectories = self._num_trajectories + dones.sum()

    return self._update_actor(obs)

  def update(self):
    self._prepare_update()

    epoch_return = 0
    epoch_extra_stats = {}

    for _ in range(self.cfg.num_batches):
      extra_stats = self.train_batch()

      epoch_return += (self._batch_return / self._num_trajectories).item()

      for k, v in extra_stats.items():
        if k not in epoch_extra_stats:
          epoch_extra_stats[k] = 0
        epoch_extra_stats[k] += v

    mean_reward = epoch_return / self.cfg.horizon / self.cfg.num_batches / self.dt

    stats = {
      "Policy/reward": mean_reward,
    }

    # Add extra stats
    for k, v in epoch_extra_stats.items():
      stats[f"Policy/{k}"] = v / self.cfg.num_batches

    return stats


class BpttPolicyLearner(BasePolicyLearner):
  """Policy Learner using direct Backpropagation Through Time (BPTT)."""

  def _on_step(self, obs, rewards, dones, extras, actions, values):
    self.actor.update_normalization(obs.detach())

    self._batch_return += rewards.sum()

  def _compute_actor_loss(self, obs):
    return -self._batch_return / self._num_trajectories


class ActorCriticPolicyLearner(BasePolicyLearner):
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

    obs = self.wm_env.get_observations()
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

  def act(self, obs):
    actions = self.actor(obs)
    values = self.critic(obs)

    self.transition.actions = actions.detach().clone()
    self.transition.values = values.detach().clone()
    self.transition.actions_log_prob = torch.zeros_like(self.transition.values)
    self.transition.action_mean = torch.zeros_like(actions)
    self.transition.action_sigma = torch.zeros_like(actions)
    self.transition.observations = TensorDict(
      {k: v.detach().clone() for k, v in obs.items()},
      batch_size=[actions.shape[0]],
      device=actions.device,
    )
    return actions, values

  def _prepare_batch(self):
    super()._prepare_batch()
    for p in self.critic.parameters():
      p.requires_grad_(False)
    self._current_steps = torch.zeros(self.wm_env.num_envs, device=self.device)

  def _on_step(self, obs, rewards, dones, extras, actions, values):
    self.actor.update_normalization(obs.detach())
    self.critic.update_normalization(obs.detach())

    self.transition.rewards = rewards.detach().clone()
    self.transition.dones = dones.detach().clone()

    if "time_outs" in extras:
      self.transition.rewards += self.cfg.gamma * torch.squeeze(
        self.transition.values * extras["time_outs"].unsqueeze(1), 1
      )
    self.rollout_storage.add_transition(self.transition)
    self.transition.clear()

    rewards_for_return = rewards.clone()
    if "time_outs" in extras:
      rewards_for_return += self.cfg.gamma * torch.squeeze(
        values * extras["time_outs"].unsqueeze(1), 1
      )

    self._batch_return += (
      self.cfg.gamma**self._current_steps * rewards_for_return
    ).sum()

    self._current_steps += 1
    self._current_steps[dones.bool()] = 0

    self._last_dones = dones
    self._last_extras = extras

  def _compute_actor_loss(self, obs):
    bootstrap = (self.cfg.gamma**self._current_steps).view(-1, 1) * self.critic(obs)
    bootstrap_mask = 1.0 - self._last_dones
    if "time_outs" in self._last_extras:
      bootstrap_mask += self._last_dones * self._last_extras["time_outs"]
    bootstrap *= bootstrap_mask.view(-1, 1)
    loss = -(self._batch_return + bootstrap.sum()) / self._num_trajectories
    return loss

  def _update_extras(self, obs):
    # Critic Update
    self.compute_returns(obs)
    for param in self.actor.parameters():
      param.requires_grad_(False)
    for param in self.critic.parameters():
      param.requires_grad_(True)
    batch_values_loss = 0
    num_value_updates = 0

    generator = self.rollout_storage.mini_batch_generator(4, 5)
    for batch in generator:
      obs_batch, _, target_values_batch, _, returns_batch, *rest = batch
      num_value_updates += 1

      value_batch = self.critic(obs_batch)

      # Value function loss
      if self.cfg.use_clipped_value_loss:
        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
          -self.cfg.clip_param, self.cfg.clip_param
        )
        value_losses = (value_batch - returns_batch).pow(2)
        value_losses_clipped = (value_clipped - returns_batch).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (returns_batch - value_batch).pow(2).mean()

      _optimize_step(self.critic_optimizer, value_loss)
      batch_values_loss += value_loss.item()
    self.rollout_storage.clear()

    return {
      "value_function": batch_values_loss / num_value_updates,
    }

  def compute_returns(self, critic_input) -> None:
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


@dataclass
class WorldModelLearnerCfg:
  lr: float = 1e-3
  num_batches: int = 20
  batch_size: int = 2048
  horizon: int = 8
  history: int = 1

  # Loss Weights (Bryson's Rule) — body-frame representation
  nominal_error_root_pos_z: float = 0.1
  nominal_error_root_gravity_b: float = 0.1
  nominal_error_root_lin_vel: float = 1.0
  nominal_error_root_ang_vel: float = 3.14
  nominal_error_joint_pos: float = 0.1
  nominal_error_joint_vel: float = 3.14

  # Loss Weights — maxcoord representation
  nominal_error_body_pos: float = 0.05
  nominal_error_body_rot: float = 0.1
  nominal_error_body_vel: float = 1.0
  nominal_error_body_avel: float = 3.14
  nominal_error_body_height: float = 0.1
  nominal_error_up_dir: float = 0.1


class WorldModelLearner:
  """Trainer for World Model (Supervised)."""

  def __init__(
    self,
    model,
    cfg: WorldModelLearnerCfg,
    dt: float,
    device: str = "cpu",
    reward_names: list[str] | None = None,
  ):
    self.model = model
    self.cfg = cfg
    self.device = device
    self.dt = dt
    self.history = cfg.history
    self.reward_names = reward_names or ["reward"]
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    representation = self.model.representation
    self.body_state_dim = representation.rep_state_dim
    self.slices = representation.loss_slices()
    self.loss_weights = representation.loss_weights(cfg, device)
    self._predict_reward = representation.has_learned_reward

    self._compiled_rollout = torch.compile(self._rollout, mode="reduce-overhead")
    self._compiled_compute_losses = torch.compile(self._compute_losses, mode="default")

  def update_normalization(self, states, actions, goals):
    representation = self.model.representation
    body_states = representation.represent(states)
    body_states_flat = body_states.reshape(-1, body_states.shape[-1])
    body_states_flat = body_states_flat.repeat(1, self.history)

    actions_flat = actions.reshape(-1, actions.shape[-1])
    # Use actions twice for last action
    inputs = torch.cat(
      [body_states_flat, actions_flat, actions, goals],
      dim=-1,
    )

    # The policy learner's _prepare_update() sets wm.eval(), which puts the
    # normalizer in eval mode. Toggle to train so update() is not a no-op.
    normalizer = self.model.input_normalizer
    normalizer.train()
    normalizer.update(inputs)
    normalizer.eval()

  def update(self, data):
    self.model.train()
    for p in self.model.parameters():
      p.requires_grad_(True)

    device = next(self.model.parameters()).device
    representation = self.model.representation

    epoch_loss_total = torch.zeros(1, device=device)
    epoch_loss_state = torch.zeros(1, device=device)
    epoch_loss_reward = torch.zeros(1, device=device)
    epoch_loss_term = torch.zeros(1, device=device)
    _metric_slice_list = list(self.slices.items())
    epoch_metrics_gpu = torch.zeros(len(_metric_slice_list), device=device)
    epoch_reward_mse = torch.zeros(len(self.reward_names), device=device)

    for full_states, actions, goals, rewards, _, dones in data:
      # history=1: start_idx is always 0 (avoid GPU randint + .item() sync).
      if self.cfg.history == 1:
        start_idx = 0
      else:
        start_idx = int(torch.randint(0, self.cfg.history, (1,)).item())
      full_len = full_states.shape[1]

      all_body_states = representation.represent(full_states)

      # Build initial history tensor (padded if needed)
      hist_context = all_body_states[:, : start_idx + 1]
      if hist_context.shape[1] < self.history:
        pad_len = self.history - hist_context.shape[1]
        pad = hist_context[:, 0:1, :].expand(-1, pad_len, -1)
        init_hist = torch.cat([pad, hist_context], dim=1)
      else:
        init_hist = hist_context[:, -self.history :, :]

      # Slice actions/goals for this rollout window
      roll_actions = actions[:, start_idx : start_idx + self.cfg.horizon, :]
      roll_goals = goals[:, start_idx : start_idx + self.cfg.horizon, :]

      init_full = full_states[:, start_idx, :]
      pred_states_all, pred_rewards_all, pred_dones_all = self._compiled_rollout(
        init_hist,
        roll_actions,
        roll_goals,
        init_full,
      )

      # Only supervise states where GT exists
      state_steps = min(self.cfg.horizon, full_len - start_idx - 1)
      pred_states = pred_states_all[:, :state_steps, :]

      gt_states = all_body_states[:, start_idx + 1 : start_idx + 1 + state_steps, :]
      gt_rewards = rewards[:, start_idx : start_idx + self.cfg.horizon]
      gt_dones = dones[:, start_idx : start_idx + self.cfg.horizon]

      # 2. Vectorized Loss Calculation (always FP32, compiled for kernel fusion)
      loss, loss_state, loss_reward, loss_term, sq_err, reward_mse = (
        self._compiled_compute_losses(
          pred_states,
          pred_rewards_all,
          pred_dones_all,
          gt_states,
          gt_rewards,
          gt_dones,
        )
      )

      # 3. Backprop
      _optimize_step(self.optimizer, loss, self.model.parameters(), max_norm=1.0)

      # 4. Logging (GPU accumulation — no sync here, .item() deferred to after loop)
      epoch_loss_total += loss.detach()
      epoch_loss_state += loss_state.detach()
      epoch_loss_reward += loss_reward.detach()
      epoch_loss_term += loss_term.detach()

      with torch.no_grad():
        for i, (_, sl) in enumerate(_metric_slice_list):
          epoch_metrics_gpu[i] += sq_err[..., sl].mean()

        if reward_mse.shape[0] == len(self.reward_names):
          epoch_reward_mse += reward_mse.detach()

    # Averages — single .item() sync per scalar after all batches are done
    n = self.cfg.num_batches
    log_dict = {
      "Loss/total": epoch_loss_total.item() / n,
      "Loss/state": epoch_loss_state.item() / n,
      "Loss/reward": epoch_loss_reward.item() / n,
      "Loss/termination": epoch_loss_term.item() / n,
    }
    avg_metrics_gpu = epoch_metrics_gpu / n
    for i, (name, _) in enumerate(_metric_slice_list):
      log_dict[f"Loss/{name}"] = avg_metrics_gpu[i].item()

    avg_reward_mse = epoch_reward_mse / self.cfg.num_batches
    for i, name in enumerate(self.reward_names):
      log_dict[f"Loss/reward_{name}"] = avg_reward_mse[i].item()

    return log_dict

  def _rollout(
    self,
    init_hist: torch.Tensor,
    actions: torch.Tensor,
    goals: torch.Tensor,
    init_full_state: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Autoregressive rollout for `horizon` steps using the world model."""
    horizon = actions.shape[1]
    curr_body_hist = init_hist
    curr_full = init_full_state

    pred_states: list[torch.Tensor] = []
    pred_rewards: list[torch.Tensor] = []
    pred_dones: list[torch.Tensor] = []

    for k in range(horizon):
      curr_a = actions[:, k, :]
      last_a = actions[:, k - 1, :] if k > 0 else torch.zeros_like(curr_a)
      curr_g = goals[:, k, :]

      curr_full, next_body_state, pred_r, pred_d = self.model.transition(
        curr_body_hist, curr_a, last_a, curr_g, curr_full, True
      )

      curr_body_hist = shift_history(curr_body_hist, next_body_state)

      pred_states.append(next_body_state)
      pred_rewards.append(pred_r)
      pred_dones.append(pred_d)

    return (
      torch.stack(pred_states, dim=1),
      torch.stack(pred_rewards, dim=1),
      torch.stack(pred_dones, dim=1),
    )

  def _compute_losses(
    self,
    pred_states,
    pred_rewards,
    pred_dones,
    gt_states,
    gt_rewards,
    gt_dones,
  ):
    # Mean Squared Error on State
    sq_err = (pred_states - gt_states) ** 2

    # Weighted Mean on body-state channels.
    loss_state = (sq_err * self.loss_weights).mean()

    # Reward Loss (skip when reward is computed analytically, not learned)
    if self._predict_reward:
      reward_mse = (pred_rewards - gt_rewards).pow(2)
      reward_mse_per_id = reward_mse.mean(dim=(0, 1))
      loss_reward = reward_mse_per_id.sum()
    else:
      loss_reward = torch.zeros(1, device=pred_states.device).squeeze()
      reward_mse_per_id = torch.zeros(pred_rewards.shape[-1], device=pred_states.device)

    # Termination Loss — float masks instead of bool indexing (avoids CPU-GPU sync).
    # Timeouts (gt_dones==2) masked out; terminations (gt_dones==1) are positives.
    valid_mask = (gt_dones != 2.0).float()
    gt_d_binary = (gt_dones == 1.0).float()

    num_pos = gt_d_binary.sum()
    num_valid = valid_mask.sum()
    pos_weight = torch.clamp((num_valid - num_pos) / (num_pos + 1e-6), max=50.0)

    ew = valid_mask * (1.0 + (pos_weight - 1.0) * gt_d_binary)
    bce_raw = nn.functional.binary_cross_entropy_with_logits(
      pred_dones, gt_d_binary, reduction="none"
    )
    loss_term = (bce_raw * ew).sum() / (num_valid + 1e-6)

    loss = loss_state + loss_reward + loss_term

    return loss, loss_state, loss_reward, loss_term, sq_err, reward_mse_per_id
