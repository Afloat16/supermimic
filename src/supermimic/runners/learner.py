from typing import Any

import torch
import torch.nn as nn


class PolicyLearner:
  """Trainer for Policy using Differentiable World Model."""

  def __init__(self, policy: nn.Module, world_model: nn.Module, cfg: Any):
    self.policy = policy
    self.wm = world_model
    self.cfg = cfg
    self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.cfg.policy_lr)

  def get_action(self, state, goal):
    with torch.no_grad():
      inputs = torch.cat([state, goal], dim=-1)
      return self.policy(inputs)

  def update(self, replay_buffer):
    self.policy.train()
    self.wm.eval()

    loader = replay_buffer.mini_batch_generator(
      self.cfg.horizon, self.cfg.num_batches_policy, self.cfg.batch_size_policy
    )

    epoch_reward = 0

    for states, _, goals, _, _ in loader:
      curr_state = states[:, 0, :]

      rewards = []

      for t in range(self.cfg.horizon):
        curr_goal = goals[:, t, :]
        inputs = torch.cat([curr_state, curr_goal], dim=-1)
        action = self.policy(inputs)
        noise = torch.randn_like(action) * self.cfg.policy_noise

        action = action + noise

        out = self.wm(curr_state, action, curr_goal)
        delta = out[..., :-1]
        pred_r = out[..., -1:]

        curr_state = curr_state + delta
        rewards.append(pred_r)

      total_reward = torch.stack(rewards).sum()
      loss = -total_reward
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      epoch_reward += total_reward.item()

    avg_reward = (
      epoch_reward
      / self.cfg.num_batches_policy
      / self.cfg.horizon
      / self.cfg.batch_size_policy
      / self.cfg.dt
    )
    print(f"[Policy] Reward: {avg_reward:.4f}")
    return {"Policy/reward": avg_reward}


class WorldModelLearner:
  """Trainer for World Model (Supervised)."""

  def __init__(self, model: nn.Module, cfg: Any):
    self.model = model
    self.cfg = cfg
    self.dyn_params = list(self.model.backbone.parameters()) + list(
      self.model.dyn_head.parameters()
    )
    self.rew_params = list(self.model.rew_head.parameters())

    self.opt_dyn = torch.optim.Adam(self.dyn_params, lr=self.cfg.model_lr)
    self.opt_rew = torch.optim.Adam(self.rew_params, lr=10 * self.cfg.model_lr)

    # Pre-calculate slice indices based on model input size
    self.dim = self.model.dyn_head[-1].out_features
    self.num_joints = (self.dim - 15) // 2
    self.slices = {
      "root_pos": slice(0, 3),
      "root_ori": slice(3, 9),
      "root_lin_vel": slice(9, 12),
      "root_ang_vel": slice(12, 15),
      "joint_pos": slice(15, 15 + self.num_joints),
      "joint_vel": slice(15 + self.num_joints, None),
    }

  def compute_stats(self, replay_buffer):
    print("[INFO] Computing normalization statistics...")
    # Flatten (Batch, Time, Dim) -> (N, Dim) directly
    states = replay_buffer.replay_buf[0][:, : replay_buffer.num_transitions].reshape(
      -1, self.dim
    )
    actions = replay_buffer.replay_buf[1][:, : replay_buffer.num_transitions].reshape(
      -1, replay_buffer.replay_buf[1].shape[-1]
    )
    goals = replay_buffer.replay_buf[2][:, : replay_buffer.num_transitions].reshape(
      -1, replay_buffer.replay_buf[2].shape[-1]
    )

    # Mask root position for stats
    states_masked = states.clone()
    states_masked[:, :2] = 0.0

    inputs = torch.cat([states_masked, actions, goals], dim=-1)
    self.model.input_mean.copy_(inputs.mean(dim=0))
    self.model.input_std.copy_(inputs.std(dim=0))
    print(f"[INFO] Stats -- Mean: {inputs.mean():.4f}, Std: {inputs.std():.4f}")

  def _get_loss_weights(self, device):
    """Creates a weight tensor matching state dimension (Broadcasting ready)."""
    W = torch.ones(self.dim, device=device)
    c = self.cfg
    weights = {
      "root_pos": 1.0 / c.nominal_error_root_pos**2,
      "root_ori": 1.0 / c.nominal_error_root_ori**2,
      "root_lin_vel": 1.0 / c.nominal_error_root_lin_vel**2,
      "root_ang_vel": 1.0 / c.nominal_error_root_ang_vel**2,
      "joint_pos": 1.0 / c.nominal_error_joint_pos**2,
      "joint_vel": 1.0 / c.nominal_error_joint_vel**2,
    }
    for name, sl in self.slices.items():
      W[sl] = weights[name]
    return W

  def update(self, replay_buffer):
    if replay_buffer.num_transitions < self.cfg.window_size:
      print("[WARN] Insufficient data.")
      return {}

    self.model.train()

    device = next(self.model.parameters()).device
    loss_weights = self._get_loss_weights(device)

    loader = replay_buffer.mini_batch_generator(
      self.cfg.window_size + 1, self.cfg.num_batches_model, self.cfg.batch_size_model
    )

    epoch_loss_total = 0
    epoch_loss_state = 0
    epoch_loss_reward = 0
    epoch_metrics = {k: 0.0 for k in self.slices.keys()}

    for states, actions, goals, rewards, _ in loader:
      # 1. Unroll Dynamics (Auto-regressive)
      curr_state = states[:, 0, :]
      preds = []
      pred_rewards = []

      for t in range(self.cfg.window_size):
        # Output is [State Delta, Reward]
        out = self.model(curr_state, actions[:, t, :], goals[:, t, :])
        delta = out[..., :-1]
        pred_r = out[..., -1:]

        curr_state = curr_state + delta
        preds.append(curr_state)
        pred_rewards.append(pred_r)

      # Stack to (Batch, Window, Dim)
      preds = torch.stack(preds, dim=1)
      pred_rewards = torch.stack(pred_rewards, dim=1)

      targets = states[:, 1:, :]
      target_rewards = rewards[:, : self.cfg.window_size, :]

      # 2. Vectorized Loss Calculation
      sq_err = (preds - targets) ** 2
      loss_state = (sq_err * loss_weights).mean()
      # Reward Loss (Scale by 1/dt^2 to match reward rate error)
      loss_reward = (pred_rewards - target_rewards).pow(2).mean() / (self.cfg.dt**2)

      loss = loss_state + loss_reward

      # 3. Backprop
      self.opt_dyn.zero_grad(set_to_none=True)
      self.opt_rew.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.dyn_params, max_norm=1.0)
      torch.nn.utils.clip_grad_norm_(self.rew_params, max_norm=1.0)
      self.opt_dyn.step()
      self.opt_rew.step()

      # 4. Logging
      epoch_loss_total += loss.item()
      epoch_loss_state += loss_state.item()
      epoch_loss_reward += loss_reward.item()

      with torch.no_grad():
        for name, sl in self.slices.items():
          epoch_metrics[name] += sq_err[..., sl].mean().item()

    # Averages
    avg_loss_total = epoch_loss_total / self.cfg.num_batches_model
    avg_loss_state = epoch_loss_state / self.cfg.num_batches_model
    avg_loss_reward = epoch_loss_reward / self.cfg.num_batches_model

    log_dict = {
      "Loss/total": avg_loss_total,
      "Loss/state": avg_loss_state,
      "Loss/reward": avg_loss_reward,
    }
    for k, v in epoch_metrics.items():
      log_dict[f"Loss/{k}"] = v / self.cfg.num_batches_model

    print(
      f"Model update |"
      f" Loss: {avg_loss_total:.6f} | State: {avg_loss_state:.6f} | Reward: {avg_loss_reward:.6f}"
    )

    return log_dict
