from dataclasses import dataclass

import torch
import torch.nn as nn
from mjlab.utils.lab_api.math import wrap_to_pi

from supermimic.utils.helpers import (
  YawAlignedStateLayout,
  apply_yaw_correction,
  compute_yaw_aligned_state,
  construct_policy_input,
)
from supermimic.utils.math import get_yaw_from_quat

from .world_model_env import WorldModelEnv


@dataclass
class PolicyLearnerCfg:
  lr: float = 1e-4
  num_batches: int = 5
  batch_size: int = 2048
  noise: float = 0.1
  horizon: int = 64
  motion_file: str | None = None
  die_penalty: float = -0.0


class PolicyLearner:
  """Trainer for Policy using Differentiable World Model."""

  def __init__(
    self,
    policy: nn.Module,
    world_model: nn.Module,
    cfg: PolicyLearnerCfg,
    env,
    dt: float,
    layout: YawAlignedStateLayout,
    device: str = "cpu",
  ):
    self.policy = policy
    self.wm = world_model
    self.cfg = cfg
    self.env = env
    self.device = device
    self.dt = dt
    self.layout = layout
    self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.cfg.lr)

    self.wm_env = WorldModelEnv(
      world_model=self.wm,
      motion_file=cfg.motion_file,
      env=self.env,
      batch_size=self.cfg.batch_size,
      horizon=self.cfg.horizon,
      layout=self.layout,
      device=self.device,
    )

  def compute_stats(self, replay_buffer):
    if not self.policy.normalization:
      return

    # Flatten (Batch, Time, Dim) -> Windows
    states = replay_buffer.replay_buf[0][:, : replay_buffer.num_transitions]
    goals = replay_buffer.replay_buf[2][:, : replay_buffer.num_transitions]

    states_flat = states.reshape(-1, states.shape[-1])
    goals_flat = goals.reshape(-1, goals.shape[-1])
    policy_goals_flat = torch.cat([goals_flat[..., :-9], goals_flat[..., -6:]], dim=-1)
    inputs = construct_policy_input(states_flat, policy_goals_flat, self.layout)

    self.policy.actor_obs_normalizer._mean.copy_(inputs.mean(dim=0))
    self.policy.actor_obs_normalizer._std.copy_(inputs.std(dim=0))

  def update(self):
    self.policy.train()
    self.wm.eval()

    epoch_reward = 0
    # Hook to capture World Model output for differentiable loss
    wm_outputs = []

    def hook(module, args, output):
      wm_outputs.append(output)

    handle = self.wm.register_forward_hook(hook)

    for _ in range(self.cfg.num_batches):
      wm_outputs.clear()
      # Reset environment at start of batch to get fresh detached states
      # This breaks the computation graph from the previous batch
      self.wm_env.detach()
      obs = self.wm_env.reset()

      rewards = []
      curr_survival = torch.ones((self.cfg.batch_size, 1), device=self.device)

      for t in range(self.cfg.horizon):
        policy_input = obs["policy"]
        action = self.policy(policy_input)

        obs, reward, _, _ = self.wm_env.step(action)

        pred_d = wm_outputs[-1][..., -1:]
        prev_survival = curr_survival
        survival_rate = torch.sigmoid(-pred_d)
        curr_survival = prev_survival * survival_rate

        step_reward = reward.unsqueeze(-1) * prev_survival
        step_reward = step_reward + self.cfg.die_penalty * (1 - curr_survival)
        rewards.append(step_reward)

      total_reward = torch.stack(rewards).sum()
      loss = -total_reward
      self.optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
      self.optimizer.step()

      epoch_reward += total_reward.item()

    handle.remove()

    avg_reward = (
      epoch_reward
      / self.cfg.num_batches
      / self.cfg.horizon
      / self.cfg.batch_size
      / self.dt
    )
    return {"Policy/reward": avg_reward}


@dataclass
class WorldModelLearnerCfg:
  lr: float = 1e-4
  num_batches: int = 100
  batch_size: int = 2048
  horizon: int = 8
  history: int = 1

  # Loss Weights (Bryson's Rule)
  nominal_error_root_pos_z: float = 0.1
  nominal_error_root_gravity_b: float = 0.1
  nominal_error_root_lin_vel: float = 1.0
  nominal_error_root_ang_vel: float = 3.14
  nominal_error_joint_pos: float = 0.1
  nominal_error_joint_vel: float = 3.14
  # Consistency terms
  nominal_error_root_pos_xy: float = 1e10
  nominal_error_root_yaw: float = 1e10


class WorldModelLearner:
  """Trainer for World Model (Supervised)."""

  def __init__(
    self,
    model: nn.Module,
    cfg: WorldModelLearnerCfg,
    dt: float,
    layout: YawAlignedStateLayout,
    device: str = "cpu",
    reward_names: list[str] | None = None,
  ):
    self.model = model
    self.cfg = cfg
    self.device = device
    self.dt = dt
    self.layout = layout
    self.history = cfg.history
    self.reward_names = reward_names or ["reward"]
    self.dyn_params = list(self.model.backbone.parameters()) + list(
      self.model.dyn_head.parameters()
    )
    self.rew_params = list(self.model.rew_head.parameters())
    self.term_params = list(self.model.term_head.parameters())

    self.opt_dyn = torch.optim.Adam(self.dyn_params, lr=self.cfg.lr)
    self.opt_rew = torch.optim.Adam(self.rew_params, lr=10 * self.cfg.lr)
    self.opt_term = torch.optim.Adam(self.term_params, lr=self.cfg.lr)

    self.slices = {
      "root_pos_z": self.layout.z,
      "root_gravity_b": self.layout.g_b,
      "root_lin_vel": self.layout.v_h,
      "root_ang_vel": self.layout.w_h,
      "joint_pos": self.layout.q,
      "joint_vel": self.layout.dq,
      "root_pos_xy": slice(self.layout.dim, self.layout.dim + 2),
      "root_yaw": slice(self.layout.dim + 2, self.layout.dim + 3),
    }

  def compute_stats(self, replay_buffer):
    states = replay_buffer.replay_buf[0][:, : replay_buffer.num_transitions]
    actions = replay_buffer.replay_buf[1][:, : replay_buffer.num_transitions]
    goals = replay_buffer.replay_buf[2][:, : replay_buffer.num_transitions]

    yaw_states = compute_yaw_aligned_state(states, self.layout)
    yaw_states_flat = yaw_states.reshape(-1, yaw_states.shape[-1])
    yaw_states_flat = yaw_states_flat.repeat(1, self.history)

    actions_aligned = actions.reshape(-1, actions.shape[-1])
    last_actions_aligned = actions_aligned
    goals_aligned = goals.reshape(-1, goals.shape[-1])

    inputs = torch.cat(
      [yaw_states_flat, actions_aligned, last_actions_aligned, goals_aligned],
      dim=-1,
    )
    self.model.input_mean.copy_(inputs.mean(dim=0))
    self.model.input_std.copy_(inputs.std(dim=0))

  def _get_loss_weights(self, device):
    """Creates a weight tensor matching state dimension (Broadcasting ready)."""
    # Layout dim + 2 (xy) + 1 (yaw)
    W = torch.ones(self.layout.dim + 3, device=device)
    c = self.cfg
    weights = {
      "root_pos_z": 1.0 / c.nominal_error_root_pos_z**2,
      "root_gravity_b": 1.0 / c.nominal_error_root_gravity_b**2,
      "root_lin_vel": 1.0 / c.nominal_error_root_lin_vel**2,
      "root_ang_vel": 1.0 / c.nominal_error_root_ang_vel**2,
      "joint_pos": 1.0 / c.nominal_error_joint_pos**2,
      "joint_vel": 1.0 / c.nominal_error_joint_vel**2,
      "root_pos_xy": 1.0 / c.nominal_error_root_pos_xy**2,
      "root_yaw": 1.0 / c.nominal_error_root_yaw**2,
    }
    for name, sl in self.slices.items():
      W[sl] = weights[name]
    return W

  def update(self, replay_buffer):
    if replay_buffer.num_transitions < self.cfg.horizon:
      print("[WARN] Insufficient data.")
      return {}

    self.model.train()

    device = next(self.model.parameters()).device
    loss_weights = self._get_loss_weights(device)

    full_len = self.cfg.history - 1 + self.cfg.horizon
    loader = replay_buffer.mini_batch_generator(
      full_len, self.cfg.num_batches, self.cfg.batch_size
    )

    epoch_loss_total = 0
    epoch_loss_state = 0
    epoch_loss_reward = 0
    epoch_loss_term = 0
    epoch_metrics = {k: 0.0 for k in self.slices.keys()}
    epoch_reward_mse = torch.zeros(len(self.reward_names), device=device)

    for full_states, actions, goals, rewards, dones in loader:
      start_idx = torch.randint(0, self.cfg.history, (1,), device=device).item()
      # 1. Unroll Dynamics (Auto-regressive)
      all_yaw_states = compute_yaw_aligned_state(full_states, self.layout)

      # Determine start index and initialize history with padding if needed
      # History context ends at start_idx (inclusive)
      hist_context = all_yaw_states[:, : start_idx + 1]
      if hist_context.shape[1] < self.history:
        pad_len = self.history - hist_context.shape[1]
        pad = hist_context[:, 0:1, :].repeat(1, pad_len, 1)
        curr_yaw_state_hist = torch.cat([pad, hist_context], dim=1)
      else:
        curr_yaw_state_hist = hist_context[:, -self.history :, :]

      pred_states = []
      pred_rewards = []
      pred_dones = []

      # We start at t = start_idx to t = start_idx + horizon - 1
      # We predict t = start_idx + 1 to t = start_idx + horizon
      # Drop the prediction state if t = full_len
      # Keep rewards/dones corresponding to state and action t = 0 to full_len - 1
      for k in range(self.cfg.horizon):
        t = start_idx + k
        curr_action = actions[:, t, :]

        # Last Action: if t=0, use zeros, else action[t-1]
        if t == 0:
          last_action = torch.zeros_like(curr_action)
        else:
          last_action = actions[:, t - 1, :]

        curr_goal = goals[:, t, :]

        out = self.model(curr_yaw_state_hist, curr_action, last_action, curr_goal, True)

        # Dynamic slicing based on num_rewards
        num_rewards = self.model.num_rewards
        delta = out[..., : -num_rewards - 1]
        pred_r = out[..., -num_rewards - 1 : -1]
        pred_d = out[..., -1:]

        # Last state in history
        last_state = curr_yaw_state_hist[:, -1, :]
        next_yaw_state = last_state + delta
        next_yaw_state = apply_yaw_correction(next_yaw_state, self.dt, self.layout)

        # Update History
        curr_yaw_state_hist = torch.cat(
          [curr_yaw_state_hist[:, 1:, :], next_yaw_state.unsqueeze(1)], dim=1
        )

        pred_states.append(next_yaw_state)
        pred_rewards.append(pred_r)
        pred_dones.append(pred_d)

      # Only supervise next-state when GT x_{t+1} exists inside the sampled window
      state_steps = min(self.cfg.horizon, full_len - start_idx - 1)

      # Stack to (Batch, Window, Dim)
      pred_states = torch.stack(pred_states[:state_steps], dim=1)

      v_tp1 = pred_states[..., self.layout.v_h]
      w_tp1 = pred_states[..., self.layout.w_h]
      pred_dpos_xys = v_tp1[..., 0:2] * self.dt
      pred_dyaws = w_tp1[..., 2] * self.dt

      pred_rewards = torch.stack(pred_rewards, dim=1)
      pred_dones = torch.stack(pred_dones, dim=1)

      # Extract Ground Truth Targets
      gt_states = all_yaw_states[:, start_idx + 1 : start_idx + 1 + state_steps, :]

      idx_pos, idx_quat = 0, 3
      root_pos = full_states[..., idx_pos : idx_pos + 3]
      root_quat = full_states[..., idx_quat : idx_quat + 4]
      psi = get_yaw_from_quat(root_quat)
      p0 = root_pos[:, start_idx : start_idx + state_steps, :]
      p1 = root_pos[:, start_idx + 1 : start_idx + 1 + state_steps, :]
      psi0 = psi[:, start_idx : start_idx + state_steps]
      psi1 = psi[:, start_idx + 1 : start_idx + 1 + state_steps]
      dp_w = p1 - p0
      c = torch.cos(psi1)
      s = torch.sin(psi1)
      dx, dy = dp_w[..., 0], dp_w[..., 1]
      gt_dpos_xys = torch.stack([c * dx + s * dy, -s * dx + c * dy], dim=-1)
      gt_dyaws = wrap_to_pi(psi1 - psi0)

      gt_rewards = rewards[:, start_idx : start_idx + self.cfg.horizon]
      gt_dones = dones[:, start_idx : start_idx + self.cfg.horizon]

      # 2. Vectorized Loss Calculation
      loss, loss_state, loss_reward, loss_term, sq_err, reward_mse = (
        self._compute_losses(
          pred_states,
          pred_dpos_xys,
          pred_dyaws,
          pred_rewards,
          pred_dones,
          gt_states,
          gt_dpos_xys,
          gt_dyaws,
          gt_rewards,
          gt_dones,
          loss_weights,
        )
      )

      # 3. Backprop
      self.opt_dyn.zero_grad(set_to_none=True)
      self.opt_rew.zero_grad(set_to_none=True)
      self.opt_term.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.dyn_params, max_norm=1.0)
      torch.nn.utils.clip_grad_norm_(self.rew_params, max_norm=1.0)
      torch.nn.utils.clip_grad_norm_(self.term_params, max_norm=1.0)
      self.opt_dyn.step()
      self.opt_rew.step()
      self.opt_term.step()

      # 4. Logging
      epoch_loss_total += loss.item()
      epoch_loss_state += loss_state.item()
      epoch_loss_reward += loss_reward.item()
      epoch_loss_term += loss_term.item()

      with torch.no_grad():
        for name, sl in self.slices.items():
          epoch_metrics[name] += sq_err[..., sl].mean().item()

        # Accumulate reward stats
        # reward_mse is (R,)
        if reward_mse.shape[0] == len(self.reward_names):
          epoch_reward_mse += reward_mse.detach()

    # Averages
    avg_loss_total = epoch_loss_total / self.cfg.num_batches
    avg_loss_state = epoch_loss_state / self.cfg.num_batches
    avg_loss_reward = epoch_loss_reward / self.cfg.num_batches
    avg_loss_term = epoch_loss_term / self.cfg.num_batches

    log_dict = {
      "Loss/total": avg_loss_total,
      "Loss/state": avg_loss_state,
      "Loss/reward": avg_loss_reward,
      "Loss/termination": avg_loss_term,
    }
    for k, v in epoch_metrics.items():
      log_dict[f"Loss/{k}"] = v / self.cfg.num_batches

    # Log individual reward terms
    avg_reward_mse = epoch_reward_mse / self.cfg.num_batches

    for i, name in enumerate(self.reward_names):
      log_dict[f"Loss/reward_{name}"] = avg_reward_mse[i].item()

    return log_dict

  def _compute_losses(
    self,
    pred_states,
    pred_dpos_xys,
    pred_dyaws,
    pred_rewards,
    pred_dones,
    gt_states,
    gt_dpos_xys,
    gt_dyaws,
    gt_rewards,
    gt_dones,
    loss_weights,
  ):
    # Mean Squared Error on State
    sq_err_state = (pred_states - gt_states) ** 2

    # Consistency Squared Errors
    sq_err_xy = (pred_dpos_xys - gt_dpos_xys) ** 2
    sq_err_yaw = wrap_to_pi(pred_dyaws - gt_dyaws).pow(2).unsqueeze(-1)

    # Concatenate all errors
    sq_err = torch.cat([sq_err_state, sq_err_xy, sq_err_yaw], dim=-1)

    # Weighted Mean (integrated consistency)
    loss_state = (sq_err * loss_weights).mean()

    # Reward Loss
    reward_mse = (pred_rewards - gt_rewards).pow(2)
    reward_mse_per_id = reward_mse.mean(dim=(0, 1))
    loss_reward = reward_mse_per_id.sum()

    # Termination Loss
    valid_term_mask = gt_dones != 2.0  # Filter out timeouts (2.0) for termination loss
    valid_pred_dones = pred_dones[valid_term_mask]
    valid_dones = gt_dones[valid_term_mask]
    num_terminations = valid_dones.sum()

    if num_terminations > 0:
      total_samples = valid_dones.numel()
      pos_weight = torch.clamp(
        (total_samples - num_terminations) / (num_terminations + 1e-6), max=50.0
      )
      loss_term = nn.functional.binary_cross_entropy_with_logits(
        valid_pred_dones, valid_dones, pos_weight=pos_weight
      )
    else:
      loss_term = torch.tensor(0.0, device=pred_dones.device, requires_grad=True)

    loss = loss_state + loss_reward + loss_term

    return loss, loss_state, loss_reward, loss_term, sq_err, reward_mse_per_id
