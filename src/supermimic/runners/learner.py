from dataclasses import dataclass

import torch
import torch.nn as nn
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from mjlab.utils.lab_api.math import (
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
  wrap_to_pi,
)

from supermimic.models import (
  YawAlignedStateLayout,
  apply_yaw_correction,
  compute_relative_yaw_obs,
  compute_yaw_aligned_state,
  get_yaw_from_quat,
)


@dataclass
class PolicyLearnerCfg:
  lr: float = 1e-4
  num_batches: int = 5
  batch_size: int = 1024
  noise: float = 0.25
  horizon: int = 128
  motion_file: str | None = None
  die_penalty: float = -0.0


class SimpleMotionSampler:
  def __init__(self, cfg: PolicyLearnerCfg, device):
    self.device = device
    body_indexes = torch.tensor([0], device=self.device, dtype=torch.long)
    self.motion = MotionLoader(cfg.motion_file, body_indexes, device=self.device)

  def sample(self, batch_size: int, horizon: int):
    # Sample random start times
    start_times = torch.randint(
      0, self.motion.time_step_total - horizon, (batch_size,), device=self.device
    )

    curr_state = self._fetch_state_at_time(start_times)
    goals, target_yaws = self._fetch_targets(start_times, horizon)

    return curr_state, goals, target_yaws

  def _fetch_state_at_time(self, times):
    root_pos = self.motion.body_pos_w[times, 0]
    root_ori = self.motion.body_quat_w[times, 0]
    root_lin_vel = self.motion.body_lin_vel_w[times, 0]
    root_ang_vel = self.motion.body_ang_vel_w[times, 0]
    joint_pos = self.motion.joint_pos[times]
    joint_vel = self.motion.joint_vel[times]

    # Apply noise
    root_pos, root_ori = self._apply_pose_noise(root_pos, root_ori)
    root_lin_vel, root_ang_vel = self._apply_vel_noise(root_lin_vel, root_ang_vel)
    joint_pos = self._apply_joint_noise(joint_pos)

    # Assemble state: [root_pos(3), root_quat(4), joint_pos(N), root_lin_vel(3), root_ang_vel(3), joint_vel(N)]
    return torch.cat(
      [root_pos, root_ori, joint_pos, root_lin_vel, root_ang_vel, joint_vel], dim=-1
    )

  def _fetch_targets(self, start_times, horizon):
    goals = []
    target_yaws = []
    for t in range(horizon):
      target_times = start_times + t
      goals.append(
        torch.cat(
          [self.motion.joint_pos[target_times], self.motion.joint_vel[target_times]],
          dim=-1,
        )
      )
      target_quat = self.motion.body_quat_w[target_times, 0]
      target_yaws.append(get_yaw_from_quat(target_quat).unsqueeze(-1))

    return torch.stack(goals, dim=1), torch.stack(target_yaws, dim=1)

  def _apply_pose_noise(self, root_pos, root_ori):
    batch_size = root_pos.shape[0]
    noise_pose = self._get_noise("pose", (batch_size, 6))
    root_pos_noisy = root_pos + noise_pose[:, :3]
    root_ori_noisy = quat_mul(
      quat_from_euler_xyz(noise_pose[:, 3], noise_pose[:, 4], noise_pose[:, 5]),
      root_ori,
    )
    return root_pos_noisy, root_ori_noisy

  def _apply_vel_noise(self, lin_vel, ang_vel):
    batch_size = lin_vel.shape[0]
    noise_vel = self._get_noise("velocity", (batch_size, 6))
    return lin_vel + noise_vel[:, :3], ang_vel + noise_vel[:, 3:]

  def _apply_joint_noise(self, joint_pos):
    joint_range = (-0.1, 0.1)
    noise = sample_uniform(
      joint_range[0], joint_range[1], joint_pos.shape, device=self.device
    )
    return joint_pos + noise

  def _get_noise(self, ranges_key, size):
    if ranges_key == "pose":
      ranges_dict = {
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      }
    else:
      ranges_dict = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.2, 0.2),
        "roll": (-0.52, 0.52),
        "pitch": (-0.52, 0.52),
        "yaw": (-0.78, 0.78),
      }

    ranges = torch.tensor(
      [ranges_dict.get(k, (0, 0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]],
      device=self.device,
    )
    return sample_uniform(ranges[:, 0], ranges[:, 1], size, device=self.device)


class PolicyLearner:
  """Trainer for Policy using Differentiable World Model."""

  def __init__(
    self,
    policy: nn.Module,
    world_model: nn.Module,
    cfg: PolicyLearnerCfg,
    dt: float,
    layout: YawAlignedStateLayout,
    device: str = "cpu",
  ):
    self.policy = policy
    self.wm = world_model
    self.cfg = cfg
    self.device = device
    self.dt = dt
    self.layout = layout
    self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.cfg.lr)
    self.sampler = SimpleMotionSampler(cfg, device)

  def compute_stats(self, replay_buffer):
    if not self.policy.normalization:
      return

    # Flatten (Batch, Time, Dim) -> Windows
    states = replay_buffer.replay_buf[0][:, : replay_buffer.num_transitions]
    goals = replay_buffer.replay_buf[2][:, : replay_buffer.num_transitions]

    # Flatten (Batch, Time, Dim) -> (N, Dim) - Use single state, no history for policy
    states_flat = states.reshape(-1, states.shape[-1])
    goals_flat = goals.reshape(-1, goals.shape[-1])

    yaw_state_flat = compute_yaw_aligned_state(states_flat, self.layout)

    # Dummy relative yaw for stats
    rel_yaw_dummy = torch.zeros((yaw_state_flat.shape[0], 2), device=self.device)
    rel_yaw_dummy[:, 1] = 1.0

    inputs = torch.cat([yaw_state_flat, rel_yaw_dummy, goals_flat], dim=-1)
    self.policy.actor_obs_normalizer._mean.copy_(inputs.mean(dim=0))
    self.policy.actor_obs_normalizer._std.copy_(inputs.std(dim=0))

  def update(self):
    self.policy.train()
    self.wm.eval()

    epoch_reward = 0

    for _ in range(self.cfg.num_batches):
      full_state, goals, target_yaws = self.sampler.sample(
        self.cfg.batch_size, self.cfg.horizon
      )

      rewards = []
      # Initial history buffer: [Batch, History, Dim]
      # Replicate current state for history initialization
      curr_yaw_state = compute_yaw_aligned_state(full_state, self.layout)
      curr_yaw_state_hist = curr_yaw_state.unsqueeze(1).repeat(1, self.wm.history, 1)

      # For relative yaw, use the current state
      curr_full_state = full_state
      curr_yaw_angle = get_yaw_from_quat(curr_full_state[..., 3:7]).unsqueeze(-1)

      curr_survival = torch.ones((self.cfg.batch_size, 1), device=self.device)

      for t in range(self.cfg.horizon):
        rel_yaw_obs = compute_relative_yaw_obs(curr_yaw_angle, target_yaws[:, t, :])
        curr_goal = goals[:, t, :]

        # Policy Forward
        curr_yaw_state = curr_yaw_state_hist[:, -1, :]
        inputs = torch.cat([curr_yaw_state, rel_yaw_obs, curr_goal], dim=-1)
        action = self.policy(inputs)

        # World Model Forward
        out = self.wm(curr_yaw_state_hist, action, curr_goal)
        delta = out[..., :-2]
        pred_r = out[..., -2:-1]
        pred_d = out[..., -1:]

        prev_survival = curr_survival
        survival_rate = torch.sigmoid(-pred_d)
        curr_survival = prev_survival * survival_rate

        step_reward = pred_r * prev_survival
        step_reward = step_reward + self.cfg.die_penalty * (1 - curr_survival)
        rewards.append(step_reward)

        # Update State
        yaw_state_next = curr_yaw_state_hist[:, -1, :] + delta
        yaw_state_next = apply_yaw_correction(yaw_state_next, self.dt, self.layout)

        # Update History
        curr_yaw_state_hist = torch.cat(
          [curr_yaw_state_hist[:, 1:, :], yaw_state_next.unsqueeze(1)], dim=1
        )

        # Update yaw angle tracking
        dpsi = yaw_state_next[..., self.layout.w_h][..., 2:3] * self.dt
        curr_yaw_angle = curr_yaw_angle + dpsi

      total_reward = torch.stack(rewards).sum()
      loss = -total_reward
      self.optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
      self.optimizer.step()

      epoch_reward += total_reward.item()

    avg_reward = (
      epoch_reward
      / self.cfg.num_batches
      / self.cfg.horizon
      / self.cfg.batch_size
      / self.dt
    )
    print(f"[Policy] Reward: {avg_reward:.4f}")
    return {"Policy/reward": avg_reward}


@dataclass
class WorldModelLearnerCfg:
  lr: float = 1e-4
  num_batches: int = 100
  batch_size: int = 2048
  horizon: int = 8
  history: int = 8

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
  ):
    self.model = model
    self.cfg = cfg
    self.device = device
    self.dt = dt
    self.layout = layout
    self.history = cfg.history
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
    print("[INFO] Computing normalization statistics...")
    # Keep (Batch, Time, Dim) for history unfolding
    states = replay_buffer.replay_buf[0][:, : replay_buffer.num_transitions]

    # Actions/Goals also needed in structure
    actions = replay_buffer.replay_buf[1][:, : replay_buffer.num_transitions]
    goals = replay_buffer.replay_buf[2][:, : replay_buffer.num_transitions]

    yaw_states = compute_yaw_aligned_state(states, self.layout)

    if yaw_states.shape[1] < self.history:
      print("[WARN] Not enough data for history stats.")
      return

    # Unfold along time dim to get history windows
    yaw_states_unfolded = yaw_states.unfold(dimension=1, size=self.history, step=1)
    yaw_states_unfolded = yaw_states_unfolded.permute(0, 1, 3, 2)
    yaw_states_flat = yaw_states_unfolded.flatten(start_dim=-2)
    yaw_states_flat = yaw_states_flat.reshape(-1, yaw_states_flat.shape[-1])

    # Align Actions/Goals with the LAST step of the window
    actions_aligned = actions[:, self.history - 1 :].reshape(-1, actions.shape[-1])
    goals_aligned = goals[:, self.history - 1 :].reshape(-1, goals.shape[-1])

    inputs = torch.cat([yaw_states_flat, actions_aligned, goals_aligned], dim=-1)
    self.model.input_mean.copy_(inputs.mean(dim=0))
    self.model.input_std.copy_(inputs.std(dim=0))
    print(f"[INFO] Stats -- Mean: {inputs.mean():.4f}, Std: {inputs.std():.4f}")

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
        curr_goal = goals[:, t, :]

        out = self.model(curr_yaw_state_hist, curr_action, curr_goal, True)
        delta = out[..., :-2]
        pred_r = out[..., -2:-1]
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
      loss, loss_state, loss_reward, loss_term, sq_err = self._compute_losses(
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

    print(
      f"Model update |"
      f" Loss: {avg_loss_total:.6f} | State: {avg_loss_state:.6f} |"
      f" Reward: {avg_loss_reward:.6f} | Term: {avg_loss_term:.6f}"
    )

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

    # Reward Loss (Scale by 1/dt^2 to match reward rate error)
    loss_reward = (pred_rewards - gt_rewards).pow(2).mean() / (self.dt**2)

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

    return loss, loss_state, loss_reward, loss_term, sq_err
