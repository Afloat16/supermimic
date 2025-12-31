from dataclasses import dataclass

import torch
import torch.nn as nn
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from mjlab.utils.lab_api.math import (
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
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
  horizon: int = 64
  motion_file: str | None = None
  die_penalty: float = -0.0


class SimpleMotionSampler:
  def __init__(self, cfg: PolicyLearnerCfg, device):
    self.device = device
    body_indexes = torch.tensor([0], device=self.device, dtype=torch.long)
    self.motion = MotionLoader(cfg.motion_file, body_indexes, device=self.device)

  def sample(self, batch_size: int, horizon: int):
    # Sample random start times
    max_start_time = self.motion.time_step_total - horizon
    start_times = torch.randint(0, max_start_time, (batch_size,), device=self.device)

    # Fetch initial state and targets
    curr_state = self._fetch_state(start_times)
    goals, target_yaws = self._fetch_targets(start_times, horizon)

    return curr_state, goals, target_yaws

  def _fetch_state(self, start_times):
    root_pos = self.motion.body_pos_w[start_times, 0]
    root_ori = self.motion.body_quat_w[start_times, 0]
    root_lin_vel = self.motion.body_lin_vel_w[start_times, 0]
    root_ang_vel = self.motion.body_ang_vel_w[start_times, 0]
    joint_pos = self.motion.joint_pos[start_times]
    joint_vel = self.motion.joint_vel[start_times]

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

    # Flatten (Batch, Time, Dim) -> (N, Dim) directly
    # state: index 0, goal: index 2
    states = replay_buffer.replay_buf[0][:, : replay_buffer.num_transitions].reshape(
      -1, replay_buffer.replay_buf[0].shape[-1]
    )
    goals = replay_buffer.replay_buf[2][:, : replay_buffer.num_transitions].reshape(
      -1, replay_buffer.replay_buf[2].shape[-1]
    )

    yaw_state = compute_yaw_aligned_state(states, self.layout)

    num_samples = yaw_state.shape[0]
    rel_yaw_dummy = torch.zeros((num_samples, 2), device=self.device)
    rel_yaw_dummy[:, 1] = 1.0  # cos(0)

    inputs = torch.cat([yaw_state, rel_yaw_dummy, goals], dim=-1)
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
      curr_yaw_state = compute_yaw_aligned_state(full_state, self.layout)
      curr_yaw_angle = get_yaw_from_quat(full_state[..., 3:7]).unsqueeze(-1)
      curr_survival = torch.ones((self.cfg.batch_size, 1), device=self.device)

      for t in range(self.cfg.horizon):
        rel_yaw_obs = compute_relative_yaw_obs(curr_yaw_angle, target_yaws[:, t, :])
        curr_goal = goals[:, t, :]
        inputs = torch.cat([curr_yaw_state, rel_yaw_obs, curr_goal], dim=-1)
        action = self.policy(inputs)
        out = self.wm(curr_yaw_state, action, curr_goal)
        delta = out[..., :-2]
        pred_r = out[..., -2:-1]
        pred_d = out[..., -1:]

        prev_survival = curr_survival
        survival_rate = torch.sigmoid(-pred_d)
        curr_survival = prev_survival * survival_rate

        step_reward = pred_r * prev_survival
        step_reward = step_reward + self.cfg.die_penalty * (1 - curr_survival)
        rewards.append(step_reward)

        yaw_state_next = curr_yaw_state + delta
        curr_yaw_state = apply_yaw_correction(yaw_state_next, self.dt, self.layout)

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
  horizon: int = 16

  # Loss Weights (Bryson's Rule)
  nominal_error_root_pos: float = 0.1
  nominal_error_root_ori: float = 0.1
  nominal_error_root_lin_vel: float = 1.0
  nominal_error_root_ang_vel: float = 3.14
  nominal_error_joint_pos: float = 0.1
  nominal_error_joint_vel: float = 3.14


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
    self.dyn_params = list(self.model.backbone.parameters()) + list(
      self.model.dyn_head.parameters()
    )
    self.rew_params = list(self.model.rew_head.parameters())
    self.term_params = list(self.model.term_head.parameters())

    self.opt_dyn = torch.optim.Adam(self.dyn_params, lr=self.cfg.lr)
    self.opt_rew = torch.optim.Adam(self.rew_params, lr=10 * self.cfg.lr)
    self.opt_term = torch.optim.Adam(self.term_params, lr=10 * self.cfg.lr)

    self.slices = {
      "root_pos": self.layout.z,
      "root_ori": self.layout.g_b,
      "root_lin_vel": self.layout.v_h,
      "root_ang_vel": self.layout.w_h,
      "joint_pos": self.layout.q,
      "joint_vel": self.layout.dq,
    }

  def compute_stats(self, replay_buffer):
    print("[INFO] Computing normalization statistics...")
    # Flatten (Batch, Time, Dim) -> (N, Dim) directly
    states = replay_buffer.replay_buf[0][:, : replay_buffer.num_transitions].reshape(
      -1, replay_buffer.replay_buf[0].shape[-1]
    )
    actions = replay_buffer.replay_buf[1][:, : replay_buffer.num_transitions].reshape(
      -1, replay_buffer.replay_buf[1].shape[-1]
    )
    goals = replay_buffer.replay_buf[2][:, : replay_buffer.num_transitions].reshape(
      -1, replay_buffer.replay_buf[2].shape[-1]
    )

    yaw_states = compute_yaw_aligned_state(states, self.layout)

    inputs = torch.cat([yaw_states, actions, goals], dim=-1)
    self.model.input_mean.copy_(inputs.mean(dim=0))
    self.model.input_std.copy_(inputs.std(dim=0))
    print(f"[INFO] Stats -- Mean: {inputs.mean():.4f}, Std: {inputs.std():.4f}")

  def _get_loss_weights(self, device):
    """Creates a weight tensor matching state dimension (Broadcasting ready)."""
    W = torch.ones(self.layout.dim, device=device)
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
    if replay_buffer.num_transitions < self.cfg.horizon:
      print("[WARN] Insufficient data.")
      return {}

    self.model.train()

    device = next(self.model.parameters()).device
    loss_weights = self._get_loss_weights(device)

    loader = replay_buffer.mini_batch_generator(
      self.cfg.horizon, self.cfg.num_batches, self.cfg.batch_size
    )

    epoch_loss_total = 0
    epoch_loss_state = 0
    epoch_loss_reward = 0
    epoch_loss_term = 0
    epoch_metrics = {k: 0.0 for k in self.slices.keys()}

    for full_states, actions, goals, rewards, dones in loader:
      # 1. Unroll Dynamics (Auto-regressive)
      # Convert entire sequence to yaw-aligned state for ground truth
      target_yaw_states = compute_yaw_aligned_state(full_states, self.layout)

      # Initial state for rollout
      curr_yaw_state = target_yaw_states[:, 0, :]

      pred_states = []
      pred_rewards = []
      pred_dones = []

      for t in range(self.cfg.horizon):
        out = self.model(curr_yaw_state, actions[:, t, :], goals[:, t, :], True)
        delta = out[..., :-2]
        pred_r = out[..., -2:-1]
        pred_d = out[..., -1:]

        next_yaw_state = curr_yaw_state + delta
        next_yaw_state = apply_yaw_correction(next_yaw_state, self.dt, self.layout)

        curr_yaw_state = next_yaw_state
        pred_states.append(curr_yaw_state)
        pred_rewards.append(pred_r)
        pred_dones.append(pred_d)

      # Stack to (Batch, Window, Dim)
      pred_states = torch.stack(pred_states, dim=1)
      pred_rewards = torch.stack(pred_rewards, dim=1)
      pred_dones = torch.stack(pred_dones, dim=1)

      # 2. Vectorized Loss Calculation
      loss, loss_state, loss_reward, loss_term, sq_err = self._compute_losses(
        pred_states,
        pred_rewards,
        pred_dones,
        target_yaw_states,
        rewards,
        dones,
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
      f" Loss: {avg_loss_total:.6f} | State: {avg_loss_state:.6f} | Reward: {avg_loss_reward:.6f} | Term: {avg_loss_term:.6f}"
    )

    return log_dict

  def _compute_losses(
    self,
    pred_states,
    pred_rewards,
    pred_dones,
    target_yaw_states,
    rewards,
    dones,
    loss_weights,
  ):
    # Normalize state indices for loss (shift by 1)
    sq_err = (pred_states[:, :-1, :] - target_yaw_states[:, 1:, :]) ** 2
    loss_state = (sq_err * loss_weights).mean()

    # Reward Loss (Scale by 1/dt^2 to match reward rate error)
    loss_reward = (pred_rewards - rewards).pow(2).mean() / (self.dt**2)

    # Termination Loss
    valid_term_mask = dones != 2.0  # Filter out timeouts (2.0) for termination loss
    valid_pred_dones = pred_dones[valid_term_mask]
    valid_dones = dones[valid_term_mask]
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
