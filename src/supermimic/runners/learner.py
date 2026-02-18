from dataclasses import dataclass

import torch
import torch.nn as nn
from mjlab.utils.lab_api.math import matrix_from_quat, quat_mul, wrap_to_pi
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from supermimic.utils.helpers import (
  BodyFrameStateLayout,
  compute_body_frame_state,
)

from .world_model_env import WorldModelEnv


@dataclass
class PolicyLearnerCfg:
  lr: float = 1e-3
  num_batches: int = 1
  batch_size: int = 2048
  noise: float = 0.1
  horizon: int = 64
  motion_file: str | list[str] | None = None
  die_penalty: float = -0.0
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
    layout: BodyFrameStateLayout,
    device: str = "cpu",
  ):
    self.actor = policy
    self.wm_env = wm_env
    self.cfg = cfg
    self.device = device
    self.dt = dt
    self.layout = layout
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
    self.actor_optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
      self.actor_params, max_norm=max_norm, error_if_nonfinite=True
    )
    self.actor_optimizer.step()
    return grad_norm

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

      rewards += self.cfg.die_penalty * dones
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
    layout: BodyFrameStateLayout,
    device: str = "cpu",
  ):
    super().__init__(policy, wm_env, cfg, dt, layout, device)
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

      self.critic_optimizer.zero_grad()
      value_loss.backward()
      self.critic_optimizer.step()
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

  # Loss Weights (Bryson's Rule)
  nominal_error_root_pos_z: float = 0.1
  nominal_error_root_gravity_b: float = 0.1
  nominal_error_root_lin_vel: float = 1.0
  nominal_error_root_ang_vel: float = 3.14
  nominal_error_joint_pos: float = 0.1
  nominal_error_joint_vel: float = 3.14
  # Consistency terms
  nominal_error_root_pos_xy: float = 1e10
  nominal_error_root_rot_z: float = 1e10


class WorldModelLearner:
  """Trainer for World Model (Supervised)."""

  def __init__(
    self,
    model: nn.Module,
    cfg: WorldModelLearnerCfg,
    dt: float,
    layout: BodyFrameStateLayout,
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
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    self.slices = {
      "root_pos_z": self.layout.z,
      "root_gravity_b": self.layout.g_b,
      "root_lin_vel": self.layout.v_b,
      "root_ang_vel": self.layout.w_b,
      "joint_pos": self.layout.q,
      "joint_vel": self.layout.dq,
      "root_pos_xy": slice(self.layout.dim, self.layout.dim + 2),
      "root_rot_z": slice(self.layout.dim + 2, self.layout.dim + 3),
    }

  def update_normalization(self, states, actions, goals):
    body_states = compute_body_frame_state(states, self.layout)
    body_states_flat = body_states.reshape(-1, body_states.shape[-1])
    body_states_flat = body_states_flat.repeat(1, self.history)

    actions_flat = actions.reshape(-1, actions.shape[-1])
    # Use actions twice for last action
    inputs = torch.cat(
      [body_states_flat, actions_flat, actions, goals],
      dim=-1,
    )
    self.model.input_normalizer.update(inputs)

  def _get_loss_weights(self, device):
    """Creates a weight tensor matching state dimension (Broadcasting ready)."""
    # Layout dim + 2 (xy position) + 1 (z rotation) for SE(2) consistency
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
      "root_rot_z": 1.0 / c.nominal_error_root_rot_z**2,
    }
    for name, sl in self.slices.items():
      W[sl] = weights[name]
    return W

  def update(self, data):
    self.model.train()
    for p in self.model.parameters():
      p.requires_grad_(True)

    device = next(self.model.parameters()).device
    loss_weights = self._get_loss_weights(device)

    epoch_loss_total = 0
    epoch_loss_state = 0
    epoch_loss_reward = 0
    epoch_loss_term = 0
    epoch_metrics = {k: 0.0 for k in self.slices.keys()}
    epoch_reward_mse = torch.zeros(len(self.reward_names), device=device)

    for full_states, actions, goals, rewards, _, dones in data:
      start_idx = torch.randint(0, self.cfg.history, (1,), device=device).item()
      # 1. Unroll Dynamics (Auto-regressive)
      all_body_states = compute_body_frame_state(full_states, self.layout)

      # Determine start index and initialize history with padding if needed
      # History context ends at start_idx (inclusive)
      hist_context = all_body_states[:, : start_idx + 1]
      if hist_context.shape[1] < self.history:
        pad_len = self.history - hist_context.shape[1]
        pad = hist_context[:, 0:1, :].repeat(1, pad_len, 1)
        curr_body_states_hist = torch.cat([pad, hist_context], dim=1)
      else:
        curr_body_states_hist = hist_context[:, -self.history :, :]

      pred_states = []
      pred_rewards = []
      pred_dones = []
      full_len = full_states.shape[1]

      # We start at t = start_idx to t = start_idx + horizon - 1
      # We predict t = start_idx + 1 to t = start_idx + horizon
      # Drop the prediction state if t = full_len
      # Keep rewards/dones corresponding to state and action t = 0 to full_len - 1
      for k in range(self.cfg.horizon):
        t = start_idx + k
        curr_actions = actions[:, t, :]

        # Last Action: if t=0, use zeros, else action[t-1]
        if t == 0:
          last_actions = torch.zeros_like(curr_actions)
        else:
          last_actions = actions[:, t - 1, :]

        curr_goals = goals[:, t, :]

        next_body_state, pred_r, pred_d = self.model(
          curr_body_states_hist, curr_actions, last_actions, curr_goals, True
        )
        # No yaw correction needed in body frame

        # Update History
        curr_body_states_hist = torch.cat(
          [curr_body_states_hist[:, 1:, :], next_body_state.unsqueeze(1)], dim=1
        )

        pred_states.append(next_body_state)
        pred_rewards.append(pred_r)
        pred_dones.append(pred_d)

      # Only supervise next-state when GT x_{t+1} exists inside the sampled window
      state_steps = min(self.cfg.horizon, full_len - start_idx - 1)

      # Stack to (Batch, Window, Dim)
      pred_states = torch.stack(pred_states[:state_steps], dim=1)

      v_tp1 = pred_states[..., self.layout.v_b]
      w_tp1 = pred_states[..., self.layout.w_b]
      pred_dpos_xys = v_tp1[..., 0:2] * self.dt
      pred_dz_rot = w_tp1[..., 2:3] * self.dt

      pred_rewards = torch.stack(pred_rewards, dim=1)
      pred_dones = torch.stack(pred_dones, dim=1)

      # Extract Ground Truth Targets
      gt_states = all_body_states[:, start_idx + 1 : start_idx + 1 + state_steps, :]

      # Compute ground truth position change in body frame
      idx_pos, idx_quat = 0, 3
      root_pos = full_states[..., idx_pos : idx_pos + 3]
      root_quat = full_states[..., idx_quat : idx_quat + 4]
      p0 = root_pos[:, start_idx : start_idx + state_steps, :]
      p1 = root_pos[:, start_idx + 1 : start_idx + 1 + state_steps, :]
      q0 = root_quat[:, start_idx : start_idx + state_steps, :]
      q1 = root_quat[:, start_idx + 1 : start_idx + 1 + state_steps, :]
      dp_w = p1 - p0

      # Rotate world displacement into body frame at time t+1
      R_wb1 = matrix_from_quat(q1)
      # R_wb1^T @ dp_w gives displacement in body frame
      dp_b = torch.einsum("...ij,...j->...i", R_wb1.transpose(-2, -1), dp_w)
      gt_dpos_xys = dp_b[..., :2]  # Only xy components in body frame

      # Compute rotation around body z-axis: q_rel = q0^{-1} * q1
      # Quaternion inverse: q^{-1} = [w, -x, -y, -z] for unit quaternion
      q0_inv = torch.cat([q0[..., 0:1], -q0[..., 1:4]], dim=-1)
      q_rel = quat_mul(q0_inv, q1)
      # For small rotations around z-axis: q ≈ [cos(θ/2), 0, 0, sin(θ/2)]
      # Extract rotation angle: θ = 2 * atan2(qz, qw)
      gt_dz_rot = (2.0 * torch.atan2(q_rel[..., 3], q_rel[..., 0])).unsqueeze(
        -1
      )  # Shape: (batch, time, 1)

      gt_rewards = rewards[:, start_idx : start_idx + self.cfg.horizon]
      gt_dones = dones[:, start_idx : start_idx + self.cfg.horizon]

      # 2. Vectorized Loss Calculation
      loss, loss_state, loss_reward, loss_term, sq_err, reward_mse = (
        self._compute_losses(
          pred_states,
          pred_dpos_xys,
          pred_dz_rot,
          pred_rewards,
          pred_dones,
          gt_states,
          gt_dpos_xys,
          gt_dz_rot,
          gt_rewards,
          gt_dones,
          loss_weights,
        )
      )

      # 3. Backprop
      self.optimizer.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(
        self.model.parameters(), max_norm=1.0, error_if_nonfinite=True
      )
      self.optimizer.step()

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
    pred_dz_rot,
    pred_rewards,
    pred_dones,
    gt_states,
    gt_dpos_xys,
    gt_dz_rot,
    gt_rewards,
    gt_dones,
    loss_weights,
  ):
    # Mean Squared Error on State
    sq_err_state = (pred_states - gt_states) ** 2

    # SE(2) Consistency Squared Errors (position + rotation in body frame)
    sq_err_xy = (pred_dpos_xys - gt_dpos_xys) ** 2

    # Rotation around body z-axis consistency
    angle_diff = wrap_to_pi(pred_dz_rot - gt_dz_rot)
    sq_err_z_rot = angle_diff.pow(2)

    # Concatenate all errors
    sq_err = torch.cat([sq_err_state, sq_err_xy, sq_err_z_rot], dim=-1)

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
