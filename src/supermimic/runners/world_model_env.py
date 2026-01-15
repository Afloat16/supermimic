import torch
from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms

from supermimic.utils.helpers import (
  apply_yaw_correction,
  compute_yaw_aligned_state,
  integrate_global_state,
)


class WorldModelEnv:
  def __init__(
    self,
    world_model,
    motion_sampler,
    env,
    batch_size,
    horizon,
    layout,
    device="cpu",
  ):
    self.wm = world_model
    self.sampler = motion_sampler
    self.device = device
    self.cfg = env.cfg
    self.num_envs = batch_size
    self.horizon = horizon
    self.layout = layout

    # Shapes and Dimensions
    self.num_actions = env.num_actions
    self.dt = env.unwrapped.step_dt
    self.max_episode_length = 100

    # Action Processing
    action_manager = env.unwrapped.action_manager
    action_term = action_manager.get_term(action_manager.active_terms[0])
    self.action_scale = action_term.scale.to(self.device).float()
    if self.action_scale.dim() > 1:
      self.action_scale = self.action_scale[0:1]

    # Persistent State
    dummy_state = self.sampler._fetch_state_at_time(
      torch.zeros(1, dtype=torch.long, device=self.device)
    )
    state_dim = dummy_state.shape[-1]

    self.motion_times = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.full_state = torch.zeros((self.num_envs, state_dim), device=self.device)

    yaw_dim = compute_yaw_aligned_state(
      torch.zeros((1, state_dim), device=self.device), self.layout
    ).shape[-1]

    self.hist = torch.zeros(
      (self.num_envs, self.wm.history, yaw_dim), device=self.device
    )
    self.last_action = torch.zeros(
      (self.num_envs, self.num_actions), device=self.device
    )
    self.episode_length_buf = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

    self.current_obs = None

  def _sample_initial_states(self, indices):
    num_resets = len(indices)
    limit = self.sampler.motion.time_step_total - self.horizon
    start_times = torch.randint(0, limit, (num_resets,), device=self.device)
    full_state = self.sampler._fetch_state_at_time(start_times)
    curr_yaw_state = compute_yaw_aligned_state(full_state, self.layout)
    yaw_state_hist = curr_yaw_state.unsqueeze(1).repeat(1, self.wm.history, 1)
    last_action = torch.zeros((num_resets, self.num_actions), device=self.device)

    return (
      start_times,
      full_state.detach(),
      yaw_state_hist.detach(),
      last_action.detach(),
    )

  def reset(self, indices=None):
    if indices is None:
      indices = torch.arange(self.num_envs, device=self.device)

    num_resets = len(indices)
    if num_resets == 0:
      return self.get_observations() if self.current_obs is not None else {}

    start_times, full_state, hist, last_action = self._sample_initial_states(indices)

    # Update State
    self.full_state[indices] = full_state
    self.hist[indices] = hist
    self.last_action[indices] = last_action

    self.motion_times[indices] = start_times
    self.episode_length_buf[indices] = 0

    return self.get_observations()

  def detach(self):
    self.full_state = self.full_state.detach()
    self.hist = self.hist.detach()
    self.last_action = self.last_action.detach()

  def step(self, actions):
    # Fetch Goal for current step
    goal_data = self.sampler._fetch_targets(self.motion_times, 1).squeeze(1)
    ref_pos_w = goal_data[..., -7:-4]
    ref_quat_w = goal_data[..., -4:]
    motion_cmd = goal_data[..., :-7]

    # Current State relative to Goal
    curr_root_pos_w = self.full_state[..., 0:3]
    curr_root_quat_w = self.full_state[..., 3:7]

    rel_pos_b, rel_ori_w = subtract_frame_transforms(
      curr_root_pos_w, curr_root_quat_w, ref_pos_w, ref_quat_w
    )
    rel_ori_mat = matrix_from_quat(rel_ori_w)
    rel_ori_b_flat = rel_ori_mat[..., :2].reshape(rel_ori_mat.shape[0], -1)
    rel_pos_b_flat = rel_pos_b.view(rel_pos_b.shape[0], -1)

    curr_goal_wm = torch.cat([motion_cmd, rel_pos_b_flat, rel_ori_b_flat], dim=-1)

    # World Model Forward
    offset = motion_cmd[..., : self.num_actions]
    proc_action = actions * self.action_scale + offset

    out = self.wm(self.hist, proc_action, self.last_action, curr_goal_wm)

    # Process Outputs
    num_rewards = self.wm.num_rewards
    delta = out[..., : -num_rewards - 1]
    pred_r = out[..., -num_rewards - 1 : -1]
    pred_d = out[..., -1:]

    reward = pred_r.sum(dim=-1) * self.dt
    dones = (pred_d > 0).float().squeeze(-1)

    # Integrate State (Differentiable)
    curr_yaw_state = self.hist[:, -1, :]
    next_yaw_state = curr_yaw_state + delta
    next_yaw_state = apply_yaw_correction(next_yaw_state, self.dt, self.layout)
    next_full_state = integrate_global_state(
      self.full_state, next_yaw_state, self.dt, self.layout
    )

    # Prepare Next History
    next_hist = torch.cat([self.hist[:, 1:, :], next_yaw_state.unsqueeze(1)], dim=1)

    # Handle Time-Outs (Non-differentiable logic)
    self.motion_times += 1
    self.episode_length_buf += 1

    motion_time_outs = (
      self.motion_times >= self.sampler.motion.time_step_total - 2
    ).float()
    episode_time_outs = (self.episode_length_buf >= self.max_episode_length).float()
    time_outs = torch.clamp(motion_time_outs + episode_time_outs, 0.0, 1.0)
    combined_dones = torch.clamp(dones + time_outs, 0.0, 1.0)

    # Identify environments that need resetting
    reset_indices = torch.nonzero(combined_dones > 0.5).squeeze(-1)

    if len(reset_indices) > 0:
      r_times, r_full, r_hist, r_last_act = self._sample_initial_states(reset_indices)
      next_full_state[reset_indices] = r_full
      next_hist[reset_indices] = r_hist
      self.motion_times[reset_indices] = r_times
      self.episode_length_buf[reset_indices] = 0

    # Update Internal State
    self.full_state = next_full_state
    self.hist = next_hist

    # Update last_action logic:
    self.last_action = proc_action.clone()
    if len(reset_indices) > 0:
      self.last_action[reset_indices] = torch.zeros(
        self.num_actions, device=self.device
      )

    obs = self.get_observations()

    extras = {}
    if time_outs.sum() > 0:
      extras["time_outs"] = time_outs

    return obs, reward, combined_dones, extras

  def get_observations(self):
    goal_data = self.sampler._fetch_targets(self.motion_times, 1).squeeze(1)
    ref_pos_w = goal_data[..., -7:-4]
    ref_quat_w = goal_data[..., -4:]
    motion_cmd = goal_data[..., :-7]

    curr_root_pos_w = self.full_state[..., 0:3]
    curr_root_quat_w = self.full_state[..., 3:7]

    rel_pos_b, rel_ori_w = subtract_frame_transforms(
      curr_root_pos_w, curr_root_quat_w, ref_pos_w, ref_quat_w
    )
    rel_ori_mat = matrix_from_quat(rel_ori_w)

    rel_ori_b_flat = rel_ori_mat[..., :2].reshape(rel_ori_mat.shape[0], -1)
    rel_pos_b_flat = rel_pos_b.view(rel_pos_b.shape[0], -1)

    curr_goal_wm = torch.cat([motion_cmd, rel_pos_b_flat, rel_ori_b_flat], dim=-1)
    curr_goal_policy = torch.cat([motion_cmd, rel_ori_b_flat], dim=-1)

    curr_yaw_state = self.hist[:, -1, :]

    curr_policy_input = torch.cat([curr_yaw_state, curr_goal_policy], dim=-1)
    curr_critic_input = torch.cat([curr_yaw_state, curr_goal_wm], dim=-1)

    self.current_obs = {
      "policy": curr_policy_input,
      "critic": curr_critic_input,
    }
    return self.current_obs
