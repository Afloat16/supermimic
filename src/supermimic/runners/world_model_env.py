import torch
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
  subtract_frame_transforms,
)

from supermimic.utils.helpers import (
  apply_yaw_correction,
  compute_yaw_aligned_state,
  integrate_global_state,
)


class SimpleMotionSampler:
  def __init__(self, motion_file: str, device):
    self.device = device
    body_indexes = torch.tensor([0], device=self.device, dtype=torch.long)
    self.motion = MotionLoader(motion_file, body_indexes, device=self.device)

  def sample_state(self, times):
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

  def get_target(self, times):
    target_times = times
    target_quat = self.motion.body_quat_w[target_times, 0]
    target_pos = self.motion.body_pos_w[target_times, 0]
    joint_pos = self.motion.joint_pos[target_times]
    joint_vel = self.motion.joint_vel[target_times]

    return joint_pos, joint_vel, target_pos, target_quat

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


class WorldModelEnv:
  def __init__(
    self,
    world_model,
    motion_file: str,
    env,
    batch_size,
    horizon,
    layout,
    device="cpu",
  ):
    self.wm = world_model
    self.device = device
    self.sampler = SimpleMotionSampler(motion_file, self.device)
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
    dummy_state = self.sampler.sample_state(
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
    start_times = torch.randint(
      0, self.sampler.motion.time_step_total - 1, (num_resets,), device=self.device
    )
    full_state = self.sampler.sample_state(start_times)
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

  def _get_goal(self):
    joint_pos, joint_vel, target_pos, target_quat = self.sampler.get_target(
      self.motion_times
    )
    target_joints = torch.cat([joint_pos, joint_vel], dim=-1)
    root_pos = self.full_state[..., 0:3]
    root_quat = self.full_state[..., 3:7]

    rel_pos_b, rel_ori_w = subtract_frame_transforms(
      root_pos, root_quat, target_pos, target_quat
    )
    rel_ori_mat = matrix_from_quat(rel_ori_w)

    rel_ori_b_flat = rel_ori_mat[..., :2].reshape(rel_ori_mat.shape[0], -1)
    rel_pos_b_flat = rel_pos_b.view(rel_pos_b.shape[0], -1)

    wm_goal_obs = torch.cat([target_joints, rel_pos_b_flat, rel_ori_b_flat], dim=-1)
    policy_goal_obs = torch.cat([target_joints, rel_ori_b_flat], dim=-1)

    return target_joints, wm_goal_obs, policy_goal_obs

  def step(self, actions):
    # Fetch Goal and relative obs equivalent for WM input
    target_joints, wm_goal_obs, _ = self._get_goal()

    # World Model Forward
    offset = target_joints[..., : self.num_actions]
    proc_action = actions * self.action_scale + offset

    out = self.wm(self.hist, proc_action, self.last_action, wm_goal_obs)

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
    _, wm_goal_obs, policy_goal_obs = self._get_goal()

    curr_yaw_state = self.hist[:, -1, :]

    curr_policy_input = torch.cat([curr_yaw_state, policy_goal_obs], dim=-1)
    curr_critic_input = torch.cat([curr_yaw_state, wm_goal_obs], dim=-1)

    self.current_obs = {
      "policy": curr_policy_input,
      "critic": curr_critic_input,
    }
    return self.current_obs
