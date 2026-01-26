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
  compute_body_frame_state,
  integrate_global_state,
)


class SimpleMotionSampler:
  def __init__(self, motion_file: str | list[str], device):
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
    motion_file: str | list[str],
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
    dummy_states = self.sampler.sample_state(
      torch.zeros(1, dtype=torch.long, device=self.device)
    )
    state_dim = dummy_states.shape[-1]

    self.motion_times = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.full_states = torch.zeros((self.num_envs, state_dim), device=self.device)

    yaw_dim = compute_body_frame_state(
      torch.zeros((1, state_dim), device=self.device), self.layout
    ).shape[-1]

    self.hist = torch.zeros(
      (self.num_envs, self.wm.history, yaw_dim), device=self.device
    )
    self.last_actions = torch.zeros(
      (self.num_envs, self.num_actions), device=self.device
    )
    self.episode_length_buf = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

    self.current_obs = None
    self.survival_rate = torch.ones(self.num_envs, device=self.device)

  def _sample_initial_states(self, indices):
    num_resets = len(indices)
    start_times = torch.randint(
      0, self.sampler.motion.time_step_total - 1, (num_resets,), device=self.device
    )
    full_states = self.sampler.sample_state(start_times)
    curr_body_states = compute_body_frame_state(full_states, self.layout)
    body_states_hist = curr_body_states.unsqueeze(1).repeat(1, self.wm.history, 1)

    return start_times, full_states.detach(), body_states_hist.detach()

  def reset(self, indices=None):
    if indices is None:
      indices = torch.arange(self.num_envs, device=self.device)

    num_resets = len(indices)
    if num_resets == 0:
      return self.get_observations() if self.current_obs is not None else {}

    start_times, full_states, hist = self._sample_initial_states(indices)

    # Update State
    self.full_states[indices] = full_states
    self.hist[indices] = hist

    self.motion_times[indices] = start_times
    self.episode_length_buf[indices] = 0
    self.survival_rate[indices] = 1.0

    # Reset last actions
    self.last_actions[indices] = torch.zeros(
      (num_resets, self.num_actions), device=self.device
    )

    return self.get_observations()

  def detach(self):
    self.full_states = self.full_states.detach()
    self.hist = self.hist.detach()
    self.last_actions = self.last_actions.detach()

  def _get_goal(self):
    joint_pos, joint_vel, target_pos, target_quat = self.sampler.get_target(
      self.motion_times
    )
    target_joints = torch.cat([joint_pos, joint_vel], dim=-1)
    root_pos = self.full_states[..., 0:3]
    root_quat = self.full_states[..., 3:7]

    rel_pos_b, rel_ori_w = subtract_frame_transforms(
      root_pos, root_quat, target_pos, target_quat
    )
    rel_ori_mat = matrix_from_quat(rel_ori_w)

    rel_ori_b_flat = rel_ori_mat[..., :2].reshape(rel_ori_mat.shape[0], -1)
    rel_pos_b_flat = rel_pos_b.view(rel_pos_b.shape[0], -1)

    wm_goal_obs = torch.cat([target_joints, rel_pos_b_flat, rel_ori_b_flat], dim=-1)
    policy_goal_obs = torch.cat([target_joints, rel_ori_b_flat], dim=-1)

    return target_joints, wm_goal_obs, policy_goal_obs

  def step(self, actions, auto_reset=True):
    # Fetch Goal and relative obs equivalent for WM input
    target_joints, wm_goal_obs, _ = self._get_goal()

    # World Model Forward
    offset = target_joints[..., : self.num_actions]
    proc_actions = actions * self.action_scale + offset

    out = self.wm(self.hist, proc_actions, self.last_actions, wm_goal_obs)

    # Process Outputs
    num_rewards = self.wm.num_rewards
    delta = out[..., : -num_rewards - 1]
    pred_r = out[..., -num_rewards - 1 : -1]
    pred_d = out[..., -1:]

    rewards = pred_r.sum(dim=-1) * self.dt
    dones = (pred_d > 0).float().squeeze(-1)

    # Integrate State (Differentiable)
    curr_body_states = self.hist[:, -1, :]
    next_body_states = curr_body_states + delta

    next_full_states = integrate_global_state(
      self.full_states, next_body_states, self.dt, self.layout
    )

    # Prepare Next History
    next_hist = torch.cat([self.hist[:, 1:, :], next_body_states.unsqueeze(1)], dim=1)

    # Handle Time-Outs (Non-differentiable logic)
    self.motion_times += 1
    self.episode_length_buf += 1
    self.motion_times = torch.clamp(
      self.motion_times, 0, self.sampler.motion.time_step_total - 1
    )

    motion_time_outs = torch.isin(
      self.motion_times, self.sampler.motion.motion_ends - 2
    ).float()
    episode_time_outs = (self.episode_length_buf >= self.max_episode_length).float()
    time_outs = torch.clamp(motion_time_outs + episode_time_outs, 0.0, 1.0)
    combined_dones = torch.clamp(dones + time_outs, 0.0, 1.0)

    # Identify environments that need resetting
    self.survival_rate = self.survival_rate * (1 - combined_dones)

    if not auto_reset:
      rewards = rewards * self.survival_rate
      combined_dones = 1 - self.survival_rate

    reset_indices = []
    if auto_reset:
      reset_indices = torch.nonzero(combined_dones > 0.5).squeeze(-1)

    # Update Internal State
    self.full_states = next_full_states
    self.hist = next_hist
    self.last_actions = proc_actions

    extras = {}
    if time_outs.sum() > 0:
      extras["time_outs"] = time_outs

    if len(reset_indices) > 0:
      obs = self.reset(reset_indices)
    else:
      obs = self.get_observations()

    return obs, rewards, combined_dones, extras

  def get_observations(self):
    _, wm_goal_obs, policy_goal_obs = self._get_goal()

    curr_body_states = self.hist[:, -1, :]

    curr_policy_input = torch.cat([curr_body_states, policy_goal_obs], dim=-1)
    curr_critic_input = torch.cat([curr_body_states, wm_goal_obs], dim=-1)

    self.current_obs = {
      "policy": curr_policy_input,
      "critic": curr_critic_input,
    }
    return self.current_obs
