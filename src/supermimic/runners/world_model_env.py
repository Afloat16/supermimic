import torch
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)
from tensordict import TensorDict

from supermimic.utils.helpers import (
  compute_body_frame_state,
  integrate_global_state,
)


class WorldModelEnv:
  def __init__(
    self,
    world_model,
    motion_file: str | list[str],
    env,
    batch_size,
    horizon,
    layout,
    replay_buffer,
    device="cpu",
  ):
    self.wm = world_model
    self.device = device
    self.motion = MotionLoader(
      motion_file,
      torch.tensor([0], device=self.device, dtype=torch.long),
      device=self.device,
    )
    self.cfg = env.cfg
    self.num_envs = batch_size
    self.horizon = horizon
    self.layout = layout
    self.replay_buffer = replay_buffer

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
    state_dim = self.layout.dim + 3

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

  def _sample_initial_states_from_buffer(self, num_samples):
    """Sample initial states from replay buffer."""
    batch = self.replay_buffer._generate_batch(
      self.replay_buffer.replay_buf,
      valid_indices=None,
      sequence_length=self.wm.history,
      mini_batch_size=num_samples,
    )
    # batch[i] has shape (num_samples, history, dim)

    # Get full_states and motion_times from last timestep.
    full_states = batch[0][:, -1, :]
    motion_times = batch[4][:, -1, :].squeeze(-1).long()

    # Compute body frame states for the whole history.
    all_states = batch[0]  # (num_samples, history, state_dim)
    all_states_flat = all_states.reshape(-1, all_states.shape[-1])
    body_states_flat = compute_body_frame_state(all_states_flat, self.layout)
    body_states_hist = body_states_flat.reshape(num_samples, self.wm.history, -1)

    return (
      motion_times,
      full_states.detach(),
      body_states_hist.detach(),
    )

  def reset(self, indices=None):
    if indices is None:
      indices = torch.arange(self.num_envs, device=self.device)

    num_resets = len(indices)
    if num_resets == 0:
      return self.get_observations() if self.current_obs is not None else {}

    start_times, full_states, hist = self._sample_initial_states_from_buffer(num_resets)

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
    target_times = self.motion_times
    target_quat = self.motion.body_quat_w[target_times, 0]
    target_pos = self.motion.body_pos_w[target_times, 0]
    joint_pos = self.motion.joint_pos[target_times]
    joint_vel = self.motion.joint_vel[target_times]

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
    actor_goal_obs = torch.cat([target_joints, rel_ori_b_flat], dim=-1)

    return target_joints, wm_goal_obs, actor_goal_obs

  def step(self, actions, auto_reset=True):
    # Fetch Goal and relative obs equivalent for WM input
    target_joints, wm_goal_obs, _ = self._get_goal()

    # World Model Forward
    offset = target_joints[..., : self.num_actions]
    proc_actions = actions * self.action_scale + offset

    next_body_states, pred_r, pred_d = self.wm(
      self.hist, proc_actions, self.last_actions, wm_goal_obs
    )

    rewards = pred_r.sum(dim=-1) * self.dt
    dones = (pred_d > 0).float().squeeze(-1)

    # Integrate State (Differentiable)
    next_full_states = integrate_global_state(
      self.full_states, next_body_states, self.dt, self.layout
    )

    # Prepare Next History
    next_hist = torch.cat([self.hist[:, 1:, :], next_body_states.unsqueeze(1)], dim=1)

    # Handle Time-Outs (Non-differentiable logic)
    self.motion_times += 1
    self.episode_length_buf += 1
    self.motion_times = torch.clamp(
      self.motion_times, 0, self.motion.time_step_total - 1
    )

    motion_time_outs = torch.isin(
      self.motion_times, self.motion.motion_ends - 2
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
    _, wm_goal_obs, actor_goal_obs = self._get_goal()

    curr_body_states = self.hist[:, -1, :]

    curr_actor_input = torch.cat([curr_body_states, actor_goal_obs], dim=-1)
    curr_critic_input = torch.cat([curr_body_states, wm_goal_obs], dim=-1)

    self.current_obs = TensorDict(
      {
        "actor": curr_actor_input,
        "critic": curr_critic_input,
      },
      batch_size=[self.num_envs],
      device=self.device,
    )
    return self.current_obs
