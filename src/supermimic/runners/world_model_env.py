import torch
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from tensordict import TensorDict

from supermimic.utils.helpers import extract_action_scale_offset, shift_history


class WorldModelEnv:
  def __init__(
    self,
    world_model,
    motion_file: str | list[str],
    env,
    batch_size,
    horizon,
    replay_buffer,
    device="cpu",
  ):
    self.wm = world_model
    self.device = device

    # Get body_indexes from the env's MotionCommand for consistent motion loading
    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    body_indexes = motion_cmd.body_indexes.to(self.device)

    self.motion = MotionLoader(
      motion_file,
      body_indexes,
      device=self.device,
    )
    self.cfg = env.cfg
    self.num_envs = batch_size
    self.horizon = horizon
    self.replay_buffer = replay_buffer

    # Shapes and Dimensions
    self.num_actions = env.num_actions
    self.dt = env.unwrapped.step_dt

    # Action Processing
    self.action_scale, self.action_offset = extract_action_scale_offset(
      env, self.device
    )

    # Persistent State
    full_state_dim = self.wm.num_states
    body_state_dim = self.wm.num_rep_state

    self.motion_times = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.full_states = torch.zeros((self.num_envs, full_state_dim), device=self.device)

    self.hist = torch.zeros(
      (self.num_envs, self.wm.history, body_state_dim), device=self.device
    )
    self.last_actions = torch.zeros(
      (self.num_envs, self.num_actions), device=self.device
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

    # Compute rep states for the whole history using representation.
    all_states = batch[0]  # (num_samples, history, full_state_dim)
    body_states_hist = self.wm.representation.represent(all_states)

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
    self.survival_rate[indices] = 1.0

    # Reset last actions
    self.last_actions[indices] = torch.zeros(
      (num_resets, self.num_actions), device=self.device
    )

    self._update_goal()
    return self.get_observations()

  def detach(self):
    self.full_states = self.full_states.detach()
    self.hist = self.hist.detach()
    self.last_actions = self.last_actions.detach()

  def _update_goal(self):
    target_joints, wm_goal_obs, actor_goal_obs = self.wm.representation.compute_wm_goal(
      self.motion_times, self.full_states, self.motion
    )
    # Clone to detach from CUDA graph output buffers (reduce-overhead mode).
    self._target_joints = target_joints.clone()
    self._wm_goal_obs = wm_goal_obs.clone()
    self._actor_goal_obs = actor_goal_obs

  def step(self, actions):
    proc_actions = actions * self.action_scale + self.action_offset
    proc_last_actions = self.last_actions * self.action_scale + self.action_offset

    next_full_states, next_body_states, pred_r, pred_d = self.wm.transition(
      self.hist, proc_actions, proc_last_actions, self._wm_goal_obs, self.full_states
    )
    rewards = self.wm.representation.compute_wm_reward(
      next_body_states,
      pred_r,
      actions,
      self.last_actions,
      self.motion,
      self.motion_times,
      self.dt,
    )

    dones = (pred_d > 0).float().squeeze(-1)

    next_hist = shift_history(self.hist, next_body_states)

    next_motion_times = torch.clamp(
      self.motion_times + 1, 0, self.motion.time_step_total - 1
    )

    motion_time_outs = torch.isin(
      next_motion_times, self.motion.motion_ends - 2
    ).float()
    time_outs = motion_time_outs
    combined_dones = torch.clamp(dones + time_outs, 0.0, 1.0)

    self.survival_rate = self.survival_rate * (1 - combined_dones)
    rewards = rewards * self.survival_rate
    combined_dones = 1 - self.survival_rate
    # Clone outputs from compiled (reduce-overhead) functions to detach from
    # CUDA graph output buffers, so reset() can safely write in-place later.
    self.full_states = next_full_states.clone()
    self.hist = next_hist.clone()
    self.last_actions = actions
    self.motion_times = next_motion_times

    self._update_goal()
    return self.get_observations(), rewards, combined_dones, {"time_outs": time_outs}

  def get_observations(self):
    curr_body_states = self.hist[:, -1, :]
    self.current_obs = TensorDict(
      {
        "actor": torch.cat([curr_body_states, self._actor_goal_obs], dim=-1),
        "critic": torch.cat([curr_body_states, self._wm_goal_obs], dim=-1),
      },
      batch_size=[self.num_envs],
      device=self.device,
    )
    return self.current_obs
