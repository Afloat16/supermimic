import torch
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)
from tensordict import TensorDict

from supermimic.utils.helpers import compute_body_frame_state


def _compute_goal(
  motion_times: torch.Tensor,
  states: torch.Tensor,
  body_quat_w: torch.Tensor,
  body_pos_w: torch.Tensor,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  target_quat = body_quat_w[motion_times, 0]
  target_pos = body_pos_w[motion_times, 0]
  tj_pos = joint_pos[motion_times]
  tj_vel = joint_vel[motion_times]

  target_joints = torch.cat([tj_pos, tj_vel], dim=-1)
  root_pos = states[..., 0:3]
  root_quat = states[..., 3:7]

  rel_pos_b, rel_ori_w = subtract_frame_transforms(
    root_pos, root_quat, target_pos, target_quat
  )
  rel_ori_mat = matrix_from_quat(rel_ori_w)

  rel_ori_b_flat = rel_ori_mat[..., :2].reshape(rel_ori_mat.shape[0], -1)
  rel_pos_b_flat = rel_pos_b.view(rel_pos_b.shape[0], -1)

  wm_goal_obs = torch.cat([target_joints, rel_pos_b_flat, rel_ori_b_flat], dim=-1)
  actor_goal_obs = torch.cat([target_joints, rel_ori_b_flat], dim=-1)

  return target_joints, wm_goal_obs, actor_goal_obs


_compiled_compute_goal = torch.compile(_compute_goal, mode="reduce-overhead")


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
    self.motion = MotionLoader(
      motion_file,
      torch.tensor([0], device=self.device, dtype=torch.long),
      device=self.device,
    )
    self.cfg = env.cfg
    self.num_envs = batch_size
    self.horizon = horizon
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
    # full_states stores the full global state [pos(3), quat(4), qj(N), vel(3), ang(3), vj(N)]
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
    all_states = batch[0]  # (num_samples, history, full_state_dim)
    all_states_flat = all_states.reshape(-1, all_states.shape[-1])
    body_states_flat = compute_body_frame_state(all_states_flat)
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

    self._compute_goal()
    return self.get_observations()

  def detach(self):
    self.full_states = self.full_states.detach()
    self.hist = self.hist.detach()
    self.last_actions = self.last_actions.detach()

  def _compute_goal(self):
    target_joints, wm_goal_obs, actor_goal_obs = _compiled_compute_goal(
      self.motion_times,
      self.full_states,
      self.motion.body_quat_w,
      self.motion.body_pos_w,
      self.motion.joint_pos,
      self.motion.joint_vel,
    )
    # Clone to detach from CUDA graph output buffers (reduce-overhead mode).
    self._target_joints = target_joints.clone()
    self._wm_goal_obs = wm_goal_obs.clone()
    self._actor_goal_obs = actor_goal_obs

  def step(self, actions, auto_reset=True):
    offset = self._target_joints[..., : self.num_actions]
    proc_actions = actions * self.action_scale + offset

    next_body_states, pred_r, pred_d = self.wm.transition(
      self.hist, proc_actions, self.last_actions, self._wm_goal_obs
    )

    rewards = pred_r.sum(dim=-1) * self.dt
    dones = (pred_d > 0).float().squeeze(-1)

    next_full_states = self.wm.reconstruct(self.full_states, next_body_states)

    if self.wm.history == 1:
      next_hist = next_body_states.unsqueeze(1)
    else:
      next_hist = torch.cat([self.hist[:, 1:, :], next_body_states.unsqueeze(1)], dim=1)

    next_motion_times = torch.clamp(
      self.motion_times + 1, 0, self.motion.time_step_total - 1
    )
    next_episode_len = self.episode_length_buf + 1

    motion_time_outs = torch.isin(
      next_motion_times, self.motion.motion_ends - 2
    ).float()
    episode_time_outs = (next_episode_len >= self.max_episode_length).float()
    time_outs = torch.clamp(motion_time_outs + episode_time_outs, 0.0, 1.0)
    combined_dones = torch.clamp(dones + time_outs, 0.0, 1.0)

    if auto_reset:
      reset_mask = combined_dones > 0.5
      expand_mask = reset_mask.unsqueeze(-1)

      new_times, new_states, new_hist = self._sample_initial_states_from_buffer(
        self.num_envs
      )

      self.full_states = torch.where(
        expand_mask, new_states.to(dtype=next_full_states.dtype), next_full_states
      )
      self.hist = torch.where(
        expand_mask.unsqueeze(-1).expand_as(next_hist),
        new_hist.to(dtype=next_hist.dtype),
        next_hist,
      )
      self.motion_times = torch.where(reset_mask, new_times, next_motion_times)
      self.episode_length_buf = torch.where(
        reset_mask, torch.zeros_like(next_episode_len), next_episode_len
      )
      self.last_actions = torch.where(
        expand_mask, torch.zeros_like(proc_actions), proc_actions
      )
      self.survival_rate = torch.where(
        reset_mask, torch.ones_like(self.survival_rate), self.survival_rate
      )
    else:
      self.survival_rate = self.survival_rate * (1 - combined_dones)
      rewards = rewards * self.survival_rate
      combined_dones = 1 - self.survival_rate
      # Clone outputs from compiled (reduce-overhead) functions to detach from
      # CUDA graph output buffers, so reset() can safely write in-place later.
      self.full_states = next_full_states.clone()
      self.hist = next_hist.clone()
      self.last_actions = proc_actions
      self.motion_times = next_motion_times
      self.episode_length_buf = next_episode_len

    self._compute_goal()
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
