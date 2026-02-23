import torch
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from mjlab.utils.lab_api.math import subtract_frame_transforms
from tensordict import TensorDict

from supermimic.representations import BodyFrameRepresentation
from supermimic.utils.helpers import (
  build_maxcoord_full_state,
  compute_maxcoord_rep_state,
  quat_to_rot6d,
  shift_history,
)


def _assemble_goal(
  content: torch.Tensor,
  root_pos: torch.Tensor,
  root_quat: torch.Tensor,
  target_root_pos: torch.Tensor,
  target_root_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Compute relative root transform and assemble wm/actor goals.

  Returns:
    (wm_goal, actor_goal) where wm_goal has extra rel_pos_b dims.
  """
  rel_pos_b, rel_ori_w = subtract_frame_transforms(
    root_pos, root_quat, target_root_pos, target_root_quat
  )
  rel_ori_b_flat = quat_to_rot6d(rel_ori_w)
  rel_pos_b_flat = rel_pos_b.view(rel_pos_b.shape[0], -1)

  wm_goal = torch.cat([content, rel_pos_b_flat, rel_ori_b_flat], dim=-1)
  actor_goal = torch.cat([content, rel_ori_b_flat], dim=-1)
  return wm_goal, actor_goal


def _gather_target_joints(
  joint_pos: torch.Tensor, joint_vel: torch.Tensor, motion_times: torch.Tensor
) -> torch.Tensor:
  return torch.cat([joint_pos[motion_times], joint_vel[motion_times]], dim=-1)


def _compute_goal(
  motion_times: torch.Tensor,
  states: torch.Tensor,
  body_quat_w: torch.Tensor,
  body_pos_w: torch.Tensor,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  target_joints = _gather_target_joints(joint_pos, joint_vel, motion_times)

  wm_goal, actor_goal = _assemble_goal(
    target_joints,
    states[..., 0:3],
    states[..., 3:7],
    body_pos_w[motion_times, 0],
    body_quat_w[motion_times, 0],
  )
  return target_joints, wm_goal, actor_goal


def _compute_maxcoord_goal(
  motion_times: torch.Tensor,
  states: torch.Tensor,
  body_quat_w: torch.Tensor,
  body_pos_w: torch.Tensor,
  body_lin_vel_w: torch.Tensor,
  body_ang_vel_w: torch.Tensor,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
  num_bodies: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Compute goal for maxcoord representation.

  Goal content is target body rep state; target_joints returned for action offset.
  """
  target_joints = _gather_target_joints(joint_pos, joint_vel, motion_times)

  target_full = build_maxcoord_full_state(
    body_pos_w[motion_times],
    body_quat_w[motion_times],
    body_lin_vel_w[motion_times],
    body_ang_vel_w[motion_times],
  )
  target_rep = compute_maxcoord_rep_state(target_full, num_bodies)

  B = num_bodies
  wm_goal, actor_goal = _assemble_goal(
    target_rep,
    states[..., 0:3],
    states[..., B * 3 : B * 3 + 4],
    body_pos_w[motion_times, 0],
    body_quat_w[motion_times, 0],
  )
  return target_joints, wm_goal, actor_goal


_compiled_compute_goal = torch.compile(_compute_goal, mode="reduce-overhead")
_compiled_compute_maxcoord_goal = torch.compile(
  _compute_maxcoord_goal, mode="reduce-overhead"
)


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

    # Set up representation-specific dispatch (once, at init time).
    representation = self.wm.representation
    if isinstance(representation, BodyFrameRepresentation):
      self._goal_fn = self._body_frame_goal
      self._reward_fn = self._learned_reward
    else:
      self._goal_fn = self._maxcoord_goal
      self._reward_fn = self._analytical_reward

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

  # -- Goal dispatch methods ---------------------------------------------------

  def _body_frame_goal(self):
    return _compiled_compute_goal(
      self.motion_times,
      self.full_states,
      self.motion.body_quat_w,
      self.motion.body_pos_w,
      self.motion.joint_pos,
      self.motion.joint_vel,
    )

  def _maxcoord_goal(self):
    return _compiled_compute_maxcoord_goal(
      self.motion_times,
      self.full_states,
      self.motion.body_quat_w,
      self.motion.body_pos_w,
      self.motion.body_lin_vel_w,
      self.motion.body_ang_vel_w,
      self.motion.joint_pos,
      self.motion.joint_vel,
      self.wm.representation.num_bodies,
    )

  def _compute_goal(self):
    target_joints, wm_goal_obs, actor_goal_obs = self._goal_fn()
    # Clone to detach from CUDA graph output buffers (reduce-overhead mode).
    self._target_joints = target_joints.clone()
    self._wm_goal_obs = wm_goal_obs.clone()
    self._actor_goal_obs = actor_goal_obs

  # -- Reward dispatch methods -------------------------------------------------

  def _learned_reward(self, next_body_states, pred_r, proc_actions):
    return pred_r.sum(dim=-1) * self.dt

  def _analytical_reward(self, next_body_states, pred_r, proc_actions):
    next_mt = torch.clamp(self.motion_times + 1, 0, self.motion.time_step_total - 1)
    target_full = build_maxcoord_full_state(
      self.motion.body_pos_w[next_mt],
      self.motion.body_quat_w[next_mt],
      self.motion.body_lin_vel_w[next_mt],
      self.motion.body_ang_vel_w[next_mt],
    )
    target_rep = compute_maxcoord_rep_state(
      target_full, self.wm.representation.num_bodies
    )
    reward = self.wm.representation.compute_reward(next_body_states, target_rep)
    # Action rate penalty (L2 squared, matching tracking env weight -0.1)
    action_rate = torch.sum(torch.square(proc_actions - self.last_actions), dim=-1)
    return self.dt * (reward - 0.1 * action_rate)

  def step(self, actions, auto_reset=True):
    offset = self._target_joints[..., : self.num_actions]
    proc_actions = actions * self.action_scale + offset

    next_full_states, next_body_states, pred_r, pred_d = self.wm.transition(
      self.hist, proc_actions, self.last_actions, self._wm_goal_obs, self.full_states
    )
    rewards = self._reward_fn(next_body_states, pred_r, proc_actions)

    dones = (pred_d > 0).float().squeeze(-1)

    next_hist = shift_history(self.hist, next_body_states)

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
