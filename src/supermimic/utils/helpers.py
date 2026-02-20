import torch
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_mul,
)


def infer_num_joints_from_state_dim(state_dim: int) -> int:
  """Infers number of joints from full-state dimension.

  Full-state layout is:
  [root_pos(3), root_quat(4), qj(N), root_lin_vel(3), root_ang_vel(3), vj(N)]
  """
  if state_dim < 13 or (state_dim - 13) % 2 != 0:
    raise ValueError(
      f"Invalid full-state dim {state_dim}: expected 13 + 2 * num_joints"
    )
  return (state_dim - 13) // 2


def rep_state_dim_from_num_joints(num_joints: int) -> int:
  return 10 + 2 * num_joints


def rep_state_dim_from_state_dim(state_dim: int) -> int:
  return rep_state_dim_from_num_joints(infer_num_joints_from_state_dim(state_dim))


def full_state_offsets(num_joints: int) -> tuple[int, int, int, int]:
  """Returns start offsets for [qj, root_lin_vel, root_ang_vel, vj] in full state."""
  idx_qj = 7
  idx_lin = idx_qj + num_joints
  idx_ang = idx_lin + 3
  idx_vj = idx_ang + 3
  return idx_qj, idx_lin, idx_ang, idx_vj


def rep_state_offsets(num_joints: int) -> tuple[int, int]:
  """Returns start offsets for [qj, vj] in rep state."""
  idx_qj = 10
  idx_vj = idx_qj + num_joints
  return idx_qj, idx_vj


def decompose_state(
  full_state: torch.Tensor,
) -> tuple[
  torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
  """Decomposes full state into canonical components."""
  num_joints = infer_num_joints_from_state_dim(full_state.shape[-1])

  idx_qj, idx_lin, idx_ang, idx_vj = full_state_offsets(num_joints)

  root_pos = full_state[..., 0:3]
  root_quat = full_state[..., 3:7]
  qj = full_state[..., idx_qj:idx_lin]
  root_lin_vel = full_state[..., idx_lin:idx_ang]
  root_ang_vel = full_state[..., idx_ang:idx_vj]
  vj = full_state[..., idx_vj : idx_vj + num_joints]
  return root_pos, root_quat, qj, root_lin_vel, root_ang_vel, vj


def decompose_rep_state(
  rep_state: torch.Tensor,
) -> tuple[
  torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
  """Decomposes body-frame state into canonical components."""
  # rep state layout: [z(1), g_b(3), v_b(3), w_b(3), qj(N), vj(N)]
  if rep_state.shape[-1] < 10 or (rep_state.shape[-1] - 10) % 2 != 0:
    raise ValueError(
      f"Invalid rep-state dim {rep_state.shape[-1]}: expected 10 + 2 * num_joints"
    )
  num_joints = (rep_state.shape[-1] - 10) // 2

  idx_qj, idx_vj = rep_state_offsets(num_joints)

  z = rep_state[..., 0:1]
  g_b = rep_state[..., 1:4]
  v_b = rep_state[..., 4:7]
  w_b = rep_state[..., 7:10]
  qj = rep_state[..., idx_qj:idx_vj]
  vj = rep_state[..., idx_vj : idx_vj + num_joints]
  return z, g_b, v_b, w_b, qj, vj


def compute_body_frame_state(full_state: torch.Tensor) -> torch.Tensor:
  """Converts raw state to body-frame representation.

  Uses root body frame for velocities (no heading alignment).
  This avoids all singularities since we're just rotating to body frame.
  """
  (
    root_pos,
    root_quat,
    qj,
    root_lin_vel,
    root_ang_vel,
    vj,
  ) = decompose_state(full_state)

  z = root_pos[..., 2:3]
  world_down = torch.tensor([0.0, 0.0, -1.0], device=full_state.device).expand_as(
    root_lin_vel
  )
  g_b = quat_apply_inverse(root_quat, world_down)
  v_b = quat_apply_inverse(root_quat, root_lin_vel)
  w_b = quat_apply_inverse(root_quat, root_ang_vel)

  return torch.cat([z, g_b, v_b, w_b, qj, vj], dim=-1)


def construct_policy_input(
  full_state: torch.Tensor,
  goal: torch.Tensor,
) -> torch.Tensor:
  """Constructs the policy input from full state and goal.

  Args:
    full_state: Full state tensor [..., full_state_dim]
    goal: Goal tensor [..., goal_dim]

  Returns:
    Policy input tensor [..., policy_input_dim]
  """
  body_state = compute_body_frame_state(full_state)
  return torch.cat([body_state, goal], dim=-1)


def integrate_global_state(
  curr_global_state: torch.Tensor,
  next_body_state: torch.Tensor,
  dt: float,
) -> torch.Tensor:
  num_joints = infer_num_joints_from_state_dim(curr_global_state.shape[-1])

  z, _, v_b, w_b, qj, vj = decompose_rep_state(next_body_state)

  root_quat = curr_global_state[..., 3:7]
  v_global = quat_apply(root_quat, v_b)
  w_global = quat_apply(root_quat, w_b)

  angle = torch.norm(w_global, dim=-1, keepdim=True) * dt
  axis = w_global / (torch.norm(w_global, dim=-1, keepdim=True) + 1e-8)
  delta_q = torch.cat([torch.cos(angle / 2), torch.sin(angle / 2) * axis], dim=-1)
  q_next = quat_mul(delta_q, root_quat)
  q_next = q_next / torch.norm(q_next, dim=-1, keepdim=True)

  next_global_state = curr_global_state.clone()
  next_global_state[..., 0:3] += v_global * dt
  next_global_state[..., 2] = z.squeeze(-1)
  next_global_state[..., 3:7] = q_next

  idx_qj, idx_lin, idx_ang, idx_vj = full_state_offsets(num_joints)
  next_global_state[..., idx_qj : idx_qj + num_joints] = qj
  next_global_state[..., idx_vj:] = vj
  next_global_state[..., idx_lin : idx_lin + 3] = v_global
  next_global_state[..., idx_ang : idx_ang + 3] = w_global

  return next_global_state
