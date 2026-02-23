import torch
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_inv,
  quat_mul,
)


def shift_history(hist: torch.Tensor, new_state: torch.Tensor) -> torch.Tensor:
  """Shift a (B, H, D) history buffer by appending new_state and dropping the oldest."""
  if hist.shape[1] == 1:
    return new_state.unsqueeze(1)
  return torch.cat([hist[:, 1:, :], new_state.unsqueeze(1)], dim=1)


def quat_to_rot6d(quat: torch.Tensor) -> torch.Tensor:
  """Convert quaternions to 6D rotation representation (first two columns of rotation matrix)."""
  mat = matrix_from_quat(quat)  # (..., 3, 3)
  return mat[..., :2].flatten(-2, -1)  # (..., 6)


def unpack_obs_dims(obs) -> tuple[int, int]:
  """Extract full_state and goal dimensions from observations.

  Handles both TensorDict (keyed) and raw tensor observations.
  """
  state_obs = obs["full_state"] if hasattr(obs, "keys") else obs
  goal_obs = obs["goal"] if hasattr(obs, "keys") else obs
  return state_obs.shape[-1], goal_obs.shape[-1]


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
  world_down = torch.zeros_like(root_lin_vel)
  world_down[..., 2] = -1.0
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

  q_next = quat_integrate(root_quat, w_global, dt)

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


# ---------------------------------------------------------------------------
# Quaternion integration
# ---------------------------------------------------------------------------


def quat_integrate(
  quat: torch.Tensor, ang_vel: torch.Tensor, dt: float
) -> torch.Tensor:
  """Integrate quaternion by angular velocity over dt (axis-angle method).

  Args:
    quat: (..., 4) current quaternion (w, x, y, z)
    ang_vel: (..., 3) angular velocity in world frame
    dt: time step

  Returns:
    (..., 4) next quaternion, normalized
  """
  omega = ang_vel * dt
  norm_sq = (omega * omega).sum(dim=-1, keepdim=True)
  norm_safe = torch.sqrt(norm_sq + 1e-8)
  half_angle = norm_safe / 2
  axis = omega / norm_safe
  delta_q = torch.cat([torch.cos(half_angle), torch.sin(half_angle) * axis], dim=-1)
  q_next = quat_mul(delta_q, quat)
  return q_next / torch.norm(q_next, dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Maximal coordinate helpers
# ---------------------------------------------------------------------------

# Full state layout for maxcoord:
#   [body_pos(B*3), body_quat(B*4), body_lin_vel(B*3), body_ang_vel(B*3)]
# where B = num_bodies.  Total dim = B*13.
#
# Rep state layout (matches control_vae's state2ob):
#   [local_pos(B*3), local_rot_6d(B*6), local_vel(B*3), local_avel(B*3),
#    body_height(B), up_dir(3)]
# Total dim = B*16 + 3
#
# Prediction layout (dyn_head output):
#   [vel_delta(B*3), avel_delta(B*3)]  (in root body frame)
# Total dim = B*6


def maxcoord_full_state_dim(num_bodies: int) -> int:
  return num_bodies * 13


def maxcoord_rep_state_dim(num_bodies: int) -> int:
  return num_bodies * 16 + 3


def maxcoord_prediction_dim(num_bodies: int) -> int:
  return num_bodies * 6


def _reshape_bodies(flat: torch.Tensor, num_bodies: int, feat: int) -> torch.Tensor:
  """(..., B*feat) -> (..., B, feat)"""
  return flat.unflatten(-1, (num_bodies, feat))


def decompose_maxcoord_state(
  full_state: torch.Tensor, num_bodies: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Decompose flat maxcoord full state into per-body tensors.

  Returns:
    body_pos: (..., B, 3)
    body_quat: (..., B, 4)
    body_lin_vel: (..., B, 3)
    body_ang_vel: (..., B, 3)
  """
  B = num_bodies
  body_pos = _reshape_bodies(full_state[..., : B * 3], B, 3)
  body_quat = _reshape_bodies(full_state[..., B * 3 : B * 7], B, 4)
  body_lin_vel = _reshape_bodies(full_state[..., B * 7 : B * 10], B, 3)
  body_ang_vel = _reshape_bodies(full_state[..., B * 10 : B * 13], B, 3)
  return body_pos, body_quat, body_lin_vel, body_ang_vel


def build_maxcoord_full_state(
  body_pos: torch.Tensor,
  body_quat: torch.Tensor,
  body_lin_vel: torch.Tensor,
  body_ang_vel: torch.Tensor,
) -> torch.Tensor:
  """Build flat maxcoord full state from per-body tensors.

  Args:
    body_pos: (..., B, 3) body positions
    body_quat: (..., B, 4) body quaternions
    body_lin_vel: (..., B, 3) body linear velocities
    body_ang_vel: (..., B, 3) body angular velocities

  Returns:
    (..., B*13) flat maxcoord full state
  """
  return torch.cat(
    [
      body_pos.flatten(-2, -1),
      body_quat.flatten(-2, -1),
      body_lin_vel.flatten(-2, -1),
      body_ang_vel.flatten(-2, -1),
    ],
    dim=-1,
  )


def compute_maxcoord_rep_state(
  full_state: torch.Tensor, num_bodies: int
) -> torch.Tensor:
  """Convert maxcoord full state to root-relative rep state.

  Follows control_vae's state2ob layout:
    [local_pos(B*3), local_rot_6d(B*6), local_vel(B*3), local_avel(B*3),
     body_height(B), up_dir(3)]

  Args:
    full_state: (..., B*13) flat maxcoord state
    num_bodies: B

  Returns:
    (..., B*16 + 3) rep state
  """
  body_pos, body_quat, body_lin_vel, body_ang_vel = decompose_maxcoord_state(
    full_state, num_bodies
  )

  # Root frame
  root_pos = body_pos[..., 0:1, :]  # (..., 1, 3)
  root_quat = body_quat[..., 0:1, :]  # (..., 1, 4)
  root_quat_inv = quat_inv(root_quat.reshape(-1, 4)).reshape(root_quat.shape)

  # Local positions: relative to root, rotated into root frame
  rel_pos_w = body_pos - root_pos  # (..., B, 3)
  local_pos = quat_apply_inverse(
    root_quat.expand_as(body_quat),
    rel_pos_w,
  )  # (..., B, 3)

  # Local rotations as 6D representation (first two columns of rotation matrix)
  rel_quat = quat_mul(
    root_quat_inv.expand_as(body_quat),
    body_quat,
  )  # (..., B, 4)
  local_rot_6d = quat_to_rot6d(rel_quat)  # (..., B, 6)

  # Local velocities: rotated into root frame
  local_vel = quat_apply_inverse(
    root_quat.expand_as(body_quat),
    body_lin_vel,
  )  # (..., B, 3)
  local_avel = quat_apply_inverse(
    root_quat.expand_as(body_quat),
    body_ang_vel,
  )  # (..., B, 3)

  # Per-body heights (Z coordinate in world frame)
  body_height = body_pos[..., 2]  # (..., B)

  # Up direction in root frame
  world_down = torch.zeros_like(root_pos[..., 0, :])  # (..., 3)
  world_down[..., 2] = -1.0
  up_dir = quat_apply_inverse(root_quat[..., 0, :], world_down)  # (..., 3)

  # Flatten per-body features and concatenate
  return torch.cat(
    [
      local_pos.flatten(-2, -1),  # B*3
      local_rot_6d.flatten(-2, -1),  # B*6
      local_vel.flatten(-2, -1),  # B*3
      local_avel.flatten(-2, -1),  # B*3
      body_height,  # B
      up_dir,  # 3
    ],
    dim=-1,
  )


def integrate_maxcoord_state(
  curr_full_state: torch.Tensor,
  vel_deltas: torch.Tensor,
  num_bodies: int,
  dt: float,
) -> torch.Tensor:
  """Integrate maxcoord state forward by velocity deltas.

  Args:
    curr_full_state: (..., B*13) current maxcoord full state
    vel_deltas: (..., B*6) velocity deltas in root body frame [dv(B*3), dw(B*3)]
    num_bodies: B
    dt: time step

  Returns:
    (..., B*13) next maxcoord full state
  """
  B = num_bodies
  body_pos, body_quat, body_lin_vel, body_ang_vel = decompose_maxcoord_state(
    curr_full_state, B
  )

  # Unpack deltas (in root body frame)
  dv_local = _reshape_bodies(vel_deltas[..., : B * 3], B, 3)  # (..., B, 3)
  dw_local = _reshape_bodies(vel_deltas[..., B * 3 : B * 6], B, 3)  # (..., B, 3)

  # Rotate deltas to world frame using root quaternion
  root_quat = body_quat[..., 0:1, :]  # (..., 1, 4)
  dv_world = quat_apply(root_quat.expand_as(body_quat), dv_local)
  dw_world = quat_apply(root_quat.expand_as(body_quat), dw_local)

  # Update velocities
  next_lin_vel = body_lin_vel + dv_world
  next_ang_vel = body_ang_vel + dw_world

  # Integrate positions
  next_pos = body_pos + next_lin_vel * dt

  # Integrate quaternions
  next_quat = quat_integrate(
    body_quat.reshape(-1, 4),
    next_ang_vel.reshape(-1, 3),
    dt,
  ).reshape(body_quat.shape)

  # Flatten back to full state
  return torch.cat(
    [
      next_pos.flatten(-2, -1),
      next_quat.flatten(-2, -1),
      next_lin_vel.flatten(-2, -1),
      next_ang_vel.flatten(-2, -1),
    ],
    dim=-1,
  )


def construct_maxcoord_policy_input(
  full_state: torch.Tensor,
  goal: torch.Tensor,
  num_bodies: int,
) -> torch.Tensor:
  """Construct policy input from maxcoord full state and goal."""
  rep_state = compute_maxcoord_rep_state(full_state, num_bodies)
  return torch.cat([rep_state, goal], dim=-1)
