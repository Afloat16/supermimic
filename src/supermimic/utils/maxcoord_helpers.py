"""Maximal-coordinate representation helper functions.

Full state layout:
  [body_pos(B*3), body_quat(B*4), body_lin_vel(B*3), body_ang_vel(B*3)]
  where B = num_bodies.  Total dim = B*13.

Rep state layout (matches control_vae's state2ob):
  [local_pos(B*3), local_rot_6d(B*6), local_vel(B*3), local_avel(B*3),
   body_height(B), up_dir(3)]
  Total dim = B*16 + 3

Prediction layout (dyn_head output):
  [vel_delta(B*3), avel_delta(B*3)]  (in root body frame)
  Total dim = B*6
"""

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul

from supermimic.utils.helpers import quat_integrate, quat_to_rot6d


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
