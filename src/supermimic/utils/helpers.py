"""Shared utility functions used across multiple modules.

Representation-specific helpers live in dedicated modules:
- :mod:`supermimic.utils.body_frame_helpers` — body-frame representation
- :mod:`supermimic.utils.maxcoord_helpers` — maximal-coordinate representation
"""

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_mul


def extract_action_scale_offset(env) -> tuple[torch.Tensor, torch.Tensor]:
  """Extract action scale and default offset from an env's action manager.

  Returns 1-D tensors (squeezed to a single row if batched).
  """
  action_manager = env.unwrapped.action_manager
  action_term = action_manager.get_term(action_manager.active_terms[0])
  device = env.unwrapped.device
  scale = action_term.scale.to(device).float()
  if scale.dim() > 1:
    scale = scale[0:1]
  offset = action_term._offset.to(device).float()
  if offset.dim() > 1:
    offset = offset[0:1]
  return scale, offset


def shift_history(hist: torch.Tensor, new_state: torch.Tensor) -> torch.Tensor:
  """Shift a (B, H, D) history buffer by appending new_state and dropping the oldest."""
  if hist.shape[1] == 1:
    return new_state.unsqueeze(1)
  return torch.cat([hist[:, 1:, :], new_state.unsqueeze(1)], dim=1)


def _matrix_to_quat(rot_mat: torch.Tensor) -> torch.Tensor:
  """Rotation matrix to quaternion — compile-safe (no boolean indexing).

  Same algorithm as pytorch3d/mjlab ``quat_from_matrix`` but uses
  ``torch.clamp`` + ``torch.gather`` instead of boolean advanced indexing,
  so it produces only static-shaped tensors and works inside
  ``torch.compile(mode="reduce-overhead")``.

  Args:
    rot_mat: (..., 3, 3) rotation matrix

  Returns:
    (..., 4) quaternion (w, x, y, z)
  """
  m00, m01, m02 = rot_mat[..., 0, 0], rot_mat[..., 0, 1], rot_mat[..., 0, 2]
  m10, m11, m12 = rot_mat[..., 1, 0], rot_mat[..., 1, 1], rot_mat[..., 1, 2]
  m20, m21, m22 = rot_mat[..., 2, 0], rot_mat[..., 2, 1], rot_mat[..., 2, 2]

  q_abs = torch.sqrt(
    torch.clamp(
      torch.stack(
        [
          1.0 + m00 + m11 + m22,
          1.0 + m00 - m11 - m22,
          1.0 - m00 + m11 - m22,
          1.0 - m00 - m11 + m22,
        ],
        dim=-1,
      ),
      min=1e-10,
    )
  )

  quat_by_rijk = torch.stack(
    [
      torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
      torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
      torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
      torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
    ],
    dim=-2,
  )  # (..., 4, 4)

  quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].clamp(min=0.1))

  best = q_abs.argmax(dim=-1)  # (...)
  best_expanded = best[..., None, None].expand(*best.shape, 1, 4)
  return quat_candidates.gather(-2, best_expanded).squeeze(-2)


def rot6d_to_quat(rot6d: torch.Tensor) -> torch.Tensor:
  """Convert 6D rotation representation back to quaternion.

  Applies Gram-Schmidt orthogonalization to recover a valid rotation matrix,
  then converts to quaternion via a compile-safe path (no boolean indexing).

  Args:
    rot6d: (..., 6) first two columns of rotation matrix, flattened

  Returns:
    (..., 4) quaternion (w, x, y, z)
  """
  cols = rot6d.unflatten(-1, (3, 2))
  a1 = cols[..., 0]  # (..., 3)
  a2 = cols[..., 1]  # (..., 3)

  b1 = torch.nn.functional.normalize(a1, dim=-1)
  dot = (b1 * a2).sum(dim=-1, keepdim=True)
  b2 = torch.nn.functional.normalize(a2 - dot * b1, dim=-1)
  b3 = torch.cross(b1, b2, dim=-1)

  rot_mat = torch.stack([b1, b2, b3], dim=-1)
  return _matrix_to_quat(rot_mat)


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


def rodrigues_rotate(v: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
  """Rotate vectors v by axis-angle omega using Rodrigues' formula.

  Args:
    v: (..., 3) vectors to rotate
    omega: (..., 3) axis-angle rotation (axis * angle)

  Returns:
    (..., 3) rotated vectors
  """
  angle_sq = (omega * omega).sum(dim=-1, keepdim=True)
  angle = torch.sqrt(angle_sq + 1e-8)
  k = omega / angle  # unit axis
  cos_a = torch.cos(angle)
  sin_a = torch.sin(angle)
  # Rodrigues: v_rot = cos(θ)·v + sin(θ)·(k × v) + (1-cos(θ))·(k·v)·k
  k_cross_v = torch.linalg.cross(k, v, dim=-1)
  k_dot_v = (k * v).sum(dim=-1, keepdim=True)
  return cos_a * v + sin_a * k_cross_v + (1.0 - cos_a) * k_dot_v * k


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
