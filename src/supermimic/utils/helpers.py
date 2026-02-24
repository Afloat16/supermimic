"""Shared utility functions used across multiple modules.

Representation-specific helpers live in dedicated modules:
- :mod:`supermimic.utils.body_frame_helpers` — body-frame representation
- :mod:`supermimic.utils.maxcoord_helpers` — maximal-coordinate representation
"""

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_mul


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
