import torch
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
)


def get_yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
  R = matrix_from_quat(quat)
  f_w = R[..., :, 0]
  return torch.atan2(f_w[..., 1], f_w[..., 0])


def quat_from_two_vectors(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
  v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
  v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)
  dot = (v1 * v2).sum(dim=-1, keepdim=True)
  w = 1 + dot
  xyz = torch.linalg.cross(v1, v2, dim=-1)
  q = torch.cat([w, xyz], dim=-1)
  q_norm = torch.norm(q, dim=-1, keepdim=True)
  return q / (q_norm + 1e-8)
