"""Gaussian-kernel tracking reward on world-frame bodies.

Mirrors ``mjlab.tasks.tracking.tracking_env_cfg.rewards``: each tracking term is
``weight * exp(-err_sq / std**2)``, plus the action-rate and joint-limit penalties;
body targets are re-anchored to the predicted root xy + yaw, matching
``MotionCommand.update_relative_body_poses`` and ``to_robot_heading``.
"""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_inv, quat_mul, yaw_quat

from supermimic.utils.helpers import rotation_error_magnitude, small_matmul


def _reanchor_targets(
  target_pos: torch.Tensor,
  target_quat: torch.Tensor,
  target_lin_vel: torch.Tensor,
  target_ang_vel: torch.Tensor,
  aux_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Re-anchor target bodies to the predicted root xy + yaw in ``aux_state``.

  The frame keeps the target root's z and roll/pitch; velocities only rotate.
  """
  delta_pos = torch.cat([aux_state[..., :2], target_pos[..., 0, 2:3]], dim=-1)
  delta_mat = matrix_from_quat(
    yaw_quat(quat_mul(aux_state[..., 3:7], quat_inv(target_quat[..., 0, :])))
  ).unsqueeze(-3)

  def rotate(vec: torch.Tensor) -> torch.Tensor:
    return small_matmul(delta_mat, vec.unsqueeze(-1))[..., 0]

  tgt_pos = delta_pos.unsqueeze(-2) + rotate(target_pos - target_pos[..., 0:1, :])
  tgt_mat = small_matmul(delta_mat, matrix_from_quat(target_quat))
  return tgt_pos, tgt_mat, rotate(target_lin_vel), rotate(target_ang_vel)


def tracking_reward(
  aux_state: torch.Tensor,
  pred_pos: torch.Tensor,
  pred_rot_mat: torch.Tensor,
  pred_lin_vel: torch.Tensor,
  pred_ang_vel: torch.Tensor,
  joint_state: torch.Tensor,
  action: torch.Tensor,
  last_action: torch.Tensor,
  target_pos: torch.Tensor,
  target_quat: torch.Tensor,
  target_lin_vel: torch.Tensor,
  target_ang_vel: torch.Tensor,
  soft_joint_pos_limits: torch.Tensor,
  dt: float,
) -> torch.Tensor:
  """Gaussian-kernel reward matching the motion-tracking env.

  ``pred_*`` are the predicted bodies in the world frame and ``aux_state`` the
  predicted root pose the targets are re-anchored to.
  """
  tgt_pos, tgt_mat, tgt_lin_vel, tgt_ang_vel = _reanchor_targets(
    target_pos, target_quat, target_lin_vel, target_ang_vel, aux_state
  )

  terms: list[tuple[torch.Tensor, float, float]] = [
    (torch.sum((pred_pos[..., 0, :] - target_pos[..., 0, :]) ** 2, dim=-1), 0.5, 0.3),
    (
      rotation_error_magnitude(
        pred_rot_mat[..., 0, :, :], matrix_from_quat(target_quat[..., 0, :])
      )
      ** 2,
      0.5,
      0.4,
    ),
    (torch.sum((pred_pos - tgt_pos) ** 2, dim=-1).mean(-1), 1.0, 0.3),
    ((rotation_error_magnitude(pred_rot_mat, tgt_mat) ** 2).mean(-1), 1.0, 0.4),
    (torch.sum((pred_lin_vel - tgt_lin_vel) ** 2, dim=-1).mean(-1), 1.0, 1.0),
    (torch.sum((pred_ang_vel - tgt_ang_vel) ** 2, dim=-1).mean(-1), 1.0, 3.14),
  ]
  reward = sum(w * torch.exp(-err / std**2) for err, w, std in terms)
  action_rate = torch.sum((action - last_action) ** 2, dim=-1)
  joint_pos, _ = joint_state.chunk(2, dim=-1)
  lo, hi = soft_joint_pos_limits.unbind(-1)
  joint_limit = ((lo - joint_pos).clamp(min=0) + (joint_pos - hi).clamp(min=0)).sum(-1)
  return dt * (reward - 0.1 * action_rate - 2.0 * joint_limit)
