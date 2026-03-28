"""Velocity WM environment: VaeWorldModelEnv with velocity tracking reward."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from mjlab.utils.lab_api.math import quat_apply

from supermimic.envs.vae_world_model_env import (
  ContinuousGoalSpec,
  GoalSpec,
  VaeWorldModelEnv,
  VaeWorldModelEnvCfg,
  get_facing_angle,
  get_forward_world,
)


@dataclass
class VelocityEnvCfg(VaeWorldModelEnvCfg):
  task_env: str = field(default="velocity", init=False)
  speed_range: tuple[float, float] = (-1.0, 3.0)
  angle_range: tuple[float, float] = (-math.pi, math.pi)
  direction_weight: float = 1.0
  speed_weight: float = 1.0

  def goal_specs(self) -> list[GoalSpec]:
    lo_s, hi_s = self.speed_range
    lo_a, hi_a = self.angle_range
    return [
      ContinuousGoalSpec("Speed (m/s)", lo_s, hi_s, step=0.1, default=0.0),
      ContinuousGoalSpec("Angle (rad)", lo_a, hi_a, step=0.05, default=0.0),
    ]


class VelocityEnv(VaeWorldModelEnv[VelocityEnvCfg]):
  """Velocity-tracking WM env: heading-relative [speed, cos(delta), sin(delta)]."""

  goal_dim = 3
  raw_goal_dim = 2

  @staticmethod
  def build_goal_obs(raw_goal: torch.Tensor, aux_state: torch.Tensor) -> torch.Tensor:
    facing_angle = get_facing_angle(aux_state)
    delta = raw_goal[:, 1] - facing_angle
    return torch.stack([raw_goal[:, 0], torch.cos(delta), torch.sin(delta)], dim=-1)

  def sample_raw_goal(self, num_envs: int) -> torch.Tensor:
    lo_s, hi_s = self.cfg.speed_range
    lo_a, hi_a = self.cfg.angle_range
    speed = torch.rand(num_envs, 1, device=self.device) * (hi_s - lo_s) + lo_s
    angle = torch.rand(num_envs, 1, device=self.device) * (hi_a - lo_a) + lo_a
    return torch.cat([speed, angle], dim=-1)

  def _compute_reward(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    is_last_mask: torch.Tensor,
  ) -> torch.Tensor:
    cfg = self.cfg
    local_vel = self.representation.local_vel(rep_state)
    B = local_vel.shape[1]
    root_quat = aux_state[:, 3:7]
    world_vel = quat_apply(root_quat.unsqueeze(1).expand(-1, B, -1), local_vel)
    com_vel_xy = (world_vel[..., :2] * self._body_mass_w.view(1, B, 1)).sum(dim=1)

    target_speed = self.goal[:, 0]
    target_angle = self.goal[:, 1]
    target_dir = torch.stack([torch.cos(target_angle), torch.sin(target_angle)], dim=-1)
    # speed=0 case: penalize any horizontal movement; otherwise track projected speed.
    projected_speed = (com_vel_xy * target_dir).sum(dim=-1)
    speed_err = torch.where(
      target_speed.abs() < 1e-3,
      com_vel_xy.abs().sum(dim=-1),
      (projected_speed - target_speed).abs() / target_speed.abs().clamp(min=1.0),
    )

    # End-of-rollout direction + speed reward, masked into the last step only —
    # a single coherent gradient avoids conflicting per-step gradients.
    forward_xy = F.normalize(get_forward_world(aux_state)[:, :2], dim=-1)
    direction_loss = 1.0 - (forward_xy * target_dir).sum(dim=-1)
    end_loss = cfg.direction_weight * direction_loss + cfg.speed_weight * speed_err
    return -is_last_mask * end_loss
