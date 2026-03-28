"""Getup WM environment: VaeWorldModelEnv with height tracking reward."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from mjlab.utils.lab_api.math import quat_apply

from supermimic.envs.vae_world_model_env import (
  DiscreteGoalSpec,
  GoalSpec,
  VaeWorldModelEnv,
  VaeWorldModelEnvCfg,
  get_forward_world,
)


@dataclass
class GetupEnvCfg(VaeWorldModelEnvCfg):
  task_env: str = field(default="getup", init=False)
  height_values: tuple[float, ...] = (0.0, 0.8)
  height_weight: float = 1.0
  face_up_weight: float = 1.0
  energy_weight: float = 1e-1
  lay_down_threshold: float = 0.4
  """Targets below this engage the lay-down auxiliary terms (face-up + potential)."""

  def goal_specs(self) -> list[GoalSpec]:
    vals = self.height_values
    return [
      DiscreteGoalSpec("Height (m)", choices=vals, default=vals[0]),
    ]


class GetupEnv(VaeWorldModelEnv[GetupEnvCfg]):
  """Height-tracking WM env: target_height passes through to policy goal."""

  goal_dim = 1
  raw_goal_dim = 1

  @staticmethod
  def build_goal_obs(raw_goal: torch.Tensor, aux_state: torch.Tensor) -> torch.Tensor:
    return raw_goal

  def sample_raw_goal(self, num_envs: int) -> torch.Tensor:
    vals = torch.tensor(self.cfg.height_values, device=self.device)
    idx = torch.randint(len(vals), (num_envs, 1), device=self.device)
    return vals[idx]

  def _compute_reward(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    is_last_mask: torch.Tensor,
  ) -> torch.Tensor:
    cfg = self.cfg
    root_height = aux_state[:, 2]
    target_height = self.goal[:, 0]
    height_loss = (root_height - target_height) ** 2
    total_loss = cfg.height_weight * height_loss

    # lay-down only: pull bodies to ground (potential) + damp motion (vel_penalty).
    is_lay_down = target_height < cfg.lay_down_threshold
    local_pos = self.representation.local_pos(rep_state)
    local_vel = self.representation.local_vel(rep_state)
    root_quat_b = aux_state[:, 3:7].unsqueeze(1).expand(-1, local_pos.shape[1], -1)
    body_height = aux_state[:, 2:3] + quat_apply(root_quat_b, local_pos)[..., 2]
    potential = (body_height * self._body_mass_w).sum(dim=-1)
    vel_penalty = 0.5 * (local_vel**2).mean(dim=(-2, -1))
    energy_loss = torch.where(is_lay_down, potential + vel_penalty, vel_penalty)
    total_loss = total_loss + cfg.energy_weight * energy_loss

    forward_world = get_forward_world(aux_state)
    face_up_loss = torch.where(
      is_lay_down, 1.0 - forward_world[:, 2], torch.zeros_like(root_height)
    )
    total_loss = total_loss + is_last_mask * cfg.face_up_weight * face_up_loss

    return -total_loss
