"""State representations for the world model.

Two representations are provided:
- BodyFrameRepresentation: joint-space body-frame representation (original)
- MaxCoordRepresentation: maximal-coordinate per-body representation
"""

from __future__ import annotations

from typing import Protocol

import torch

from supermimic.utils.helpers import (
  compute_body_frame_state,
  compute_maxcoord_rep_state,
  construct_maxcoord_policy_input,
  construct_policy_input,
  integrate_global_state,
  integrate_maxcoord_state,
  maxcoord_full_state_dim,
  maxcoord_prediction_dim,
  maxcoord_rep_state_dim,
  rep_state_dim_from_num_joints,
  rep_state_offsets,
)


class StateRepresentation(Protocol):
  full_state_dim: int
  rep_state_dim: int
  prediction_dim: int
  has_learned_reward: bool

  def represent(self, full_state: torch.Tensor) -> torch.Tensor: ...

  def apply_prediction(
    self,
    curr_full_state: torch.Tensor,
    curr_rep_state: torch.Tensor,
    prediction: torch.Tensor,
    dt: float,
  ) -> tuple[torch.Tensor, torch.Tensor]: ...

  def reconstruct(
    self, curr_full_state: torch.Tensor, rep_state: torch.Tensor, dt: float
  ) -> torch.Tensor: ...

  def loss_weights(self, cfg: object, device: torch.device | str) -> torch.Tensor: ...

  def loss_slices(self) -> dict[str, slice]: ...

  def construct_policy_input(
    self, full_state: torch.Tensor, goal: torch.Tensor
  ) -> torch.Tensor: ...

  def goal_dim_split(self) -> int:
    """Number of extra dims wm_goal has beyond actor_goal (e.g. rel_pos_b)."""
    ...


class BodyFrameRepresentation:
  """Joint-space body-frame representation.

  Full state: [root_pos(3), root_quat(4), qj(N), root_lin_vel(3), root_ang_vel(3), vj(N)]
  Rep state:  [z(1), g_b(3), v_b(3), w_b(3), qj(N), vj(N)]
  Prediction: same as rep_state (residual delta)
  """

  has_learned_reward = True

  def __init__(self, num_joints: int):
    self.num_joints = num_joints
    self.full_state_dim = 13 + 2 * num_joints
    self.rep_state_dim = rep_state_dim_from_num_joints(num_joints)
    self.prediction_dim = self.rep_state_dim  # dyn_head predicts rep-state delta

  def represent(self, full_state: torch.Tensor) -> torch.Tensor:
    flat = full_state.reshape(-1, self.full_state_dim)
    rep = compute_body_frame_state(flat)
    return rep.reshape(*full_state.shape[:-1], -1)

  def apply_prediction(
    self,
    curr_full_state: torch.Tensor,
    curr_rep_state: torch.Tensor,
    prediction: torch.Tensor,
    dt: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    next_rep = curr_rep_state + prediction
    next_full = integrate_global_state(curr_full_state, next_rep, dt)
    return next_full, next_rep

  def reconstruct(
    self, curr_full_state: torch.Tensor, rep_state: torch.Tensor, dt: float
  ) -> torch.Tensor:
    return integrate_global_state(curr_full_state, rep_state, dt)

  def loss_weights(self, cfg: object, device: torch.device | str) -> torch.Tensor:
    W = torch.ones(self.rep_state_dim, device=device)
    slices = self.loss_slices()
    weights = {
      "root_pos_z": 1.0 / getattr(cfg, "nominal_error_root_pos_z", 0.1) ** 2,
      "root_gravity_b": 1.0 / getattr(cfg, "nominal_error_root_gravity_b", 0.1) ** 2,
      "root_lin_vel": 1.0 / getattr(cfg, "nominal_error_root_lin_vel", 1.0) ** 2,
      "root_ang_vel": 1.0 / getattr(cfg, "nominal_error_root_ang_vel", 3.14) ** 2,
      "joint_pos": 1.0 / getattr(cfg, "nominal_error_joint_pos", 0.1) ** 2,
      "joint_vel": 1.0 / getattr(cfg, "nominal_error_joint_vel", 3.14) ** 2,
    }
    for name, sl in slices.items():
      W[sl] = weights[name]
    return W

  def loss_slices(self) -> dict[str, slice]:
    idx_qj, idx_vj = rep_state_offsets(self.num_joints)
    return {
      "root_pos_z": slice(0, 1),
      "root_gravity_b": slice(1, 4),
      "root_lin_vel": slice(4, 7),
      "root_ang_vel": slice(7, 10),
      "joint_pos": slice(10, 10 + self.num_joints),
      "joint_vel": slice(idx_vj, idx_vj + self.num_joints),
    }

  def construct_policy_input(
    self, full_state: torch.Tensor, goal: torch.Tensor
  ) -> torch.Tensor:
    return construct_policy_input(full_state, goal)

  def goal_dim_split(self) -> int:
    return 3  # wm_goal has 3 extra dims (rel_pos_b) beyond actor_goal


class MaxCoordRepresentation:
  """Maximal-coordinate per-body representation (following control_vae).

  Full state: [body_pos(B*3), body_quat(B*4), body_lin_vel(B*3), body_ang_vel(B*3)]
  Rep state:  [local_pos(B*3), local_rot_6d(B*6), local_vel(B*3), local_avel(B*3),
               body_height(B), up_dir(3)]
  Prediction: [vel_delta(B*3), avel_delta(B*3)] in root body frame

  Reward is computed analytically (not learned), following control_vae's pose_err.
  """

  has_learned_reward = False

  # Default reward weights (from control_vae ControlVAE.yml)
  DEFAULT_REWARD_WEIGHTS = {
    "pos": 0.2,
    "rot": 0.1,
    "vel": 0.5,
    "avel": 0.5,
    "height": 1.2,
    "up_dir": 3.0,
  }

  def __init__(self, num_bodies: int, reward_weights: dict[str, float] | None = None):
    self.num_bodies = num_bodies
    self.full_state_dim = maxcoord_full_state_dim(num_bodies)
    self.rep_state_dim = maxcoord_rep_state_dim(num_bodies)
    self.prediction_dim = maxcoord_prediction_dim(num_bodies)
    self.reward_weights = reward_weights or self.DEFAULT_REWARD_WEIGHTS.copy()

  def represent(self, full_state: torch.Tensor) -> torch.Tensor:
    flat = full_state.reshape(-1, self.full_state_dim)
    rep = compute_maxcoord_rep_state(flat, self.num_bodies)
    return rep.reshape(*full_state.shape[:-1], -1)

  def apply_prediction(
    self,
    curr_full_state: torch.Tensor,
    curr_rep_state: torch.Tensor,
    prediction: torch.Tensor,
    dt: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    next_full = integrate_maxcoord_state(
      curr_full_state, prediction, self.num_bodies, dt
    )
    next_rep = compute_maxcoord_rep_state(next_full, self.num_bodies)
    return next_full, next_rep

  def reconstruct(
    self, curr_full_state: torch.Tensor, rep_state: torch.Tensor, dt: float
  ) -> torch.Tensor:
    return curr_full_state

  def compute_reward(
    self,
    predicted_rep: torch.Tensor,
    target_rep: torch.Tensor,
    dt: float,
  ) -> torch.Tensor:
    """Compute per-env reward from pose error (following control_vae).

    Uses weighted L1 norms matching control_vae's pose_err function.
    Returns negative error as reward (so maximizing reward = minimizing error).

    Args:
      predicted_rep: (..., rep_state_dim) predicted rep state
      target_rep: (..., rep_state_dim) target rep state
      dt: simulation timestep

    Returns:
      (...,) per-env reward (negative pose error)
    """
    slices = self.loss_slices()
    delta = predicted_rep - target_rep
    w = self.reward_weights

    # L1 norms per component (summed across feature dims, not averaged)
    pos_err = w["pos"] * delta[..., slices["body_pos"]].abs().sum(-1)
    rot_err = w["rot"] * delta[..., slices["body_rot"]].abs().sum(-1)
    vel_err = dt * w["vel"] * delta[..., slices["body_vel"]].abs().sum(-1)
    avel_err = dt * w["avel"] * delta[..., slices["body_avel"]].abs().sum(-1)
    height_err = w["height"] * delta[..., slices["body_height"]].abs().sum(-1)
    up_dir_err = w["up_dir"] * delta[..., slices["up_dir"]].abs().sum(-1)

    total_err = pos_err + rot_err + vel_err + avel_err + height_err + up_dir_err
    return -total_err

  def loss_weights(self, cfg: object, device: torch.device | str) -> torch.Tensor:
    W = torch.ones(self.rep_state_dim, device=device)
    slices = self.loss_slices()
    weights = {
      "body_pos": 1.0 / getattr(cfg, "nominal_error_body_pos", 0.05) ** 2,
      "body_rot": 1.0 / getattr(cfg, "nominal_error_body_rot", 0.1) ** 2,
      "body_vel": 1.0 / getattr(cfg, "nominal_error_body_vel", 1.0) ** 2,
      "body_avel": 1.0 / getattr(cfg, "nominal_error_body_avel", 3.14) ** 2,
      "body_height": 1.0 / getattr(cfg, "nominal_error_body_height", 0.1) ** 2,
      "up_dir": 1.0 / getattr(cfg, "nominal_error_up_dir", 0.1) ** 2,
    }
    for name, sl in slices.items():
      W[sl] = weights[name]
    return W

  def loss_slices(self) -> dict[str, slice]:
    B = self.num_bodies
    off = 0
    slices = {}
    slices["body_pos"] = slice(off, off + B * 3)
    off += B * 3
    slices["body_rot"] = slice(off, off + B * 6)
    off += B * 6
    slices["body_vel"] = slice(off, off + B * 3)
    off += B * 3
    slices["body_avel"] = slice(off, off + B * 3)
    off += B * 3
    slices["body_height"] = slice(off, off + B)
    off += B
    slices["up_dir"] = slice(off, off + 3)
    return slices

  def construct_policy_input(
    self, full_state: torch.Tensor, goal: torch.Tensor
  ) -> torch.Tensor:
    return construct_maxcoord_policy_input(full_state, goal, self.num_bodies)

  def goal_dim_split(self) -> int:
    return 3  # wm_goal has 3 extra dims (rel_pos_b) beyond actor_goal
