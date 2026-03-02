"""State representations for the world model.

Two representations are provided:
- BodyFrameRepresentation: joint-space body-frame representation (original)
- MaxCoordRepresentation: maximal-coordinate per-body representation
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from mjlab.utils.lab_api.math import subtract_frame_transforms

from supermimic.utils.body_frame_helpers import (
  compute_body_frame_state,
  construct_policy_input,
  infer_num_joints_from_state_dim,
  integrate_global_state,
  rep_state_dim_from_num_joints,
  rep_state_offsets,
)
from supermimic.utils.helpers import quat_to_rot6d
from supermimic.utils.maxcoord_helpers import (
  build_maxcoord_full_state,
  compute_maxcoord_rep_state,
  construct_maxcoord_policy_input,
  integrate_maxcoord_state,
  maxcoord_full_state_dim,
  maxcoord_prediction_dim,
  maxcoord_rep_state_dim,
)


def create_representation(name: str, full_state_dim: int) -> StateRepresentation:
  """Create a state representation from its name and full-state dimension.

  Args:
    name: "body_frame" or "maxcoord"
    full_state_dim: dimension of the full state vector

  Returns:
    A BodyFrameRepresentation or MaxCoordRepresentation instance.
  """
  if name == "body_frame":
    num_joints = infer_num_joints_from_state_dim(full_state_dim)
    return BodyFrameRepresentation(num_joints)
  elif name == "maxcoord":
    if full_state_dim % 13 != 0:
      raise ValueError(
        f"Maxcoord full_state_dim ({full_state_dim}) must be divisible by 13"
      )
    return MaxCoordRepresentation(full_state_dim // 13)
  else:
    raise ValueError(
      f"Unknown representation: {name}. Must be 'body_frame' or 'maxcoord'."
    )


def _bryson_loss_weights(
  rep_state_dim: int,
  slices: dict[str, slice],
  weight_config: dict[str, tuple[str, float]],
  cfg: object,
  device: torch.device | str,
) -> torch.Tensor:
  """Compute Bryson's-rule loss weights: W[slice] = 1 / nominal_error^2.

  Args:
    rep_state_dim: total dimension of the representation state
    slices: mapping from component name to its slice in rep state
    weight_config: mapping from component name to (cfg_attr, default_value)
    cfg: config object with nominal error attributes
    device: torch device
  """
  W = torch.ones(rep_state_dim, device=device)
  for name, sl in slices.items():
    attr, default = weight_config[name]
    W[sl] = 1.0 / getattr(cfg, attr, default) ** 2
  return W


def _assemble_goal(
  content: torch.Tensor,
  root_pos: torch.Tensor,
  root_quat: torch.Tensor,
  target_root_pos: torch.Tensor,
  target_root_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Compute relative root transform and assemble wm/actor goals.

  Returns:
    (wm_goal, actor_goal) where wm_goal has extra rel_pos_b dims.
  """
  rel_pos_b, rel_ori_w = subtract_frame_transforms(
    root_pos, root_quat, target_root_pos, target_root_quat
  )
  rel_ori_b_flat = quat_to_rot6d(rel_ori_w)
  rel_pos_b_flat = rel_pos_b.view(rel_pos_b.shape[0], -1)

  wm_goal = torch.cat([content, rel_pos_b_flat, rel_ori_b_flat], dim=-1)
  actor_goal = torch.cat([content, rel_ori_b_flat], dim=-1)
  return wm_goal, actor_goal


def _gather_target_joints(
  joint_pos: torch.Tensor, joint_vel: torch.Tensor, motion_times: torch.Tensor
) -> torch.Tensor:
  return torch.cat([joint_pos[motion_times], joint_vel[motion_times]], dim=-1)


def _compute_goal(
  motion_times: torch.Tensor,
  root_pos: torch.Tensor,
  root_quat: torch.Tensor,
  body_pos_w: torch.Tensor,
  body_quat_w: torch.Tensor,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  target_joints = _gather_target_joints(joint_pos, joint_vel, motion_times)
  wm_goal, actor_goal = _assemble_goal(
    target_joints,
    root_pos,
    root_quat,
    body_pos_w[motion_times, 0],
    body_quat_w[motion_times, 0],
  )
  return target_joints, wm_goal, actor_goal


_compiled_goal = torch.compile(_compute_goal, mode="reduce-overhead")


class StateRepresentation(ABC):
  """Abstract base class for state representations.

  Subclasses must set these instance attributes in ``__init__``:
    full_state_dim, rep_state_dim, prediction_dim, has_learned_reward

  And define this class attribute:
    LOSS_WEIGHT_CONFIG: dict[str, tuple[str, float]]
  """

  full_state_dim: int
  rep_state_dim: int
  prediction_dim: int
  has_learned_reward: bool
  LOSS_WEIGHT_CONFIG: dict[str, tuple[str, float]]

  # ------------------------------------------------------------------
  # Abstract methods (must be implemented by subclasses)
  # ------------------------------------------------------------------

  @abstractmethod
  def represent(self, full_state: torch.Tensor) -> torch.Tensor: ...

  @abstractmethod
  def apply_prediction(
    self,
    curr_full_state: torch.Tensor,
    curr_rep_state: torch.Tensor,
    prediction: torch.Tensor,
    dt: float,
  ) -> tuple[torch.Tensor, torch.Tensor]: ...

  @abstractmethod
  def reconstruct(
    self, curr_full_state: torch.Tensor, rep_state: torch.Tensor, dt: float
  ) -> torch.Tensor: ...

  @abstractmethod
  def loss_slices(self) -> dict[str, slice]: ...

  @abstractmethod
  def construct_policy_input(
    self, full_state: torch.Tensor, goal: torch.Tensor
  ) -> torch.Tensor: ...

  def compute_wm_goal(
    self,
    motion_times: torch.Tensor,
    full_states: torch.Tensor,
    motion: object,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (target_joints, wm_goal, actor_goal) from motion data."""
    root_pos, root_quat = self.extract_root_pos_quat(full_states)
    return _compiled_goal(
      motion_times,
      root_pos,
      root_quat,
      motion.body_pos_w,
      motion.body_quat_w,
      motion.joint_pos,
      motion.joint_vel,
    )

  @abstractmethod
  def compute_wm_reward(
    self,
    next_body_states: torch.Tensor,
    pred_r: torch.Tensor,
    actions: torch.Tensor,
    last_actions: torch.Tensor,
    motion: object,
    motion_times: torch.Tensor,
    dt: float,
  ) -> torch.Tensor:
    """Compute per-env reward for world model env rollouts."""
    ...

  # ------------------------------------------------------------------
  # Default implementations (may be overridden)
  # ------------------------------------------------------------------

  def loss_weights(self, cfg: object, device: torch.device | str) -> torch.Tensor:
    return _bryson_loss_weights(
      self.rep_state_dim, self.loss_slices(), self.LOSS_WEIGHT_CONFIG, cfg, device
    )

  def goal_dim_split(self) -> int:
    """Number of extra dims wm_goal has beyond actor_goal (rel_pos_b)."""
    return 3

  def extract_actor_goal(self, wm_goal: torch.Tensor) -> torch.Tensor:
    """Extract actor goal from full WM goal by removing extra rel_pos_b dims."""
    extra = self.goal_dim_split()
    return torch.cat([wm_goal[..., : -(extra + 6)], wm_goal[..., -6:]], dim=-1)

  @property
  def uses_per_body_tracking(self) -> bool:
    """Whether this representation tracks individual bodies (vs. full robot qpos)."""
    return False

  def extract_root_pos_quat(
    self, full_state: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract root position and quaternion from full state."""
    return full_state[..., 0:3], full_state[..., 3:7]


class BodyFrameRepresentation(StateRepresentation):
  """Joint-space body-frame representation.

  Full state: [root_pos(3), root_quat(4), qj(N), root_lin_vel(3), root_ang_vel(3), vj(N)]
  Rep state:  [z(1), g_b(3), v_b(3), w_b(3), qj(N), vj(N)]
  Prediction: same as rep_state (residual delta)
  """

  has_learned_reward = True

  LOSS_WEIGHT_CONFIG: dict[str, tuple[str, float]] = {
    "root_pos_z": ("nominal_error_root_pos_z", 0.1),
    "root_gravity_b": ("nominal_error_root_gravity_b", 0.1),
    "root_lin_vel": ("nominal_error_root_lin_vel", 1.0),
    "root_ang_vel": ("nominal_error_root_ang_vel", 3.14),
    "joint_pos": ("nominal_error_joint_pos", 0.1),
    "joint_vel": ("nominal_error_joint_vel", 3.14),
  }

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

  def compute_wm_reward(
    self, next_body_states, pred_r, actions, last_actions, motion, motion_times, dt
  ):
    return pred_r.sum(dim=-1) * dt


class MaxCoordRepresentation(StateRepresentation):
  """Maximal-coordinate per-body representation (following control_vae).

  Full state: [body_pos(B*3), body_quat(B*4), body_lin_vel(B*3), body_ang_vel(B*3)]
  Rep state:  [local_pos(B*3), local_rot_6d(B*6), local_vel(B*3), local_avel(B*3),
               body_height(B), up_dir(3)]
  Prediction: [vel_delta(B*3), avel_delta(B*3)] in root body frame

  Reward is computed analytically (not learned), following control_vae's pose_err.
  """

  has_learned_reward = False

  LOSS_WEIGHT_CONFIG: dict[str, tuple[str, float]] = {
    "body_pos": ("nominal_error_body_pos", 0.05),
    "body_rot": ("nominal_error_body_rot", 0.1),
    "body_vel": ("nominal_error_body_vel", 1.0),
    "body_avel": ("nominal_error_body_avel", 3.14),
    "body_height": ("nominal_error_body_height", 0.1),
    "up_dir": ("nominal_error_up_dir", 0.1),
  }

  # Default reward weights (from control_vae ControlVAE.yml)
  DEFAULT_REWARD_WEIGHTS = {
    "pos": 0.2,
    "rot": 0.1,
    "vel": 0.01,
    "avel": 0.01,
    "height": 1.2,
    "up_dir": 3.0,
  }

  @property
  def uses_per_body_tracking(self) -> bool:
    return True

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
  ) -> torch.Tensor:
    """Compute per-env reward from pose error (following control_vae).

    Uses weighted L1 norms matching control_vae's pose_err function.
    Returns negative error as reward (so maximizing reward = minimizing error).

    Args:
      predicted_rep: (..., rep_state_dim) predicted rep state
      target_rep: (..., rep_state_dim) target rep state

    Returns:
      (...,) per-env reward (negative pose error)
    """
    slices = self.loss_slices()
    delta = predicted_rep - target_rep
    w = self.reward_weights

    # L1 norms per component (summed across feature dims, not averaged)
    pos_err = w["pos"] * delta[..., slices["body_pos"]].abs().sum(-1)
    rot_err = w["rot"] * delta[..., slices["body_rot"]].abs().sum(-1)
    vel_err = w["vel"] * delta[..., slices["body_vel"]].abs().sum(-1)
    avel_err = w["avel"] * delta[..., slices["body_avel"]].abs().sum(-1)
    height_err = w["height"] * delta[..., slices["body_height"]].abs().sum(-1)
    up_dir_err = w["up_dir"] * delta[..., slices["up_dir"]].abs().sum(-1)

    total_err = pos_err + rot_err + vel_err + avel_err + height_err + up_dir_err
    return -total_err

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

  def compute_wm_reward(
    self, next_body_states, pred_r, actions, last_actions, motion, motion_times, dt
  ):
    next_mt = torch.clamp(motion_times + 1, 0, motion.time_step_total - 1)
    target_full = build_maxcoord_full_state(
      motion.body_pos_w[next_mt],
      motion.body_quat_w[next_mt],
      motion.body_lin_vel_w[next_mt],
      motion.body_ang_vel_w[next_mt],
    )
    target_rep = compute_maxcoord_rep_state(target_full, self.num_bodies)
    reward = self.compute_reward(next_body_states, target_rep)
    action_rate = torch.sum(torch.square(actions - last_actions), dim=-1)
    return dt * (reward - 0.1 * action_rate)

  def extract_root_pos_quat(
    self, full_state: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract root position and quaternion from full state."""
    B = self.num_bodies
    return full_state[..., 0:3], full_state[..., B * 3 : B * 3 + 4]
