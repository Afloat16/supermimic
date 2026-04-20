"""State representations for the world model.

Two representations are provided:
- GenCoordRepresentation: generalized-coordinate (joint-space) representation
- MaxCoordRepresentation: maximal-coordinate per-body representation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from supermimic.models import _CudaGraphNormalization, build_mlp

if TYPE_CHECKING:
  from mjlab.tasks.tracking.mdp.commands import MotionLoader

from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  subtract_frame_transforms,
)

from supermimic.utils.gencoord_helpers import (
  compute_gencoord_state,
  decompose_rep_state,
  infer_num_joints_from_state_dim,
  rep_state_dim_from_num_joints,
  rep_state_offsets,
)
from supermimic.utils.helpers import (
  quat_integrate,
  quat_to_rot6d,
  rodrigues_apply,
  rodrigues_precompute,
)
from supermimic.utils.maxcoord_helpers import (
  build_maxcoord_full_state,
  compute_maxcoord_rep_state,
  decompose_maxcoord_rep_state,
  maxcoord_full_state_dim,
  maxcoord_prediction_dim,
  maxcoord_rep_state_dim,
  reconstruct_full_from_aux_and_rep,
)


def create_representation(
  name: str,
  full_state_dim: int,
  num_actions: int,
  num_goals: int,
  num_rewards: int,
  dt: float,
) -> StateRepresentation:
  """Create a state representation wired up for training and reward compute.

  Instantiates the named representation and builds its learned heads.
  Action scale/offset belong to :class:`supermimic.models.Dynamics` — the
  only place the raw→proc transform is applied — and are not threaded
  through the representation.
  """
  if name == "gencoord":
    num_joints = infer_num_joints_from_state_dim(full_state_dim)
    rep: StateRepresentation = GenCoordRepresentation(num_joints)
  elif name == "maxcoord":
    if full_state_dim % 13 != 0:
      raise ValueError(
        f"Maxcoord full_state_dim ({full_state_dim}) must be divisible by 13"
      )
    rep = MaxCoordRepresentation(full_state_dim // 13)
  else:
    raise ValueError(
      f"Unknown representation: {name}. Must be 'gencoord' or 'maxcoord'."
    )
  rep.build_learned_heads(num_actions, num_goals, num_rewards)
  rep.dt = dt
  return rep


# ---------------------------------------------------------------------------
# Goal layout constants (from _assemble_goal)
# ---------------------------------------------------------------------------
# wm_goal  = [content, rel_pos_b(3), rel_ori_b(6)]
# actor_goal = [content, rel_ori_b(6)]
GOAL_REL_POS_DIM = 3  # rel_pos_b: in wm_goal only
GOAL_REL_ORI_DIM = 6  # rel_ori_b: in both wm_goal and actor_goal


def actor_goal_dim(num_goals: int) -> int:
  """Number of goal dimensions seen by the actor (wm_goal minus rel_pos_b)."""
  return num_goals - GOAL_REL_POS_DIM


def extract_actor_goal(wm_goal: torch.Tensor) -> torch.Tensor:
  """Extract actor goal from full WM goal by removing rel_pos_b dims."""
  tail = GOAL_REL_POS_DIM + GOAL_REL_ORI_DIM
  return torch.cat([wm_goal[..., :-tail], wm_goal[..., -GOAL_REL_ORI_DIM:]], dim=-1)


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


class StateRepresentation(nn.Module, ABC):
  """Abstract base class for state representations.

  Owns the state-semantics methods (``represent``, ``apply_prediction``,
  etc.) and any learned heads (reward / termination) built via
  ``build_learned_heads``. Gradient isolation between dynamics and these
  heads is structural: dynamics is a separate ``nn.Module`` and shares no
  parameters with the representation.

  Subclasses must set these instance attributes in ``__init__``:
    full_state_dim, rep_state_dim, prediction_dim

  And define this class attribute:
    LOSS_WEIGHT_CONFIG: dict[str, tuple[str, float]]
  """

  full_state_dim: int
  rep_state_dim: int
  prediction_dim: int
  LOSS_WEIGHT_CONFIG: dict[str, tuple[str, float]]

  # ------------------------------------------------------------------
  # Abstract methods (must be implemented by subclasses)
  # ------------------------------------------------------------------

  @abstractmethod
  def represent(self, full_state: torch.Tensor) -> torch.Tensor: ...

  @abstractmethod
  def extract_aux_state(self, full_state: torch.Tensor) -> torch.Tensor:
    """Extract auxiliary state [root_pos(3), root_quat(4)] from full state."""
    ...

  @abstractmethod
  def apply_prediction(
    self,
    aux_state: torch.Tensor,
    curr_rep_state: torch.Tensor,
    prediction: torch.Tensor,
    dt: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply prediction to produce next aux_state and next rep_state.

    Returns:
      (next_aux_state, next_rep_state)
    """
    ...

  @abstractmethod
  def reconstruct_full_state(
    self, aux_state: torch.Tensor, rep_state: torch.Tensor
  ) -> torch.Tensor:
    """Reconstruct full state from aux_state + rep_state (on-demand, for viz)."""
    ...

  @abstractmethod
  def loss_slices(self) -> dict[str, slice]: ...

  @abstractmethod
  def build_learned_heads(
    self, num_actions: int, num_goals: int, num_rewards: int
  ) -> None:
    """Instantiate learned reward/termination submodules with their own normalizers."""
    ...

  @abstractmethod
  def predict_termination(
    self, rep_state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor
  ) -> torch.Tensor:
    """Termination logit."""
    ...

  @abstractmethod
  def compute_wm_reward(
    self,
    next_rep_state: torch.Tensor,
    next_aux_state: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    goal: torch.Tensor,
    motion: MotionLoader,
    motion_times: torch.Tensor,
  ) -> torch.Tensor:
    """Per-env scalar reward used for WM-env rollouts.

    ``action``/``last_action`` are raw policy outputs (pre-scale/offset) —
    the canonical form across the replay buffer and every learned head;
    :class:`supermimic.models.Dynamics` is the sole place the raw→proc
    transform is applied.
    """
    ...

  def predict_reward(
    self,
    rep_state: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    goal: torch.Tensor,
  ) -> torch.Tensor | None:
    """Per-dim reward prediction for supervised training.

    Returns ``None`` when the representation has no learned reward head
    (analytical path only). Override to return a ``(..., num_rewards)``
    tensor.
    """
    del rep_state, action, last_action, goal
    return None

  # ------------------------------------------------------------------
  # Default implementations (may be overridden)
  # ------------------------------------------------------------------

  def normalizers(self) -> list[_CudaGraphNormalization]:
    """Normalizers for state/action/goal — built by ``build_learned_heads``."""
    return [self.state_normalizer, self.action_normalizer, self.goal_normalizer]

  def compute_wm_goal(
    self,
    motion_times: torch.Tensor,
    aux_state: torch.Tensor,
    motion: MotionLoader,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (target_joints, wm_goal, actor_goal) from motion data."""
    root_pos = aux_state[..., :3]
    root_quat = aux_state[..., 3:7]
    return _compiled_goal(
      motion_times,
      root_pos,
      root_quat,
      motion.body_pos_w,
      motion.body_quat_w,
      motion.joint_pos,
      motion.joint_vel,
    )

  def loss_weights(self, cfg: object, device: torch.device | str) -> torch.Tensor:
    return _bryson_loss_weights(
      self.rep_state_dim, self.loss_slices(), self.LOSS_WEIGHT_CONFIG, cfg, device
    )

  def construct_policy_input(
    self, full_state: torch.Tensor, goal: torch.Tensor
  ) -> torch.Tensor:
    """Build policy input by concatenating rep state with goal."""
    return torch.cat([self.represent(full_state), goal], dim=-1)

  @property
  def uses_per_body_tracking(self) -> bool:
    """Whether this representation tracks individual bodies (vs. full robot qpos)."""
    return False


class GenCoordRepresentation(StateRepresentation):
  """Generalized-coordinate (joint-space) representation.

  Full state: [root_pos(3), root_quat(4), qj(N), root_lin_vel(3), root_ang_vel(3), vj(N)]
  Rep state:  [z(1), g_b(3), v_b(3), w_b(3), qj(N), vj(N)]
  Prediction: same as rep_state (residual delta)
  """

  LOSS_WEIGHT_CONFIG: dict[str, tuple[str, float]] = {
    "root_pos_z": ("nominal_error_root_pos_z", 0.1),
    "root_gravity_b": ("nominal_error_root_gravity_b", 0.1),
    "root_lin_vel": ("nominal_error_root_lin_vel", 1.0),
    "root_ang_vel": ("nominal_error_root_ang_vel", 3.14),
    "joint_pos": ("nominal_error_joint_pos", 0.1),
    "joint_vel": ("nominal_error_joint_vel", 3.14),
  }

  def __init__(self, num_joints: int):
    super().__init__()
    self.num_joints = num_joints
    self.full_state_dim = 13 + 2 * num_joints
    self.rep_state_dim = rep_state_dim_from_num_joints(num_joints)
    self.prediction_dim = self.rep_state_dim  # dynamics predicts rep-state delta

  def build_learned_heads(
    self, num_actions: int, num_goals: int, num_rewards: int
  ) -> None:
    hidden = (512, 256, 128)

    self.state_normalizer = _CudaGraphNormalization(self.rep_state_dim)
    self.action_normalizer = _CudaGraphNormalization(num_actions)
    self.goal_normalizer = _CudaGraphNormalization(num_goals)

    self.rew_mlp = build_mlp(
      self.rep_state_dim + 2 * num_actions + num_goals,
      hidden,
      output_dim=num_rewards,
    )
    self.term_mlp = build_mlp(
      self.rep_state_dim + num_actions + num_goals, hidden, output_dim=1
    )

  def represent(self, full_state: torch.Tensor) -> torch.Tensor:
    flat = full_state.reshape(-1, self.full_state_dim)
    rep = compute_gencoord_state(flat)
    return rep.reshape(*full_state.shape[:-1], -1)

  def extract_aux_state(self, full_state: torch.Tensor) -> torch.Tensor:
    return full_state[..., :7]

  def apply_prediction(
    self,
    aux_state: torch.Tensor,
    curr_rep_state: torch.Tensor,
    prediction: torch.Tensor,
    dt: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    next_rep = curr_rep_state + prediction
    z, _, v_b, w_b, _, _ = decompose_rep_state(next_rep)
    root_pos = aux_state[..., :3]
    root_quat = aux_state[..., 3:7]
    v_global = quat_apply(root_quat, v_b)
    w_global = quat_apply(root_quat, w_b)
    next_root_pos = root_pos + v_global * dt
    next_root_pos = torch.cat([next_root_pos[..., :2], z], dim=-1)
    next_root_quat = quat_integrate(root_quat, w_global, dt)
    next_aux = torch.cat([next_root_pos, next_root_quat], dim=-1)
    return next_aux, next_rep

  def reconstruct_full_state(
    self, aux_state: torch.Tensor, rep_state: torch.Tensor
  ) -> torch.Tensor:
    z, _, v_b, w_b, qj, vj = decompose_rep_state(rep_state)
    root_pos = aux_state[..., :3]
    root_quat = aux_state[..., 3:7]
    # Override root Z with rep_state's z
    root_pos = torch.cat([root_pos[..., :2], z], dim=-1)
    v_global = quat_apply(root_quat, v_b)
    w_global = quat_apply(root_quat, w_b)
    parts = [root_pos, root_quat, qj, v_global, w_global, vj]
    return torch.cat(parts, dim=-1)

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

  def predict_reward(
    self,
    rep_state: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    goal: torch.Tensor,
  ) -> torch.Tensor:
    s = self.state_normalizer(rep_state)
    a = self.action_normalizer(action)
    la = self.action_normalizer(last_action)
    g = self.goal_normalizer(goal)
    return self.rew_mlp(torch.cat([s, a, la, g], dim=-1))

  def predict_termination(
    self, rep_state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor
  ) -> torch.Tensor:
    s = self.state_normalizer(rep_state)
    a = self.action_normalizer(action)
    g = self.goal_normalizer(goal)
    return self.term_mlp(torch.cat([s, a, g], dim=-1))

  def compute_wm_reward(
    self,
    next_rep_state: torch.Tensor,
    next_aux_state: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    goal: torch.Tensor,
    motion: MotionLoader,
    motion_times: torch.Tensor,
  ) -> torch.Tensor:
    del next_aux_state, motion, motion_times
    pred_r = self.predict_reward(next_rep_state, action, last_action, goal)
    return pred_r.sum(dim=-1) * self.dt


class MaxCoordRepresentation(StateRepresentation):
  """Maximal-coordinate per-body representation (following control_vae).

  Full state: [body_pos(B*3), body_quat(B*4), body_lin_vel(B*3), body_ang_vel(B*3)]
  Rep state:  [local_pos(B*3), local_rot_6d(B*6), local_vel(B*3), local_avel(B*3),
               root_height(1), up_dir(3)]
  Aux state:  [root_pos(3), root_quat(4)]
  Prediction: [vel_delta(B*3), avel_delta(B*3)] in root body frame

  Reward is computed analytically (not learned), following control_vae's pose_err.
  """

  LOSS_WEIGHT_CONFIG: dict[str, tuple[str, float]] = {
    "body_pos": ("nominal_error_body_pos", 0.05),
    "body_rot": ("nominal_error_body_rot", 0.1),
    "body_vel": ("nominal_error_body_vel", 1.0),
    "body_avel": ("nominal_error_body_avel", 3.14),
    "root_height": ("nominal_error_root_height", 0.1),
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
    super().__init__()
    self.num_bodies = num_bodies
    self.full_state_dim = maxcoord_full_state_dim(num_bodies)
    self.rep_state_dim = maxcoord_rep_state_dim(num_bodies)
    self.prediction_dim = maxcoord_prediction_dim(num_bodies)
    self.reward_weights = reward_weights or self.DEFAULT_REWARD_WEIGHTS.copy()

  def build_learned_heads(
    self, num_actions: int, num_goals: int, num_rewards: int
  ) -> None:
    del num_rewards  # reward is analytical for MaxCoord
    hidden = (512, 256, 128)
    self.state_normalizer = _CudaGraphNormalization(self.rep_state_dim)
    self.action_normalizer = _CudaGraphNormalization(num_actions)
    self.goal_normalizer = _CudaGraphNormalization(num_goals)
    self.term_mlp = build_mlp(
      self.rep_state_dim + num_actions + num_goals, hidden, output_dim=1
    )

  def represent(self, full_state: torch.Tensor) -> torch.Tensor:
    flat = full_state.reshape(-1, self.full_state_dim)
    rep = compute_maxcoord_rep_state(flat, self.num_bodies)
    return rep.reshape(*full_state.shape[:-1], -1)

  def extract_aux_state(self, full_state: torch.Tensor) -> torch.Tensor:
    B = self.num_bodies
    return torch.cat([full_state[..., :3], full_state[..., B * 3 : B * 3 + 4]], dim=-1)

  # rep_state layout: [local_pos(B*3), local_rot_6d(B*6), local_vel(B*3),
  # local_avel(B*3), root_height(1), up_dir(3)] — total 15*B + 4.

  def local_vel(self, rep_state: torch.Tensor) -> torch.Tensor:
    B = self.num_bodies
    return rep_state[..., B * 9 : B * 12].unflatten(-1, (B, 3))

  def root_height(self, rep_state: torch.Tensor) -> torch.Tensor:
    return rep_state[..., self.num_bodies * 15]

  def apply_prediction(
    self,
    aux_state: torch.Tensor,
    curr_rep_state: torch.Tensor,
    prediction: torch.Tensor,
    dt: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate state by prediction deltas in root body frame.

    Integrates positions and rotations in the current root frame,
    then applies a frame-change rotation to the new root frame.
    """
    B = self.num_bodies
    root_pos = aux_state[..., :3]
    root_quat = aux_state[..., 3:7]

    local_pos, local_rot_6d, local_vel, local_avel, _, _ = decompose_maxcoord_rep_state(
      curr_rep_state, B
    )

    # Prediction deltas (in root body frame)
    dv_local = prediction[..., : B * 3].unflatten(-1, (B, 3))
    dw_local = prediction[..., B * 3 : B * 6].unflatten(-1, (B, 3))

    # --- Step 1: Apply deltas in local frame ---
    next_vel = local_vel + dv_local
    next_avel = local_avel + dw_local

    # --- Step 2: Integrate in current root frame ---
    # Positions (with drift correction: subtract root_pos_local)
    root_pos_local = local_pos[..., 0:1, :]
    rel_vel = next_vel - next_vel[..., 0:1, :]
    next_pos = (local_pos - root_pos_local) + rel_vel * dt
    # Rotations (per-body omega, shared trig for 2 rot6d cols)
    k_body, cos_body, sin_body = rodrigues_precompute(next_avel * dt)
    cols = local_rot_6d.unflatten(-1, (3, 2))
    c1_mid = rodrigues_apply(cols[..., 0], k_body, cos_body, sin_body)
    c2_mid = rodrigues_apply(cols[..., 1], k_body, cos_body, sin_body)

    # --- Step 3: Root pose in world frame (with drift correction) ---
    next_root_pos = root_pos + quat_apply(
      root_quat,
      root_pos_local.squeeze(-2) + next_vel[..., 0, :] * dt,
    )
    next_root_quat = quat_integrate(
      root_quat, quat_apply(root_quat, next_avel[..., 0, :]), dt
    )

    # --- Step 4: Frame change (old root → new root) ---
    frame_omega = -next_avel[..., 0:1, :] * dt
    k_f, cos_f, sin_f = rodrigues_precompute(frame_omega)

    next_local_pos = rodrigues_apply(next_pos, k_f, cos_f, sin_f)
    next_local_vel = rodrigues_apply(next_vel, k_f, cos_f, sin_f)
    next_local_avel = rodrigues_apply(next_avel, k_f, cos_f, sin_f)
    next_local_rot_6d = torch.stack(
      [
        rodrigues_apply(c1_mid, k_f, cos_f, sin_f),
        rodrigues_apply(c2_mid, k_f, cos_f, sin_f),
      ],
      dim=-1,
    ).flatten(-2, -1)

    # --- Step 5: Aux state & trailing fields ---
    world_down = torch.zeros_like(next_root_pos)
    world_down[..., 2] = -1.0
    next_up_dir = quat_apply_inverse(next_root_quat, world_down)

    next_aux = torch.cat([next_root_pos, next_root_quat], dim=-1)
    next_rep = torch.cat(
      [
        next_local_pos.flatten(-2, -1),
        next_local_rot_6d.flatten(-2, -1),
        next_local_vel.flatten(-2, -1),
        next_local_avel.flatten(-2, -1),
        next_root_pos[..., 2:3],  # root_height
        next_up_dir,
      ],
      dim=-1,
    )
    return next_aux, next_rep

  def reconstruct_full_state(
    self, aux_state: torch.Tensor, rep_state: torch.Tensor
  ) -> torch.Tensor:
    return reconstruct_full_from_aux_and_rep(aux_state, rep_state, self.num_bodies)

  def compute_reward(
    self,
    predicted_rep: torch.Tensor,
    target_rep: torch.Tensor,
    predicted_body_height: torch.Tensor,
    target_body_height: torch.Tensor,
  ) -> torch.Tensor:
    """Compute per-env reward from pose error (following control_vae).

    Uses weighted L1 norms matching control_vae's pose_err function.
    Returns negative error as reward (so maximizing reward = minimizing error).

    Args:
      predicted_rep: (..., rep_state_dim) predicted rep state
      target_rep: (..., rep_state_dim) target rep state
      predicted_body_height: (..., B) predicted per-body heights from aux_state
      target_body_height: (..., B) target per-body heights from motion data

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
    height_err = w["height"] * (predicted_body_height - target_body_height).abs().sum(
      -1
    )
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
    slices["root_height"] = slice(off, off + 1)
    off += 1
    slices["up_dir"] = slice(off, off + 3)
    return slices

  def predict_termination(
    self, rep_state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor
  ) -> torch.Tensor:
    s = self.state_normalizer(rep_state)
    a = self.action_normalizer(action)
    g = self.goal_normalizer(goal)
    return self.term_mlp(torch.cat([s, a, g], dim=-1))

  def compute_wm_reward(
    self,
    next_rep_state: torch.Tensor,
    next_aux_state: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    goal: torch.Tensor,
    motion: MotionLoader,
    motion_times: torch.Tensor,
  ) -> torch.Tensor:
    del goal
    target_full = build_maxcoord_full_state(
      motion.body_pos_w[motion_times],
      motion.body_quat_w[motion_times],
      motion.body_lin_vel_w[motion_times],
      motion.body_ang_vel_w[motion_times],
    )
    target_rep = compute_maxcoord_rep_state(target_full, self.num_bodies)
    # Derive body heights on-demand from next aux + next rep.
    root_pos = next_aux_state[..., :3]
    root_quat = next_aux_state[..., 3:7]
    local_pos = next_rep_state[..., : self.num_bodies * 3].unflatten(
      -1, (self.num_bodies, 3)
    )
    root_quat_b = root_quat.unsqueeze(-2).expand(*local_pos.shape[:-1], 4)
    pred_body_height = root_pos[..., 2:3] + quat_apply(root_quat_b, local_pos)[..., 2]
    target_body_height = motion.body_pos_w[motion_times][..., 2]
    reward = self.compute_reward(
      next_rep_state, target_rep, pred_body_height, target_body_height
    )
    action_rate = torch.sum(torch.square(action - last_action), dim=-1)
    return self.dt * (reward - 0.1 * action_rate)
