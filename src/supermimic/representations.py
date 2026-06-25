"""State representations for the world model.

Two representations are provided:
- GenCoordRepresentation: generalized-coordinate (joint-space) representation
- MaxCoordRepresentation: maximal-coordinate per-body representation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  subtract_frame_transforms,
)

if TYPE_CHECKING:
  from mjlab.tasks.tracking.mdp.commands import MotionLoader

from supermimic.models import _CudaGraphNormalization, build_mlp
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
  compute_maxcoord_rep_state,
  decompose_maxcoord_rep_state,
  maxcoord_full_state_dim,
  maxcoord_prediction_dim,
  maxcoord_rep_state_dim,
  reconstruct_full_from_aux_and_rep,
)
from supermimic.utils.maxcoord_reward import maxcoord_wm_reward


def create_representation(
  name: str,
  full_state_dim: int,
  num_actions: int,
  num_goals: int,
  num_rewards: int,
  sim_ctx_dim: int,
  dt: float,
  default_joint_pos: torch.Tensor,
  has_state_estimation: bool,
  actor_obs_mode: str = "rep_state",
) -> StateRepresentation:
  """Create a state representation wired up for training and reward compute.

  ``has_state_estimation`` must match the mjlab env's actor-obs layout:
  when True the layout includes ``base_lin_vel``; when False that term is
  omitted (matches mjlab's no-state-estimation deploy mode).
  """
  if name == "gencoord":
    num_joints = infer_num_joints_from_state_dim(full_state_dim)
    rep: StateRepresentation = GenCoordRepresentation(num_joints, default_joint_pos)
  elif name == "maxcoord":
    if full_state_dim % 13 != 0:
      raise ValueError(
        f"Maxcoord full_state_dim ({full_state_dim}) must be divisible by 13"
      )
    rep = MaxCoordRepresentation(full_state_dim // 13, default_joint_pos.shape[0])
  else:
    raise ValueError(
      f"Unknown representation: {name}. Must be 'gencoord' or 'maxcoord'."
    )
  rep.has_state_estimation = has_state_estimation
  rep.actor_obs_mode = actor_obs_mode
  rep.build_learned_heads(num_actions, num_goals, num_rewards, sim_ctx_dim)
  if actor_obs_mode == "proprio":
    rep.register_buffer(
      "_proprio_noise_bounds",
      rep.proprio_noise_bounds(num_actions, default_joint_pos.device),
      persistent=False,
    )
  rep.dt = dt
  return rep


# ---------------------------------------------------------------------------
# Goal layout constants
# ---------------------------------------------------------------------------
# Reference-motion frames exposed in the goal window: a strided future window of
# NUM_GOAL_FRAMES frames spaced GOAL_FRAME_STRIDE motion steps apart. Motion runs
# at the control rate (one frame per env step, ~50 Hz / 0.02 s), so stride 5
# ≈ 0.1 s spacing and a ~0.9 s lookahead with 10 frames.
NUM_GOAL_FRAMES = 10
GOAL_FRAME_STRIDE = 5

# wm_goal  = [content, rel_pos_b(3*N), rel_ori_b(6*N)]
# actor_goal = [content, rel_ori_b(6*N)]
# content = NUM_GOAL_FRAMES joint targets: per frame [joint_pos, joint_vel].
GOAL_REL_POS_DIM = 3 * NUM_GOAL_FRAMES  # rel_pos_b: in wm_goal only
GOAL_REL_ORI_DIM = 6 * NUM_GOAL_FRAMES  # rel_ori_b: in both wm_goal and actor_goal


def actor_goal_dim(num_goals: int) -> int:
  """Number of goal dimensions seen by the actor (wm_goal minus rel_pos_b)."""
  return num_goals - GOAL_REL_POS_DIM


PROPRIO_NOISE_HALF_WIDTHS: dict[str, float] = {
  "projected_gravity": 0.05,
  "base_lin_vel": 0.5,
  "base_ang_vel": 0.2,
  "joint_pos": 0.01,
  "joint_vel": 0.5,
  "actions": 0.0,
}


def term_major_to_temporal(
  flat: torch.Tensor, term_widths: list[int], history_length: int
) -> torch.Tensor:
  """Convert mjlab's term-major flat obs to a ``(B, H, sum(w_i))`` frame-major window.

  Input layout ``[term0_t0..tH-1, term1_t0..tH-1, ...]`` is what the deployed
  actor obs group emits; the decoder consumes frame-major. Applied at the ONNX
  export boundary so deploy and training see the same window.
  """
  parts = []
  off = 0
  for w in term_widths:
    chunk = flat[..., off : off + history_length * w]
    parts.append(chunk.unflatten(-1, (history_length, w)))
    off += history_length * w
  return torch.cat(parts, dim=-1)


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
  target_pos: torch.Tensor,
  target_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Assemble ``(wm_goal, actor_goal)``; only wm_goal carries the rel_pos dims."""
  rel_pos, rel_ori = assemble_anchor_window(
    root_pos, root_quat, target_pos, target_quat
  )
  wm_goal = torch.cat([content, rel_pos, rel_ori], dim=-1)
  actor_goal = torch.cat([content, rel_ori], dim=-1)
  return wm_goal, actor_goal


def _build_sensor_proprio(
  projected_gravity: torch.Tensor,
  v_b: torch.Tensor,
  w_b: torch.Tensor,
  joint_pos_rel: torch.Tensor,
  joint_vel: torch.Tensor,
  has_state_estimation: bool,
) -> torch.Tensor:
  """Cat the state-derived proprio layout.

  ``[projected_gravity(3), (base_lin_vel(3),) base_ang_vel(3),
  joint_pos_rel(N), joint_vel(N)]``.
  """
  parts = [projected_gravity]
  if has_state_estimation:
    parts.append(v_b)
  parts.extend([w_b, joint_pos_rel, joint_vel])
  return torch.cat(parts, dim=-1)


def goal_window_indices(
  motion: MotionLoader, motion_times: torch.Tensor
) -> torch.Tensor:
  """Clamped windowed frame indices ``[t, t+stride, ..., t+(N-1)*stride]``, ``(E, N)``.

  Future offsets are clamped to the last frame of each env's current clip so the
  window never bleeds into the next concatenated clip — matching how the env
  advances ``motion_times``.
  """
  clip = torch.searchsorted(motion.motion_ends, motion_times, right=True)
  # Past-the-end guard: searchsorted returns numel() at the final frame.
  clip = clip.clamp(max=motion.motion_ends.numel() - 1)
  last_valid = motion.motion_ends[clip] - 1
  offsets = (
    torch.arange(NUM_GOAL_FRAMES, device=motion_times.device) * GOAL_FRAME_STRIDE
  )
  return torch.minimum(motion_times[:, None] + offsets, last_valid[:, None])


def motion_goal_window(
  motion: MotionLoader, motion_times: torch.Tensor
) -> torch.Tensor:
  """Windowed joint targets as the goal content, ``(E, N * 2J)`` — per frame
  ``[joint_pos(J), joint_vel(J)]``.
  """
  idx = goal_window_indices(motion, motion_times)
  frames = torch.cat([motion.joint_pos[idx], motion.joint_vel[idx]], dim=-1)
  return frames.reshape(frames.shape[0], -1)


def motion_anchor_window(
  motion: MotionLoader, motion_times: torch.Tensor, anchor_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
  """Anchor-body world pose ``(pos (E, N, 3), quat (E, N, 4))`` at the same
  windowed indices as :func:`motion_goal_window`.
  """
  idx = goal_window_indices(motion, motion_times)
  pos = motion.body_pos_w[:, anchor_index][idx]
  quat = motion.body_quat_w[:, anchor_index][idx]
  return pos, quat


def assemble_anchor_window(
  root_pos: torch.Tensor,
  root_quat: torch.Tensor,
  target_pos: torch.Tensor,
  target_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Per-frame motion anchor expressed relative to the robot's current root.

  ``root_*``: ``(E, 3)`` / ``(E, 4)``; ``target_*``: ``(E, N, 3)`` / ``(E, N, 4)``.
  Returns flattened ``(rel_pos (E, N*3), rel_ori_rot6d (E, N*6))``. The world-frame
  origin offset cancels in the relative transform, so callers only need the root
  and target anchors expressed in a common frame.
  """
  num_envs, num_frames = target_pos.shape[:2]
  root_pos = root_pos[:, None, :].expand(num_envs, num_frames, 3).reshape(-1, 3)
  root_quat = root_quat[:, None, :].expand(num_envs, num_frames, 4).reshape(-1, 4)
  rel_pos, rel_ori_w = subtract_frame_transforms(
    root_pos, root_quat, target_pos.reshape(-1, 3), target_quat.reshape(-1, 4)
  )
  rel_pos = rel_pos.reshape(num_envs, -1)
  rel_ori = quat_to_rot6d(rel_ori_w).reshape(num_envs, -1)
  return rel_pos, rel_ori


_compiled_assemble_goal = torch.compile(_assemble_goal, mode="reduce-overhead")


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
  num_joints: int
  has_state_estimation: bool = False
  actor_obs_mode: Literal["rep_state", "proprio"] = "rep_state"
  obs_head: nn.Module | None
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
    self,
    num_actions: int,
    num_goals: int,
    num_rewards: int,
    sim_ctx_dim: int,
  ) -> None:
    """Instantiate learned reward/termination submodules with their own normalizers."""
    ...

  @abstractmethod
  def predict_termination(
    self, rep_state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor
  ) -> torch.Tensor:
    """Termination logit."""
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

  def local_pos(self, rep_state: torch.Tensor) -> torch.Tensor:
    """Per-body position in the root-body frame, shape ``(..., num_bodies, 3)``."""
    del rep_state
    raise NotImplementedError(
      f"{type(self).__name__} does not expose local_pos; "
      "posterior tasks that use body geometry require MaxCoordRepresentation."
    )

  def local_vel(self, rep_state: torch.Tensor) -> torch.Tensor:
    """Per-body linear velocity in the root-body frame, shape ``(..., num_bodies, 3)``."""
    del rep_state
    raise NotImplementedError(
      f"{type(self).__name__} does not expose local_vel; "
      "posterior tasks that use body geometry require MaxCoordRepresentation."
    )

  @abstractmethod
  def compute_wm_obs(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> torch.Tensor:
    """Synthesize state-derived proprio from WM-predicted state."""
    ...

  @property
  def wm_obs_dim(self) -> int:
    """Width of state-derived proprio synthesized by compute_wm_obs."""
    base = 6 + 2 * self.num_joints
    return base + 3 if self.has_state_estimation else base

  def proprio_obs_dim(self, num_actions: int) -> int:
    """Dim of the deploy-shape sensor proprio."""
    return self.wm_obs_dim + num_actions

  def proprio_term_widths(self, num_actions: int) -> list[int]:
    """Per-term widths of the sensor proprio, in the concat order set by
    ``_sensor_actor_group``."""
    widths = [3]
    if self.has_state_estimation:
      widths.append(3)
    widths += [3, self.num_joints, self.num_joints, num_actions]
    return widths

  def proprio_noise_bounds(
    self, num_actions: int, device: torch.device | str
  ) -> torch.Tensor:
    """Per-dim uniform-noise half-widths matching the sensor proprio layout."""
    N = self.num_joints
    nh = PROPRIO_NOISE_HALF_WIDTHS
    spec: list[tuple[int, float]] = [(3, nh["projected_gravity"])]
    if self.has_state_estimation:
      spec.append((3, nh["base_lin_vel"]))
    spec += [
      (3, nh["base_ang_vel"]),
      (N, nh["joint_pos"]),
      (N, nh["joint_vel"]),
      (num_actions, nh["actions"]),
    ]
    return torch.cat([torch.full((w,), v, device=device) for w, v in spec])

  def actor_obs_dim(self, num_actions: int, num_goals: int, history_length: int) -> int:
    """Dim of the actor input — ``history_length`` sensor frames (proprio or
    rep_state) cat'd with the goal."""
    sensor_or_state = (
      self.rep_state_dim
      if self.actor_obs_mode == "rep_state"
      else self.proprio_obs_dim(num_actions)
    )
    return history_length * sensor_or_state + actor_goal_dim(num_goals)

  def add_proprio_noise(self, obs: torch.Tensor) -> torch.Tensor:
    """Add per-dim uniform proprio jitter — the single noise site shared by the
    synthesized sensor and the real-sim / imagination history rings."""
    return obs + ((torch.rand_like(obs) * 2 - 1) * self._proprio_noise_bounds).detach()

  def compute_actor_sensor(
    self,
    rep_state: torch.Tensor,
    num_actions: int,
    sim_ctx: torch.Tensor,
    aux_state: torch.Tensor | None = None,
    last_action: torch.Tensor | None = None,
    sim_actor_obs: torch.Tensor | None = None,
    noisy: bool = False,
  ) -> torch.Tensor:
    """Sensor-or-state half of the actor input (no goal).

    Returns ``rep_state`` in rep_state mode. In proprio mode, returns
    ``sim_actor_obs`` if provided (mjlab's ``obs["actor"]``) else
    synthesizes via :meth:`compute_wm_obs`; ``noisy=True`` adds per-dim
    uniform jitter scaled by the precomputed ``_proprio_noise_bounds``. ``aux_state`` and
    ``last_action`` are required only when synthesizing (proprio mode
    with no ``sim_actor_obs``).
    """
    if self.actor_obs_mode == "rep_state":
      return rep_state
    if sim_actor_obs is not None:
      obs = sim_actor_obs
    else:
      if aux_state is None or last_action is None:
        raise ValueError(
          "aux_state and last_action are required to synthesize proprio "
          "obs (no sim_actor_obs provided)"
        )
      obs = self.compute_wm_obs(rep_state, aux_state, sim_ctx)
      obs = torch.cat([obs, last_action], dim=-1)
    if noisy:
      obs = self.add_proprio_noise(obs)
    return obs

  def build_actor_obs(
    self, sensor_history: torch.Tensor, actor_goal: torch.Tensor
  ) -> torch.Tensor:
    """Flatten the ``(B, history_length, sensor_dim)`` window and cat the goal."""
    return torch.cat([sensor_history.flatten(start_dim=1), actor_goal], dim=-1)

  # ------------------------------------------------------------------
  # Default implementations (may be overridden)
  # ------------------------------------------------------------------

  def normalizers(self) -> list[_CudaGraphNormalization]:
    """Normalizers for state/action/goal — built by ``build_learned_heads``."""
    return [self.state_normalizer, self.action_normalizer, self.goal_normalizer]

  def compute_wm_goal(
    self,
    motion: MotionLoader,
    motion_times: torch.Tensor,
    aux_state: torch.Tensor,
    anchor_index: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(wm_goal, actor_goal)`` from the motion window at ``motion_times``.

    The single goal-build path shared by the WM env, the task goal extractor, and
    the viewer so every site produces an identical layout.
    """
    content = motion_goal_window(motion, motion_times)
    target_pos, target_quat = motion_anchor_window(motion, motion_times, anchor_index)
    root_pos, root_quat = aux_state[..., :3], aux_state[..., 3:7]
    return _compiled_assemble_goal(
      content, root_pos, root_quat, target_pos, target_quat
    )

  def compute_wm_reward_from_reference(
    self,
    next_rep_state: torch.Tensor,
    next_aux_state: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    goal: torch.Tensor,
    target_body_pos_w: torch.Tensor,
    target_body_quat_w: torch.Tensor,
    target_body_lin_vel_w: torch.Tensor,
    target_body_ang_vel_w: torch.Tensor,
  ) -> torch.Tensor:
    """Per-env scalar reward used for WM-env rollouts.

    ``action``/``last_action`` are raw policy outputs (pre-scale/offset) —
    the canonical form across the replay buffer and every learned head;
    :class:`supermimic.models.Dynamics` is the sole place the raw→proc
    transform is applied.
    """
    del (
      next_rep_state,
      next_aux_state,
      action,
      last_action,
      goal,
      target_body_pos_w,
      target_body_quat_w,
      target_body_lin_vel_w,
      target_body_ang_vel_w,
    )
    raise NotImplementedError

  def loss_weights(self, cfg: object, device: torch.device | str) -> torch.Tensor:
    return _bryson_loss_weights(
      self.rep_state_dim, self.loss_slices(), self.LOSS_WEIGHT_CONFIG, cfg, device
    )

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

  def __init__(self, num_joints: int, default_joint_pos: torch.Tensor):
    super().__init__()
    self.num_joints = num_joints
    self.full_state_dim = 13 + 2 * num_joints
    self.rep_state_dim = rep_state_dim_from_num_joints(num_joints)
    self.prediction_dim = self.rep_state_dim  # dynamics predicts rep-state delta
    self._default_joint_pos = default_joint_pos
    self.obs_head = None

  def build_learned_heads(
    self,
    num_actions: int,
    num_goals: int,
    num_rewards: int,
    sim_ctx_dim: int,
  ) -> None:
    del sim_ctx_dim  # GenCoord proprio is derived directly from rep_state.
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

  def compute_wm_reward_from_reference(
    self,
    next_rep_state: torch.Tensor,
    next_aux_state: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    goal: torch.Tensor,
    target_body_pos_w: torch.Tensor,
    target_body_quat_w: torch.Tensor,
    target_body_lin_vel_w: torch.Tensor,
    target_body_ang_vel_w: torch.Tensor,
  ) -> torch.Tensor:
    del next_aux_state, target_body_pos_w, target_body_quat_w
    del target_body_lin_vel_w, target_body_ang_vel_w
    pred_r = self.predict_reward(next_rep_state, action, last_action, goal)
    return pred_r.sum(dim=-1) * self.dt

  def compute_wm_obs(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> torch.Tensor:
    del aux_state, sim_ctx
    _, g_b, v_b, w_b, qj, vj = decompose_rep_state(rep_state)
    return _build_sensor_proprio(
      g_b,
      v_b,
      w_b,
      qj - self._default_joint_pos,
      vj,
      self.has_state_estimation,
    )


class MaxCoordRepresentation(StateRepresentation):
  """Maximal-coordinate per-body representation (following control_vae).

  Full state: [body_pos(B*3), body_quat(B*4), body_lin_vel(B*3), body_ang_vel(B*3)]
  Rep state:  [local_pos(B*3), local_rot_6d(B*6), local_vel(B*3), local_avel(B*3),
               root_height(1), up_dir(3)]
  Aux state:  [root_pos(3), root_quat(4)]
  Prediction: [vel_delta(B*3), avel_delta(B*3)] in root body frame

  Reward is computed analytically (not learned) via Gaussian-kernel terms
  matching the motion-tracking env (see ``maxcoord_wm_reward``).
  """

  LOSS_WEIGHT_CONFIG: dict[str, tuple[str, float]] = {
    "body_pos": ("nominal_error_body_pos", 0.05),
    "body_rot": ("nominal_error_body_rot", 0.1),
    "body_vel": ("nominal_error_body_vel", 1.0),
    "body_avel": ("nominal_error_body_avel", 3.14),
    "root_height": ("nominal_error_root_height", 0.1),
    "up_dir": ("nominal_error_up_dir", 0.1),
  }

  @property
  def uses_per_body_tracking(self) -> bool:
    return True

  def __init__(self, num_bodies: int, num_joints: int):
    super().__init__()
    self.num_bodies = num_bodies
    self.full_state_dim = maxcoord_full_state_dim(num_bodies)
    self.rep_state_dim = maxcoord_rep_state_dim(num_bodies)
    self.prediction_dim = maxcoord_prediction_dim(num_bodies)
    self.num_joints = num_joints
    self.obs_head = None

  def build_learned_heads(
    self,
    num_actions: int,
    num_goals: int,
    num_rewards: int,
    sim_ctx_dim: int,
  ) -> None:
    del num_rewards  # reward is analytical for MaxCoord
    hidden = (512, 256, 128)
    self.state_normalizer = _CudaGraphNormalization(self.rep_state_dim)
    self.action_normalizer = _CudaGraphNormalization(num_actions)
    self.goal_normalizer = _CudaGraphNormalization(num_goals)
    self.term_mlp = build_mlp(
      self.rep_state_dim + num_actions + num_goals, hidden, output_dim=1
    )
    if self.actor_obs_mode == "proprio":
      self.obs_head = build_mlp(
        self.rep_state_dim + sim_ctx_dim,
        (256, 256),
        output_dim=2 * self.num_joints,
        use_layer_norm=True,
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

  def local_pos(self, rep_state: torch.Tensor) -> torch.Tensor:
    B = self.num_bodies
    return rep_state[..., : B * 3].unflatten(-1, (B, 3))

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

  def compute_wm_reward_from_reference(
    self,
    next_rep_state: torch.Tensor,
    next_aux_state: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    goal: torch.Tensor,
    target_body_pos_w: torch.Tensor,
    target_body_quat_w: torch.Tensor,
    target_body_lin_vel_w: torch.Tensor,
    target_body_ang_vel_w: torch.Tensor,
  ) -> torch.Tensor:
    del goal
    return maxcoord_wm_reward(
      next_aux_state,
      next_rep_state,
      action,
      last_action,
      target_body_pos_w,
      target_body_quat_w,
      target_body_lin_vel_w,
      target_body_ang_vel_w,
      self.dt,
    )

  def compute_wm_obs(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> torch.Tensor:
    if self.obs_head is None:
      raise RuntimeError(
        "MaxCoordRepresentation.obs_head is None — set actor_obs_mode='proprio' "
        "and pass default_joint_pos to create_representation to enable proprio obs"
      )
    root_quat = aux_state[..., 3:7]
    g_w = aux_state.new_tensor([0.0, 0.0, -1.0]).expand(*aux_state.shape[:-1], 3)
    projected_gravity = quat_apply_inverse(root_quat, g_w)
    B = self.num_bodies
    v_b = rep_state[..., B * 9 : B * 9 + 3]
    w_b = rep_state[..., B * 12 : B * 12 + 3]
    joint_output = self.obs_head(torch.cat([rep_state, sim_ctx], dim=-1))
    N = self.num_joints
    joint_pos_rel = joint_output[..., :N]
    joint_vel = joint_output[..., N : 2 * N]
    return _build_sensor_proprio(
      projected_gravity,
      v_b,
      w_b,
      joint_pos_rel,
      joint_vel,
      self.has_state_estimation,
    )
