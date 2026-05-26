"""Differentiable world model environment for motion tracking.

Replaces the physics simulator with a learned ``Dynamics`` + ``StateRepresentation``
while a reference motion clip supplies per-frame goals. Follows a template-method
pattern inspired by mjlab's ManagerBasedRlEnv:

  reset()  →  _on_reset()
  step()   →  _wm_step()  →  _compute_commands()  →  get_observations()

``_wm_step`` bundles dynamics + termination + reward into one compiled forward per
step so the cudagraph_trees pool sees a single graph per logical env step (no
inter-graph fragmentation from eager obs work). Subclasses implement only
``_get_policy_obs`` — the actor observation, which is the one thing that differs
between a single-actor policy and the multi-encoder actor — and may override
``_build_task_goals`` to emit extra per-task goal tensors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict

from supermimic.replay_buffer import ReplayBuffer

if TYPE_CHECKING:
  from supermimic.models import Dynamics
  from supermimic.representations import StateRepresentation
  from supermimic.sim_context import SimContext


class WorldModelEnv(ABC):
  """Differentiable world model env for motion tracking.

  Owns the reference-motion buffers, the bundled ``_wm_step``, goal updates,
  and the reset/observation plumbing. ``_get_policy_obs`` is the only abstract
  method; subclasses choose how to assemble the actor observation.
  """

  def __init__(
    self,
    dynamics: Dynamics,
    representation: StateRepresentation,
    env: Any,
    batch_size: int,
    horizon: int,
    replay_buffer: ReplayBuffer,
    sim_context: SimContext,
    device: str = "cpu",
  ) -> None:
    self.device = device
    self.dynamics = dynamics
    self.representation = representation
    self.num_envs = batch_size
    self.horizon = horizon
    self.replay_buffer = replay_buffer
    self.sim_context = sim_context

    # Persistent state buffers.
    self.aux_state = torch.zeros(batch_size, 7, device=device)
    self.rep_state = torch.zeros(
      batch_size, representation.rep_state_dim, device=device
    )
    # Raw policy outputs (pre-scale/offset). Dynamics handles the transform;
    # reward/termination heads consume these directly.
    self.last_actions = torch.zeros(batch_size, dynamics.num_actions, device=device)
    self.sim_ctx = torch.zeros(batch_size, sim_context.output_dim, device=device)

    # Reference-motion buffers.
    self._num_actions = env.num_actions
    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    self.motion = motion_cmd.motion
    self._ref_body_state = None
    self.motion_times = torch.zeros(batch_size, dtype=torch.long, device=device)
    self._update_goal()

  # ------------------------------------------------------------------
  # Subclass surface
  # ------------------------------------------------------------------

  @abstractmethod
  def _get_policy_obs(self) -> TensorDict:
    """Build the policy observation TensorDict: a ``"critic"`` key plus the
    actor's input keys."""
    ...

  def _build_task_goals(self) -> None:  # noqa: B027
    """Build extra per-task goal tensors after a goal update. Default: none."""

  # ------------------------------------------------------------------
  # Template methods
  # ------------------------------------------------------------------

  @property
  def num_actions(self) -> int:
    """Dimension of the action space for the high-level policy."""
    return self._num_actions

  def reset(self) -> TensorDict:
    """Sample initial states, reset buffers, and refresh goals."""
    batch = self._sample_initial_states()
    self.aux_state, self.rep_state = self._unpack_initial_states(batch)
    self.last_actions = batch["prev_action"].detach()
    self.sim_ctx = self.sim_context.encode_batch(batch).detach()
    self._on_reset(batch)
    return self.get_observations()

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
    next_aux_state, next_rep_state, term_logit, rewards = self._wm_step(
      self.rep_state,
      self.aux_state,
      actions,
      self.last_actions,
      self._wm_goal_obs,
      self.sim_ctx,
    )
    self.rep_state = next_rep_state
    self.aux_state = next_aux_state
    self.last_actions = actions

    term_prob = self._compute_term_prob(term_logit)
    self._compute_commands()

    return self.get_observations(), rewards, term_prob, {}

  def _wm_step(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    goal: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bundled dynamics + termination + reward for one env step.

    Returns ``(next_aux_state, next_rep_state, term_logit, reward)``. ``action``
    / ``last_action`` are raw policy outputs; Dynamics applies the scale/offset
    transform internally. The reward tracks the cached reference body state.
    """
    next_aux_state, next_rep_state = self.dynamics(
      rep_state, action, aux_state, sim_ctx
    )
    term_logit = self.representation.predict_termination(next_rep_state, action, goal)
    ref_body_pos_w, ref_body_quat_w, ref_body_lin_vel_w, ref_body_ang_vel_w = (
      self._ref_body_state
    )
    reward = self.representation.compute_wm_reward_from_reference(
      next_rep_state,
      next_aux_state,
      action,
      last_action,
      goal,
      ref_body_pos_w,
      ref_body_quat_w,
      ref_body_lin_vel_w,
      ref_body_ang_vel_w,
    )
    return next_aux_state, next_rep_state, term_logit, reward

  def _on_reset(self, batch: TensorDict) -> None:
    """Seed motion times from the sampled batch and refresh goals."""
    self.motion_times = batch["motion_time"]
    self._update_goal()

  def _compute_commands(self) -> None:
    """Advance the reference clip one frame (clamped at clip ends) and refresh."""
    next_motion_times = self.motion_times + 1
    at_end = torch.isin(next_motion_times, self.motion.motion_ends)
    self.motion_times = torch.where(at_end, self.motion_times, next_motion_times)
    self._update_goal()

  def _update_goal(self) -> None:
    """Index the reference motion at ``motion_times`` and rebuild goals."""
    ref_joint_pos = self.motion.joint_pos[self.motion_times]
    ref_joint_vel = self.motion.joint_vel[self.motion_times]
    ref_body_pos_w = self.motion.body_pos_w[self.motion_times]
    ref_body_quat_w = self.motion.body_quat_w[self.motion_times]
    ref_body_lin_vel_w = self.motion.body_lin_vel_w[self.motion_times]
    ref_body_ang_vel_w = self.motion.body_ang_vel_w[self.motion_times]
    self._ref_body_state = (
      ref_body_pos_w,
      ref_body_quat_w,
      ref_body_lin_vel_w,
      ref_body_ang_vel_w,
    )
    target_joints = torch.cat([ref_joint_pos, ref_joint_vel], dim=-1)
    wm_goal_obs, actor_goal_obs = self.representation.compute_wm_goal_from_reference(
      target_joints,
      ref_body_pos_w[:, 0],
      ref_body_quat_w[:, 0],
      self.aux_state,
    )
    # Clone to detach from the CUDA graph output buffers (reduce-overhead
    # mode): whichever goal a subclass routes into its obs TensorDict must
    # survive until AC captures it into rollout storage, but the next compiled
    # goal-update would otherwise overwrite the buffer it points at.
    self._wm_goal_obs = wm_goal_obs.clone()
    self._actor_goal_obs = actor_goal_obs.clone()
    self._build_task_goals()

  # ------------------------------------------------------------------
  # Shared helpers
  # ------------------------------------------------------------------

  def _sample_initial_states(self) -> TensorDict:
    """Sample initial state snapshots from the replay buffer."""
    return self.replay_buffer.sample(1, self.num_envs).squeeze(1)

  def _unpack_initial_states(
    self, batch: TensorDict
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract (aux_state, rep_state) from a sampled batch."""
    full_states = batch["state"]
    aux_state = self.representation.extract_aux_state(full_states)
    rep_state = self.representation.represent(full_states)
    return aux_state.detach(), rep_state.detach()

  def _compute_term_prob(self, term_logit: torch.Tensor) -> torch.Tensor:
    """Convert raw WM termination logit to probability."""
    return torch.sigmoid(term_logit).squeeze(-1).clone()

  def get_observations(self) -> TensorDict:
    """Build observations via subclass ``_get_policy_obs()``."""
    return self._get_policy_obs()
