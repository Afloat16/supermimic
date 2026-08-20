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
    history_length: int,
    termination_softness: float,
    device: str = "cpu",
  ) -> None:
    self.device = device
    self.dynamics = dynamics
    self.representation = representation
    self.num_envs = batch_size
    self.horizon = horizon
    self.replay_buffer = replay_buffer
    self.sim_context = sim_context
    self.history_length = history_length

    # Persistent state buffers.
    self.aux_state = torch.zeros(batch_size, 7, device=device)
    self.rep_state = torch.zeros(
      batch_size, representation.rep_state_dim, device=device
    )
    # Raw policy outputs (pre-scale/offset). Dynamics handles the transform;
    # the reward path consumes these directly.
    self.last_actions = torch.zeros(batch_size, dynamics.num_actions, device=device)
    self.sim_ctx = torch.zeros(batch_size, sim_context.output_dim, device=device)

    # Analytic per-step reward ceiling: positive tracking-reward weights, dt-scaled.
    self.max_step_reward = env.unwrapped.step_dt * sum(
      term.weight for term in env.cfg.rewards.values() if term.weight > 0.0
    )

    # Reference-motion buffers.
    self._num_actions = env.num_actions
    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    self.motion = motion_cmd.motion
    self._anchor_index = motion_cmd.motion_anchor_body_index

    if termination_softness <= 0.0:
      raise ValueError(
        f"termination_softness must be positive, got {termination_softness}"
      )
    terms = env.cfg.terminations
    self._termination_softness = termination_softness
    self._anchor_pos_threshold = terms["anchor_pos"].params["threshold"]
    self._anchor_ori_threshold = terms["anchor_ori"].params["threshold"]

    self._ref_body_state = None
    self.motion_times = torch.zeros(batch_size, dtype=torch.long, device=device)
    self._update_goal()

    # The learner sizes its normalizers from get_observations() before reset(),
    # so the ring must exist pre-reset; reset() overwrites these.
    self._latest_sensor = self._synth_sensor()
    if history_length > 1:
      self._past_sensors = self._latest_sensor.unsqueeze(1).repeat(
        1, history_length - 1, 1
      )

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
    next_aux_state, next_rep_state, term_prob, rewards = self._wm_step(
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

    self._compute_commands()

    if self.history_length > 1:
      # .clone(): the ring outlives this step, but torch.compile reduce-overhead
      # reuses the compiled-step output buffer the latest sensor points at.
      self._past_sensors = torch.cat(
        [self._past_sensors[:, 1:], self._latest_sensor.unsqueeze(1).clone()], dim=1
      )
    self._latest_sensor = self._synth_sensor()

    return self.get_observations(), rewards, term_prob, {}

  def _synth_sensor(self) -> torch.Tensor:
    return self.representation.compute_actor_sensor(
      self.rep_state,
      self.num_actions,
      self.sim_ctx,
      aux_state=self.aux_state,
      last_action=self.last_actions,
      noisy=True,
    )

  def _sensor_history(self) -> torch.Tensor:
    """The ``(B, history_length, sensor_dim)`` window; newest frame last."""
    latest = self._latest_sensor.unsqueeze(1)
    if self.history_length == 1:
      return latest
    return torch.cat([self._past_sensors, latest], dim=1)

  def _critic_obs(self) -> torch.Tensor:
    """The privileged critic input — full WM state, not the actor's sensor obs."""
    return torch.cat([self.rep_state, self._wm_goal_obs, self.sim_ctx], dim=-1)

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

    Returns ``(next_aux_state, next_rep_state, term_prob, reward)``. ``action``
    / ``last_action`` are raw policy outputs; Dynamics applies the scale/offset
    transform internally. Termination and reward both score the post-step state
    against the cached reference body state, the way the simulator scores it
    before advancing the clip.
    """
    next_aux_state, next_rep_state = self.dynamics(
      rep_state, action, aux_state, sim_ctx
    )
    ref_body_pos_w, ref_body_quat_w, ref_body_lin_vel_w, ref_body_ang_vel_w = (
      self._ref_body_state
    )
    term_prob = self._compute_term_prob(next_aux_state, ref_body_pos_w, ref_body_quat_w)
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
    return next_aux_state, next_rep_state, term_prob, reward

  def _compute_term_prob(
    self,
    aux_state: torch.Tensor,
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
  ) -> torch.Tensor:
    """Soft termination from the real threshold rules."""
    anchor_pos_w = ref_body_pos_w[:, self._anchor_index]
    anchor_quat_w = ref_body_quat_w[:, self._anchor_index]
    root_pos, root_quat = aux_state[..., :3], aux_state[..., 3:7]
    height_error = (anchor_pos_w[..., 2] - root_pos[..., 2]).abs()
    # Gravity's z in a body frame is 2(x^2 + y^2) - 1 for a unit quaternion.
    ref_gravity_z = 2 * (anchor_quat_w[..., 1:3] ** 2).sum(-1) - 1.0
    root_gravity_z = 2 * (root_quat[..., 1:3] ** 2).sum(-1) - 1.0
    tilt_error = (ref_gravity_z - root_gravity_z).abs()
    normalized_error = torch.maximum(
      height_error / self._anchor_pos_threshold,
      tilt_error / self._anchor_ori_threshold,
    )
    return torch.sigmoid((normalized_error - 1.0) / self._termination_softness)

  def _on_reset(self, batch: TensorDict) -> None:
    """Seed motion times from the sampled batch and refresh goals."""
    self.motion_times = batch["motion_time"]
    self._update_goal()
    if self.history_length > 1:
      self._past_sensors = self._warmstart_past_sensors()
    self._latest_sensor = self._synth_sensor()

  def _compute_commands(self) -> None:
    """Advance the reference clip one frame (clamped at clip ends) and refresh."""
    next_motion_times = self.motion_times + 1
    at_end = torch.isin(next_motion_times, self.motion.motion_ends)
    self.motion_times = torch.where(at_end, self.motion_times, next_motion_times)
    self._update_goal()

  def _update_goal(self) -> None:
    """Index the reference motion at ``motion_times`` and rebuild goals."""
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
    wm_goal_obs, actor_goal_obs = self.representation.compute_wm_goal(
      self.motion, self.motion_times, self.aux_state, self._anchor_index
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
    """Sample initial state snapshots from the replay buffer.

    With history_length > 1 the warm start draws the init state 1-step
    (H-independent) and keeps its real in-episode predecessors to seed the ring.
    """
    if self.history_length == 1:
      return self.replay_buffer.sample(1, self.num_envs).squeeze(1)
    window, padded = self.replay_buffer.sample_causal(
      self.history_length, self.num_envs
    )
    self._init_sensor_window = window["actor_obs"].detach()
    self._init_pad_mask = padded
    return window[:, -1]

  def _warmstart_past_sensors(self) -> torch.Tensor:
    """Noise the sampled predecessor window; clamp-padded slots (before the
    episode start) share the oldest real frame's single noise realization, the
    way the real-sim ring's reset padding does."""
    noisy = self.representation.add_proprio_noise(self._init_sensor_window[:, :-1])
    pad = self._init_pad_mask[:, :-1]  # padding is a leading run (sample_causal)
    first_real = (~pad).int().argmax(dim=1)
    shared = noisy[torch.arange(self.num_envs, device=self.device), first_real]
    return torch.where(pad.unsqueeze(-1), shared.unsqueeze(1), noisy)

  def _unpack_initial_states(
    self, batch: TensorDict
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract (aux_state, rep_state) from a sampled batch."""
    full_states = batch["state"]
    aux_state = self.representation.extract_aux_state(full_states)
    rep_state = self.representation.represent(full_states)
    return aux_state.detach(), rep_state.detach()

  def get_observations(self) -> TensorDict:
    """Build observations via subclass ``_get_policy_obs()``."""
    return self._get_policy_obs()
