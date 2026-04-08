"""Abstract base class for differentiable world model environments.

Shared infrastructure for world model environments: replay buffer sampling,
WM forward stepping, and observation building.

Follows the template-method pattern inspired by mjlab's ManagerBasedRlEnv:
  reset()  →  _on_reset()
  step()   →  _process_actions()  →  _wm_forward()  →  _compute_dones()
             →  _compute_rewards()  →  _compute_commands()  →  get_observations()
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from tensordict import TensorDict

from supermimic.models import WorldModel
from supermimic.replay_buffer import ReplayBuffer


class WorldModelEnv(ABC):
  """Abstract base for differentiable world model environments.

  Provides a template-method ``step()``:

  1. ``_process_actions(actions)`` — process actions (scale/offset, VAE decode).
  2. ``_wm_forward(proc_actions)`` — run the world model, update state.
  3. ``_compute_dones(pred_d)`` — sigmoid on termination logits.
  4. ``_compute_rewards(...)`` — per-env reward.
  5. ``_compute_commands()`` — update goals (advance motion, etc.).
  6. ``get_observations()`` — build policy obs.
  """

  def __init__(
    self,
    world_model: WorldModel,
    action_scale: torch.Tensor,
    action_offset: torch.Tensor,
    batch_size: int,
    horizon: int,
    replay_buffer: ReplayBuffer,
    device: str = "cpu",
  ) -> None:
    self.wm = world_model
    self.device = device
    self.num_envs = batch_size
    self.horizon = horizon
    self.replay_buffer = replay_buffer

    # Joint-level action processing (for WM input).
    self.action_scale = action_scale.to(device)
    self.action_offset = action_offset.to(device)
    self.dt = world_model.dt

    rep = world_model.representation
    self._rep_state_dim = rep.rep_state_dim

    # Persistent state buffers.
    self.aux_states = torch.zeros(batch_size, 7, device=device)
    self.rep_state = torch.zeros(batch_size, rep.rep_state_dim, device=device)
    self._last_proc_actions = torch.zeros(
      batch_size, world_model.num_actions, device=device
    )
    self.actions = torch.zeros(batch_size, world_model.num_actions, device=device)
    self.last_actions = torch.zeros(batch_size, world_model.num_actions, device=device)
    self._first_step = True

  # ------------------------------------------------------------------
  # Abstract methods — subclasses must implement
  # ------------------------------------------------------------------

  @property
  @abstractmethod
  def num_actions(self) -> int:
    """Dimension of the action space for the high-level policy."""
    ...

  @abstractmethod
  def _on_reset(self, batch: list[torch.Tensor | None]) -> None:
    """Subclass-specific reset logic.

    Called after initial states are sampled and base buffers are reset.

    Args:
      batch: List of tensors from replay buffer, one per field:
        [full_states, actions, goals, rewards, motion_times, dones],
        each of shape ``(batch_size, dim)``.
    """
    ...

  @abstractmethod
  def _get_wm_goal(self) -> torch.Tensor:
    """Return the goal tensor for the WM forward pass.

    Shape: ``(num_envs, wm.num_goals)``.
    """
    ...

  @abstractmethod
  def _get_policy_obs(self) -> TensorDict:
    """Build the policy observation TensorDict.

    Must contain at least ``"actor"`` and ``"critic"`` keys.
    """
    ...

  def _process_actions(self, actions: torch.Tensor) -> torch.Tensor:
    """Convert policy actions to processed joint-level actions.

    Base implementation stores ``actions`` and shifts ``last_actions``.
    Subclasses should call ``super()._process_actions(actions)`` first,
    then return processed actions.

    Args:
      actions: (B, num_actions) raw policy output.

    Returns:
      (B, joint_action_dim) processed actions ready for the world model.
    """
    self.last_actions = self.actions
    self.actions = actions
    return actions

  @abstractmethod
  def _compute_rewards(
    self,
    actions: torch.Tensor,
    next_rep_state: torch.Tensor,
    next_aux_state: torch.Tensor,
    pred_reward: torch.Tensor,
  ) -> torch.Tensor:
    """Compute per-env scalar reward after the WM step.

    Args:
      actions: (B, num_actions) raw policy output (before processing).
      next_rep_state: (B, rep_state_dim) predicted next state.
      next_aux_state: (B, 7) predicted next aux state [root_pos, root_quat].
      pred_reward: (B, num_rewards) predicted reward from WM reward head.

    Returns:
      (B,) per-env scalar reward.
    """
    ...

  def _compute_commands(self) -> None:  # noqa: B027
    """Update goals/commands after the WM step.

    Default: no-op. Override to advance motion times, resample task
    commands, etc. Called after rewards, before observations.
    """

  # ------------------------------------------------------------------
  # Template methods
  # ------------------------------------------------------------------

  def reset(self) -> TensorDict:
    """Sample initial states, reset buffers, and delegate to subclass."""
    batch = self._sample_initial_states()
    self.aux_states, self.rep_state = self._unpack_initial_states(batch)
    self._last_proc_actions.zero_()
    self.actions.zero_()
    self.last_actions.zero_()
    self._first_step = True
    self._on_reset(batch)
    return self.get_observations()

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
    """process_actions → wm_forward → terminations → rewards → commands → obs."""
    proc_actions = self._process_actions(actions)

    # On first step after reset, set last actions to current to avoid
    # spurious action rate penalty, and detach proc_actions for WM.
    if self._first_step:
      self.last_actions = actions
      self._last_proc_actions = proc_actions.detach()
      self._first_step = False

    next_aux, next_rep_state, pred_r, pred_d = self._wm_forward(proc_actions)
    dones = self._compute_dones(pred_d)
    rewards = self._compute_rewards(actions, next_rep_state, next_aux, pred_r)
    self._compute_commands()

    return self.get_observations(), rewards, dones, {}

  # ------------------------------------------------------------------
  # Shared helpers
  # ------------------------------------------------------------------

  def _sample_initial_states(self) -> list[torch.Tensor | None]:
    """Sample initial state snapshots from the replay buffer."""
    batch = self.replay_buffer._generate_batch(
      self.replay_buffer.replay_buf,
      valid_indices=None,
      sequence_length=1,
      mini_batch_size=self.num_envs,
    )
    return [t.squeeze(1) if t is not None else None for t in batch]

  def _unpack_initial_states(
    self, batch: list[torch.Tensor | None]
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract (aux_states, rep_state) from a sampled batch."""
    full_states = batch[0]
    aux_states = self.wm.representation.extract_aux_state(full_states)
    rep_state = self.wm.representation.represent(full_states)
    return aux_states.detach(), rep_state.detach()

  def _wm_forward(
    self, proc_actions: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the world model forward pass and update internal state.

    Returns:
      (next_aux, next_rep_state, pred_reward, pred_done)
    """
    next_aux, next_rep_state, pred_r, pred_d = self.wm(
      self.rep_state,
      proc_actions,
      self._last_proc_actions,
      self._get_wm_goal(),
      self.aux_states,
    )

    self.rep_state = next_rep_state
    self.aux_states = next_aux
    self._last_proc_actions = proc_actions

    return next_aux, next_rep_state, pred_r, pred_d

  def _compute_dones(self, pred_d: torch.Tensor) -> torch.Tensor:
    """Convert raw WM termination logit to probability."""
    return torch.sigmoid(pred_d).squeeze(-1).clone()

  def get_observations(self) -> TensorDict:
    """Build observations via subclass ``_get_policy_obs()``."""
    return self._get_policy_obs()

  def _build_obs_tensordict(
    self,
    actor_goal: torch.Tensor,
    critic_goal: torch.Tensor | None = None,
  ) -> TensorDict:
    """Helper: concatenate rep_state with goal(s) and wrap in TensorDict."""
    if critic_goal is None:
      critic_goal = actor_goal
    return TensorDict(
      {
        "actor": torch.cat([self.rep_state, actor_goal], dim=-1),
        "critic": torch.cat([self.rep_state, critic_goal], dim=-1),
      },
      batch_size=[self.num_envs],
      device=self.device,
    )
