"""Abstract base class for differentiable world model environments.

Shared infrastructure for world model environments: replay buffer sampling,
dynamics stepping, and observation building.

Follows the template-method pattern inspired by mjlab's ManagerBasedRlEnv:
  reset()  →  _on_reset()
  step()   →  _wm_step()  →  _compute_commands()  →  get_observations()

``_wm_step`` bundles dynamics + termination + reward into one compiled
forward per step so the cudagraph_trees pool sees a single graph per
logical env step (no inter-graph fragmentation from eager obs work).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from supermimic.replay_buffer import ReplayBuffer

if TYPE_CHECKING:
  from supermimic.models import Dynamics
  from supermimic.representations import StateRepresentation


class WorldModelEnv(ABC):
  """Abstract base for differentiable world model environments.

  Provides a template-method ``step()``:

  1. ``_wm_step(...)`` — bundled dynamics + termination + reward. One
     compiled graph per step. Subclasses implement this to define the
     env's step semantics (rep-routed reward, task reward, VAE decode,
     etc.). Raw policy outputs flow in; ``Dynamics`` handles the
     raw→proc transform internally.
  2. ``_compute_commands()`` — update goals (advance motion, etc.).
  3. ``get_observations()`` — build policy obs.
  """

  def __init__(
    self,
    dynamics: Dynamics,
    representation: StateRepresentation,
    batch_size: int,
    horizon: int,
    replay_buffer: ReplayBuffer,
    device: str = "cpu",
  ) -> None:
    self.device = device
    self.dynamics = dynamics
    self.representation = representation
    self.num_envs = batch_size
    self.horizon = horizon
    self.replay_buffer = replay_buffer

    # Persistent state buffers.
    self.aux_state = torch.zeros(batch_size, 7, device=device)
    self.rep_state = torch.zeros(
      batch_size, representation.rep_state_dim, device=device
    )
    # Raw policy outputs (pre-scale/offset). Dynamics handles the transform;
    # reward/termination heads consume these directly.
    self.last_actions = torch.zeros(batch_size, dynamics.num_actions, device=device)
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
  def _on_reset(self, batch: TensorDict) -> None:
    """Subclass-specific reset logic.

    Called after initial states are sampled and base buffers are reset.

    Args:
      batch: TensorDict of one-step samples (batch_size=[num_envs]) from the
        replay buffer; keys match the buffer schema.
    """
    ...

  @abstractmethod
  def _get_wm_goal(self) -> torch.Tensor:
    """Return the goal tensor for the dynamics forward pass.

    Shape: ``(num_envs, num_goals)``.
    """
    ...

  @abstractmethod
  def _get_policy_obs(self) -> TensorDict:
    """Build the policy observation TensorDict.

    Must contain at least ``"actor"`` and ``"critic"`` keys.
    """
    ...

  @abstractmethod
  def _wm_step(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    goal: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bundled dynamics + termination + reward for one env step.

    Returns ``(next_aux, next_rep_state, term_logit, reward)``. ``action`` /
    ``last_action`` are raw policy outputs; Dynamics applies the
    scale/offset transform internally.
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
    self.aux_state, self.rep_state = self._unpack_initial_states(batch)
    self.last_actions.zero_()
    self._first_step = True
    self._on_reset(batch)
    return self.get_observations()

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
    """_wm_step (one compiled graph) → commands → obs."""
    # Zero out the action-rate signal on the first step after reset.
    if self._first_step:
      self.last_actions = actions.detach()
      self._first_step = False

    last_actions = self.last_actions
    goal = self._get_wm_goal()

    next_aux, next_rep_state, term_logit, rewards = self._wm_step(
      self.rep_state, self.aux_state, actions, last_actions, goal
    )
    self.rep_state = next_rep_state
    self.aux_state = next_aux
    self.last_actions = actions

    term_prob = self._compute_term_prob(term_logit)
    self._compute_commands()

    return self.get_observations(), rewards, term_prob, {}

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
