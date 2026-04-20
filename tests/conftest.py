"""Shared test fixtures and mock factories."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
from tensordict import TensorDict


def make_mock_dynamics(
  rep_state_dim: int = 214,
  num_actions: int = 29,
) -> MagicMock:
  """Mock ``Dynamics`` with ``forward(rep_state, action, aux_state) -> (next_aux, next_rep)``."""
  dyn = MagicMock()
  dyn.num_rep_state = rep_state_dim
  dyn.num_actions = num_actions
  dyn.dt = 0.02

  def forward(
    rep_state: torch.Tensor,
    action: torch.Tensor,
    aux_state: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    B = rep_state.shape[0]
    d = rep_state.device
    return (torch.randn(B, 7, device=d), torch.randn(B, rep_state_dim, device=d))

  dyn.side_effect = forward
  return dyn


def make_mock_representation(
  rep_state_dim: int = 214,
  full_state_dim: int = 182,
  num_bodies: int = 14,
) -> MagicMock:
  """Mock ``StateRepresentation`` with MaxCoord layout slice helpers."""
  rep = MagicMock()
  rep.rep_state_dim = rep_state_dim
  rep.full_state_dim = full_state_dim
  rep.num_bodies = num_bodies
  rep.dt = 0.02

  rep.represent.side_effect = lambda x: torch.randn(
    *x.shape[:-1], rep_state_dim, device=x.device
  )
  rep.extract_aux_state.side_effect = lambda x: torch.randn(
    x.shape[0], 7, device=x.device
  )
  rep.local_vel.side_effect = lambda rs: rs[
    ..., num_bodies * 9 : num_bodies * 12
  ].unflatten(-1, (num_bodies, 3))
  rep.root_height.side_effect = lambda rs: rs[..., num_bodies * 15]
  return rep


def make_mock_vae(latent_dim: int = 64, action_dim: int = 29) -> MagicMock:
  vae = MagicMock()
  vae.fixed_std = 0.3
  vae.encode.side_effect = lambda obs, **kw: torch.zeros(
    obs.shape[0], latent_dim, device=obs.device
  )
  vae.decode.side_effect = lambda z, obs: torch.zeros(
    z.shape[0], action_dim, device=z.device
  )
  return vae


def make_mock_buffer(full_state_dim: int = 182, goal_dim: int = 3) -> MagicMock:
  buf = MagicMock()
  buf.num_transitions = 100

  def sample(sequence_length: int, batch_size: int) -> TensorDict:
    return TensorDict(
      {
        "state": torch.randn(batch_size, sequence_length, full_state_dim),
        "action": torch.randn(batch_size, sequence_length, 29),
        "goal": torch.randn(batch_size, sequence_length, goal_dim),
        "reward": torch.randn(batch_size, sequence_length, 1),
        "motion_time": torch.zeros(batch_size, sequence_length, dtype=torch.long),
        "terminated": torch.zeros(batch_size, sequence_length, dtype=torch.bool),
        "truncated": torch.zeros(batch_size, sequence_length, dtype=torch.bool),
      },
      batch_size=[batch_size, sequence_length],
    )

  buf.sample = sample
  return buf
