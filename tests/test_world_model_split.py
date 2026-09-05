"""Tests for the Dynamics / StateRepresentation split contracts."""

import torch

from supermimic.models import Dynamics
from supermimic.representations import create_representation

NUM_JOINTS = 20
NUM_ACTIONS = 4
NUM_GOALS = 5
NUM_REWARDS = 1
FULL_STATE_DIM = 13 + 2 * NUM_JOINTS


def _build() -> tuple[Dynamics, torch.nn.Module]:
  rep = create_representation(
    "gencoord",
    FULL_STATE_DIM,
    NUM_ACTIONS,
    NUM_GOALS,
    NUM_REWARDS,
    dt=0.02,
    decimation=4,
    default_joint_pos=torch.zeros(NUM_JOINTS),
    has_state_estimation=False,
    actor_obs_mode="rep_state",
    reward_hidden_dims=(16, 16),
  )
  dyn = Dynamics(
    rep,
    NUM_ACTIONS,
    0,
    torch.zeros(NUM_ACTIONS),
    torch.zeros(NUM_ACTIONS),
    hidden_dims=(16, 16),
  )
  return dyn, rep


def test_reward_backward_leaves_dynamics_grads_none():
  dyn, rep = _build()
  B = 2
  next_rep = torch.randn(B, rep.rep_state_dim)
  action = torch.randn(B, NUM_ACTIONS)
  last_action = torch.randn(B, NUM_ACTIONS)
  goal = torch.randn(B, NUM_GOALS)

  loss = rep.predict_reward(next_rep, action, last_action, goal).sum()
  loss.backward()

  assert all(p.grad is None for p in dyn.parameters())
  assert any(p.grad is not None for p in rep.parameters())


def test_dynamics_backward_leaves_rep_head_grads_none():
  dyn, rep = _build()
  B = 2
  rep_state = torch.randn(B, rep.rep_state_dim)
  action = torch.randn(B, NUM_ACTIONS)
  aux = torch.zeros(B, 7)
  aux[..., 3] = 1.0  # unit quaternion

  _, next_rep = dyn(rep_state, action, aux, torch.zeros(B, 0))
  next_rep.sum().backward()

  rew_params = list(rep.reward_head.parameters()) if hasattr(rep, "reward_head") else []
  for p in rew_params:
    assert p.grad is None


def test_checkpoint_bundle_round_trip():
  dyn, rep = _build()
  bundle = {"dynamics": dyn.state_dict(), "representation": rep.state_dict()}

  dyn2, rep2 = _build()
  dyn2.load_state_dict(bundle["dynamics"])
  rep2.load_state_dict(bundle["representation"])

  B = 2
  rep_state = torch.randn(B, rep.rep_state_dim)
  action = torch.randn(B, NUM_ACTIONS)
  last_action = torch.randn(B, NUM_ACTIONS)
  goal = torch.randn(B, NUM_GOALS)
  aux = torch.zeros(B, 7)
  aux[..., 3] = 1.0

  sim_ctx = torch.zeros(B, 0)
  for a, b in zip(
    dyn(rep_state, action, aux, sim_ctx),
    dyn2(rep_state, action, aux, sim_ctx),
    strict=True,
  ):
    torch.testing.assert_close(a, b)
  torch.testing.assert_close(
    rep.predict_reward(rep_state, action, last_action, goal),
    rep2.predict_reward(rep_state, action, last_action, goal),
  )
