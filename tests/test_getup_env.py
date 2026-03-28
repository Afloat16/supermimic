"""Tests for GetupEnv."""

import torch

from supermimic.envs.getup_env import GetupEnv, GetupEnvCfg
from tests.conftest import (
  make_mock_buffer,
  make_mock_dynamics,
  make_mock_representation,
  make_mock_vae,
)

BATCH = 8


class TestGetupEnv:
  def _create_env(self, **cfg_kw):
    cfg = GetupEnvCfg(**cfg_kw)
    num_bodies = 14
    return GetupEnv(
      cfg=cfg,
      body_masses=torch.ones(num_bodies),
      dynamics=make_mock_dynamics(),
      representation=make_mock_representation(num_bodies=num_bodies),
      vae=make_mock_vae(),
      batch_size=BATCH,
      horizon=16,
      replay_buffer=make_mock_buffer(goal_dim=1),
      latent_dim=64,
      device="cpu",
    )

  def test_sample_raw_goal_in_height_values(self):
    env = self._create_env(height_values=(0.0, 0.4, 0.8))
    cmd = env.sample_raw_goal(BATCH)
    assert cmd.shape == (BATCH, 1)
    allowed = torch.tensor([0.0, 0.4, 0.8])
    assert torch.isin(cmd, allowed).all()

  def test_height_reward_monotonicity(self):
    """At target height, reward > away-from-target."""
    env = self._create_env()
    env.reset()
    env.goal.fill_(0.8)
    aux_at = torch.zeros(BATCH, 7)
    aux_at[:, 2] = 0.8
    aux_at[:, 3] = 1.0  # identity quat
    aux_off = torch.zeros(BATCH, 7)
    aux_off[:, 2] = 0.2
    aux_off[:, 3] = 1.0
    rep = torch.zeros(BATCH, 214)
    is_last = torch.zeros(BATCH)
    r_at = env._compute_reward(rep, aux_at, is_last)
    r_off = env._compute_reward(rep, aux_off, is_last)
    assert r_at.mean() > r_off.mean()

  def test_face_up_loss_masked_outside_last_step(self):
    """face_up loss multiplies by is_last_mask — zero contribution when mask=0."""
    env = self._create_env(face_up_weight=1.0, height_weight=0.0, energy_weight=0.0)
    env.reset()
    env.goal.fill_(0.0)  # lay-down (target < 0.4)
    aux = torch.zeros(BATCH, 7)
    aux[:, 3] = 1.0  # identity quat → forward_world[:, 2] = 0 → face_up_loss = 1
    rep = torch.zeros(BATCH, 214)
    r_last = env._compute_reward(rep, aux, torch.ones(BATCH))
    r_not_last = env._compute_reward(rep, aux, torch.zeros(BATCH))
    assert (r_not_last > r_last).all()
