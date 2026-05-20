"""Tests for VelocityEnv."""

import math

import torch

from supermimic.envs.velocity_env import VelocityEnv, VelocityEnvCfg
from tests.conftest import (
  make_mock_buffer,
  make_mock_dynamics,
  make_mock_representation,
  make_mock_sim_context,
  make_mock_vae,
)

BATCH = 8


class TestVelocityEnv:
  def _create_env(self, **cfg_kw):
    cfg = VelocityEnvCfg(**cfg_kw)
    num_bodies = 14
    return VelocityEnv(
      cfg=cfg,
      body_masses=torch.ones(num_bodies),
      dynamics=make_mock_dynamics(),
      representation=make_mock_representation(num_bodies=num_bodies),
      vae=make_mock_vae(),
      batch_size=BATCH,
      horizon=16,
      replay_buffer=make_mock_buffer(goal_dim=2),
      latent_dim=64,
      sim_context=make_mock_sim_context(),
      device="cpu",
    )

  def test_sample_raw_goal_within_range(self):
    env = self._create_env(speed_range=(0.0, 5.0), angle_range=(-3.14, 3.14))
    cmd = env.sample_raw_goal(BATCH)
    assert cmd.shape == (BATCH, 2)
    assert cmd[:, 0].min() >= 0.0
    assert cmd[:, 0].max() <= 5.0
    assert cmd[:, 1].min() >= -3.14
    assert cmd[:, 1].max() <= 3.14

  def test_build_goal_obs_heading_relative(self):
    """yaw=π/2, raw=[1.0, π/2] → delta=0 → [speed, cos(0), sin(0)] = [1, 1, 0]."""
    aux = torch.zeros(1, 7)
    aux[0, 3] = math.cos(math.pi / 4)  # quat (w,x,y,z) for yaw=π/2 about z
    aux[0, 6] = math.sin(math.pi / 4)
    raw = torch.tensor([[1.0, math.pi / 2]])
    obs = VelocityEnv.build_goal_obs(raw, aux)
    assert torch.allclose(obs, torch.tensor([[1.0, 1.0, 0.0]]), atol=1e-5)
