"""Tests for MotionTrackingEnv (single-actor PPO-runner WM env)."""

from __future__ import annotations

import torch

from supermimic.envs.motion_tracking_env import MotionTrackingEnv
from tests.conftest import (
  ACTION_DIM,
  make_mock_buffer,
  make_mock_command_env,
  make_mock_dynamics,
  make_mock_sim_context,
  make_mock_wm_representation,
)

BATCH = 8


def _make_env() -> MotionTrackingEnv:
  rep = make_mock_wm_representation(BATCH)
  rep.build_actor_obs.side_effect = lambda rs, goal, *a, **kw: torch.cat(
    [rs, goal], dim=-1
  )

  return MotionTrackingEnv(
    dynamics=make_mock_dynamics(),
    representation=rep,
    env=make_mock_command_env(),
    batch_size=BATCH,
    horizon=8,
    replay_buffer=make_mock_buffer(),
    sim_context=make_mock_sim_context(),
    device="cpu",
  )


class TestMotionTrackingEnv:
  def test_reset_obs_has_actor_and_critic(self):
    env = _make_env()
    obs = env.reset()
    assert "actor" in obs.keys()
    assert "critic" in obs.keys()
    assert obs["actor"].shape[0] == BATCH
    assert obs["critic"].shape[0] == BATCH

  def test_step_advances_motion_times(self):
    env = _make_env()
    env.reset()
    times_before = env.motion_times.clone()
    env.step(torch.zeros(BATCH, ACTION_DIM))
    assert (env.motion_times == times_before + 1).all()

  def test_step_returns_correct_shapes(self):
    env = _make_env()
    env.reset()
    obs, rewards, dones, _ = env.step(torch.zeros(BATCH, ACTION_DIM))
    assert obs["actor"].shape[0] == BATCH
    assert rewards.shape == (BATCH,)
    assert dones.shape == (BATCH,)

  def test_num_actions_is_joint_dim(self):
    env = _make_env()
    assert env.num_actions == ACTION_DIM
