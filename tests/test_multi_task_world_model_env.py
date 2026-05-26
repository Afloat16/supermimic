"""Tests for MultiTaskWorldModelEnv."""

from __future__ import annotations

import torch

from supermimic.envs.multi_task_world_model_env import MultiTaskWorldModelEnv
from supermimic.representations import GenCoordRepresentation
from supermimic.tasks.task_spec import (
  TaskSpec,
  default_encoder_factory,
  make_motion_task,
)
from tests.conftest import (
  ACTION_DIM,
  MOTION_GOAL_DIM,
  VELOCITY_GOAL_DIM,
  identity_aux,
  make_mock_buffer,
  make_mock_command_env,
  make_mock_dynamics,
  make_mock_sim_context,
  make_mock_wm_representation,
)

BATCH = 8


def _specs():
  motion_goal = torch.randn(BATCH, MOTION_GOAL_DIM)
  velocity_goal = torch.randn(BATCH, VELOCITY_GOAL_DIM)

  return [
    TaskSpec(
      name="motion",
      goal_dim=MOTION_GOAL_DIM,
      encoder_factory=default_encoder_factory,
      extract_training_goal=lambda m, mt, aux: motion_goal,
    ),
    TaskSpec(
      name="velocity",
      goal_dim=VELOCITY_GOAL_DIM,
      encoder_factory=default_encoder_factory,
      extract_training_goal=lambda m, mt, aux: velocity_goal,
    ),
  ]


def _make_env() -> MultiTaskWorldModelEnv:
  rep = make_mock_wm_representation(BATCH)

  return MultiTaskWorldModelEnv(
    dynamics=make_mock_dynamics(),
    representation=rep,
    env=make_mock_command_env(),
    batch_size=BATCH,
    horizon=8,
    replay_buffer=make_mock_buffer(goal_dim=87),
    sim_context=make_mock_sim_context(),
    task_specs=_specs(),
    device="cpu",
  )


class TestMultiTaskWorldModelEnv:
  def test_reset_obs_has_per_task_goals(self):
    env = _make_env()
    obs = env.reset()
    assert "obs" in obs.keys()
    assert "critic" in obs.keys()
    assert "goal_motion" in obs.keys()
    assert "goal_velocity" in obs.keys()
    assert obs["goal_motion"].shape == (BATCH, MOTION_GOAL_DIM)
    assert obs["goal_velocity"].shape == (BATCH, VELOCITY_GOAL_DIM)

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
    assert obs["obs"].shape[0] == BATCH
    assert rewards.shape == (BATCH,)
    assert dones.shape == (BATCH,)

  def test_num_actions_is_joint_dim(self):
    """The actor outputs joint actions; env consumes them directly."""
    env = _make_env()
    assert env.num_actions == ACTION_DIM


class TestRepresentationApiContract:
  """Pin the real representation's ``*_from_reference`` API that the env and
  motion TaskSpec call, against a non-mocked representation.
  """

  def test_uses_from_reference_api(self):
    rep = GenCoordRepresentation(12, torch.zeros(12))
    assert not hasattr(rep, "compute_wm_goal")
    assert not hasattr(rep, "compute_wm_reward")
    aux = torch.zeros(4, 7)
    aux[:, 3] = 1.0
    wm_goal, actor_goal = rep.compute_wm_goal_from_reference(
      torch.zeros(4, 24), torch.zeros(4, 3), aux[:, 3:7], aux
    )
    assert wm_goal.shape[0] == 4
    assert actor_goal.shape[0] == 4

  def test_env_update_goals_against_real_representation(self):
    """Drive the env -> rep call path with a REAL representation: a wrong
    method name or swapped arg order would error/NaN here, where the mocked
    env tests cannot catch it.
    """
    rep = GenCoordRepresentation(29, torch.zeros(29))
    env = MultiTaskWorldModelEnv(
      dynamics=make_mock_dynamics(),
      representation=rep,
      env=make_mock_command_env(),
      batch_size=BATCH,
      horizon=8,
      replay_buffer=make_mock_buffer(),
      sim_context=make_mock_sim_context(),
      task_specs=[make_motion_task(rep, num_goals=87)],
      device="cpu",
    )
    env.aux_state = identity_aux(BATCH)
    env._update_goal()
    assert torch.isfinite(env._wm_goal_obs).all()
    assert torch.isfinite(env._task_goal_obs["motion"]).all()
