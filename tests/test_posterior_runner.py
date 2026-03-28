"""Tests for PosteriorRunner task dispatch."""

from __future__ import annotations

import pytest

from supermimic.envs import TASK_ENV_REGISTRY
from supermimic.envs.getup_env import GetupEnv, GetupEnvCfg
from supermimic.envs.velocity_env import VelocityEnv, VelocityEnvCfg
from supermimic.runners.posterior_runner import PosteriorRunner, PosteriorRunnerCfg


class TestTaskEnvRegistry:
  def test_registry_keys_match_cfg_discriminators(self):
    """Each *EnvCfg's ``task_env`` discriminator must be a registry key."""
    for name, (_, cfg_cls) in TASK_ENV_REGISTRY.items():
      assert cfg_cls().task_env == name

  def test_registry_entries_are_typed_correctly(self):
    """Each registry value is (env_cls, env_cfg_cls) — both classes."""
    for name, (env_cls, cfg_cls) in TASK_ENV_REGISTRY.items():
      assert isinstance(env_cls, type), f"{name}: env_cls must be a class"
      assert isinstance(cfg_cls, type), f"{name}: cfg_cls must be a class"
      assert hasattr(env_cls, "raw_goal_dim"), f"{name}: env missing raw_goal_dim"
      assert hasattr(cfg_cls, "goal_specs"), f"{name}: cfg missing goal_specs()"

  def test_velocity_and_getup_classes_match(self):
    assert TASK_ENV_REGISTRY["velocity"] == (VelocityEnv, VelocityEnvCfg)
    assert TASK_ENV_REGISTRY["getup"] == (GetupEnv, GetupEnvCfg)


class TestBufferSchemaDispatch:
  @pytest.mark.parametrize(
    "task_cfg", [VelocityEnvCfg(), GetupEnvCfg()], ids=["velocity", "getup"]
  )
  def test_goal_field_width_matches_env_raw_goal_dim(self, task_cfg):
    """_buffer_schema sizes the goal field by the dispatched env's raw_goal_dim."""
    runner = PosteriorRunner.__new__(PosteriorRunner)
    runner.cfg = PosteriorRunnerCfg(task=task_cfg)
    schema = runner._buffer_schema(num_full_state=182, num_actions=29, num_rewards=1)
    expected = TASK_ENV_REGISTRY[task_cfg.task_env][0].raw_goal_dim
    assert schema["goal"].dim == expected

  def test_other_fields_passthrough_from_base(self):
    runner = PosteriorRunner.__new__(PosteriorRunner)
    runner.cfg = PosteriorRunnerCfg(task=VelocityEnvCfg())
    schema = runner._buffer_schema(num_full_state=182, num_actions=29, num_rewards=1)
    assert schema["state"].dim == 182
    assert schema["action"].dim == 29
    assert schema["reward"].dim == 1
    assert "terminated" in schema
    assert "truncated" in schema
