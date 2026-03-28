"""Task-env registry: the single source of truth for `task_env` dispatch."""

from __future__ import annotations

from supermimic.envs.getup_env import GetupEnv, GetupEnvCfg
from supermimic.envs.vae_world_model_env import VaeWorldModelEnv
from supermimic.envs.velocity_env import VelocityEnv, VelocityEnvCfg

TASK_ENV_REGISTRY: dict[str, tuple[type[VaeWorldModelEnv], type]] = {
  "velocity": (VelocityEnv, VelocityEnvCfg),
  "getup": (GetupEnv, GetupEnvCfg),
}
