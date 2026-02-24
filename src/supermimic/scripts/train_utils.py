"""Shared utilities for training scripts."""

from pathlib import Path

import torch
import wandb
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends


def download_registry_motions(registry_name: str | None) -> str | list[str] | None:
  """Download motion files from W&B registry artifacts.

  Supports comma-separated registry names for multi-motion.
  Returns a single path string if one motion, a list if multiple, or None.
  """
  if not registry_name:
    return None

  names = [n.strip() for n in registry_name.split(",")]
  api = wandb.Api()

  paths = []
  for name in names:
    if ":" not in name:
      name = name + ":latest"
    print(f"[INFO] Downloading motion from registry: {name}")
    artifact = api.artifact(name)
    paths.append(str(Path(artifact.download()) / "motion.npz"))

  print(f"[INFO] Loaded {len(paths)} motion(s)")
  return paths if len(paths) > 1 else paths[0]


def setup_device(device: str | None = None) -> str:
  """Configure torch backends and resolve device."""
  configure_torch_backends()
  return device or ("cuda:0" if torch.cuda.is_available() else "cpu")


def create_env(env_cfg, num_envs: int, device: str):
  """Create a wrapped RL environment."""
  env_cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  return RslRlVecEnvWrapper(env)
