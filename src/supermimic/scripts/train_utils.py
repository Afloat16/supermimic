"""Shared utilities for training scripts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import wandb
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

_ARTIFACT_CACHE_DIR = Path.home() / ".cache" / "supermimic" / "artifacts"


def download_artifact_cached(artifact: Any, root: str | None = None) -> str:
  """Download a wandb artifact with digest-based caching.

  Skips ``artifact.download()`` entirely when the local cache already holds
  files for the same content digest. Re-downloads automatically when the
  artifact content changes (different digest).

  Returns the path to the downloaded artifact directory.
  """
  # Build a collision-free cache key via hashing
  raw_key = f"{artifact.entity}/{artifact.project}/{artifact.name}"
  key = hashlib.sha256(raw_key.encode()).hexdigest()[:16]
  _ARTIFACT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
  meta_file = _ARTIFACT_CACHE_DIR / f"{key}.json"

  if meta_file.exists():
    try:
      meta = json.loads(meta_file.read_text())
      if meta.get("digest") == artifact.digest:
        cached_path = Path(meta["path"])
        if cached_path.exists():
          print(
            f"[INFO] Using cached artifact: {artifact.name} "
            f"(digest: {artifact.digest[:8]}...)"
          )
          return str(cached_path)
    except Exception:
      pass

  print(
    f"[INFO] Downloading artifact: {artifact.name} (digest: {artifact.digest[:8]}...)"
  )
  kwargs = {}
  if root is not None:
    kwargs["root"] = root
  path = artifact.download(**kwargs)

  meta_file.write_text(json.dumps({"digest": artifact.digest, "path": path}))

  return path


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
    artifact = api.artifact(name)
    paths.append(str(Path(download_artifact_cached(artifact)) / "motion.npz"))

  print(f"[INFO] Loaded {len(paths)} motion(s)")
  return paths if len(paths) > 1 else paths[0]


def setup_device(device: str | None = None) -> str:
  """Configure torch backends and resolve device."""
  configure_torch_backends()
  return device or ("cuda:0" if torch.cuda.is_available() else "cpu")


def create_env(
  env_cfg: ManagerBasedRlEnvCfg, num_envs: int, device: str
) -> RslRlVecEnvWrapper:
  """Create a wrapped RL environment."""
  env_cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  return RslRlVecEnvWrapper(env)
