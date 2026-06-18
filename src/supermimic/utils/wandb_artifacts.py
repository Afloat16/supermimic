"""W&B model-artifact helpers: digest-cached download and versioned logging."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import wandb

_ARTIFACT_CACHE_DIR = Path.home() / ".cache" / "supermimic" / "artifacts"


def download_artifact_cached(artifact: Any) -> Path:
  """Download a wandb artifact with digest-based caching.

  Skips ``artifact.download()`` entirely when the local cache already holds
  files for the same content digest. Re-downloads automatically when the
  artifact content changes (different digest).

  Returns the path to the downloaded artifact directory.
  """
  # entity/project/name contains "/"; hash it to a flat cache filename.
  raw_key = f"{artifact.entity}/{artifact.project}/{artifact.name}"
  key = hashlib.sha256(raw_key.encode()).hexdigest()[:16]
  _ARTIFACT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
  meta_file = _ARTIFACT_CACHE_DIR / f"{key}.json"

  try:
    meta = json.loads(meta_file.read_text())
    if meta["digest"] == artifact.digest and Path(meta["path"]).exists():
      print(
        f"[INFO] Using cached artifact: {artifact.name} "
        f"(digest: {artifact.digest[:8]}...)"
      )
      return Path(meta["path"])
  except (OSError, json.JSONDecodeError, KeyError, TypeError):
    pass

  print(
    f"[INFO] Downloading artifact: {artifact.name} (digest: {artifact.digest[:8]}...)"
  )
  path = artifact.download()
  meta_file.write_text(json.dumps({"digest": artifact.digest, "path": path}))
  return Path(path)


def download_model_dir(run: Any) -> Path:
  """Local dir of a run's ``:latest`` model-checkpoint artifact (digest-cached)."""
  artifact = next(
    (a for a in run.logged_artifacts() if a.type == "model" and "latest" in a.aliases),
    None,
  )
  if artifact is None:
    raise RuntimeError(f"No 'model' artifact found for run {run.id}")
  return download_artifact_cached(artifact)


def log_model_artifact(
  files: list[str], version: int, keep_last: int = 5, milestone_every: int = 25
) -> None:
  """Log ``files`` as a new version of the run's ``model`` checkpoint artifact."""
  assert wandb.run is not None, "log_model_artifact needs an active wandb run"
  artifact = wandb.Artifact(wandb.run.name, type="model")
  for path in files:
    artifact.add_file(path)
  wandb.run.log_artifact(artifact)
  stale = version - keep_last
  if stale >= 0 and stale % milestone_every != 0:
    name = f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.name}:v{stale}"
    wandb.Api().artifact(name, type="model").delete()
