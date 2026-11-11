from pathlib import Path

import pytest

from supermimic.utils import wandb_artifacts


class Artifact:
  entity, project, name, digest = "sample", "project", "model:v1", "0123456789abcdef"

  def __init__(self, root):
    self.root = root
    self.downloads = 0
    self.verifications = 0

  def download(self):
    self.downloads += 1
    self.root.mkdir(exist_ok=True)
    (self.root / "weights.pt").write_bytes(b"complete checkpoint")
    return str(self.root)

  def verify(self, root):
    self.verifications += 1
    if (Path(root) / "weights.pt").read_bytes() != b"complete checkpoint":
      raise ValueError("Digest mismatch")


@pytest.mark.parametrize("corruption", ["missing", "changed", "metadata"])
def test_broken_cache_is_downloaded_again(tmp_path, monkeypatch, corruption):
  cache = tmp_path / "cache"
  monkeypatch.setattr(wandb_artifacts, "_ARTIFACT_CACHE_DIR", cache)
  artifact = Artifact(tmp_path / "model")
  root = wandb_artifacts.download_artifact_cached(artifact)
  if corruption == "missing":
    (root / "weights.pt").unlink()
  elif corruption == "changed":
    (root / "weights.pt").write_bytes(b"corrupt checkpoint!")
  else:
    next(cache.glob("*.json")).write_bytes(b"\xff")
  assert wandb_artifacts.download_artifact_cached(artifact) == root
  assert artifact.downloads == 2
  assert (root / "weights.pt").read_bytes() == b"complete checkpoint"


def test_complete_cache_avoids_another_download(tmp_path, monkeypatch):
  monkeypatch.setattr(wandb_artifacts, "_ARTIFACT_CACHE_DIR", tmp_path / "cache")
  artifact = Artifact(tmp_path / "model")
  first = wandb_artifacts.download_artifact_cached(artifact)
  assert wandb_artifacts.download_artifact_cached(artifact) == first
  assert artifact.downloads == 1
  assert artifact.verifications == 1
