from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from supermimic.utils import wandb_artifacts


@pytest.mark.parametrize("upload_fails", [False, True])
def test_retention_waits_for_a_published_replacement(monkeypatch, upload_fails):
  events = []
  artifact = Mock()
  artifact.add_file.side_effect = lambda path: events.append("add")

  def wait():
    events.append("wait")
    if upload_fails:
      raise RuntimeError("upload failed")
    return artifact

  artifact.wait.side_effect = wait
  run = SimpleNamespace(
    id="run",
    entity="sample",
    project="project",
    log_artifact=lambda value: (events.append("log"), value)[1],
  )
  versions = [
    SimpleNamespace(
      version=f"v{i}", aliases=[], delete=lambda i=i: events.append(f"delete-{i}")
    )
    for i in range(1, 4)
  ]
  api = Mock()
  api.artifacts.side_effect = lambda *args: (events.append("list"), versions)[1]
  monkeypatch.setattr(
    wandb_artifacts.wandb, "Artifact", lambda *args, **kwargs: artifact
  )
  monkeypatch.setattr(wandb_artifacts.wandb, "run", run)
  monkeypatch.setattr(wandb_artifacts.wandb, "Api", lambda: api)
  wandb_artifacts.log_model_artifact(["checkpoint.pt"], keep_last=1)
  assert events[:3] == ["add", "log", "wait"]
  if upload_fails:
    assert events == ["add", "log", "wait"]
  else:
    assert events[3:] == ["list", "delete-1", "delete-2"]
