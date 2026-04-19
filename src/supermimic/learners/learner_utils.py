"""Shared learner utilities."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn
import wandb


def load_wandb_checkpoint(
  run_path: str,
  task_id: str,
  dynamics: nn.Module,
  representation: nn.Module,
  device: str,
) -> None:
  """Download and load a world-model checkpoint from W&B.

  Loads the ``{"dynamics", "representation"}`` state-dict bundle written by
  ``SuperMimicRunner``.
  """
  print(f"[INFO] Loading World Model from W&B: {run_path}")

  run_id = run_path.split("/")[-1]
  download_dir = Path("logs") / task_id / "wandb_checkpoints" / run_id
  checkpoint_path = download_dir / "world_model.pt"

  if not checkpoint_path.exists():
    print(f"[INFO] Downloading world_model.pt from {run_path}...")
    api = wandb.Api()
    run = api.run(run_path)
    run.file("world_model.pt").download(root=str(download_dir), replace=True)

  bundle = torch.load(checkpoint_path, map_location=device)
  dynamics.load_state_dict(bundle["dynamics"])
  representation.load_state_dict(bundle["representation"])
  print(f"[INFO] World Model loaded from {checkpoint_path}")


def _optimize_step(
  optimizer: torch.optim.Optimizer,
  loss: torch.Tensor,
  params: Iterable[torch.Tensor] | None = None,
  max_norm: float | None = None,
) -> torch.Tensor | None:
  """Run one optimizer step: zero_grad -> backward -> optional clip -> step."""
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  grad_norm = None
  if params is not None and max_norm is not None:
    grad_norm = torch.nn.utils.clip_grad_norm_(
      params, max_norm=max_norm, error_if_nonfinite=True
    )
  optimizer.step()
  return grad_norm
