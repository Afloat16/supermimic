"""Shared learner utilities."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import wandb

from supermimic.utils.wandb_artifacts import download_model_dir


def load_wandb_checkpoint(
  wandb_run_path: str,
  dynamics: nn.Module,
  representation: nn.Module,
  device: str,
) -> None:
  """Load a W&B run's world-model bundle in place into dynamics and representation."""
  print(f"[INFO] Loading World Model from W&B: {wandb_run_path}")
  run = wandb.Api().run(wandb_run_path)
  bundle = torch.load(download_model_dir(run) / "world_model.pt", map_location=device)
  dynamics.load_state_dict(bundle["dynamics"])
  representation.load_state_dict(bundle["representation"])


def _optimize_step(
  optimizer: torch.optim.Optimizer,
  loss: torch.Tensor,
  params: Iterable[torch.Tensor] | None = None,
  max_norm: float | None = None,
  clip_value: float = 0.0,
) -> torch.Tensor | None:
  """Run one optimizer step: zero_grad -> backward -> optional clip -> step."""
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  grad_norm = None
  if params is not None and max_norm is not None:
    # Don't set error_if_nonfinite=True: it forces a per-step CPU<->GPU sync.
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
    if clip_value > 0:
      torch.nn.utils.clip_grad_value_(params, clip_value)
  optimizer.step()
  return grad_norm
