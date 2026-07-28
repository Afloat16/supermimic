"""Shared learner utilities."""

from __future__ import annotations

from collections.abc import Iterable

import torch


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
