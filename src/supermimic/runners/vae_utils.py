"""Utilities for loading VAE checkpoints and configuring VAE prior envs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import wandb

from supermimic.models import load_actor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg


@dataclass
class VAEDims:
  """Dimensions inferred from a VAE state dict."""

  obs_dim: int
  goal_dim: int
  action_dim: int
  latent_dim: int


def _infer_vae_dims(state_dict: dict[str, torch.Tensor]) -> VAEDims:
  prior_weights = sorted(
    k for k in state_dict if k.startswith("prior_encoder.") and k.endswith(".weight")
  )
  decoder_weights = sorted(
    k for k in state_dict if k.startswith("decoder.") and k.endswith(".weight")
  )
  latent_dim = state_dict[prior_weights[-1]].shape[0]
  obs_dim = state_dict[prior_weights[0]].shape[1]
  action_dim = state_dict[decoder_weights[-1]].shape[0]
  normalizer_width = state_dict["obs_normalizer._mean"].shape[-1]
  goal_dim = normalizer_width - obs_dim
  return VAEDims(
    obs_dim=obs_dim, goal_dim=goal_dim, action_dim=action_dim, latent_dim=latent_dim
  )


def load_vae_from_wandb(
  wandb_path: str, device: str
) -> tuple[torch.nn.Module, VAEDims, str]:
  """Fetch + load a frozen VAE from a W&B run; return ``(vae, dims, representation)``."""
  api = wandb.Api()
  run = api.run(wandb_path)

  runner_cfg = run.config.get("runner_cfg", {})
  representation = runner_cfg.get("representation", "maxcoord")
  fixed_std = runner_cfg.get("fixed_std", 0.3)
  kl_weight = runner_cfg.get("kl_weight", 5e-5)

  run_id = wandb_path.split("/")[-1]
  download_dir = Path("logs") / "vae_checkpoints" / run_id
  checkpoint_path = download_dir / "actor.pt"

  if not checkpoint_path.exists():
    print(f"[INFO] Downloading VAE from {wandb_path}...")
    run.file("actor.pt").download(root=str(download_dir), replace=True)
  else:
    print(f"[INFO] Using cached VAE: {checkpoint_path}")

  print(
    f"[INFO] VAE: rep={representation}, fixed_std={fixed_std}, "
    f"kl_weight={kl_weight}, path={checkpoint_path}"
  )

  state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
  dims = _infer_vae_dims(state_dict)
  vae = load_actor(
    state_dict,
    dims.obs_dim,
    dims.goal_dim,
    dims.action_dim,
    device,
    fixed_std=fixed_std,
    kl_weight=kl_weight,
  )
  for p in vae.parameters():
    p.requires_grad_(False)
  return vae, dims, representation


def modify_env_cfg_for_vae_prior(cfg: ManagerBasedRlEnvCfg) -> None:
  """Strip motion-tracking terminations + domain rand for VAE-prior use."""
  from mjlab.managers.termination_manager import TerminationTermCfg

  cfg.terminations = {
    k: v
    for k, v in cfg.terminations.items()
    if isinstance(v, TerminationTermCfg) and v.time_out
  }
  cfg.events = {k: v for k, v in cfg.events.items() if k.startswith("reset_")}
  if "motion" in cfg.commands:
    cfg.commands["motion"].sampling_mode = "uniform"
