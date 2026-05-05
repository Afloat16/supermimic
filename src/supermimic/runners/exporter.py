"""ONNX exporter for SuperMimic actors (MLP or VAE, prior-only)."""

from __future__ import annotations

import os
from typing import cast

import torch
import torch.nn as nn
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.tracking.mdp import MotionCommand

from supermimic.models import VAEActor


def _get_device(module: nn.Module) -> torch.device:
  try:
    return next(module.parameters()).device
  except StopIteration:
    return torch.device("cpu")


def _export_onnx(
  module: nn.Module,
  sample_inputs: tuple[torch.Tensor, ...],
  input_names: list[str],
  output_names: list[str],
  path: str,
  verbose: bool = False,
) -> None:
  module.to("cpu")
  module.eval()
  torch.onnx.export(
    module,
    sample_inputs,
    path,
    export_params=True,
    opset_version=18,
    verbose=verbose,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={},
    dynamo=False,
  )


class _VaeActorOnnx(nn.Module):
  """VAEActor adapter exposing the rsl_rl ``as_onnx`` interface.

  Deploy semantics: ``obs → decode(encode(obs), obs)`` (prior-only). The
  ``input_size`` attribute and ``forward(obs) -> action`` signature mirror
  ``MLPModel.as_onnx`` so :class:`_ActorWrapper` treats both paths uniformly.
  """

  def __init__(self, vae: VAEActor) -> None:
    super().__init__()
    self.vae = vae
    self.input_size = vae.obs_dim

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.vae.decode(self.vae.encode(obs), obs)


class _ActorWrapper(nn.Module):
  """Bundles actor + per-body motion lookup tables for the tracking ONNX contract."""

  def __init__(
    self, env: ManagerBasedRlEnv, actor: nn.Module, verbose: bool = False
  ) -> None:
    super().__init__()
    self.actor = (
      _VaeActorOnnx(actor)
      if isinstance(actor, VAEActor)
      else actor.as_onnx(verbose=verbose)
    )
    self.obs_dim = self.actor.input_size

    cmd = cast(MotionCommand, env.command_manager.get_term("motion"))
    motion = cmd.motion
    self.register_buffer("_joint_pos", motion.joint_pos.detach().cpu())
    self.register_buffer("_joint_vel", motion.joint_vel.detach().cpu())
    self.register_buffer("_body_pos_w", motion.body_pos_w.detach().cpu())
    self.register_buffer("_body_quat_w", motion.body_quat_w.detach().cpu())
    self.register_buffer("_body_lin_vel_w", motion.body_lin_vel_w.detach().cpu())
    self.register_buffer("_body_ang_vel_w", motion.body_ang_vel_w.detach().cpu())
    self._time_step_total = int(self._joint_pos.shape[0])

  def forward(
    self, x: torch.Tensor, time_step: torch.Tensor
  ) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
  ]:
    t = torch.clamp(time_step.long().squeeze(-1), max=self._time_step_total - 1)
    return (
      self.actor(x),
      self._joint_pos[t],
      self._joint_vel[t],
      self._body_pos_w[t],
      self._body_quat_w[t],
      self._body_lin_vel_w[t],
      self._body_ang_vel_w[t],
    )


_ACTOR_OUTPUT_NAMES = [
  "actions",
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
]


def export_actor_as_onnx(
  env: ManagerBasedRlEnv,
  actor: nn.Module,
  path: str,
  run_name: str,
  filename: str = "policy.onnx",
  verbose: bool = False,
) -> None:
  """Export a SuperMimic actor to ONNX with bundled motion lookup tables.

  Graph: ``(obs, time_step) → (action, joint_pos, joint_vel,
  body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w)``. Deploy
  assembles ``obs`` from ``observation_names`` in the ONNX metadata.
  VAE deploy is prior-only (``decode(encode(obs), obs)``).
  """
  os.makedirs(path, exist_ok=True)
  onnx_path = os.path.join(path, filename)
  original_device = _get_device(actor)
  try:
    wrapper = _ActorWrapper(env, actor, verbose)
    _export_onnx(
      wrapper,
      (torch.zeros(1, wrapper.obs_dim), torch.zeros(1, 1)),
      ["obs", "time_step"],
      _ACTOR_OUTPUT_NAMES,
      onnx_path,
      verbose,
    )
  finally:
    actor.to(original_device)

  motion_term = cast(MotionCommand, env.command_manager.get_term("motion"))
  metadata = get_base_metadata(env, run_name)
  if not isinstance(actor, VAEActor):
    actor_goal_names = [
      n
      for n in env.observation_manager.active_terms["goal"]
      if n != "motion_anchor_pos_b"
    ]
    metadata["observation_names"] = (
      list(metadata["observation_names"]) + actor_goal_names
    )
  metadata.update(
    {
      "anchor_body_name": motion_term.cfg.anchor_body_name,
      "body_names": list(motion_term.cfg.body_names),
    }
  )
  attach_metadata_to_onnx(onnx_path, metadata)


class _PosteriorWrapper(nn.Module):
  """High-level posterior + frozen VAE decoder as one ``(obs) → action`` module."""

  def __init__(
    self,
    mlp: nn.Module,
    vae: VAEActor,
    latent_scale: float,
    sensor_dim: int,
    verbose: bool = False,
  ) -> None:
    super().__init__()
    self._mlp = mlp.as_onnx(verbose=verbose)
    self._vae = vae
    self._latent_scale = latent_scale
    self._sensor_dim = sensor_dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    residual = self._latent_scale * torch.tanh(self._mlp(x))
    obs = x[..., : self._sensor_dim]
    return self._vae.decode(self._vae.encode(obs) + residual, obs)


def export_posterior_as_onnx(
  env: ManagerBasedRlEnv,
  mlp: nn.Module,
  vae: VAEActor,
  latent_scale: float,
  sensor_dim: int,
  goal_dim: int,
  path: str,
  run_name: str,
  filename: str = "policy.onnx",
  verbose: bool = False,
) -> None:
  """Export high-level posterior + frozen VAE decoder as a single ONNX policy.

  Graph: ``(obs) → action`` where ``obs = cat([sensor, command])``,
  matching mjlab's velocity-task layout (command embedded in actor obs).
  The MLP reads the full obs; the VAE prior reads only the sensor prefix.
  """
  os.makedirs(path, exist_ok=True)
  onnx_path = os.path.join(path, filename)
  mlp_device = _get_device(mlp)
  vae_device = _get_device(vae)
  try:
    wrapper = _PosteriorWrapper(mlp, vae, latent_scale, sensor_dim, verbose)
    _export_onnx(
      wrapper,
      (torch.zeros(1, sensor_dim + goal_dim),),
      ["obs"],
      ["actions"],
      onnx_path,
      verbose,
    )
  finally:
    mlp.to(mlp_device)
    vae.to(vae_device)

  metadata = get_base_metadata(env, run_name)
  metadata["observation_names"] = list(metadata["observation_names"]) + ["command"]
  metadata["command_names"] = ["twist"]
  attach_metadata_to_onnx(onnx_path, metadata)
