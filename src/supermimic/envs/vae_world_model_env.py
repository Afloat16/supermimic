"""VAE world model env for posterior training."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import quat_apply
from tensordict import TensorDict

from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.models import Dynamics, VAEActor
from supermimic.replay_buffer import ReplayBuffer
from supermimic.representations import StateRepresentation
from supermimic.sim_context import SimContext


@dataclass(frozen=True)
class ContinuousGoalSpec:
  """Slider control for one raw-goal dim."""

  name: str
  min: float
  max: float
  step: float
  default: float


@dataclass(frozen=True)
class DiscreteGoalSpec:
  """Dropdown control for one raw-goal dim with a fixed value set."""

  name: str
  choices: tuple[float, ...]
  default: float


type GoalSpec = ContinuousGoalSpec | DiscreteGoalSpec


@dataclass
class VaeWorldModelEnvCfg:
  """Base cfg shared by all VAE-prior task envs. Task cfgs inherit and extend."""

  latent_scale_mult: float = 0.3
  """Tanh bound on latent residual, in units of VAE σ_φ."""


def get_forward_world(aux_state: torch.Tensor) -> torch.Tensor:
  """Body +x rotated into the world frame (Z-up)."""
  root_quat = aux_state[:, 3:7]
  forward_body = torch.zeros(aux_state.shape[0], 3, device=aux_state.device)
  forward_body[:, 0] = 1.0
  return quat_apply(root_quat, forward_body)


def get_facing_angle(aux_state: torch.Tensor) -> torch.Tensor:
  """Yaw of body +x projected onto the world XY plane."""
  forward_world = get_forward_world(aux_state)
  return torch.atan2(forward_world[:, 1], forward_world[:, 0])


class VaeWorldModelEnv[CfgT: VaeWorldModelEnvCfg](WorldModelEnv):
  """WM env with frozen-VAE decode; subclasses define task goals and rewards."""

  cfg: CfgT
  goal_dim: int
  raw_goal_dim: int

  def __init__(
    self,
    cfg: CfgT,
    dynamics: Dynamics,
    representation: StateRepresentation,
    vae: VAEActor,
    batch_size: int,
    horizon: int,
    replay_buffer: ReplayBuffer,
    body_masses: torch.Tensor,
    latent_dim: int,
    sim_context: SimContext,
    device: str = "cpu",
  ) -> None:
    super().__init__(
      dynamics=dynamics,
      representation=representation,
      batch_size=batch_size,
      horizon=horizon,
      replay_buffer=replay_buffer,
      sim_context=sim_context,
      device=device,
    )
    self.cfg = cfg
    self.vae = vae
    self._latent_dim = latent_dim
    self.goal = torch.zeros(batch_size, self.raw_goal_dim, device=device)
    self._step_count = 0
    self._is_last_mask = torch.zeros(batch_size, device=device)
    self._body_mass_w = (body_masses / body_masses.sum()).to(device)
    self._latent_scale = cfg.latent_scale_mult * vae.fixed_std
    # Sensor written by ``_get_policy_obs`` and read by the next ``step``'s
    # ``decode_residual``. The actor and the VAE encoder must see the same
    # noise sample, so we cache the obs the actor was given and reuse it
    # downstream rather than recompute (which would draw new noise).
    self._pending_vae_obs: torch.Tensor | None = None

  @property
  def num_actions(self) -> int:
    return self._latent_dim

  def _on_reset(self, batch: TensorDict) -> None:
    self._step_count = 0
    self._is_last_mask = torch.zeros(self.num_envs, device=self.device)
    self.goal = batch["goal"]

  def _get_wm_goal(self) -> torch.Tensor:
    # Posterior tasks have no motion goal; _wm_step ignores this argument.
    return torch.zeros(self.num_envs, 1, device=self.device)

  def _compute_term_prob(self, term_logit: torch.Tensor) -> torch.Tensor:
    return torch.zeros(term_logit.shape[0], device=term_logit.device)

  def _get_policy_obs(self) -> TensorDict:
    goal_obs = self.build_goal_obs(self.goal, self.aux_state)
    sensor = self.representation.compute_actor_sensor(
      self.rep_state,
      self.dynamics.num_actions,
      self.sim_ctx,
      aux_state=self.aux_state,
      last_action=self.last_actions,
      noisy=self.representation.actor_obs_mode == "proprio",
    )
    self._pending_vae_obs = sensor
    return TensorDict(
      {
        "actor": torch.cat([sensor, goal_obs], dim=-1),
        "critic": torch.cat([self.rep_state, goal_obs, self.sim_ctx], dim=-1),
      },
      batch_size=[self.num_envs],
      device=self.device,
    )

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
    """VAE decode latent → joint actions eagerly, then delegate to base step.

    VAE decode is kept eager (outside ``_wm_step``) because VAE's internal
    branches and ``self._kl_raw`` write add trace-time overhead that dominates
    the cudagraph gains when folded into the bundled compile.
    """
    self._step_count += 1
    # Allocate fresh: in-place fill_ would mutate a tensor that BPTT backward
    # has captured for gradient computation, triggering a version-counter error.
    self._is_last_mask = torch.full(
      (self.num_envs,),
      float(self._step_count >= self.horizon),
      device=self.device,
    )
    assert self._pending_vae_obs is not None, "step() called before get_observations()"
    joint_actions = self.decode_residual(self._pending_vae_obs, actions)
    return super().step(joint_actions)

  def _wm_step(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    goal: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del last_action, goal
    next_aux, next_rep = self.dynamics(rep_state, action, aux_state, sim_ctx)
    pred_d = torch.zeros(rep_state.shape[0], 1, device=rep_state.device)
    reward = self._compute_reward(next_rep, next_aux, self._is_last_mask)
    return next_aux, next_rep, pred_d, reward

  @staticmethod
  @abstractmethod
  def build_goal_obs(raw_goal: torch.Tensor, aux_state: torch.Tensor) -> torch.Tensor:
    """Convert raw goals to policy goal obs; called every step as the robot moves."""
    ...

  @abstractmethod
  def _compute_reward(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    is_last_mask: torch.Tensor,
  ) -> torch.Tensor: ...

  @abstractmethod
  def sample_raw_goal(self, num_envs: int) -> torch.Tensor: ...

  def decode_residual(
    self, vae_obs: torch.Tensor, residual: torch.Tensor
  ) -> torch.Tensor:
    """Apply tanh-bounded residual to the VAE prior and decode to joint actions."""
    z = self.vae.encode(vae_obs) + self._latent_scale * torch.tanh(residual)
    return self.vae.decode(z, vae_obs)
