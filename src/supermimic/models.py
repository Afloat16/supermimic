from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.modules import EmpiricalNormalization
from tensordict import TensorDict

from supermimic.representations import StateRepresentation


class _CudaGraphNormalization(EmpiricalNormalization):
  """EmpiricalNormalization with in-place _std update for CUDA graph compatibility.

  The upstream update() does ``self._std = torch.sqrt(self._var)`` which creates
  a new tensor, invalidating data pointers captured by CUDA graphs
  (torch.compile reduce-overhead mode). This subclass uses copy_() instead.
  """

  @torch.jit.unused
  def update(self, x: torch.Tensor) -> None:
    std_buf = self._std
    super().update(x)
    std_buf.copy_(self._std)
    self._std = std_buf


def build_mlp(
  input_dim: int,
  hidden_dims: tuple[int, ...],
  output_dim: int | None = None,
  use_layer_norm: bool = False,
  act: type[nn.Module] = nn.GELU,
) -> nn.Sequential | tuple[nn.Sequential, int]:
  """Build an MLP from layer specs.

  When *output_dim* is given, returns the ``nn.Sequential`` directly.
  Otherwise returns ``(sequential, last_hidden_dim)``.
  """
  layers: list[nn.Module] = []
  curr = input_dim
  for h in hidden_dims:
    layers.append(nn.Linear(curr, h))
    if use_layer_norm:
      layers.append(nn.LayerNorm(h))
    layers.append(act())
    curr = h
  if output_dim is not None:
    layers.append(nn.Linear(curr, output_dim))
    return nn.Sequential(*layers)
  return nn.Sequential(*layers), curr


class WorldModel(nn.Module):
  """World Model predicting next full-state and reward given (state, action, goal)."""

  def __init__(
    self,
    representation: StateRepresentation,
    num_actions: int,
    num_goals: int,
    num_rewards: int,
    dt: float,
    history: int = 1,
    backbone_hidden_dims: tuple[int, ...] = (512, 512, 512),
    dyn_hidden_dims: tuple[int, ...] = (),
    aux_hidden_dims: tuple[int, ...] = (512, 256, 128),
    act: type[nn.Module] = nn.GELU,
  ):
    super().__init__()

    self.representation = representation
    self.num_rep_state = representation.rep_state_dim
    self.num_actions = num_actions
    self.num_goals = num_goals
    self.num_rewards = num_rewards
    self.history = history
    self.dt = dt

    self.state_normalizer = _CudaGraphNormalization(self.num_rep_state * history)
    self.action_normalizer = _CudaGraphNormalization(num_actions)
    self.goal_normalizer = _CudaGraphNormalization(num_goals)

    # ---- Backbone: (rep_state_hist, action) -> z ----
    backbone_input_dim = (self.num_rep_state * history) + num_actions
    self.backbone, backbone_out_dim = build_mlp(
      backbone_input_dim, backbone_hidden_dims, use_layer_norm=True, act=act
    )

    # ---- Dynamics head: z -> prediction ----
    self.dyn_head = build_mlp(
      backbone_out_dim,
      dyn_hidden_dims,
      output_dim=representation.prediction_dim,
      use_layer_norm=True,
      act=act,
    )

    # ---- Reward head: z + action + last_action + goal -> reward ----
    aux_input_dim = backbone_out_dim + num_actions + num_actions + num_goals
    self.rew_head = build_mlp(
      aux_input_dim, aux_hidden_dims, output_dim=num_rewards, act=act
    )

    # ---- Termination head: z + action + last_action + goal -> termination ----
    self.term_head = build_mlp(aux_input_dim, aux_hidden_dims, output_dim=1, act=act)

  def forward(
    self,
    rep_state_hist: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    goal: torch.Tensor,
    aux_state: torch.Tensor,
    detach_backbone: bool = False,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict next state, reward, and termination.

    Args:
      rep_state_hist: (B, history, rep_state_dim)
      action: (B, action_dim)
      last_action: (B, action_dim)
      goal: (B, goal_dim)
      aux_state: (B, 7) [root_pos(3), root_quat(4)]

    Returns:
      (next_aux_state, next_rep_state, reward, termination)
    """
    latest_rep_state = rep_state_hist[..., -1, :]

    state_norm = self.state_normalizer(rep_state_hist.flatten(start_dim=-2))
    action_norm = self.action_normalizer(action)
    last_action_norm = self.action_normalizer(last_action)
    goal_norm = self.goal_normalizer(goal)

    z = self.backbone(torch.cat([state_norm, action_norm], dim=-1))
    prediction = self.dyn_head(z)

    next_aux, next_rep_state = self.representation.apply_prediction(
      aux_state, latest_rep_state, prediction, self.dt
    )

    z_in = z.detach() if detach_backbone else z
    aux_input = torch.cat([z_in, action_norm, last_action_norm, goal_norm], dim=-1)
    reward = self.rew_head(aux_input)
    termination = self.term_head(aux_input)

    return next_aux, next_rep_state, reward, termination

  def normalizers(self) -> list[_CudaGraphNormalization]:
    return [self.state_normalizer, self.action_normalizer, self.goal_normalizer]


class VAEActor(nn.Module):
  """CVAE actor with prior/posterior encoders and MLP decoder.

  Drop-in replacement for rsl_rl MLPModel. Forward takes a TensorDict
  with an ``"actor"`` key (concatenated obs + goal) and returns actions.
  """

  def __init__(
    self,
    obs_dim: int,
    goal_dim: int,
    action_dim: int,
    latent_dim: int = 64,
    hidden_dims: tuple[int, ...] = (512, 256, 128),
    fixed_std: float = 0.3,
    kl_weight: float = 5e-5,
  ):
    super().__init__()
    self.obs_dim = obs_dim
    self.goal_dim = goal_dim
    self.fixed_std = fixed_std
    self.kl_weight = kl_weight
    self._kl_raw: torch.Tensor | None = None

    self.obs_normalizer = _CudaGraphNormalization(obs_dim + goal_dim)
    self.prior_encoder = build_mlp(obs_dim, hidden_dims, output_dim=latent_dim)
    self.posterior_encoder = build_mlp(
      obs_dim + goal_dim, hidden_dims, output_dim=latent_dim
    )
    self.decoder = build_mlp(latent_dim + obs_dim, hidden_dims, output_dim=action_dim)

  def _normalize(
    self, obs: torch.Tensor, goal: torch.Tensor | None = None
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalize obs (and goal) through the joint normalizer."""
    if goal is not None:
      x_norm = self.obs_normalizer(torch.cat([obs, goal], dim=-1))
      return x_norm[..., : self.obs_dim], x_norm[..., self.obs_dim :]
    # obs-only: pad with zeros so normalizer sees the expected input width.
    x = torch.cat([obs, obs.new_zeros(*obs.shape[:-1], self.goal_dim)], dim=-1)
    return self.obs_normalizer(x)[..., : self.obs_dim], None

  def _encode(
    self, obs_n: torch.Tensor, goal_n: torch.Tensor | None = None
  ) -> torch.Tensor:
    """Encode from already-normalized inputs.

    Uses posterior when *goal_n* is provided, prior otherwise.
    The posterior predicts a *residual* on top of the prior mean:
    ``mu = mu_prior + mu_residual``.
    Reparameterization noise and KL loss are only applied in training mode.
    """
    z = self.prior_encoder(obs_n)
    self._kl_raw = None
    if goal_n is not None:
      residual = self.posterior_encoder(torch.cat([obs_n, goal_n], dim=-1))
      z = z + residual
      if self.training:
        z = z + self.fixed_std * torch.randn_like(z)
        self._kl_raw = 0.5 * (residual**2 / self.fixed_std**2).sum(dim=-1)
    return z

  def _decode(self, z: torch.Tensor, obs_n: torch.Tensor) -> torch.Tensor:
    """Decode from already-normalized obs."""
    return self.decoder(torch.cat([z, obs_n], dim=-1))

  def encode(self, obs: torch.Tensor, goal: torch.Tensor | None = None) -> torch.Tensor:
    """Encode into latent code, using posterior if goal is provided during training."""
    obs_n, goal_n = self._normalize(obs, goal)
    return self._encode(obs_n, goal_n)

  def decode(self, z: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Decode latent code and observation into actions."""
    obs_n, _ = self._normalize(obs)
    return self._decode(z, obs_n)

  def forward(self, obs_td: TensorDict) -> torch.Tensor:
    x = obs_td["actor"]
    x_norm = self.obs_normalizer(x)
    obs_n = x_norm[..., : self.obs_dim]
    goal_n = x_norm[..., self.obs_dim :]

    z = self._encode(obs_n, goal_n)
    return self._decode(z, obs_n)

  def auxiliary_loss(self) -> torch.Tensor | float:
    if self._kl_raw is None:
      return 0.0
    return self.kl_weight * self._kl_raw

  def auxiliary_stats(self) -> dict[str, float]:
    if self._kl_raw is None:
      return {}
    return {"kl": self._kl_raw.mean().item()}

  @torch.jit.unused
  def update_normalization(self, obs_td: TensorDict) -> None:
    self.obs_normalizer.update(obs_td["actor"])


def load_actor(
  state_dict: dict[str, torch.Tensor],
  obs_dim: int,
  goal_dim: int,
  action_dim: int,
  device: str = "cpu",
) -> nn.Module:
  """Load an actor from a state_dict, auto-detecting the type from its keys."""
  if "prior_encoder.0.weight" in state_dict:
    latent_dim = state_dict["decoder.0.weight"].shape[1] - obs_dim
    actor = VAEActor(
      obs_dim=obs_dim,
      goal_dim=goal_dim,
      action_dim=action_dim,
      latent_dim=latent_dim,
    ).to(device)
  else:
    from rsl_rl.models import MLPModel

    input_dim = obs_dim + goal_dim
    obs_td = TensorDict(
      {"actor": torch.zeros(1, input_dim, device=device)},
      batch_size=[1],
      device=device,
    )
    actor = MLPModel(
      obs=obs_td,
      obs_groups={"actor": ["actor"]},
      obs_set="actor",
      output_dim=action_dim,
      hidden_dims=(512, 256, 128),
      obs_normalization=True,
    ).to(device)
  actor.load_state_dict(state_dict)
  actor.eval()
  return actor
