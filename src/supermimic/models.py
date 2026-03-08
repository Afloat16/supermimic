from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.modules import EmpiricalNormalization

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
