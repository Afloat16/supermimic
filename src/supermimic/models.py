from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.modules import EmpiricalNormalization

from supermimic.representations import StateRepresentation


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
    act=nn.GELU,
  ):
    super().__init__()

    self.representation = representation
    self.num_states = representation.full_state_dim
    self.num_rep_state = representation.rep_state_dim
    self.num_actions = num_actions
    self.num_goals = num_goals
    self.num_rewards = num_rewards
    self.history = history
    self.dt = dt

    full_input_dim = (
      (self.num_rep_state * history) + num_actions + num_actions + num_goals
    )
    self.input_normalizer = EmpiricalNormalization(full_input_dim)

    def build_mlp(input_dim, hidden_dims, output_dim=None, use_layer_norm=False):
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

    # ---- Backbone: (rep_state_hist, action) -> z ----
    backbone_input_dim = full_input_dim - num_actions - num_goals
    self.backbone, backbone_out_dim = build_mlp(
      backbone_input_dim, backbone_hidden_dims, use_layer_norm=True
    )

    # ---- Dynamics head: z -> prediction ----
    self.dyn_head = build_mlp(
      backbone_out_dim,
      dyn_hidden_dims,
      output_dim=representation.prediction_dim,
      use_layer_norm=True,
    )

    # ---- Reward head: z + action + last_action + goal -> reward ----
    aux_input_dim = backbone_out_dim + num_actions + num_actions + num_goals
    self.rew_head = build_mlp(aux_input_dim, aux_hidden_dims, output_dim=num_rewards)

    # ---- Termination head: z + action + last_action + goal -> termination ----
    self.term_head = build_mlp(aux_input_dim, aux_hidden_dims, output_dim=1)

  def forward(
    self,
    state: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    goal: torch.Tensor,
    detach_backbone: bool = False,
  ) -> tuple[torch.Tensor, ...]:
    latest_full_state = state[..., -1, :]
    return self.transition(
      self.represent(state),
      action,
      last_action,
      goal,
      latest_full_state,
      detach_backbone,
    )

  def transition(
    self,
    rep_state_hist: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    goal: torch.Tensor,
    curr_full_state: torch.Tensor,
    detach_backbone: bool = False,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict next state, reward, and termination.

    Returns:
      (next_full_state, next_rep_state, reward, termination)
    """
    latest_rep_state = rep_state_hist[..., -1, :]

    flat = rep_state_hist.flatten(start_dim=-2)
    full_input = torch.cat([flat, action, last_action, goal], dim=-1)
    full_norm = self.input_normalizer(full_input)

    state_hist_dim = self.num_rep_state * self.history
    state_norm = full_norm[..., :state_hist_dim]
    action_norm = full_norm[..., state_hist_dim : state_hist_dim + self.num_actions]
    last_action_norm = full_norm[
      ..., state_hist_dim + self.num_actions : state_hist_dim + 2 * self.num_actions
    ]
    goal_norm = full_norm[..., state_hist_dim + 2 * self.num_actions :]

    z = self.backbone(torch.cat([state_norm, action_norm], dim=-1))
    prediction = self.dyn_head(z)

    next_full_state, next_rep_state = self.representation.apply_prediction(
      curr_full_state, latest_rep_state, prediction, self.dt
    )

    z_in = z.detach() if detach_backbone else z
    aux_input = torch.cat([z_in, action_norm, last_action_norm, goal_norm], dim=-1)
    reward = self.rew_head(aux_input)
    termination = self.term_head(aux_input)

    return next_full_state, next_rep_state, reward, termination

  def represent(self, state: torch.Tensor) -> torch.Tensor:
    return self.representation.represent(state)
