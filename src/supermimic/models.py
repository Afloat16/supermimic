import torch
import torch.nn as nn
from rsl_rl.modules import EmpiricalNormalization


class WorldModel(nn.Module):
  """World Model predicting next state and reward given (obs, action, goal)."""

  def __init__(
    self,
    num_states: int,
    num_actions: int,
    num_goals: int,
    num_rewards: int,
    history: int = 1,
    backbone_hidden_dims: tuple[int, ...] = (512, 512, 512),
    dyn_hidden_dims: tuple[int, ...] = (),
    aux_hidden_dims: tuple[int, ...] = (512, 256, 128),
    act=nn.ELU,
  ):
    super().__init__()

    self.num_states = num_states
    self.num_actions = num_actions
    self.num_goals = num_goals
    self.num_rewards = num_rewards
    self.history = history

    full_input_dim = (num_states * history) + num_actions + num_actions + num_goals
    self.input_normalizer = EmpiricalNormalization(full_input_dim)

    def build_mlp(input_dim, hidden_dims, output_dim=None):
      layers: list[nn.Module] = []
      curr = input_dim
      for h in hidden_dims:
        layers.append(nn.Linear(curr, h))
        layers.append(act())
        curr = h
      if output_dim is not None:
        layers.append(nn.Linear(curr, output_dim))
        return nn.Sequential(*layers)
      return nn.Sequential(*layers), curr

    # ---- Backbone: (state_hist, action) -> z ----
    backbone_input_dim = full_input_dim - num_actions - num_goals
    self.backbone, backbone_out_dim = build_mlp(
      backbone_input_dim, backbone_hidden_dims
    )

    # ---- Dynamics head: z -> next-state delta ----
    self.dyn_head = build_mlp(backbone_out_dim, dyn_hidden_dims, output_dim=num_states)

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
    """
    Returns:
      next_state: [..., num_states] predicted next state
      reward: [..., num_rewards] predicted rewards
      termination: [..., 1] termination logits
    """
    # Keep the latest state for next-state reconstruction.
    latest_state = state[..., -1, :]

    # state is expected to be [..., history, num_states] -> [..., history * num_states]
    state = state.flatten(start_dim=-2)

    # Normalize all inputs together to match the statistics buffers
    full_input = torch.cat([state, action, last_action, goal], dim=-1)
    full_norm = self.input_normalizer(full_input)

    # Split normalized inputs
    # state: [0 : state_hist_dim]
    # action: [state_hist_dim : state_hist_dim + num_actions]
    # last_action: [state_hist_dim + num_actions : state_hist_dim + 2 * num_actions]
    # goal: [state_hist_dim + 2 * num_actions : ]
    state_hist_dim = self.num_states * self.history
    state_norm = full_norm[..., :state_hist_dim]
    action_norm = full_norm[..., state_hist_dim : state_hist_dim + self.num_actions]
    last_action_norm = full_norm[
      ..., state_hist_dim + self.num_actions : state_hist_dim + 2 * self.num_actions
    ]
    goal_norm = full_norm[..., state_hist_dim + 2 * self.num_actions :]

    # Backbone Input: (state_hist, action)
    backbone_in = torch.cat([state_norm, action_norm], dim=-1)
    z = self.backbone(backbone_in)

    delta = self.dyn_head(z)
    next_state = latest_state + delta

    # Reward head and Termination head conditioned on goal and action
    z_in = z.detach() if detach_backbone else z
    aux_input = torch.cat([z_in, action_norm, last_action_norm, goal_norm], dim=-1)
    reward = self.rew_head(aux_input)
    termination = self.term_head(aux_input)

    return next_state, reward, termination
