import torch
import torch.nn as nn


class WorldModel(nn.Module):
    """World Model predicting next state given current state and action."""

    def __init__(self, num_obs: int, num_actions: int, hidden_dims: list[int] = [2048, 2048, 2048]):
        super().__init__()

        # Input: State + Action
        input_dim = num_obs + num_actions
        output_dim = num_obs

        layers = []
        curr_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ELU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))

        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next state (or delta) given obs and action."""
        x = torch.cat([obs, action], dim=-1)
        x = (x - self.input_mean) / (self.input_std + 1e-6)
        return self.net(x)
