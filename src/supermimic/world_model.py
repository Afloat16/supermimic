import torch
import torch.nn as nn

from supermimic.policy import EmpiricalNormalization


class WorldModel(nn.Module):
    """World Model predicting next state given current state and action."""

    def __init__(self, num_obs: int, num_actions: int, hidden_dims: list[int] = [512, 512, 256]):
        super().__init__()

        # Input: State + Action
        input_dim = num_obs + num_actions
        output_dim = num_obs

        # Input normalizer
        self.input_normalizer = EmpiricalNormalization(input_dim)

        layers = []
        curr_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ELU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next state (or delta) given obs and action."""
        x = torch.cat([obs, action], dim=-1)
        x = self.input_normalizer(x)
        return self.net(x)
