import torch
import torch.nn as nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values.
    
    Copied from rsl_rl to maintain compatibility.
    """

    def __init__(self, shape, eps=1e-2, until=None):
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        """Normalize mean and variance of values based on empirical values."""
        return (x - self._mean) / (self._std + self.eps)


class Mlp(nn.Module):
    """Simple MLP policy matching the rsl_rl ActorCritic structure for the actor part."""

    def __init__(self, num_obs: int, num_actions: int, hidden_dims: list[int] = [512, 256, 128], normalization: bool = False):
        super().__init__()
        
        self.normalization = normalization
        if self.normalization:
            # Match rsl_rl naming: self.actor_obs_normalizer
            self.actor_obs_normalizer = EmpiricalNormalization(num_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()

        layers = []
        input_dim = num_obs
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ELU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, num_actions))

        # Match rsl_rl naming: self.actor
        self.actor = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        norm_obs = self.actor_obs_normalizer(obs)
        return self.actor(norm_obs)

    def load_state_dict_from_rsl_rl_ckpt(self, ckpt_path: str, device: str) -> None:
        """Load weights from an rsl_rl checkpoint."""
        ckpt = torch.load(ckpt_path, map_location=device)
        model_state_dict = ckpt["model_state_dict"]

        filtered_state_dict = {}
        for k, v in model_state_dict.items():
            if k.startswith("actor"):
                filtered_state_dict[k] = v
            if self.normalization and k.startswith("actor_obs_normalizer"):
                filtered_state_dict[k] = v

        self.load_state_dict(filtered_state_dict, strict=False)
        print(f"[INFO] Loaded weights for {list(filtered_state_dict.keys())[:3]}...")
