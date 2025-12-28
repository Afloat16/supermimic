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


class Policy(nn.Module):
    """Simple MLP policy matching the rsl_rl ActorCritic structure for the actor part."""

    def __init__(self, num_obs: int, num_actions: int, hidden_dims: list[int] = [512, 256, 128],
                 normalization: bool = False):
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


class WorldModel(nn.Module):
    """World Model predicting state delta and reward given (obs, action, goal)."""

    def __init__(
            self,
            num_obs: int,
            num_actions: int,
            num_goals: int,
            backbone_hidden_dims: list[int] = [2048, 2048, 2048, 2048, 2048],
            dyn_hidden_dims: list[int] = [],
            rew_hidden_dims: list[int] = [512, 256, 128],
            act=nn.ELU,
    ):
        super().__init__()

        self.num_obs = num_obs
        self.num_actions = num_actions
        self.num_goals = num_goals

        full_input_dim = num_obs + num_actions + num_goals
        self.register_buffer("input_mean", torch.zeros(full_input_dim))
        self.register_buffer("input_std", torch.ones(full_input_dim))

        # ---- Backbone: (obs, action, goal) -> z ----
        backbone_input_dim = num_obs + num_actions + num_goals
        backbone_layers = []
        curr = backbone_input_dim
        for h in backbone_hidden_dims:
            backbone_layers += [nn.Linear(curr, h), act()]
            curr = h
        self.backbone = nn.Sequential(*backbone_layers)
        backbone_out_dim = curr

        # ---- Dynamics head: z -> delta_state ----
        dyn_layers = []
        curr = backbone_out_dim
        for h in dyn_hidden_dims:
            dyn_layers += [nn.Linear(curr, h), act()]
            curr = h
        dyn_layers += [nn.Linear(curr, num_obs)]
        self.dyn_head = nn.Sequential(*dyn_layers)

        # ---- Reward head: z + goal -> reward ----
        rew_layers = []
        curr = backbone_out_dim + num_goals
        for h in rew_hidden_dims:
            rew_layers += [nn.Linear(curr, h), act()]
            curr = h
        rew_layers += [nn.Linear(curr, 1)]
        self.rew_head = nn.Sequential(*rew_layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, goal: torch.Tensor,
                detach_reward_backbone: bool = False) -> torch.Tensor:
        """
        Returns: Tensor[..., num_obs + 1] = [delta_state, reward]
        """

        # Mask out root_pos x and y (first 2 dims) for translation invariance
        obs_masked = obs.clone()
        obs_masked[..., :2] = 0.0

        # Normalize all inputs together to match the statistics buffers
        full_input = torch.cat([obs_masked, action, goal], dim=-1)
        full_norm = (full_input - self.input_mean) / (self.input_std + 1e-6)

        # Split normalized inputs
        split_idx = self.num_obs + self.num_actions
        backbone_in = full_norm
        goal_norm = full_norm[..., split_idx:]

        z = self.backbone(backbone_in)

        delta = self.dyn_head(z)

        # Reward head conditioned on goal
        z_in = z.detach() if detach_reward_backbone else z
        rew_input = torch.cat([z_in, goal_norm], dim=-1)
        reward = self.rew_head(rew_input)

        return torch.cat([delta, reward], dim=-1)
