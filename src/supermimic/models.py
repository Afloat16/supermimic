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

  def __init__(
    self,
    num_obs: int,
    num_actions: int,
    hidden_dims: tuple[int, ...] = (2048, 2048, 2048, 2048, 2048),
    normalization: bool = False,
  ):
    super().__init__()

    self.normalization = normalization
    self.is_recurrent = False
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
    num_rewards: int = 1,
    history: int = 1,
    backbone_hidden_dims: tuple[int, ...] = (2048, 2048, 2048, 2048, 2048),
    dyn_hidden_dims: tuple[int, ...] = (),
    aux_hidden_dims: tuple[int, ...] = (512, 256, 128),
    act=nn.ELU,
  ):
    super().__init__()

    self.num_obs = num_obs
    self.num_actions = num_actions
    self.num_goals = num_goals
    self.num_rewards = num_rewards
    self.history = history

    full_input_dim = (num_obs * history) + num_actions + num_actions + num_goals
    self.register_buffer("input_mean", torch.zeros(full_input_dim))
    self.register_buffer("input_std", torch.ones(full_input_dim))

    # ---- Backbone: (obs_hist, action) -> z ----
    backbone_input_dim = full_input_dim - num_actions - num_goals
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

    # ---- Reward head: z + action + last_action + goal -> reward ----
    rew_layers = []
    curr = backbone_out_dim + num_actions + num_actions + num_goals
    for h in aux_hidden_dims:
      rew_layers += [nn.Linear(curr, h), act()]
      curr = h
    rew_layers += [nn.Linear(curr, num_rewards)]
    self.rew_head = nn.Sequential(*rew_layers)

    # ---- Termination head: z + action + last_action + goal -> termination ----
    term_layers = []
    curr = backbone_out_dim + num_actions + num_actions + num_goals
    term_hidden_dims = aux_hidden_dims
    for h in term_hidden_dims:
      term_layers += [nn.Linear(curr, h), act()]
      curr = h
    term_layers += [nn.Linear(curr, 1)]
    self.term_head = nn.Sequential(*term_layers)

  def forward(
    self,
    obs: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    goal: torch.Tensor,
    detach_reward_backbone: bool = False,
  ) -> torch.Tensor:
    """
    Returns: Tensor[..., num_obs + num_rewards + 1] = [delta_state, reward, termination_logit]
    """
    # obs is expected to be [..., history, num_obs] -> [..., history * num_obs]
    obs = obs.flatten(start_dim=-2)

    # Normalize all inputs together to match the statistics buffers
    full_input = torch.cat([obs, action, last_action, goal], dim=-1)
    full_norm = (full_input - self.input_mean) / (self.input_std + 1e-6)

    # Split normalized inputs
    # obs: [0 : obs_hist_dim]
    # action: [obs_hist_dim : obs_hist_dim + num_actions]
    # last_action: [obs_hist_dim + num_actions : obs_hist_dim + 2 * num_actions]
    # goal: [obs_hist_dim + 2 * num_actions : ]
    obs_hist_dim = self.num_obs * self.history
    obs_norm = full_norm[..., :obs_hist_dim]
    action_norm = full_norm[..., obs_hist_dim : obs_hist_dim + self.num_actions]
    last_action_norm = full_norm[
      ..., obs_hist_dim + self.num_actions : obs_hist_dim + 2 * self.num_actions
    ]
    goal_norm = full_norm[..., obs_hist_dim + 2 * self.num_actions :]

    # Backbone Input: (obs_hist, action)
    backbone_in = torch.cat([obs_norm, action_norm], dim=-1)
    z = self.backbone(backbone_in)

    delta = self.dyn_head(z)

    # Reward head and Termination head conditioned on goal and action
    z_in = z.detach() if detach_reward_backbone else z
    aux_input = torch.cat([z_in, action_norm, last_action_norm, goal_norm], dim=-1)
    reward = self.rew_head(aux_input)
    termination = self.term_head(aux_input)

    return torch.cat([delta, reward, termination], dim=-1)
