import torch
import torch.nn as nn
from rsl_rl.modules import EmpiricalNormalization


class Policy(nn.Module):
  """Simple MLP policy matching the rsl_rl ActorCritic structure for the actor part."""

  def __init__(
    self,
    num_obs: int,
    num_actions: int,
    hidden_dims: tuple[int, ...] = (512, 256, 128),
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

  def update_normalizer(self, obs: torch.Tensor) -> None:
    if self.normalization:
      self.actor_obs_normalizer.update(obs)

  def load_state_dict_from_rsl_rl_ckpt(self, ckpt_path: str, device: str) -> None:
    """Load weights from an rsl_rl checkpoint (supports both 3.x and 4.0 formats)."""
    ckpt = torch.load(ckpt_path, map_location=device)

    filtered_state_dict = {}

    if "actor_state_dict" in ckpt:
      # rsl_rl 4.0 format: separate actor_state_dict with mlp.*/obs_normalizer.* keys
      actor_state_dict = ckpt["actor_state_dict"]
      for k, v in actor_state_dict.items():
        if k.startswith("mlp."):
          # Map mlp.* -> actor.* (supermimic internal naming)
          new_key = "actor." + k[len("mlp.") :]
          filtered_state_dict[new_key] = v
        elif self.normalization and k.startswith("obs_normalizer."):
          # Map obs_normalizer.* -> actor_obs_normalizer.*
          new_key = "actor_obs_normalizer." + k[len("obs_normalizer.") :]
          filtered_state_dict[new_key] = v
    else:
      # rsl_rl 3.x format: combined model_state_dict with actor.*/actor_obs_normalizer.* keys
      model_state_dict = ckpt["model_state_dict"]
      for k, v in model_state_dict.items():
        if k.startswith("actor"):
          filtered_state_dict[k] = v
        if self.normalization and k.startswith("actor_obs_normalizer"):
          filtered_state_dict[k] = v

    self.load_state_dict(filtered_state_dict, strict=False)
    print(f"[INFO] Loaded weights for {list(filtered_state_dict.keys())[:3]}...")


class Critic(nn.Module):
  """Critic matching the rsl_rl Critic structure."""

  def __init__(
    self,
    num_obs: int,
    hidden_dims: tuple[int, ...] = (512, 256, 128),
    normalization: bool = False,
  ):
    super().__init__()

    self.normalization = normalization
    if self.normalization:
      self.critic_obs_normalizer = EmpiricalNormalization(num_obs)
    else:
      self.critic_obs_normalizer = nn.Identity()

    layers = []
    input_dim = num_obs
    for hidden_dim in hidden_dims:
      layers.append(nn.Linear(input_dim, hidden_dim))
      layers.append(nn.ELU())
      input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, 1))

    self.critic = nn.Sequential(*layers)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    norm_obs = self.critic_obs_normalizer(obs)
    return self.critic(norm_obs)

  def update_normalizer(self, obs: torch.Tensor) -> None:
    if self.normalization:
      self.critic_obs_normalizer.update(obs)

  def load_state_dict_from_rsl_rl_ckpt(self, ckpt_path: str, device: str) -> None:
    """Load weights from an rsl_rl checkpoint (supports both 3.x and 4.0 formats)."""
    ckpt = torch.load(ckpt_path, map_location=device)

    filtered_state_dict = {}

    if "critic_state_dict" in ckpt:
      # rsl_rl 4.0 format: separate critic_state_dict with mlp.*/obs_normalizer.* keys
      critic_state_dict = ckpt["critic_state_dict"]
      for k, v in critic_state_dict.items():
        if k.startswith("mlp."):
          # Map mlp.* -> critic.* (supermimic internal naming)
          new_key = "critic." + k[len("mlp.") :]
          filtered_state_dict[new_key] = v
        elif self.normalization and k.startswith("obs_normalizer."):
          # Map obs_normalizer.* -> critic_obs_normalizer.*
          new_key = "critic_obs_normalizer." + k[len("obs_normalizer.") :]
          filtered_state_dict[new_key] = v
    else:
      # rsl_rl 3.x format: combined model_state_dict with critic.*/critic_obs_normalizer.* keys
      model_state_dict = ckpt["model_state_dict"]
      for k, v in model_state_dict.items():
        if k.startswith("critic"):
          filtered_state_dict[k] = v
        if self.normalization and k.startswith("critic_obs_normalizer"):
          filtered_state_dict[k] = v

    self.load_state_dict(filtered_state_dict, strict=False)
    print(f"[INFO] Loaded weights for {list(filtered_state_dict.keys())[:3]}...")


class WorldModel(nn.Module):
  """World Model predicting state delta and reward given (obs, action, goal)."""

  def __init__(
    self,
    num_states: int,
    num_actions: int,
    num_goals: int,
    num_rewards: int = 1,
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

    # ---- Backbone: (state_hist, action) -> z ----
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
    dyn_layers += [nn.Linear(curr, num_states)]
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
    state: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    goal: torch.Tensor,
    detach_backbone: bool = False,
  ) -> torch.Tensor:
    """
    Returns: Tensor[..., num_states + num_rewards + 1] = [delta_state, reward, termination_logit]
    """
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

    # Reward head and Termination head conditioned on goal and action
    z_in = z.detach() if detach_backbone else z
    aux_input = torch.cat([z_in, action_norm, last_action_norm, goal_norm], dim=-1)
    reward = self.rew_head(aux_input)
    termination = self.term_head(aux_input)

    return torch.cat([delta, reward, termination], dim=-1)
