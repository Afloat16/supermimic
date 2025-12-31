import torch
import torch.nn as nn
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
)


def get_yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
  R = matrix_from_quat(quat)
  f_w = R[..., :, 0]
  return torch.atan2(f_w[..., 1], f_w[..., 0])


def compute_relative_yaw_obs(
  curr_yaw: torch.Tensor, target_yaw: torch.Tensor
) -> torch.Tensor:
  diff = target_yaw - curr_yaw
  return torch.cat([torch.sin(diff), torch.cos(diff)], dim=-1)


def compute_yaw_aligned_state(full_state: torch.Tensor) -> torch.Tensor:
  """Converts raw state to yaw-aligned representation."""
  # full_state: [root_pos(3), root_quat(4), q(N), root_lin_vel(3), root_ang_vel(3), dq(N)]
  num_joints = (full_state.shape[-1] - 13) // 2

  idx_pos = 0
  idx_quat = 3
  idx_q = 7
  idx_lin = 7 + num_joints
  idx_ang = 7 + num_joints + 3
  idx_dq = 7 + num_joints + 3 + 3

  root_pos = full_state[..., idx_pos : idx_pos + 3]
  root_quat = full_state[..., idx_quat : idx_quat + 4]
  q = full_state[..., idx_q : idx_q + num_joints]
  root_lin_vel = full_state[..., idx_lin : idx_lin + 3]
  root_ang_vel = full_state[..., idx_ang : idx_ang + 3]
  dq = full_state[..., idx_dq : idx_dq + num_joints]

  R_wb = matrix_from_quat(root_quat)
  f_w = R_wb[..., :, 0]
  psi = torch.atan2(f_w[..., 1], f_w[..., 0])

  z = root_pos[..., 2:3]
  g_b = -1.0 * R_wb[..., 2, :]  # R_wb^T @ [0,0,-1]

  # C(psi) = Rz(-psi). Rotate velocity to heading frame
  c = torch.cos(psi)
  s = torch.sin(psi)

  vx, vy, vz = root_lin_vel[..., 0], root_lin_vel[..., 1], root_lin_vel[..., 2]
  v_h = torch.stack([c * vx + s * vy, -s * vx + c * vy, vz], dim=-1)

  wx, wy, wz = root_ang_vel[..., 0], root_ang_vel[..., 1], root_ang_vel[..., 2]
  w_h = torch.stack([c * wx + s * wy, -s * wx + c * wy, wz], dim=-1)

  return torch.cat([z, g_b, v_h, w_h, q, dq], dim=-1)


def apply_yaw_correction(yaw_state_next: torch.Tensor, dt: float) -> torch.Tensor:
  """Re-canonicalize yaw for the next step prediction."""
  # Indices: z(1), g_b(3), v_h(3), w_h(3), q(N), dq(N)
  v_h = yaw_state_next[..., 4:7]
  w_h = yaw_state_next[..., 7:10]

  dpsi = w_h[..., 2] * dt
  c = torch.cos(dpsi)
  s = torch.sin(dpsi)

  vx, vy, vz = v_h[..., 0], v_h[..., 1], v_h[..., 2]
  wx, wy, wz = w_h[..., 0], w_h[..., 1], w_h[..., 2]

  yaw_state_new = yaw_state_next.clone()
  # Rotate vectors by Rz(-dpsi)
  yaw_state_new[..., 4] = c * vx + s * vy
  yaw_state_new[..., 5] = -s * vx + c * vy
  yaw_state_new[..., 7] = c * wx + s * wy
  yaw_state_new[..., 8] = -s * wx + c * wy

  return yaw_state_new


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
    hidden_dims: list[int] = [512, 256, 128],
    normalization: bool = False,
  ):
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
    aux_hidden_dims: list[int] = [512, 256, 128],
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
    for h in aux_hidden_dims:
      rew_layers += [nn.Linear(curr, h), act()]
      curr = h
    rew_layers += [nn.Linear(curr, 1)]
    self.rew_head = nn.Sequential(*rew_layers)

    # ---- Termination head: z + goal -> termination ----
    term_layers = []
    curr = backbone_out_dim + num_goals
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
    goal: torch.Tensor,
    detach_reward_backbone: bool = False,
  ) -> torch.Tensor:
    """
    Returns: Tensor[..., num_obs + 2] = [delta_state, reward, termination_logit]
    """

    # Normalize all inputs together to match the statistics buffers
    full_input = torch.cat([obs, action, goal], dim=-1)
    full_norm = (full_input - self.input_mean) / (self.input_std + 1e-6)

    # Split normalized inputs
    split_idx = self.num_obs + self.num_actions
    backbone_in = full_norm
    goal_norm = full_norm[..., split_idx:]

    z = self.backbone(backbone_in)

    delta = self.dyn_head(z)

    # Reward head and Termination head conditioned on goal
    z_in = z.detach() if detach_reward_backbone else z
    aux_input = torch.cat([z_in, goal_norm], dim=-1)

    reward = self.rew_head(aux_input)
    termination = self.term_head(aux_input)

    return torch.cat([delta, reward, termination], dim=-1)
