from dataclasses import dataclass

import torch
import torch.nn as nn
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_mul,
)


@dataclass(frozen=True)
class YawAlignedStateLayout:
  num_joints: int
  z: slice = slice(0, 1)
  g_b: slice = slice(1, 4)
  v_h: slice = slice(4, 7)
  w_h: slice = slice(7, 10)

  @property
  def q(self) -> slice:
    return slice(10, 10 + self.num_joints)

  @property
  def dq(self) -> slice:
    return slice(10 + self.num_joints, 10 + 2 * self.num_joints)

  @property
  def dim(self) -> int:
    return 10 + 2 * self.num_joints


def get_yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
  R = matrix_from_quat(quat)
  f_w = R[..., :, 0]
  return torch.atan2(f_w[..., 1], f_w[..., 0])


def quat_from_two_vectors(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
  v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
  v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)
  dot = (v1 * v2).sum(dim=-1, keepdim=True)
  w = 1 + dot
  xyz = torch.linalg.cross(v1, v2, dim=-1)
  q = torch.cat([w, xyz], dim=-1)
  q_norm = torch.norm(q, dim=-1, keepdim=True)
  return q / (q_norm + 1e-8)


def compute_relative_yaw_obs(
  curr_yaw: torch.Tensor, target_yaw: torch.Tensor
) -> torch.Tensor:
  diff = target_yaw - curr_yaw
  return torch.cat([torch.sin(diff), torch.cos(diff)], dim=-1)


def compute_yaw_aligned_state(
  full_state: torch.Tensor, layout: YawAlignedStateLayout
) -> torch.Tensor:
  """Converts raw state to yaw-aligned representation."""
  # full_state: [root_pos(3), root_quat(4), q(N), root_lin_vel(3), root_ang_vel(3), dq(N)]
  num_joints = layout.num_joints
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


def apply_yaw_correction(
  yaw_state_next: torch.Tensor, dt: float, layout: YawAlignedStateLayout
) -> torch.Tensor:
  """Re-canonicalize yaw for the next step prediction."""
  # Indices: z(1), g_b(3), v_h(3), w_h(3), q(N), dq(N)

  v_h = yaw_state_next[..., layout.v_h]
  w_h = yaw_state_next[..., layout.w_h]

  dpsi = w_h[..., 2] * dt
  c = torch.cos(dpsi)
  s = torch.sin(dpsi)

  vx, vy, vz = v_h[..., 0], v_h[..., 1], v_h[..., 2]
  wx, wy, wz = w_h[..., 0], w_h[..., 1], w_h[..., 2]

  yaw_state_new = yaw_state_next.clone()

  v_start = layout.v_h.start
  w_start = layout.w_h.start

  yaw_state_new[..., v_start] = c * vx + s * vy
  yaw_state_new[..., v_start + 1] = -s * vx + c * vy
  yaw_state_new[..., w_start] = c * wx + s * wy
  yaw_state_new[..., w_start + 1] = -s * wx + c * wy

  return yaw_state_new


def construct_policy_input_from_yaw_state(
  yaw_state: torch.Tensor,
  goal: torch.Tensor,
) -> torch.Tensor:
  """Constructs the policy input from yaw-aligned state and goal.

  Args:
    yaw_state: Yaw-aligned state tensor [..., yaw_state_dim]
    goal: Goal tensor [..., goal_dim]

  Returns:
    Policy input tensor [..., policy_input_dim]
  """
  return torch.cat([yaw_state, goal], dim=-1)


def construct_policy_input(
  full_state: torch.Tensor,
  goal: torch.Tensor,
  layout: YawAlignedStateLayout,
) -> torch.Tensor:
  """Constructs the policy input from full state and goal.

  Args:
    full_state: Full state tensor [..., full_state_dim]
    goal: Goal tensor [..., goal_dim]
    layout: YawAlignedStateLayout

  Returns:
    Policy input tensor [..., policy_input_dim]
  """
  yaw_state = compute_yaw_aligned_state(full_state, layout)
  return construct_policy_input_from_yaw_state(yaw_state, goal)


def integrate_global_state(
  curr_global_state: torch.Tensor,
  next_yaw_state: torch.Tensor,
  dt: float,
  layout: YawAlignedStateLayout,
) -> torch.Tensor:
  """Integrates the global state using angular velocity as a prior and projected gravity
  as a hard constraint (Geodesic Correction).

  Args:
    curr_global_state: Current global state [..., dim]
      Expected layout: [root_pos(3), root_quat(4), q(N), root_lin_vel(3), root_ang_vel(3), dq(N)]
    next_yaw_state: Next yaw-aligned state prediction [..., yaw_state_dim]
    dt: Time step
    layout: YawAlignedStateLayout

  Returns:
    Next global state [..., dim]
  """

  num_joints = layout.num_joints

  # 1. Unpack Predictions
  # ---------------------
  g_b = next_yaw_state[..., layout.g_b]
  v_h = next_yaw_state[..., layout.v_h]
  w_h = next_yaw_state[..., layout.w_h]

  # 2. Compute Global Velocities (v_global, w_global)
  # -------------------------------------------------

  root_quat = curr_global_state[..., 3:7]
  R_wb = matrix_from_quat(root_quat)
  f_w = R_wb[..., :, 0]
  psi = torch.atan2(f_w[..., 1], f_w[..., 0])

  # Semi-implicit yaw update using (t+1) yaw-state yaw rate
  dpsi = w_h[..., 2] * dt
  psi_next = psi + dpsi

  # Rotate velocities to Global Frame
  c = torch.cos(psi_next)
  s = torch.sin(psi_next)
  vx, vy, vz = v_h[..., 0], v_h[..., 1], v_h[..., 2]
  v_global = torch.stack([c * vx - s * vy, s * vx + c * vy, vz], dim=-1)
  wx, wy, wz = w_h[..., 0], w_h[..., 1], w_h[..., 2]
  w_global = torch.stack([c * wx - s * wy, s * wx + c * wy, wz], dim=-1)

  # 3. Integrate Orientation (The Prior)
  # -----------------------------------------
  angle = torch.norm(w_global, dim=-1, keepdim=True) * dt
  axis = w_global / (torch.norm(w_global, dim=-1, keepdim=True) + 1e-8)
  ca = torch.cos(angle / 2)
  sa = torch.sin(angle / 2)
  delta_q = torch.cat([ca, sa * axis], dim=-1)
  q_integrated = quat_mul(delta_q, root_quat)
  q_integrated = q_integrated / torch.norm(q_integrated, dim=-1, keepdim=True)

  # 4. Apply Gravity Constraint (The Correction)
  # --------------------------------------------
  # We want R_final * g_b = world_down.
  # Currently, our integrated estimate predicts: g_est = R_integrated * g_b

  # Map the predicted body gravity (g_b) into world frame using integrated q
  # (Rotate vector g_b by quaternion q_integrated)
  g_est = quat_apply(q_integrated, g_b)

  world_down = torch.tensor([0.0, 0.0, -1.0], device=next_yaw_state.device).expand_as(
    g_b
  )

  # Find the shortest rotation to snap g_est to world_down
  # This minimizes the angular change while satisfying the constraint.
  q_corr = quat_from_two_vectors(g_est, world_down)

  # Apply correction: q_final = q_corr * q_integrated
  q_final = quat_mul(q_corr, q_integrated)
  q_final = q_final / torch.norm(q_final, dim=-1, keepdim=True)

  # 5. Assemble Next State
  # ----------------------
  next_global_state = curr_global_state.clone()

  # Position
  next_global_state[..., 0:3] += v_global * dt
  next_global_state[..., 2] = next_yaw_state[..., layout.z].squeeze(-1)

  # Orientation
  next_global_state[..., 3:7] = q_final

  # Joints & Velocities
  idx_q = 7
  idx_lin = 7 + num_joints
  idx_ang = idx_lin + 3
  idx_dq = idx_ang + 3

  next_global_state[..., idx_q : idx_q + num_joints] = next_yaw_state[..., layout.q]
  next_global_state[..., idx_dq:] = next_yaw_state[..., layout.dq]
  next_global_state[..., idx_lin : idx_lin + 3] = v_global
  next_global_state[..., idx_ang : idx_ang + 3] = w_global

  return next_global_state


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
    backbone_hidden_dims: list[int] = [2048, 2048, 2048, 2048, 2048],
    dyn_hidden_dims: list[int] = [],
    aux_hidden_dims: list[int] = [512, 256, 128],
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
