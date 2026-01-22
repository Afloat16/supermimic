from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_mul,
)

from supermimic.utils.math import quat_from_two_vectors


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

  vx, vy = v_h[..., 0], v_h[..., 1]
  wx, wy = w_h[..., 0], w_h[..., 1]

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
