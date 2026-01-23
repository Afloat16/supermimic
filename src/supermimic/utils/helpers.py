from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_mul,
)

from supermimic.utils.math import quat_from_two_vectors


@dataclass(frozen=True)
class BodyFrameStateLayout:
  num_joints: int
  z: slice = slice(0, 1)
  g_b: slice = slice(1, 4)
  v_b: slice = slice(4, 7)
  w_b: slice = slice(7, 10)

  @property
  def q(self) -> slice:
    return slice(10, 10 + self.num_joints)

  @property
  def dq(self) -> slice:
    return slice(10 + self.num_joints, 10 + 2 * self.num_joints)

  @property
  def dim(self) -> int:
    return 10 + 2 * self.num_joints


def compute_body_frame_state(
  full_state: torch.Tensor, layout: BodyFrameStateLayout
) -> torch.Tensor:
  """Converts raw state to body-frame representation.

  Uses root body frame for velocities (no heading alignment).
  This avoids all singularities since we're just rotating to body frame.
  """
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

  z = root_pos[..., 2:3]
  g_b = -1.0 * R_wb[..., 2, :]  # R_wb^T @ [0,0,-1]

  # Rotate velocities to body frame: v_b = R_wb^T @ v_w
  # R_wb^T is the transpose, which rotates world to body
  v_b = torch.einsum("...ij,...j->...i", R_wb.transpose(-2, -1), root_lin_vel)
  w_b = torch.einsum("...ij,...j->...i", R_wb.transpose(-2, -1), root_ang_vel)

  return torch.cat([z, g_b, v_b, w_b, q, dq], dim=-1)


def construct_policy_input_from_body_state(
  body_state: torch.Tensor,
  goal: torch.Tensor,
) -> torch.Tensor:
  """Constructs the policy input from body-frame state and goal.

  Args:
    body_state: Body-frame state tensor [..., body_state_dim]
    goal: Goal tensor [..., goal_dim]

  Returns:
    Policy input tensor [..., policy_input_dim]
  """
  return torch.cat([body_state, goal], dim=-1)


def construct_policy_input(
  full_state: torch.Tensor,
  goal: torch.Tensor,
  layout: BodyFrameStateLayout,
) -> torch.Tensor:
  """Constructs the policy input from full state and goal.

  Args:
    full_state: Full state tensor [..., full_state_dim]
    goal: Goal tensor [..., goal_dim]
    layout: BodyFrameStateLayout

  Returns:
    Policy input tensor [..., policy_input_dim]
  """
  body_state = compute_body_frame_state(full_state, layout)
  return construct_policy_input_from_body_state(body_state, goal)


def integrate_global_state(
  curr_global_state: torch.Tensor,
  next_body_state: torch.Tensor,
  dt: float,
  layout: BodyFrameStateLayout,
) -> torch.Tensor:
  """Integrates the global state using angular velocity as a prior and projected gravity
  as a hard constraint (Geodesic Correction).

  Args:
    curr_global_state: Current global state [..., dim]
      Expected layout: [root_pos(3), root_quat(4), q(N), root_lin_vel(3), root_ang_vel(3), dq(N)]
    next_body_state: Next body-frame state prediction [..., body_state_dim]
    dt: Time step
    layout: BodyFrameStateLayout

  Returns:
    Next global state [..., dim]
  """

  num_joints = layout.num_joints

  # 1. Unpack Predictions
  # ---------------------
  g_b = next_body_state[..., layout.g_b]
  v_b = next_body_state[..., layout.v_b]
  w_b = next_body_state[..., layout.w_b]

  # 2. Convert Body-Frame Velocities to Global Frame
  # --------------------------------------------------
  # v_b and w_b are in body frame from the prediction
  # Convert to global: v_w = R_wb @ v_b

  root_quat = curr_global_state[..., 3:7]
  R_wb = matrix_from_quat(root_quat)

  v_b = next_body_state[..., layout.v_b]  # Actually body-frame linear velocity
  w_b = next_body_state[..., layout.w_b]  # Actually body-frame angular velocity

  v_global = torch.einsum("...ij,...j->...i", R_wb, v_b)
  w_global = torch.einsum("...ij,...j->...i", R_wb, w_b)

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

  world_down = torch.tensor([0.0, 0.0, -1.0], device=next_body_state.device).expand_as(
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
  next_global_state[..., 2] = next_body_state[..., layout.z].squeeze(-1)

  # Orientation
  next_global_state[..., 3:7] = q_final

  # Joints & Velocities
  idx_q = 7
  idx_lin = 7 + num_joints
  idx_ang = idx_lin + 3
  idx_dq = idx_ang + 3

  next_global_state[..., idx_q : idx_q + num_joints] = next_body_state[..., layout.q]
  next_global_state[..., idx_dq:] = next_body_state[..., layout.dq]
  next_global_state[..., idx_lin : idx_lin + 3] = v_global
  next_global_state[..., idx_ang : idx_ang + 3] = w_global

  return next_global_state
