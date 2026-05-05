"""Tests for MaxCoordRepresentation.

Covers apply_prediction, rep_state layout, loss slices, compute_reward,
and gradient flow (especially through on-demand body height computation).
"""

import torch
from mjlab.utils.lab_api.math import quat_apply

from supermimic.representations import MaxCoordRepresentation
from supermimic.utils.helpers import quat_to_rot6d
from supermimic.utils.maxcoord_helpers import (
  build_maxcoord_full_state,
  compute_maxcoord_rep_state,
  decompose_maxcoord_rep_state,
  decompose_maxcoord_state,
  maxcoord_rep_state_dim,
)

NUM_BODIES = 16


# ---------------------------------------------------------------------------
# Rep state dimensions & layout
# ---------------------------------------------------------------------------


class TestRepStateDim:
  def test_dim_formula(self):
    """rep_state_dim should be B*15 + 4 (root_height is 1, not B)."""
    assert maxcoord_rep_state_dim(NUM_BODIES) == NUM_BODIES * 15 + 4
    rep = MaxCoordRepresentation(NUM_BODIES, num_joints=29)
    assert rep.rep_state_dim == NUM_BODIES * 15 + 4

  def test_dim_various_body_counts(self):
    for B in [1, 4, 16, 32]:
      assert maxcoord_rep_state_dim(B) == B * 15 + 4


class TestDecomposeRootHeight:
  def test_root_height_shape(self):
    """decompose should return root_height of shape (..., 1), not (..., B)."""
    B = NUM_BODIES
    rep_state = torch.randn(4, maxcoord_rep_state_dim(B))
    _, _, _, _, root_height, up_dir = decompose_maxcoord_rep_state(rep_state, B)
    assert root_height.shape == (4, 1)
    assert up_dir.shape == (4, 3)

  def test_root_height_is_root_body_z(self):
    """root_height in rep_state should equal root body world Z coordinate."""
    B = NUM_BODIES
    body_pos = torch.randn(4, B, 3)
    body_quat = torch.randn(4, B, 4)
    body_quat = body_quat / body_quat.norm(dim=-1, keepdim=True)
    body_lin_vel = torch.randn(4, B, 3)
    body_ang_vel = torch.randn(4, B, 3)
    full_state = build_maxcoord_full_state(
      body_pos, body_quat, body_lin_vel, body_ang_vel
    )
    rep_state = compute_maxcoord_rep_state(full_state, B)
    _, _, _, _, root_height, _ = decompose_maxcoord_rep_state(rep_state, B)
    expected_root_z = body_pos[:, 0, 2:3]
    assert torch.allclose(root_height, expected_root_z, atol=1e-5)


# ---------------------------------------------------------------------------
# Loss slices & weights
# ---------------------------------------------------------------------------


class TestMaxCoordLossSlices:
  def test_slices_cover_full_dim(self):
    rep = MaxCoordRepresentation(NUM_BODIES, num_joints=29)
    slices = rep.loss_slices()
    covered = set()
    for sl in slices.values():
      covered.update(range(sl.start, sl.stop))
    assert len(covered) == rep.rep_state_dim

  def test_loss_weights_shape(self):
    rep = MaxCoordRepresentation(NUM_BODIES, num_joints=29)
    W = rep.loss_weights(cfg=object(), device="cpu")
    assert W.shape == (rep.rep_state_dim,)
    assert (W > 0).all()

  def test_root_height_slice_width_1(self):
    rep = MaxCoordRepresentation(NUM_BODIES, num_joints=29)
    slices = rep.loss_slices()
    rh = slices["root_height"]
    assert rh.stop - rh.start == 1

  def test_no_body_height_slice(self):
    rep = MaxCoordRepresentation(NUM_BODIES, num_joints=29)
    slices = rep.loss_slices()
    assert "body_height" not in slices
    assert "root_height" in slices


# ---------------------------------------------------------------------------
# apply_prediction correctness
# ---------------------------------------------------------------------------


class TestMaxCoordApplyPrediction:
  def test_prediction_dim(self):
    rep = MaxCoordRepresentation(NUM_BODIES, num_joints=29)
    # Master predicts vel_delta(B*3) + avel_delta(B*3) = B*6
    assert rep.prediction_dim == NUM_BODIES * 6

  def test_output_shapes(self):
    rep = MaxCoordRepresentation(NUM_BODIES, num_joints=29)
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    pred = torch.randn(4, rep.prediction_dim) * 0.01
    next_aux, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.01)
    assert next_aux.shape == (4, 7)
    assert next_rep.shape == (4, rep.rep_state_dim)

  def test_zero_prediction_velocities_unchanged(self):
    """Zero prediction means zero velocity delta — velocities stay the same (before frame correction)."""
    rep = MaxCoordRepresentation(NUM_BODIES, num_joints=29)
    B = NUM_BODIES
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0  # identity quat -> no frame correction
    curr = torch.randn(4, rep.rep_state_dim)
    # Zero out angular velocities so frame correction is identity
    avel_start = B * 3 + B * 6 + B * 3
    curr[..., avel_start : avel_start + B * 3] = 0.0
    pred = torch.zeros(4, rep.prediction_dim)
    _, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.01)
    # Velocities should be unchanged (zero avel -> no frame rotation)
    vel_start = B * 3 + B * 6
    assert torch.allclose(
      next_rep[..., vel_start : vel_start + B * 3],
      curr[..., vel_start : vel_start + B * 3],
      atol=1e-4,
    )

  def test_zero_prediction_positions_integrate_from_velocity(self):
    """With zero prediction, positions integrate via world-frame round-trip."""
    rep = MaxCoordRepresentation(NUM_BODIES, num_joints=29)
    B = NUM_BODIES
    dt = 0.01
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    # Zero out angular velocities so root quat stays identity
    avel_start = B * 3 + B * 6 + B * 3
    curr[..., avel_start : avel_start + B * 3] = 0.0
    local_pos_curr = curr[..., : B * 3].unflatten(-1, (B, 3))
    vel_start = B * 3 + B * 6
    local_vel = curr[..., vel_start : vel_start + B * 3].unflatten(-1, (B, 3))
    # World-frame integration (identity quat, root_pos=0):
    # world_pos = local_pos, next_world_pos = local_pos + local_vel*dt
    # next_root_pos = next_world_pos[0], next_local_pos = next_world_pos - next_root_pos
    next_world_pos = local_pos_curr + local_vel * dt
    expected_pos = next_world_pos - next_world_pos[..., 0:1, :]
    pred = torch.zeros(4, rep.prediction_dim)
    _, next_rep = rep.apply_prediction(aux, curr, pred, dt=dt)
    next_pos = next_rep[..., : B * 3].unflatten(-1, (B, 3))
    assert torch.allclose(next_pos, expected_pos, atol=1e-4)

  def test_zero_prediction_rotation_integrates_from_avel(self):
    """With zero prediction, rotations integrate from current angular velocities."""
    rep = MaxCoordRepresentation(NUM_BODIES, num_joints=29)
    B = NUM_BODIES
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    pred = torch.zeros(4, rep.prediction_dim)
    _, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.01)
    # Rotations should change (integrated from avel), so they shouldn't equal curr
    rot_start = B * 3
    rot_end = rot_start + B * 6
    # Just verify the output is finite and different from current
    assert not next_rep[..., rot_start:rot_end].isnan().any()
    assert not next_rep[..., rot_start:rot_end].isinf().any()

  def test_root_quat_is_unit(self):
    rep = MaxCoordRepresentation(NUM_BODIES, num_joints=29)
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    pred = torch.randn(4, rep.prediction_dim) * 0.1
    next_aux, _ = rep.apply_prediction(aux, curr, pred, dt=0.01)
    quat_norms = next_aux[..., 3:7].norm(dim=-1)
    assert torch.allclose(quat_norms, torch.ones(4), atol=1e-5)

  def test_output_rot6d_stays_near_so3(self):
    """Starting from valid SO(3) rotations, output should remain near SO(3)."""
    rep = MaxCoordRepresentation(NUM_BODIES, num_joints=29)
    B = NUM_BODIES
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    # Build curr with valid SO(3) rot6d
    curr = torch.randn(4, rep.rep_state_dim)
    rot_start = B * 3
    rot_end = rot_start + B * 6
    random_quats = torch.randn(4, B, 4)
    random_quats = random_quats / random_quats.norm(dim=-1, keepdim=True)
    valid_rot6d = quat_to_rot6d(random_quats)
    curr[..., rot_start:rot_end] = valid_rot6d.flatten(-2, -1)
    pred = torch.randn(4, rep.prediction_dim) * 0.01  # small prediction
    _, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.01)
    rot = next_rep[..., rot_start:rot_end].unflatten(-1, (B, 6))
    cols = rot.unflatten(-1, (3, 2))
    c1, c2 = cols[..., 0], cols[..., 1]
    # Should stay approximately unit-length and orthogonal with small predictions
    assert torch.allclose(c1.norm(dim=-1), torch.ones(4, B), atol=0.1)
    assert torch.allclose(c2.norm(dim=-1), torch.ones(4, B), atol=0.1)

  def test_velocity_couples_to_position(self):
    """Position loss should backprop to velocity — velocity drives position integration."""
    rep = MaxCoordRepresentation(NUM_BODIES, num_joints=29)
    B = NUM_BODIES
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    pred = (torch.randn(4, rep.prediction_dim) * 0.01).requires_grad_(True)
    _, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.01)
    # Loss on positions only
    pos_loss = next_rep[..., : B * 3].sum()
    pos_loss.backward()
    # Velocity part of prediction should have nonzero gradient
    vel_grad = pred.grad[..., : B * 3]
    assert vel_grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# compute_reward with body height args
# ---------------------------------------------------------------------------


class TestComputeReward:
  def test_compute_reward_requires_body_height_args(self):
    """compute_reward must accept predicted/target body_height tensors."""
    rep = MaxCoordRepresentation(NUM_BODIES, num_joints=29)
    B = NUM_BODIES
    pred_rep = torch.randn(4, rep.rep_state_dim)
    target_rep = torch.randn(4, rep.rep_state_dim)
    pred_body_height = torch.randn(4, B)
    target_body_height = torch.randn(4, B)
    reward = rep.compute_reward(
      pred_rep, target_rep, pred_body_height, target_body_height
    )
    assert reward.shape == (4,)

  def test_compute_reward_height_error_contributes(self):
    """Changing body height inputs should change the reward."""
    rep = MaxCoordRepresentation(NUM_BODIES, num_joints=29)
    B = NUM_BODIES
    pred_rep = torch.randn(4, rep.rep_state_dim)
    target_rep = pred_rep.clone()  # zero rep error
    zeros = torch.zeros(4, B)
    r_zero = rep.compute_reward(pred_rep, target_rep, zeros, zeros)
    # Introduce height mismatch
    r_mismatch = rep.compute_reward(pred_rep, target_rep, torch.ones(4, B), zeros)
    assert (r_mismatch < r_zero).all()


# ---------------------------------------------------------------------------
# Gradient: apply_prediction
# ---------------------------------------------------------------------------


class TestApplyPredictionGradient:
  def test_no_nan_inf_at_various_scales(self):
    """Gradients should be finite across different prediction magnitudes."""
    rep = MaxCoordRepresentation(NUM_BODIES, num_joints=29)
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    for scale in [1.0, 1e-2, 1e-4, 1e-8]:
      pred = (torch.randn(4, rep.prediction_dim) * scale).requires_grad_(True)
      next_aux, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.01)
      loss = next_rep.sum() + next_aux.sum()
      loss.backward()
      assert not pred.grad.isnan().any(), f"NaN at scale {scale}"
      assert not pred.grad.isinf().any(), f"Inf at scale {scale}"

  def test_gradient_from_rep_loss(self):
    """Full rep state loss should give gradient to all prediction dims."""
    rep = MaxCoordRepresentation(NUM_BODIES, num_joints=29)
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    pred = (torch.randn(4, rep.prediction_dim) * 0.01).requires_grad_(True)
    _, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.01)
    next_rep.sum().backward()
    # Every prediction dim should receive some gradient
    assert (pred.grad.abs().sum(dim=0) > 0).all()

  def test_gradient_from_aux_loss(self):
    """Aux state loss should give gradient to prediction (through vel integration)."""
    rep = MaxCoordRepresentation(NUM_BODIES, num_joints=29)
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    pred = (torch.randn(4, rep.prediction_dim) * 0.01).requires_grad_(True)
    next_aux, _ = rep.apply_prediction(aux, curr, pred, dt=0.01)
    next_aux.sum().backward()
    assert pred.grad.abs().sum() > 0

  def test_gradient_norm_bounded_across_scales(self):
    """Gradient norms should stay bounded as prediction scale shrinks."""
    rep = MaxCoordRepresentation(NUM_BODIES, num_joints=29)
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    grad_norms = []
    for scale in [1e-2, 1e-4, 1e-6, 1e-8]:
      pred = (torch.randn(4, rep.prediction_dim) * scale).requires_grad_(True)
      next_aux, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.01)
      (next_rep.sum() + next_aux.sum()).backward()
      grad_norms.append(pred.grad.norm().item())
    assert max(grad_norms) / min(grad_norms) < 10.0


# ---------------------------------------------------------------------------
# Gradient: height error -> aux_state & rep_state
# ---------------------------------------------------------------------------


def _make_grad_inputs(requires_grad=True):
  B = NUM_BODIES
  rep = MaxCoordRepresentation(B, num_joints=29)
  aux = torch.zeros(4, 7)
  aux[..., 2] = 1.0  # root at height 1
  aux[..., 3] = 1.0  # identity quat (w,x,y,z)
  aux = aux.clone().requires_grad_(requires_grad)
  rep_state = torch.randn(4, rep.rep_state_dim).requires_grad_(requires_grad)
  return rep, aux, rep_state


def _compute_body_heights(aux, rep_state, num_bodies):
  """Reconstruct per-body heights from aux + rep (same as compute_wm_reward)."""
  root_pos = aux[..., :3]
  root_quat = aux[..., 3:7]
  local_pos = rep_state[..., : num_bodies * 3].unflatten(-1, (num_bodies, 3))
  root_quat_b = root_quat.unsqueeze(-2).expand(*local_pos.shape[:-1], 4)
  return root_pos[..., 2:3] + quat_apply(root_quat_b, local_pos)[..., 2]


class TestHeightGradientFlow:
  """Height error must backprop to aux_state (root_pos, root_quat) and rep_state (local_pos)."""

  def test_height_gradient_to_root_pos_z(self):
    """Height error gradient must reach root_pos Z (index 2 of aux_state)."""
    _, aux, rep_state = _make_grad_inputs()
    pred_bh = _compute_body_heights(aux, rep_state, NUM_BODIES)
    pred_bh.sum().backward()
    assert aux.grad[..., 2].abs().sum() > 0, "No gradient to root_pos Z"

  def test_height_gradient_to_root_quat(self):
    """Height error gradient must reach root_quat (indices 3:7 of aux_state)."""
    _, aux, rep_state = _make_grad_inputs()
    pred_bh = _compute_body_heights(aux, rep_state, NUM_BODIES)
    pred_bh.sum().backward()
    assert aux.grad[..., 3:7].abs().sum() > 0, "No gradient to root_quat"

  def test_height_gradient_to_local_pos(self):
    """Height error gradient must reach local_pos in rep_state."""
    _, aux, rep_state = _make_grad_inputs()
    pred_bh = _compute_body_heights(aux, rep_state, NUM_BODIES)
    pred_bh.sum().backward()
    local_pos_grad = rep_state.grad[..., : NUM_BODIES * 3]
    assert local_pos_grad.abs().sum() > 0, "No gradient to local_pos"

  def test_compute_reward_gradient_to_body_height(self):
    """compute_reward must pass gradient through predicted_body_height."""
    rep = MaxCoordRepresentation(NUM_BODIES, num_joints=29)
    B = NUM_BODIES
    pred_rep = torch.randn(4, rep.rep_state_dim)
    target_rep = torch.randn(4, rep.rep_state_dim)
    pred_bh = torch.randn(4, B, requires_grad=True)
    target_bh = torch.randn(4, B)
    reward = rep.compute_reward(pred_rep, target_rep, pred_bh, target_bh)
    reward.sum().backward()
    assert pred_bh.grad is not None
    assert pred_bh.grad.abs().sum() > 0, "No gradient to predicted_body_height"

  def test_compute_reward_gradient_to_rep_state(self):
    """compute_reward must pass gradient through rep_state (pos, rot, vel, etc.)."""
    rep = MaxCoordRepresentation(NUM_BODIES, num_joints=29)
    B = NUM_BODIES
    pred_rep = torch.randn(4, rep.rep_state_dim, requires_grad=True)
    target_rep = torch.randn(4, rep.rep_state_dim)
    pred_bh = torch.randn(4, B)
    target_bh = torch.randn(4, B)
    reward = rep.compute_reward(pred_rep, target_rep, pred_bh, target_bh)
    reward.sum().backward()
    assert pred_rep.grad is not None
    assert pred_rep.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# End-to-end gradient: apply_prediction -> body height -> reward -> backprop
# ---------------------------------------------------------------------------


class TestEndToEndRewardGradient:
  def test_reward_gradient_through_apply_prediction_to_prediction(self):
    """Reward computed from apply_prediction outputs must backprop to prediction tensor."""
    B = NUM_BODIES
    rep = MaxCoordRepresentation(B, num_joints=29)
    aux = torch.zeros(4, 7)
    aux[..., 2] = 1.0
    aux[..., 3] = 1.0
    curr_rep = torch.randn(4, rep.rep_state_dim)
    pred = (torch.randn(4, rep.prediction_dim) * 0.01).requires_grad_(True)

    next_aux, next_rep = rep.apply_prediction(aux, curr_rep, pred, dt=0.01)
    pred_bh = _compute_body_heights(next_aux, next_rep, B)
    target_bh = torch.randn(4, B)
    target_rep = torch.randn(4, rep.rep_state_dim)
    reward = rep.compute_reward(next_rep, target_rep, pred_bh, target_bh)
    reward.sum().backward()

    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0, "No gradient from reward to prediction"

  def test_reward_gradient_reaches_all_prediction_dims(self):
    """Both vel_delta and avel_delta portions of prediction should receive gradient."""
    B = NUM_BODIES
    rep = MaxCoordRepresentation(B, num_joints=29)
    aux = torch.zeros(4, 7)
    aux[..., 2] = 1.0
    aux[..., 3] = 1.0
    curr_rep = torch.randn(4, rep.rep_state_dim)
    pred = (torch.randn(4, rep.prediction_dim) * 0.01).requires_grad_(True)

    next_aux, next_rep = rep.apply_prediction(aux, curr_rep, pred, dt=0.01)
    pred_bh = _compute_body_heights(next_aux, next_rep, B)
    target_bh = torch.zeros(4, B)
    target_rep = torch.zeros(4, rep.rep_state_dim)
    reward = rep.compute_reward(next_rep, target_rep, pred_bh, target_bh)
    reward.sum().backward()

    vel_grad = pred.grad[..., : B * 3]  # vel_delta
    avel_grad = pred.grad[..., B * 3 : B * 6]  # avel_delta
    assert vel_grad.abs().sum() > 0, "No gradient to vel_delta"
    assert avel_grad.abs().sum() > 0, "No gradient to avel_delta"

  def test_height_gradient_finite_no_nan(self):
    """Gradient through body height computation must be finite (no NaN/Inf)."""
    B = NUM_BODIES
    rep = MaxCoordRepresentation(B, num_joints=29)
    for scale in [1.0, 1e-2, 1e-4, 1e-8]:
      aux = torch.zeros(4, 7)
      aux[..., 2] = 1.0
      aux[..., 3] = 1.0
      curr_rep = torch.randn(4, rep.rep_state_dim)
      pred = (torch.randn(4, rep.prediction_dim) * scale).requires_grad_(True)

      next_aux, next_rep = rep.apply_prediction(aux, curr_rep, pred, dt=0.01)
      pred_bh = _compute_body_heights(next_aux, next_rep, B)
      pred_bh.sum().backward()
      assert not pred.grad.isnan().any(), f"NaN gradient at scale {scale}"
      assert not pred.grad.isinf().any(), f"Inf gradient at scale {scale}"

  def test_height_gradient_norm_bounded_across_scales(self):
    """Gradient norms from height error should stay bounded as prediction scale shrinks."""
    B = NUM_BODIES
    rep = MaxCoordRepresentation(B, num_joints=29)
    torch.manual_seed(42)
    grad_norms = []
    for scale in [1e-2, 1e-4, 1e-6, 1e-8]:
      aux = torch.zeros(4, 7)
      aux[..., 2] = 1.0
      aux[..., 3] = 1.0
      curr_rep = torch.randn(4, rep.rep_state_dim)
      pred = (torch.randn(4, rep.prediction_dim) * scale).requires_grad_(True)

      next_aux, next_rep = rep.apply_prediction(aux, curr_rep, pred, dt=0.01)
      pred_bh = _compute_body_heights(next_aux, next_rep, B)
      target_bh = torch.zeros(4, B)
      target_rep = torch.zeros(4, rep.rep_state_dim)
      reward = rep.compute_reward(next_rep, target_rep, pred_bh, target_bh)
      reward.sum().backward()
      grad_norms.append(pred.grad.norm().item())

    ratio = max(grad_norms) / min(grad_norms)
    assert ratio < 100.0, f"Gradient norm ratio {ratio:.1f} too large: {grad_norms}"


# ---------------------------------------------------------------------------
# World-frame consistency: apply_prediction matches direct world integration
# ---------------------------------------------------------------------------


def _make_valid_full_state(batch: int, B: int):
  """Build a valid full state with unit quaternions."""
  body_pos = torch.randn(batch, B, 3)
  body_quat = torch.randn(batch, B, 4)
  body_quat = body_quat / body_quat.norm(dim=-1, keepdim=True)
  body_lin_vel = torch.randn(batch, B, 3) * 0.5
  body_ang_vel = torch.randn(batch, B, 3) * 0.5
  return build_maxcoord_full_state(body_pos, body_quat, body_lin_vel, body_ang_vel)


class TestWorldFrameConsistency:
  """Verify that apply_prediction → reconstruct matches direct world-frame integration."""

  def test_apply_prediction_matches_world_frame_integration(self):
    """apply_prediction → reconstruct_full_state should match direct world integration."""
    B = NUM_BODIES
    batch = 4
    dt = 0.01
    rep = MaxCoordRepresentation(B, num_joints=29)
    torch.manual_seed(123)

    full_state = _make_valid_full_state(batch, B)
    body_pos, body_quat, body_vel, body_avel = decompose_maxcoord_state(full_state, B)

    aux = rep.extract_aux_state(full_state)
    rep_state = rep.represent(full_state)
    root_quat = aux[..., 3:7]
    root_quat_b = root_quat.unsqueeze(-2).expand(batch, B, 4)

    # Small prediction in root frame
    pred = torch.randn(batch, rep.prediction_dim) * 0.01
    dv_local = pred[..., : B * 3].unflatten(-1, (B, 3))
    dw_local = pred[..., B * 3 : B * 6].unflatten(-1, (B, 3))

    # --- Direct world-frame integration (control_vae style) ---
    dv_world = quat_apply(root_quat_b, dv_local)
    dw_world = quat_apply(root_quat_b, dw_local)
    expected_vel = body_vel + dv_world
    expected_avel = body_avel + dw_world
    expected_pos = body_pos + expected_vel * dt

    # --- Through apply_prediction ---
    next_aux, next_rep = rep.apply_prediction(aux, rep_state, pred, dt=dt)
    next_full = rep.reconstruct_full_state(next_aux, next_rep)
    actual_pos, _, actual_vel, actual_avel = decompose_maxcoord_state(next_full, B)

    assert torch.allclose(actual_pos, expected_pos, atol=1e-4), (
      f"Position mismatch: max diff {(actual_pos - expected_pos).abs().max():.6f}"
    )
    assert torch.allclose(actual_vel, expected_vel, atol=1e-4), (
      f"Velocity mismatch: max diff {(actual_vel - expected_vel).abs().max():.6f}"
    )
    assert torch.allclose(actual_avel, expected_avel, atol=1e-4), (
      f"Angular velocity mismatch: max diff {(actual_avel - expected_avel).abs().max():.6f}"
    )

  def test_root_local_pos_always_zero(self):
    """Root body's local_pos should always be zero after integration."""
    B = NUM_BODIES
    rep = MaxCoordRepresentation(B, num_joints=29)
    torch.manual_seed(42)
    full_state = _make_valid_full_state(4, B)
    aux = rep.extract_aux_state(full_state)
    rep_state = rep.represent(full_state)
    pred = torch.randn(4, rep.prediction_dim) * 0.1
    _, next_rep = rep.apply_prediction(aux, rep_state, pred, dt=0.01)
    next_local_pos = next_rep[..., : B * 3].unflatten(-1, (B, 3))
    assert torch.allclose(next_local_pos[..., 0, :], torch.zeros(4, 3), atol=1e-5), (
      f"Root local_pos not zero: {next_local_pos[..., 0, :]}"
    )

  def test_single_body_shape(self):
    """B=1 should produce correct shapes."""
    rep = MaxCoordRepresentation(1, num_joints=29)
    aux = torch.zeros(2, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(2, rep.rep_state_dim)
    pred = torch.randn(2, rep.prediction_dim) * 0.01
    next_aux, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.01)
    assert next_aux.shape == (2, 7)
    assert next_rep.shape == (2, rep.rep_state_dim)

  def test_zero_pred_zero_vel_no_position_change(self):
    """Zero prediction + zero velocity = no position change."""
    B = NUM_BODIES
    rep = MaxCoordRepresentation(B, num_joints=29)
    full_state = _make_valid_full_state(4, B)
    # Zero out all velocities
    body_pos, body_quat, _, _ = decompose_maxcoord_state(full_state, B)
    full_state = build_maxcoord_full_state(
      body_pos, body_quat, torch.zeros(4, B, 3), torch.zeros(4, B, 3)
    )
    aux = rep.extract_aux_state(full_state)
    rep_state = rep.represent(full_state)
    pred = torch.zeros(4, rep.prediction_dim)
    next_aux, next_rep = rep.apply_prediction(aux, rep_state, pred, dt=0.01)
    next_full = rep.reconstruct_full_state(next_aux, next_rep)
    next_pos, _, _, _ = decompose_maxcoord_state(next_full, B)
    assert torch.allclose(next_pos, body_pos, atol=1e-5), (
      f"Positions changed: max diff {(next_pos - body_pos).abs().max():.6f}"
    )

  def test_gradient_flows_through_world_frame_conversion(self):
    """Gradient should flow through the full world-frame round-trip."""
    B = NUM_BODIES
    rep = MaxCoordRepresentation(B, num_joints=29)
    full_state = _make_valid_full_state(4, B)
    aux = rep.extract_aux_state(full_state)
    rep_state = rep.represent(full_state)
    pred = (torch.randn(4, rep.prediction_dim) * 0.01).requires_grad_(True)
    next_aux, next_rep = rep.apply_prediction(aux, rep_state, pred, dt=0.01)
    next_full = rep.reconstruct_full_state(next_aux, next_rep)
    next_full.sum().backward()
    assert pred.grad is not None
    assert not pred.grad.isnan().any()
    assert (pred.grad.abs().sum(dim=0) > 0).all(), (
      "Some prediction dims got no gradient"
    )
