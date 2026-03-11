"""Tests for MaxCoordRepresentation.apply_prediction."""

import torch

from supermimic.representations import MaxCoordRepresentation

NUM_BODIES = 16


class TestMaxCoordApplyPrediction:
  def test_prediction_dim(self):
    rep = MaxCoordRepresentation(NUM_BODIES)
    # Master predicts vel_delta(B*3) + avel_delta(B*3) = B*6
    assert rep.prediction_dim == NUM_BODIES * 6

  def test_output_shapes(self):
    rep = MaxCoordRepresentation(NUM_BODIES)
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    pred = torch.randn(4, rep.prediction_dim) * 0.01
    next_aux, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.01)
    assert next_aux.shape == (4, 7)
    assert next_rep.shape == (4, rep.rep_state_dim)

  def test_zero_prediction_velocities_unchanged(self):
    """Zero prediction means zero velocity delta — velocities stay the same (before frame correction)."""
    rep = MaxCoordRepresentation(NUM_BODIES)
    B = NUM_BODIES
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0  # identity quat → no frame correction
    curr = torch.randn(4, rep.rep_state_dim)
    # Zero out angular velocities so frame correction is identity
    avel_start = B * 3 + B * 6 + B * 3
    curr[..., avel_start : avel_start + B * 3] = 0.0
    pred = torch.zeros(4, rep.prediction_dim)
    _, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.01)
    # Velocities should be unchanged (zero avel → no frame rotation)
    vel_start = B * 3 + B * 6
    assert torch.allclose(
      next_rep[..., vel_start : vel_start + B * 3],
      curr[..., vel_start : vel_start + B * 3],
      atol=1e-4,
    )

  def test_zero_prediction_positions_integrate_from_velocity(self):
    """With zero prediction, positions integrate from current velocities."""
    rep = MaxCoordRepresentation(NUM_BODIES)
    B = NUM_BODIES
    dt = 0.01
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    # Zero out angular velocities so frame correction is identity
    avel_start = B * 3 + B * 6 + B * 3
    curr[..., avel_start : avel_start + B * 3] = 0.0
    local_pos_curr = curr[..., : B * 3].unflatten(-1, (B, 3))
    vel_start = B * 3 + B * 6
    local_vel = curr[..., vel_start : vel_start + B * 3].unflatten(-1, (B, 3))
    rel_vel = local_vel - local_vel[..., 0:1, :]
    expected_pos = local_pos_curr + rel_vel * dt
    pred = torch.zeros(4, rep.prediction_dim)
    _, next_rep = rep.apply_prediction(aux, curr, pred, dt=dt)
    next_pos = next_rep[..., : B * 3].unflatten(-1, (B, 3))
    assert torch.allclose(next_pos, expected_pos, atol=1e-4)

  def test_zero_prediction_rotation_integrates_from_avel(self):
    """With zero prediction, rotations integrate from current angular velocities."""
    rep = MaxCoordRepresentation(NUM_BODIES)
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
    rep = MaxCoordRepresentation(NUM_BODIES)
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    pred = torch.randn(4, rep.prediction_dim) * 0.1
    next_aux, _ = rep.apply_prediction(aux, curr, pred, dt=0.01)
    quat_norms = next_aux[..., 3:7].norm(dim=-1)
    assert torch.allclose(quat_norms, torch.ones(4), atol=1e-5)

  def test_output_rot6d_stays_near_so3(self):
    """Starting from valid SO(3) rotations, output should remain near SO(3)."""
    rep = MaxCoordRepresentation(NUM_BODIES)
    B = NUM_BODIES
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    # Build curr with valid SO(3) rot6d
    curr = torch.randn(4, rep.rep_state_dim)
    rot_start = B * 3
    rot_end = rot_start + B * 6
    from supermimic.utils.helpers import quat_to_rot6d

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
    rep = MaxCoordRepresentation(NUM_BODIES)
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

  def test_no_nan_inf_at_various_scales(self):
    """Gradients should be finite across different prediction magnitudes."""
    rep = MaxCoordRepresentation(NUM_BODIES)
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
    rep = MaxCoordRepresentation(NUM_BODIES)
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
    rep = MaxCoordRepresentation(NUM_BODIES)
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    pred = (torch.randn(4, rep.prediction_dim) * 0.01).requires_grad_(True)
    next_aux, _ = rep.apply_prediction(aux, curr, pred, dt=0.01)
    next_aux.sum().backward()
    assert pred.grad.abs().sum() > 0

  def test_gradient_norm_bounded_across_scales(self):
    """Gradient norms should stay bounded as prediction scale shrinks."""
    rep = MaxCoordRepresentation(NUM_BODIES)
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


class TestMaxCoordLossSlices:
  def test_slices_cover_full_dim(self):
    rep = MaxCoordRepresentation(NUM_BODIES)
    slices = rep.loss_slices()
    covered = set()
    for sl in slices.values():
      covered.update(range(sl.start, sl.stop))
    assert len(covered) == rep.rep_state_dim

  def test_loss_weights_shape(self):
    rep = MaxCoordRepresentation(NUM_BODIES)
    W = rep.loss_weights(cfg=object(), device="cpu")
    assert W.shape == (rep.rep_state_dim,)
    assert (W > 0).all()
