"""Tests for BodyFrameRepresentation and create_representation factory."""

import torch

from supermimic.representations import (
  BodyFrameRepresentation,
  MaxCoordRepresentation,
  create_representation,
)

# ---------------------------------------------------------------------------
# create_representation factory
# ---------------------------------------------------------------------------


class TestCreateRepresentation:
  def test_body_frame(self):
    rep = create_representation("body_frame", 13 + 2 * 20)  # 20 joints
    assert isinstance(rep, BodyFrameRepresentation)
    assert rep.num_joints == 20

  def test_maxcoord(self):
    rep = create_representation("maxcoord", 13 * 16)  # 16 bodies
    assert isinstance(rep, MaxCoordRepresentation)
    assert rep.num_bodies == 16

  def test_unknown_raises(self):
    try:
      create_representation("unknown", 100)
      assert False, "Should have raised ValueError"
    except ValueError:
      pass


# ---------------------------------------------------------------------------
# BodyFrameRepresentation
# ---------------------------------------------------------------------------

NUM_JOINTS = 20


class TestBodyFrameApplyPrediction:
  def test_output_shapes(self):
    rep = BodyFrameRepresentation(NUM_JOINTS)
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    pred = torch.randn(4, rep.prediction_dim) * 0.01
    next_aux, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.01)
    assert next_aux.shape == (4, 7)
    assert next_rep.shape == (4, rep.rep_state_dim)

  def test_prediction_dim_equals_rep_state_dim(self):
    rep = BodyFrameRepresentation(NUM_JOINTS)
    assert rep.prediction_dim == rep.rep_state_dim

  def test_root_quat_is_unit(self):
    rep = BodyFrameRepresentation(NUM_JOINTS)
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    pred = torch.randn(4, rep.prediction_dim) * 0.1
    next_aux, _ = rep.apply_prediction(aux, curr, pred, dt=0.01)
    quat_norms = next_aux[..., 3:7].norm(dim=-1)
    assert torch.allclose(quat_norms, torch.ones(4), atol=1e-5)

  def test_root_z_from_rep_state(self):
    """Root Z should be overridden by rep state's z component."""
    rep = BodyFrameRepresentation(NUM_JOINTS)
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    pred = torch.randn(4, rep.prediction_dim) * 0.01
    next_aux, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.01)
    # z is the first element of rep state
    assert torch.allclose(next_aux[..., 2], next_rep[..., 0], atol=1e-6)

  def test_no_nan_inf_at_various_scales(self):
    """Gradients should be finite across different prediction magnitudes."""
    rep = BodyFrameRepresentation(NUM_JOINTS)
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

  def test_gradient_stable_across_scales(self):
    """Gradient norms should not explode as prediction shrinks."""
    rep = BodyFrameRepresentation(NUM_JOINTS)
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


class TestBodyFrameLossSlices:
  def test_slices_cover_full_dim(self):
    rep = BodyFrameRepresentation(NUM_JOINTS)
    slices = rep.loss_slices()
    covered = set()
    for sl in slices.values():
      covered.update(range(sl.start, sl.stop))
    assert len(covered) == rep.rep_state_dim

  def test_loss_weights_shape(self):
    rep = BodyFrameRepresentation(NUM_JOINTS)
    W = rep.loss_weights(cfg=object(), device="cpu")
    assert W.shape == (rep.rep_state_dim,)
    assert (W > 0).all()
