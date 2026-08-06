"""Tests for GenCoordRepresentation and create_representation factory."""

import torch

from supermimic.representations import (
  GenCoordRepresentation,
  MaxCoordRepresentation,
  create_representation,
)
from tests.conftest import identity_aux

# ---------------------------------------------------------------------------
# create_representation factory
# ---------------------------------------------------------------------------


class TestCreateRepresentation:
  def _dummy_args(self, num_actions: int = 4, num_joints: int = 20):
    return {
      "num_actions": num_actions,
      "num_goals": 5,
      "num_rewards": 1,
      "sim_ctx_dim": 0,
      "dt": 0.02,
      "default_joint_pos": torch.zeros(num_joints),
      "has_state_estimation": False,
      "actor_obs_mode": "rep_state",
      "reward_hidden_dims": (16, 16),
      "term_hidden_dims": (16, 16),
      "obs_hidden_dims": (16, 16),
    }

  def test_gencoord(self):
    rep = create_representation("gencoord", 13 + 2 * 20, **self._dummy_args())
    assert isinstance(rep, GenCoordRepresentation)
    assert rep.num_joints == 20

  def test_maxcoord(self):
    rep = create_representation("maxcoord", 13 * 16, **self._dummy_args(num_joints=29))
    assert isinstance(rep, MaxCoordRepresentation)
    assert rep.num_bodies == 16

  def test_unknown_raises(self):
    try:
      create_representation("unknown", 100, **self._dummy_args())
      assert False, "Should have raised ValueError"
    except ValueError:
      pass


# ---------------------------------------------------------------------------
# GenCoordRepresentation
# ---------------------------------------------------------------------------

NUM_JOINTS = 20


class TestGenCoordApplyPrediction:
  def test_output_shapes(self):
    rep = GenCoordRepresentation(NUM_JOINTS, torch.zeros(NUM_JOINTS))
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    pred = torch.randn(4, rep.prediction_dim) * 0.01
    next_aux, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.01)
    assert next_aux.shape == (4, 7)
    assert next_rep.shape == (4, rep.rep_state_dim)

  def test_prediction_dim_is_accel_level(self):
    rep = GenCoordRepresentation(NUM_JOINTS, torch.zeros(NUM_JOINTS))
    assert rep.prediction_dim == 6 + NUM_JOINTS

  def test_root_quat_is_unit(self):
    rep = GenCoordRepresentation(NUM_JOINTS, torch.zeros(NUM_JOINTS))
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    pred = torch.randn(4, rep.prediction_dim) * 0.1
    next_aux, _ = rep.apply_prediction(aux, curr, pred, dt=0.01)
    quat_norms = next_aux[..., 3:7].norm(dim=-1)
    assert torch.allclose(quat_norms, torch.ones(4), atol=1e-5)

  def test_root_z_from_rep_state(self):
    """Root Z should be overridden by rep state's z component."""
    rep = GenCoordRepresentation(NUM_JOINTS, torch.zeros(NUM_JOINTS))
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim)
    pred = torch.randn(4, rep.prediction_dim) * 0.01
    next_aux, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.01)
    # z is the first element of rep state
    assert torch.allclose(next_aux[..., 2], next_rep[..., 0], atol=1e-6)

  def test_no_nan_inf_at_various_scales(self):
    """Gradients should be finite across different prediction magnitudes."""
    rep = GenCoordRepresentation(NUM_JOINTS, torch.zeros(NUM_JOINTS))
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
    rep = GenCoordRepresentation(NUM_JOINTS, torch.zeros(NUM_JOINTS))
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


class TestGenCoordAccelPrediction:
  def _rep(self):
    return GenCoordRepresentation(NUM_JOINTS, torch.zeros(NUM_JOINTS))

  def test_kinematic_consistency(self):
    """qj integrates vj; z and gravity are derived from the root pose."""
    from mjlab.utils.lab_api.math import quat_apply_inverse

    rep = self._rep()
    dt = 0.02
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim) * 0.3
    pred = torch.randn(4, rep.prediction_dim) * 0.1
    next_aux, next_rep = rep.apply_prediction(aux, curr, pred, dt=dt)

    N = NUM_JOINTS
    qj, vj = curr[:, 10 : 10 + N], curr[:, 10 + N : 10 + 2 * N]
    next_qj, next_vj = next_rep[:, 10 : 10 + N], next_rep[:, 10 + N : 10 + 2 * N]
    torch.testing.assert_close(next_vj, vj + pred[:, 6 : 6 + N])
    torch.testing.assert_close(next_qj, qj + next_vj * dt)
    torch.testing.assert_close(next_rep[:, 0:1], next_aux[:, 2:3])
    down = torch.zeros(4, 3)
    down[:, 2] = -1.0
    torch.testing.assert_close(
      next_rep[:, 1:4], quat_apply_inverse(next_aux[:, 3:7], down)
    )
    assert torch.allclose(next_aux[:, 3:7].norm(dim=-1), torch.ones(4), atol=1e-5)

  def test_gradients_finite_and_cover_prediction(self):
    rep = self._rep()
    aux = torch.zeros(4, 7)
    aux[..., 3] = 1.0
    curr = torch.randn(4, rep.rep_state_dim) * 0.3
    pred = (torch.randn(4, rep.prediction_dim) * 0.1).requires_grad_(True)
    next_aux, next_rep = rep.apply_prediction(aux, curr, pred, dt=0.02)
    (next_rep.sum() + next_aux.sum()).backward()
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad.abs().sum(dim=0) > 0).all()


class TestGenCoordComputeWmObs:
  def test_output_shape(self):
    rep = GenCoordRepresentation(NUM_JOINTS, torch.zeros(NUM_JOINTS))
    aux_state = identity_aux(4)
    rep_state = torch.randn(4, rep.rep_state_dim)
    sim_ctx = torch.zeros(4, 0)

    obs = rep.compute_wm_obs(rep_state, aux_state, sim_ctx)

    assert obs.shape == (4, rep.wm_obs_dim)
    # 3 (proj_grav) + 3 (base_ang_vel) + N + N
    assert rep.wm_obs_dim == 6 + 2 * NUM_JOINTS

  def test_joint_pos_rel_subtracts_default(self):
    default = torch.linspace(0.1, 0.5, NUM_JOINTS)
    rep = GenCoordRepresentation(NUM_JOINTS, default)
    aux_state = identity_aux(2)
    rep_state = torch.zeros(2, rep.rep_state_dim)
    qj = torch.linspace(0.2, 0.7, NUM_JOINTS).expand(2, -1)
    rep_state[:, 10 : 10 + NUM_JOINTS] = qj
    sim_ctx = torch.zeros(2, 0)

    obs = rep.compute_wm_obs(rep_state, aux_state, sim_ctx)

    # joint_pos_rel starts after proj_grav(3) + base_ang_vel(3) = 6
    jp_rel = obs[:, 6 : 6 + NUM_JOINTS]
    assert torch.allclose(jp_rel, qj - default, atol=1e-6)

  def test_maxcoord_raises_when_head_uninitialized(self):
    # rep_state mode never builds the obs head — compute_wm_obs must raise.
    rep = MaxCoordRepresentation(num_bodies=4, num_joints=10)
    rep.build_learned_heads(
      num_actions=5,
      num_goals=1,
      num_rewards=0,
      sim_ctx_dim=0,
      reward_hidden_dims=(16, 16),
      term_hidden_dims=(16, 16),
      obs_hidden_dims=(16, 16),
    )
    aux_state = identity_aux(2)
    rep_state = torch.zeros(2, rep.rep_state_dim)
    sim_ctx = torch.zeros(2, 0)
    try:
      rep.compute_wm_obs(rep_state, aux_state, sim_ctx)
      assert False, "Should have raised RuntimeError"
    except RuntimeError:
      pass


class TestMaxCoordObsHead:
  @staticmethod
  def _build(num_bodies: int, num_joints: int, num_actions: int):
    rep = MaxCoordRepresentation(num_bodies, num_joints)
    rep.actor_obs_mode = "proprio"
    rep.build_learned_heads(
      num_actions=num_actions,
      num_goals=1,
      num_rewards=0,
      sim_ctx_dim=0,
      reward_hidden_dims=(16, 16),
      term_hidden_dims=(16, 16),
      obs_hidden_dims=(16, 16),
    )
    return rep

  def test_output_shape(self):
    num_bodies = 4
    num_joints = 10
    num_actions = num_joints
    rep = self._build(num_bodies, num_joints, num_actions)
    aux_state = identity_aux(3)
    rep_state = torch.randn(3, rep.rep_state_dim)
    sim_ctx = torch.zeros(3, 0)

    obs = rep.compute_wm_obs(rep_state, aux_state, sim_ctx)

    assert obs.shape == (3, rep.wm_obs_dim)
    # 3 (proj_grav) + 3 (base_ang_vel) + N + N
    assert rep.wm_obs_dim == 6 + 2 * num_joints

  def test_head_is_trainable_submodule(self):
    rep = self._build(num_bodies=4, num_joints=10, num_actions=10)
    head_params = list(rep.obs_head.parameters())
    assert len(head_params) > 0
    for p in head_params:
      assert p.requires_grad

  def test_noise_bounds_shape(self):
    rep = self._build(num_bodies=4, num_joints=10, num_actions=10)
    bounds = rep.proprio_noise_bounds(num_actions=10, device="cpu")
    assert bounds.shape == (rep.proprio_obs_dim(10),)


class TestGenCoordNoiseBounds:
  def test_shape_matches_obs_dim(self):
    rep = GenCoordRepresentation(NUM_JOINTS, torch.zeros(NUM_JOINTS))
    for num_actions in (15, NUM_JOINTS, 40):
      bounds = rep.proprio_noise_bounds(num_actions, device="cpu")
      assert bounds.shape == (rep.proprio_obs_dim(num_actions),)

  def test_last_action_is_clean(self):
    rep = GenCoordRepresentation(NUM_JOINTS, torch.zeros(NUM_JOINTS))
    num_actions = NUM_JOINTS
    bounds = rep.proprio_noise_bounds(num_actions, device="cpu")
    assert (bounds[-num_actions:] == 0).all()

  def test_term_widths_cover_obs_dim(self):
    """Deploy term widths must tile the full sensor — a term added to
    _sensor_actor_group without updating proprio_term_widths breaks ONNX."""
    rep = GenCoordRepresentation(NUM_JOINTS, torch.zeros(NUM_JOINTS))
    for num_actions in (15, NUM_JOINTS, 40):
      widths = rep.proprio_term_widths(num_actions)
      assert sum(widths) == rep.proprio_obs_dim(num_actions)

  def test_slice_values_match_mjlab(self):
    rep = GenCoordRepresentation(NUM_JOINTS, torch.zeros(NUM_JOINTS))
    num_actions = NUM_JOINTS
    bounds = rep.proprio_noise_bounds(num_actions, device="cpu")
    # [proj_grav(3), base_ang_vel(3), joint_pos(N), joint_vel(N), last_action(A)]
    assert torch.allclose(bounds[:3], torch.full((3,), 0.05))
    assert torch.allclose(bounds[3:6], torch.full((3,), 0.2))
    assert torch.allclose(bounds[6 : 6 + NUM_JOINTS], torch.full((NUM_JOINTS,), 0.01))
    assert torch.allclose(
      bounds[6 + NUM_JOINTS : 6 + 2 * NUM_JOINTS], torch.full((NUM_JOINTS,), 0.5)
    )


class TestProprioNoiseSpecConsistency:
  """env_cfg's `_sensor_actor_group` and rep.proprio_noise_bounds must agree."""

  def test_env_cfg_terms_match_half_widths(self):
    from supermimic.representations import PROPRIO_NOISE_HALF_WIDTHS
    from supermimic.tasks.env_cfgs import _sensor_actor_group

    group = _sensor_actor_group(has_state_estimation=True)
    for name, half in PROPRIO_NOISE_HALF_WIDTHS.items():
      term = group.terms[name]
      if half == 0:
        assert term.noise is None, f"{name}: expected no noise, got {term.noise}"
      else:
        assert term.noise.n_min == -half, name
        assert term.noise.n_max == half, name


class TestGenCoordLossSlices:
  def test_slices_cover_full_dim(self):
    rep = GenCoordRepresentation(NUM_JOINTS, torch.zeros(NUM_JOINTS))
    slices = rep.loss_slices()
    covered = set()
    for sl in slices.values():
      covered.update(range(sl.start, sl.stop))
    assert len(covered) == rep.rep_state_dim

  def test_loss_weights_shape(self):
    rep = GenCoordRepresentation(NUM_JOINTS, torch.zeros(NUM_JOINTS))
    W = rep.loss_weights(cfg=object(), device="cpu")
    assert W.shape == (rep.rep_state_dim,)
    assert (W > 0).all()
