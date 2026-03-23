"""Tests for VAEActor module."""

import torch
from tensordict import TensorDict

from supermimic.models import VAEActor, load_actor

OBS_DIM = 68
GOAL_DIM = 87
ACTION_DIM = 29
BATCH = 16


def _make_obs_td(batch_size=BATCH):
  """Create a TensorDict matching WorldModelEnv.get_observations() format."""
  return TensorDict(
    {"actor": torch.randn(batch_size, OBS_DIM + GOAL_DIM)},
    batch_size=[batch_size],
  )


def _make_actor(**kwargs):
  return VAEActor(
    obs_dim=OBS_DIM,
    goal_dim=GOAL_DIM,
    action_dim=ACTION_DIM,
    **kwargs,
  )


class TestVAEActorForwardShape:
  def test_output_shape(self):
    actor = _make_actor()
    actor.train()
    obs = _make_obs_td()
    actions = actor(obs)
    assert actions.shape == (BATCH, ACTION_DIM)

  def test_single_env(self):
    actor = _make_actor()
    actor.train()
    obs = _make_obs_td(batch_size=1)
    actions = actor(obs)
    assert actions.shape == (1, ACTION_DIM)


class TestAuxiliaryLoss:
  def test_positive_in_training(self):
    actor = _make_actor()
    actor.train()
    actor(_make_obs_td())
    assert isinstance(actor.auxiliary_loss(), torch.Tensor)
    assert (actor.auxiliary_loss() > 0).all()

  def test_stats_reports_raw_kl(self):
    actor = _make_actor(kl_weight=0.1)
    actor.train()
    actor(_make_obs_td())
    stats = actor.auxiliary_stats()
    assert "kl" in stats
    assert abs(actor.auxiliary_loss().sum().item() / stats["kl"] / BATCH - 0.1) < 1e-5

  def test_smaller_with_zero_goal(self):
    """KL with zero goal should be smaller than KL with large random goal."""
    actor = _make_actor()
    actor.train()

    # Large random goal -> large KL
    obs_data = torch.randn(BATCH, OBS_DIM)
    large_goal = torch.randn(BATCH, GOAL_DIM) * 10
    obs_large = TensorDict(
      {"actor": torch.cat([obs_data, large_goal], dim=-1)},
      batch_size=[BATCH],
    )
    actor(obs_large)
    kl_large = actor.auxiliary_stats()["kl"]

    # Zero goal -> smaller KL
    obs_zero = TensorDict(
      {"actor": torch.cat([obs_data, torch.zeros(BATCH, GOAL_DIM)], dim=-1)},
      batch_size=[BATCH],
    )
    actor(obs_zero)
    kl_zero = actor.auxiliary_stats()["kl"]

    assert kl_zero < kl_large

  def test_zero_in_eval(self):
    actor = _make_actor()
    actor.eval()
    actor(_make_obs_td())
    assert actor.auxiliary_loss() == 0.0
    assert actor.auxiliary_stats() == {}


class TestEvalMode:
  def test_deterministic(self):
    """Two eval-mode calls with same input should produce identical output."""
    actor = _make_actor()
    actor.eval()
    obs = _make_obs_td()
    a1 = actor(obs)
    a2 = actor(obs)
    assert torch.allclose(a1, a2)

  def test_prior_ignores_goal(self):
    """encode(obs) without goal gives same result regardless of goal."""
    actor = _make_actor()
    actor.eval()
    obs = torch.randn(BATCH, OBS_DIM)
    z1 = actor.encode(obs)
    z2 = actor.encode(obs)
    assert torch.allclose(z1, z2)
    a1 = actor.decode(z1, obs)
    a2 = actor.decode(z2, obs)
    assert torch.allclose(a1, a2)

  def test_posterior_uses_goal(self):
    """encode(obs, goal) with different goals gives different latents."""
    actor = _make_actor()
    actor.eval()
    obs = torch.randn(BATCH, OBS_DIM)
    z1 = actor.encode(obs, torch.randn(BATCH, GOAL_DIM))
    z2 = actor.encode(obs, torch.randn(BATCH, GOAL_DIM))
    assert not torch.allclose(z1, z2)


class TestGradientFlow:
  def test_gradients_flow_through_encoders_and_decoder(self):
    actor = _make_actor()
    actor.train()
    obs = _make_obs_td()
    actions = actor(obs)
    loss = actions.sum() + actor.auxiliary_loss().sum()
    loss.backward()
    for name, p in actor.named_parameters():
      assert p.grad is not None, f"No gradient for {name}"
      assert not p.grad.isnan().any(), f"NaN gradient for {name}"


class TestLoadActor:
  def test_roundtrip_vae(self):
    actor = _make_actor(latent_dim=32)
    actor.train()
    obs = _make_obs_td()
    actor(obs)
    state_dict = actor.state_dict()
    loaded = load_actor(state_dict, OBS_DIM, GOAL_DIM, ACTION_DIM)
    assert isinstance(loaded, VAEActor)
    assert loaded.obs_dim == OBS_DIM
    assert torch.allclose(loaded(obs), actor.eval()(obs))

  def test_roundtrip_mlp(self):
    from rsl_rl.models import MLPModel

    obs_td = TensorDict({"actor": torch.zeros(1, OBS_DIM + GOAL_DIM)}, batch_size=[1])
    mlp = MLPModel(
      obs=obs_td,
      obs_groups={"actor": ["actor"]},
      obs_set="actor",
      output_dim=ACTION_DIM,
      hidden_dims=(512, 256, 128),
      obs_normalization=True,
    )
    loaded = load_actor(mlp.state_dict(), OBS_DIM, GOAL_DIM, ACTION_DIM)
    assert isinstance(loaded, MLPModel)

  def test_vae_infers_latent_dim(self):
    actor = _make_actor(latent_dim=32)
    loaded = load_actor(actor.state_dict(), OBS_DIM, GOAL_DIM, ACTION_DIM)
    # Decoder input is latent_dim + obs_dim
    assert loaded.decoder[0].in_features == 32 + OBS_DIM


class TestNormalization:
  def test_update_normalization(self):
    actor = _make_actor()
    obs = _make_obs_td()
    actor.update_normalization(obs.detach())
    # Should not raise
