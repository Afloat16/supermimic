"""Tests for VaeWorldModelEnv base class."""

import torch

from supermimic.envs.vae_world_model_env import VaeWorldModelEnv, VaeWorldModelEnvCfg
from tests.conftest import (
  make_mock_buffer,
  make_mock_dynamics,
  make_mock_representation,
  make_mock_vae,
)


class ConcreteTestEnv(VaeWorldModelEnv):
  goal_dim = 3
  raw_goal_dim = 3

  @staticmethod
  def build_goal_obs(raw_goal, aux_state):
    return raw_goal

  def _compute_reward(self, rep_state, aux_state, is_last_mask):
    return torch.zeros(rep_state.shape[0], device=rep_state.device)

  def sample_raw_goal(self, num_envs: int) -> torch.Tensor:
    return torch.zeros(num_envs, self.raw_goal_dim, device=self.device)


BATCH = 8
CFG = VaeWorldModelEnvCfg(latent_scale_mult=2.0)


class TestVaeWorldModelEnv:
  def _create_env(self):
    return ConcreteTestEnv(
      cfg=CFG,
      dynamics=make_mock_dynamics(),
      representation=make_mock_representation(),
      vae=make_mock_vae(),
      batch_size=BATCH,
      horizon=16,
      replay_buffer=make_mock_buffer(),
      body_masses=torch.ones(14),
      latent_dim=64,
      device="cpu",
    )

  def test_reset_returns_correct_obs_shape(self):
    env = self._create_env()
    obs = env.reset()
    assert obs["actor"].shape == (BATCH, 214 + 3)

  def test_step_returns_correct_shapes(self):
    env = self._create_env()
    env.reset()
    obs, rewards, dones, _ = env.step(torch.zeros(BATCH, 64))
    assert obs["actor"].shape == (BATCH, 214 + 3)
    assert rewards.shape == (BATCH,)
    assert dones.shape == (BATCH,)

  def test_is_last_mask_fresh_tensor_each_step(self):
    """In-place mutation across steps would break BPTT backward."""
    saved_masks: list[torch.Tensor] = []

    class SpyEnv(ConcreteTestEnv):
      def _compute_reward(self, rep_state, aux_state, is_last_mask):
        saved_masks.append(is_last_mask)
        return torch.zeros(rep_state.shape[0], device=rep_state.device)

    env = SpyEnv(
      cfg=CFG,
      dynamics=make_mock_dynamics(),
      representation=make_mock_representation(),
      vae=make_mock_vae(),
      batch_size=BATCH,
      horizon=4,
      replay_buffer=make_mock_buffer(),
      body_masses=torch.ones(14),
      latent_dim=64,
      device="cpu",
    )
    env.reset()
    env.step(torch.zeros(BATCH, 64))
    env.step(torch.zeros(BATCH, 64))
    assert saved_masks[0].data_ptr() != saved_masks[1].data_ptr()

  def test_is_last_mask_set_only_on_horizon_step(self):
    """is_last_mask becomes 1.0 on step `horizon`, stays 0.0 before."""
    saved_masks: list[torch.Tensor] = []

    class SpyEnv(ConcreteTestEnv):
      def _compute_reward(self, rep_state, aux_state, is_last_mask):
        saved_masks.append(is_last_mask.clone())
        return torch.zeros(rep_state.shape[0], device=rep_state.device)

    env = SpyEnv(
      cfg=CFG,
      dynamics=make_mock_dynamics(),
      representation=make_mock_representation(),
      vae=make_mock_vae(),
      batch_size=BATCH,
      horizon=4,
      replay_buffer=make_mock_buffer(),
      body_masses=torch.ones(14),
      latent_dim=64,
      device="cpu",
    )
    env.reset()
    for _ in range(4):
      env.step(torch.zeros(BATCH, 64))

    for i in range(3):
      assert (saved_masks[i] == 0).all()
    assert (saved_masks[3] == 1).all()

  def test_decode_residual_bounded_by_latent_scale(self):
    """tanh saturates: |z − vae.encode(rs)| ≤ latent_scale_mult·fixed_std."""
    env = self._create_env()
    vae_obs = torch.zeros(BATCH, 214)
    env.decode_residual(vae_obs, torch.full((BATCH, 64), 1e6))  # tanh ≈ +1
    z = env.vae.decode.call_args.args[0]
    expected = CFG.latent_scale_mult * make_mock_vae().fixed_std
    assert torch.allclose(z, torch.full_like(z, expected), atol=1e-3)

  def test_decode_residual_zero_residual_is_pure_prior(self):
    """residual=0 → z = vae.encode(rs); the VAE prior is unchanged."""
    env = self._create_env()
    vae_obs = torch.zeros(BATCH, 214)
    env.decode_residual(vae_obs, torch.zeros(BATCH, 64))
    z = env.vae.decode.call_args.args[0]
    assert torch.allclose(z, torch.zeros_like(z))
