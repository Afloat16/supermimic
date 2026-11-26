"""Tests for WorldModelEnv."""

from __future__ import annotations

import torch

from supermimic.envs.world_model_env import WorldModelEnv, sample_start
from supermimic.representations import (
  GOAL_REL_POS_DIM,
  NUM_GOAL_FRAMES,
  GenCoordRepresentation,
  _assemble_goal,
)
from supermimic.tasks.task_spec import TaskSpec, make_motion_task
from tests.conftest import (
  ACTION_DIM,
  MOTION_GOAL_DIM,
  VELOCITY_GOAL_DIM,
  identity_aux,
  make_actor_specs,
  make_mock_buffer,
  make_mock_command_env,
  make_mock_dynamics,
  make_mock_wm_representation,
)

SOFTNESS = 0.02
BATCH = 8


def _make_env(history_length: int = 1) -> WorldModelEnv:
  velocity_goal = torch.randn(BATCH, VELOCITY_GOAL_DIM)
  velocity = TaskSpec(
    name="velocity",
    goal_dim=VELOCITY_GOAL_DIM,
    extract_training_goal=lambda m, mt, aux: velocity_goal,
  )
  return WorldModelEnv(
    dynamics=make_mock_dynamics(),
    representation=make_mock_wm_representation(BATCH),
    env=make_mock_command_env(),
    task_specs=[*make_actor_specs(velocity=False), velocity],
    history_length=history_length,
    termination_softness=SOFTNESS,
  )


class TestWorldModelEnv:
  def test_reset_obs_has_per_task_goals(self):
    env = _make_env()
    obs = env.reset(sample_start(make_mock_buffer(), 1, BATCH))
    assert torch.equal(obs["history"], env._latest_sensor.unsqueeze(1))
    assert "critic" in obs.keys()
    assert obs["goal_motion"].shape == (BATCH, MOTION_GOAL_DIM)
    assert obs["goal_velocity"].shape == (BATCH, VELOCITY_GOAL_DIM)

  def test_step_advances_motion_times(self):
    env = _make_env()
    env.reset(sample_start(make_mock_buffer(), 1, BATCH))
    times_before = env.motion_times.clone()
    env.step(torch.zeros(BATCH, ACTION_DIM))
    assert (env.motion_times == times_before + 1).all()


class TestHistoryRing:
  H = 4

  def test_reset_seeds_past_from_causal_window(self):
    env = _make_env(history_length=self.H)
    obs = env.reset(sample_start(make_mock_buffer(), self.H, BATCH))
    history = obs["history"]
    assert history.shape == (BATCH, self.H, env._latest_sensor.shape[-1])
    # Past slots hold the sampled window's real predecessors (frame k == k);
    # the newest slot is the freshly synthesized sensor of the init state.
    for k in range(self.H - 1):
      assert torch.allclose(history[:, k], torch.full_like(history[:, k], float(k)))
    assert torch.equal(history[:, -1], env._latest_sensor)

  def test_reset_noises_seeds_and_shares_padding(self):
    """Warm-start predecessors are noised; clamp-padded slots (before the
    episode start) share ONE noise realization, as the collection ring does."""
    env = _make_env(history_length=self.H)
    env.representation.add_proprio_noise.side_effect = lambda x: x + torch.rand_like(x)
    start = sample_start(make_mock_buffer(), self.H, BATCH)
    clean = start["sensor_window"]  # frame k holds k
    start["pad_mask"][:, :2] = True  # slots 0,1 clamp-pad
    history = env.reset(start)["history"]
    # The padded slots carry the SAME noised realization of the oldest real frame.
    assert torch.equal(history[:, 0], history[:, 1])
    assert torch.equal(history[:, 0], history[:, 2])
    # ...and they are noised, not the clean buffer value.
    assert not torch.allclose(history[:, 0], clean[:, 2])

  def test_step_shifts_and_appends_fresh_frame(self):
    env = _make_env(history_length=self.H)
    old = env.reset(sample_start(make_mock_buffer(), self.H, BATCH))
    obs = env.step(torch.zeros(BATCH, ACTION_DIM))[0]
    # Drop-oldest / append-newest: new[:, :-1] == old[:, 1:].
    assert torch.allclose(obs["history"][:, :-1], old["history"][:, 1:])
    assert torch.equal(obs["history"][:, -1], env._latest_sensor)

  def test_past_frames_keep_autograd_graph(self):
    """Backward through a PAST history slot must reach the action that produced
    it: rolled-off frames keep their autograd graph (a detach on the ring
    append would sever this)."""
    env = _make_env(history_length=self.H)
    env.reset(sample_start(make_mock_buffer(), self.H, BATCH))
    D = env._latest_sensor.shape[-1]
    env.representation.compute_actor_sensor.side_effect = lambda rs, ctx, **kw: (
      kw["last_action"].sum(-1, keepdim=True).expand(-1, D) * 2.0
    )

    a1 = torch.randn(BATCH, ACTION_DIM, requires_grad=True)
    env.step(a1)  # synthesizes the sensor that depends on a1
    obs = env.step(torch.zeros(BATCH, ACTION_DIM))[0]  # rolls it into a past slot

    obs["history"][:, -2].sum().backward()
    assert a1.grad is not None
    assert torch.allclose(a1.grad, torch.full_like(a1, 2.0 * D))


class TestRepresentationApiContract:
  """Pin the real representation's goal/reward API that the env and motion
  TaskSpec call, against a non-mocked representation.
  """

  def test_goal_assembly_shapes(self):
    aux = torch.zeros(4, 7)
    aux[:, 3] = 1.0
    anchor_pos = torch.zeros(4, NUM_GOAL_FRAMES, 3)
    anchor_quat = torch.zeros(4, NUM_GOAL_FRAMES, 4)
    anchor_quat[..., 0] = 1.0
    wm_goal, actor_goal = _assemble_goal(
      torch.zeros(4, 24), aux[:, :3], aux[:, 3:7], anchor_pos, anchor_quat
    )
    assert wm_goal.shape[0] == 4
    assert actor_goal.shape[0] == 4
    # The actor sees the same goal minus the world-model-only rel_pos dims.
    assert wm_goal.shape[1] - actor_goal.shape[1] == GOAL_REL_POS_DIM

  def test_env_update_goals_against_real_representation(self):
    """Drive the env -> rep call path with a REAL representation: a wrong
    method name or swapped arg order would error/NaN here, where the mocked
    env tests cannot catch it.
    """
    rep = GenCoordRepresentation(29, torch.zeros(29))
    env = WorldModelEnv(
      dynamics=make_mock_dynamics(),
      representation=rep,
      env=make_mock_command_env(),
      task_specs=[make_motion_task(rep, num_goals=87)],
      history_length=1,
      termination_softness=SOFTNESS,
    )
    env.aux_state = identity_aux(BATCH)
    env.motion_times = torch.zeros(BATCH, dtype=torch.long)
    env._update_goal()
    assert torch.isfinite(env._wm_goal_obs).all()
    assert torch.isfinite(env._task_goal_obs["motion"]).all()
