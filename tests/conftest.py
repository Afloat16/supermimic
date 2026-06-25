"""Shared test fixtures and helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
from tensordict import TensorDict

from supermimic.sim_context import FootFrictionSpec, SimContext
from supermimic.tasks.task_spec import TaskSpec, default_encoder_factory

__all__ = [
  "identity_aux",
  "make_actor_obs",
  "make_actor_specs",
  "make_mock_buffer",
  "make_mock_command_env",
  "make_mock_dynamics",
  "make_mock_motion",
  "make_mock_representation",
  "make_mock_wm_representation",
  "make_mock_sim_context",
]

# Shared dims for multi-encoder actor tests (G1 MaxCoord).
OBS_DIM = 68
MOTION_GOAL_DIM = 84
VELOCITY_GOAL_DIM = 3
ACTION_DIM = 29
LATENT_DIM = 32


def make_actor_specs(velocity: bool = True) -> list[TaskSpec]:
  """Motion (+ optional velocity) specs whose extract_training_goal is a stub.

  Actor-forward tests drive the encoders directly and never extract goals
  from a motion clip, so the stub guards against accidental use.
  """

  def stub(*args, **kwargs):
    raise AssertionError("extract_training_goal called in actor test")

  specs = [
    TaskSpec(
      name="motion",
      goal_dim=MOTION_GOAL_DIM,
      encoder_factory=default_encoder_factory,
      extract_training_goal=stub,
    )
  ]
  if velocity:
    specs.append(
      TaskSpec(
        name="velocity",
        goal_dim=VELOCITY_GOAL_DIM,
        encoder_factory=default_encoder_factory,
        extract_training_goal=stub,
      )
    )
  return specs


def make_actor_obs(batch: int, history_length: int = 1) -> TensorDict:
  """Obs TensorDict with history + per-task goal keys for the multi-encoder
  actor. The history repeats the current frame (reset padding)."""
  obs = torch.randn(batch, OBS_DIM)
  return TensorDict(
    {
      "history": obs.unsqueeze(1).repeat(1, history_length, 1),
      "goal_motion": torch.randn(batch, MOTION_GOAL_DIM),
      "goal_velocity": torch.randn(batch, VELOCITY_GOAL_DIM),
    },
    batch_size=[batch],
  )


def make_mock_sim_context() -> SimContext:
  """One-spec SimContext for env-style tests; matches the ``foot_friction`` buffer field."""
  spec = FootFrictionSpec()
  spec.mean = torch.zeros(1)
  spec.scale = torch.ones(1)
  return SimContext([spec])


def make_mock_dynamics(
  rep_state_dim: int = 214,
  num_actions: int = 29,
) -> MagicMock:
  """Mock ``Dynamics`` matching ``forward(rep_state, action, aux_state, sim_ctx)``."""
  dyn = MagicMock()
  dyn.num_rep_state = rep_state_dim
  dyn.num_actions = num_actions
  dyn.dt = 0.02

  def forward(
    rep_state: torch.Tensor,
    action: torch.Tensor,
    aux_state: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    del sim_ctx
    B = rep_state.shape[0]
    d = rep_state.device
    return (torch.randn(B, 7, device=d), torch.randn(B, rep_state_dim, device=d))

  dyn.side_effect = forward
  return dyn


def make_mock_representation(
  rep_state_dim: int = 214,
  full_state_dim: int = 182,
  num_bodies: int = 14,
) -> MagicMock:
  """Mock ``StateRepresentation`` with MaxCoord layout slice helpers."""
  rep = MagicMock()
  rep.rep_state_dim = rep_state_dim
  rep.full_state_dim = full_state_dim
  rep.num_bodies = num_bodies
  rep.dt = 0.02

  rep.actor_obs_mode = "rep_state"
  rep.represent.side_effect = lambda x: torch.randn(
    *x.shape[:-1], rep_state_dim, device=x.device
  )
  rep.extract_aux_state.side_effect = lambda x: torch.randn(
    x.shape[0], 7, device=x.device
  )
  rep.compute_actor_sensor.side_effect = lambda rs, *a, **kw: rs
  rep.add_proprio_noise.side_effect = lambda obs: obs
  rep.local_pos.side_effect = lambda rs: rs[..., : num_bodies * 3].unflatten(
    -1, (num_bodies, 3)
  )
  rep.local_vel.side_effect = lambda rs: rs[
    ..., num_bodies * 9 : num_bodies * 12
  ].unflatten(-1, (num_bodies, 3))
  rep.root_height.side_effect = lambda rs: rs[..., num_bodies * 15]
  return rep


def make_mock_wm_representation(batch: int) -> MagicMock:
  """Mock representation wired for the WM env's goal/reward call path."""
  rep = make_mock_representation()
  rep.compute_wm_goal.side_effect = lambda *a, **kw: (
    torch.randn(batch, 87),  # wm_goal_obs
    torch.randn(batch, MOTION_GOAL_DIM),  # actor_goal_obs
  )
  rep.predict_termination.side_effect = lambda rs, a, g: torch.zeros(rs.shape[0], 1)
  rep.compute_wm_reward_from_reference.side_effect = lambda *a, **kw: torch.zeros(batch)
  return rep


def make_mock_buffer(full_state_dim: int = 182, goal_dim: int = 3) -> MagicMock:
  buf = MagicMock()
  buf.num_transitions = 100

  def sample(sequence_length: int, batch_size: int) -> TensorDict:
    return TensorDict(
      {
        "state": torch.randn(batch_size, sequence_length, full_state_dim),
        "action": torch.randn(batch_size, sequence_length, 29),
        "prev_action": torch.randn(batch_size, sequence_length, 29),
        "goal": torch.randn(batch_size, sequence_length, goal_dim),
        "reward": torch.randn(batch_size, sequence_length, 1),
        "motion_time": torch.zeros(batch_size, sequence_length, dtype=torch.long),
        "terminated": torch.zeros(batch_size, sequence_length, dtype=torch.bool),
        "truncated": torch.zeros(batch_size, sequence_length, dtype=torch.bool),
        "foot_friction": torch.zeros(batch_size, sequence_length, 1),
      },
      batch_size=[batch_size, sequence_length],
    )

  buf.sample = sample

  def sample_causal(
    sequence_length: int, batch_size: int
  ) -> tuple[TensorDict, torch.Tensor]:
    td = sample(sequence_length, batch_size)
    # Distinct per frame: actor_obs[:, k] == k, to verify the ring seed order.
    # Dim matches the mock representation's sensor (= rep_state, 214).
    td["actor_obs"] = torch.arange(sequence_length).float().view(
      1, sequence_length, 1
    ) * torch.ones(batch_size, sequence_length, 214)
    padded = torch.zeros(batch_size, sequence_length, dtype=torch.bool)
    return td, padded

  buf.sample_causal = sample_causal
  return buf


def make_mock_motion(num_frames: int = 10, num_bodies: int = 14) -> MagicMock:
  """Mocked MotionLoader with the body/joint fields the WM env indexes."""
  motion = MagicMock()
  motion.body_pos_w = torch.randn(num_frames, num_bodies, 3)
  motion.body_quat_w = torch.zeros(num_frames, num_bodies, 4)
  motion.body_quat_w[..., 0] = 1.0
  motion.body_lin_vel_w = torch.randn(num_frames, num_bodies, 3)
  motion.body_ang_vel_w = torch.randn(num_frames, num_bodies, 3)
  motion.joint_pos = torch.randn(num_frames, 29)
  motion.joint_vel = torch.randn(num_frames, 29)
  motion.motion_ends = torch.tensor([num_frames], dtype=torch.long)
  return motion


def make_mock_command_env(num_actions: int = ACTION_DIM) -> MagicMock:
  """Mocked env exposing ``num_actions`` and the motion command term."""
  env = MagicMock()
  env.num_actions = num_actions
  term = env.unwrapped.command_manager.get_term.return_value
  term.motion = make_mock_motion()
  term.motion_anchor_body_index = 0
  return env


def identity_aux(batch_size: int) -> torch.Tensor:
  """Return a ``(batch_size, 7)`` aux state with identity quaternion."""
  aux = torch.zeros(batch_size, 7)
  aux[..., 3] = 1.0
  return aux
