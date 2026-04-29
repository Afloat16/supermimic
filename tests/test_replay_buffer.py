import pytest
import torch
from tensordict import TensorDict

from supermimic.replay_buffer import FieldSpec, ReplayBuffer


def _schema(state_dim: int = 3) -> dict[str, FieldSpec]:
  return {
    "state": FieldSpec(state_dim),
    "terminated": FieldSpec(dtype=torch.bool),
    "truncated": FieldSpec(dtype=torch.bool),
  }


def _step_td(
  num_envs: int,
  stamp: float,
  terminated: bool = False,
  truncated: bool = False,
  state_dim: int = 3,
) -> TensorDict:
  return TensorDict(
    {
      "state": torch.full((num_envs, state_dim), stamp),
      "terminated": torch.full((num_envs,), terminated, dtype=torch.bool),
      "truncated": torch.full((num_envs,), truncated, dtype=torch.bool),
    },
    batch_size=[num_envs],
  )


def test_insert_increments_counter():
  buf = ReplayBuffer(_schema(), buffer_size=10, num_envs=4)
  assert buf.num_transitions == 0
  for i in range(5):
    buf.insert(_step_td(4, float(i)))
  assert buf.num_transitions == 5


def test_insert_caps_at_buffer_size():
  buf = ReplayBuffer(_schema(), buffer_size=10, num_envs=4)
  for i in range(25):
    buf.insert(_step_td(4, float(i)))
  assert buf.num_transitions == 10


def test_insert_wrap_overwrites():
  torch.manual_seed(0)
  buf = ReplayBuffer(_schema(), buffer_size=10, num_envs=4)
  for i in range(20):
    buf.insert(_step_td(4, float(i)))
  # Most recent buffer_size stamps are 10..19 — older data is overwritten.
  td = buf.sample(sequence_length=1, batch_size=200)
  stamps = td["state"][..., 0, 0]
  assert (stamps >= 10).all() and (stamps <= 19).all()


def test_window_content_consecutive():
  torch.manual_seed(0)
  buf = ReplayBuffer(_schema(), buffer_size=20, num_envs=4)
  for i in range(15):
    buf.insert(_step_td(4, float(i)))
  td = buf.sample(sequence_length=4, batch_size=200)
  stamps = td["state"][..., 0]
  diffs = stamps[:, 1:] - stamps[:, :-1]
  assert (diffs == 1.0).all()


def test_window_never_crosses_step_after_wrap():
  # A window straddling the write pointer would jump backward in stamp.
  torch.manual_seed(0)
  buf = ReplayBuffer(_schema(), buffer_size=10, num_envs=4)
  for i in range(25):
    buf.insert(_step_td(4, float(i)))
  td = buf.sample(sequence_length=4, batch_size=500)
  stamps = td["state"][..., 0]
  diffs = stamps[:, 1:] - stamps[:, :-1]
  assert (diffs == 1.0).all()


def test_terminated_at_interior_excludes_window():
  torch.manual_seed(0)
  buf = ReplayBuffer(_schema(), buffer_size=10, num_envs=4)
  for i in range(10):
    buf.insert(_step_td(4, float(i), terminated=(i == 5)))
  td = buf.sample(sequence_length=4, batch_size=500)
  assert (~td["terminated"][:, :-1]).all()


def test_terminated_at_final_position_allowed():
  torch.manual_seed(0)
  buf = ReplayBuffer(_schema(), buffer_size=10, num_envs=4)
  for i in range(10):
    buf.insert(_step_td(4, float(i), terminated=(i == 5)))
  td = buf.sample(sequence_length=4, batch_size=500)
  assert td["terminated"][:, -1].any()


def test_truncated_at_interior_excludes_window():
  torch.manual_seed(0)
  buf = ReplayBuffer(_schema(), buffer_size=10, num_envs=4)
  for i in range(10):
    buf.insert(_step_td(4, float(i), truncated=(i == 5)))
  td = buf.sample(sequence_length=4, batch_size=500)
  assert (~td["truncated"][:, :-1]).all()


def test_truncated_at_final_position_allowed():
  torch.manual_seed(0)
  buf = ReplayBuffer(_schema(), buffer_size=10, num_envs=4)
  for i in range(10):
    buf.insert(_step_td(4, float(i), truncated=(i == 5)))
  td = buf.sample(sequence_length=4, batch_size=500)
  assert td["truncated"][:, -1].any()


def test_warmup_only_filled_starts():
  torch.manual_seed(0)
  buf = ReplayBuffer(_schema(), buffer_size=20, num_envs=4)
  for i in range(5):
    buf.insert(_step_td(4, float(i)))
  td = buf.sample(sequence_length=3, batch_size=500)
  stamps = td["state"][..., 0]
  assert (stamps >= 0).all() and (stamps <= 4).all()


def test_wrap_overwrites_stale_dones():
  # Fill with terminated=True everywhere; overwrite via wrap with terminated=False.
  # Sampled windows must reflect the new (clean) data, not the stale terminations.
  torch.manual_seed(0)
  buf = ReplayBuffer(_schema(), buffer_size=10, num_envs=4)
  for i in range(10):
    buf.insert(_step_td(4, float(i), terminated=True))
  for i in range(10, 20):
    buf.insert(_step_td(4, float(i), terminated=False))
  td = buf.sample(sequence_length=4, batch_size=500)
  stamps = td["state"][..., 0]
  assert (stamps >= 10).all() and (stamps <= 19).all()
  assert (~td["terminated"]).all()


def test_env_stratified_uniform_under_skewed_validity():
  # env 0: terminated every 4th step (75% of L=4 windows blocked).
  # env 1-3: no terminations (all windows valid).
  # Expected env-id histogram: uniform (~12500 per env).
  torch.manual_seed(42)
  E, T, B = 4, 100, 50_000
  buf = ReplayBuffer(_schema(state_dim=2), buffer_size=T, num_envs=E)
  env_ids = torch.arange(E, dtype=torch.float32)
  for i in range(T):
    state = torch.stack([torch.full((E,), float(i)), env_ids], dim=-1)
    term = torch.zeros(E, dtype=torch.bool)
    if i % 4 == 0:
      term[0] = True
    truncated = torch.zeros(E, dtype=torch.bool)
    buf.insert(
      TensorDict(
        {"state": state, "terminated": term, "truncated": truncated},
        batch_size=[E],
      )
    )
  td = buf.sample(sequence_length=4, batch_size=B)
  envs = td["state"][:, 0, 1].long()
  counts = torch.bincount(envs, minlength=E).float()
  expected = B / E
  chi2 = ((counts - expected) ** 2 / expected).sum().item()
  # df=3, alpha=1e-4 critical ≈ 21
  assert chi2 < 21.0, (
    f"chi2={chi2} suggests non-uniform env sampling, counts={counts.tolist()}"
  )


def test_sample_raises_when_no_valid_windows():
  buf = ReplayBuffer(_schema(), buffer_size=10, num_envs=4)
  with pytest.raises(RuntimeError, match="no valid sequences"):
    buf.sample(sequence_length=4, batch_size=10)


def test_sample_raises_on_L_greater_than_buffer_size():
  buf = ReplayBuffer(_schema(), buffer_size=10, num_envs=4)
  for i in range(5):
    buf.insert(_step_td(4, float(i)))
  with pytest.raises(ValueError, match="exceeds buffer_size"):
    buf.sample(sequence_length=11, batch_size=4)


def test_insert_rejects_schema_mismatch():
  buf = ReplayBuffer(_schema(), buffer_size=10, num_envs=4)
  bad = TensorDict(
    {
      "state": torch.zeros(4, 3),
      "terminated": torch.zeros(4, dtype=torch.bool),
    },
    batch_size=[4],
  )
  with pytest.raises(ValueError, match="schema mismatch"):
    buf.insert(bad)


def test_insert_rejects_extra_key():
  buf = ReplayBuffer(_schema(), buffer_size=10, num_envs=4)
  bad = TensorDict(
    {
      "state": torch.zeros(4, 3),
      "terminated": torch.zeros(4, dtype=torch.bool),
      "truncated": torch.zeros(4, dtype=torch.bool),
      "extra": torch.zeros(4),
    },
    batch_size=[4],
  )
  with pytest.raises(ValueError, match="schema mismatch"):
    buf.insert(bad)


def test_learner_schema_round_trip():
  # Mirrors the production schema in supermimic_runner.py: 7 keys with mixed
  # dtypes and shapes. Pins the runner ↔ wm_learner destructure contract.
  schema = {
    "state": FieldSpec(71),
    "action": FieldSpec(29),
    "goal": FieldSpec(8),
    "reward": FieldSpec(5),
    "motion_time": FieldSpec(dtype=torch.long),
    "terminated": FieldSpec(dtype=torch.bool),
    "truncated": FieldSpec(dtype=torch.bool),
  }
  buf = ReplayBuffer(schema, buffer_size=20, num_envs=4)
  step = TensorDict(
    {
      "state": torch.randn(4, 71),
      "action": torch.randn(4, 29),
      "goal": torch.randn(4, 8),
      "reward": torch.randn(4, 5),
      "motion_time": torch.zeros(4, dtype=torch.long),
      "terminated": torch.zeros(4, dtype=torch.bool),
      "truncated": torch.zeros(4, dtype=torch.bool),
    },
    batch_size=[4],
  )
  for _ in range(8):
    buf.insert(step)
  out = buf.sample(sequence_length=8, batch_size=4)
  assert set(out.keys()) == set(schema.keys())
  for name, spec in schema.items():
    expected_shape = (4, 8, spec.dim) if spec.dim else (4, 8)
    assert out[name].shape == expected_shape, f"{name}: {tuple(out[name].shape)}"
    assert out[name].dtype == spec.dtype, f"{name}: {out[name].dtype}"


def test_env_stratified_uniform_after_wrap():
  # Same setup as the chi-squared test but with the buffer wrapped past the
  # write pointer, exercising the full-buffer pointer-guard branch.
  torch.manual_seed(42)
  E, T, B = 4, 100, 50_000
  buf = ReplayBuffer(_schema(state_dim=2), buffer_size=T, num_envs=E)
  env_ids = torch.arange(E, dtype=torch.float32)
  for i in range(T + 37):  # +37 → step lands mid-buffer post-wrap
    state = torch.stack([torch.full((E,), float(i)), env_ids], dim=-1)
    term = torch.zeros(E, dtype=torch.bool)
    if i % 4 == 0:
      term[0] = True
    truncated = torch.zeros(E, dtype=torch.bool)
    buf.insert(
      TensorDict(
        {"state": state, "terminated": term, "truncated": truncated},
        batch_size=[E],
      )
    )
  td = buf.sample(sequence_length=4, batch_size=B)
  envs = td["state"][:, 0, 1].long()
  counts = torch.bincount(envs, minlength=E).float()
  expected = B / E
  chi2 = ((counts - expected) ** 2 / expected).sum().item()
  assert chi2 < 21.0, f"chi2={chi2}, counts={counts.tolist()}"


def test_sample_one_valid_env_with_oversized_batch():
  # Only env 0 has valid windows (envs 1..3 have terminations everywhere in
  # the interior). batch_size > num_envs exercises multinomial replacement
  # over a single nonzero weight; all samples must come from env 0.
  torch.manual_seed(0)
  E = 4
  buf = ReplayBuffer(_schema(state_dim=2), buffer_size=20, num_envs=E)
  env_ids = torch.arange(E, dtype=torch.float32)
  for i in range(20):
    state = torch.stack([torch.full((E,), float(i)), env_ids], dim=-1)
    term = torch.ones(E, dtype=torch.bool)
    term[0] = False  # only env 0 is sampleable
    buf.insert(
      TensorDict(
        {
          "state": state,
          "terminated": term,
          "truncated": torch.zeros(E, dtype=torch.bool),
        },
        batch_size=[E],
      )
    )
  td = buf.sample(sequence_length=3, batch_size=64)  # 64 > E=4
  assert (td["state"][:, 0, 1] == 0).all()


def test_init_rejects_missing_episode_keys():
  with pytest.raises(ValueError, match="terminated"):
    ReplayBuffer({"state": FieldSpec(3)}, buffer_size=10, num_envs=4)


def test_wrap_aware_sampling_reaches_all_time_windows():
  torch.manual_seed(0)
  T, L = 10, 4
  buf = ReplayBuffer(_schema(), buffer_size=T, num_envs=4)
  for i in range(25):
    buf.insert(_step_td(4, float(i)))
  td = buf.sample(sequence_length=L, batch_size=4000)
  stamps = td["state"][..., 0]
  diffs = stamps[:, 1:] - stamps[:, :-1]
  assert (diffs == 1.0).all(), "non-contiguous window sampled"
  expected = {15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0}
  actual = set(stamps[:, 0].unique().tolist())
  missing = expected - actual
  assert not missing, f"missing start_stamps {missing}"


def test_insert_rejects_wrong_batch_size():
  buf = ReplayBuffer(_schema(), buffer_size=10, num_envs=4)
  for bad in [
    TensorDict(
      {
        "state": torch.zeros(2, 3),
        "terminated": torch.zeros(2, dtype=torch.bool),
        "truncated": torch.zeros(2, dtype=torch.bool),
      },
      batch_size=[2],
    ),  # wrong num_envs
    TensorDict(
      {
        "state": torch.zeros(4, 1, 3),
        "terminated": torch.zeros(4, 1, dtype=torch.bool),
        "truncated": torch.zeros(4, 1, dtype=torch.bool),
      },
      batch_size=[4, 1],
    ),  # already-unsqueezed
  ]:
    with pytest.raises(ValueError, match="batch_size"):
      buf.insert(bad)
