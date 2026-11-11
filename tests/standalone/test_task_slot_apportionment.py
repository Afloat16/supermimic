from types import SimpleNamespace

import pytest

from supermimic.models import MultiEncoderAE


def actor(mix):
  return MultiEncoderAE(
    2,
    1,
    [SimpleNamespace(name=n, goal_dim=1) for n in mix],
    mix,
    latent_dim=1,
    hidden_dims=(2,),
    latent_bc_weight=0,
    history_length=1,
  )


@pytest.mark.parametrize(
  "mix",
  [
    {"a": 0.5, "b": 0.5, "c": 0.0},
    {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
    {"a": 0.0, "b": 0.3, "c": 0.3, "d": 0.4},
  ],
)
def test_task_ranges_partition_batch_without_disabled_slots(mix):
  model = actor(mix)
  for size in range(1, 33):
    ranges = model.slot_ranges(size)
    assert ranges[0][0] == 0 and ranges[-1][1] == size
    for i, (lo, hi) in enumerate(ranges):
      assert 0 <= lo <= hi <= size
      if list(mix.values())[i] == 0:
        assert lo == hi
      if i:
        assert ranges[i - 1][1] == lo


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_fractions_fail_at_configuration(value):
  with pytest.raises(ValueError):
    actor({"a": value, "b": 0.0})


@pytest.mark.parametrize("weight", [0.4999996, 0.5000004])
def test_accepted_rounding_tolerance_never_enables_zero_weight_task(weight):
  model = actor({"a": weight, "b": weight, "c": 0.0})
  ranges = model.slot_ranges(1_000_001)
  assert ranges[-1] == (1_000_001, 1_000_001)
  assert ranges[0][1] == ranges[1][0]
