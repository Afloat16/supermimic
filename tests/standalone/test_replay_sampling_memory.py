import torch
from tensordict import TensorDict

from supermimic.replay_buffer import FieldSpec, ReplayBuffer


def filled_buffer():
  buf = ReplayBuffer(
    {
      "stamp": FieldSpec(),
      "env": FieldSpec(),
      "terminated": FieldSpec(dtype=torch.bool),
      "truncated": FieldSpec(dtype=torch.bool),
    },
    128,
    2,
  )
  for step in range(140):
    buf.insert(
      TensorDict(
        {
          "stamp": torch.full((2,), float(step)),
          "env": torch.arange(2).float(),
          "terminated": torch.tensor([step % 17 == 0, step % 3 == 0]),
          "truncated": torch.zeros(2, dtype=torch.bool),
        },
        [2],
      )
    )
  return buf


def test_sampling_avoids_batch_by_capacity_cdf_allocation():
  buf = filled_buffer()
  batch = 4096
  with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
  ) as profile:
    result = buf.sample(3, batch)
  largest = max(e.cpu_memory_usage for e in profile.events())
  assert largest < batch * buf.buffer_size * 8
  assert result.batch_size == torch.Size([batch, 3])


def test_wrapped_sampling_preserves_validity_and_env_balance():
  torch.manual_seed(42)
  result = filled_buffer().sample(3, 20000)
  torch.testing.assert_close(
    result["stamp"][:, 1:] - result["stamp"][:, :-1], torch.ones(20000, 2)
  )
  assert result["stamp"].min() >= 12
  assert result["stamp"].max() <= 139
  assert not result["terminated"][:, :-1].any()
  assert 0.47 < result["env"][:, 0].mean() < 0.53
