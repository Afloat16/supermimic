import pytest
import torch
from tensordict import TensorDict
from torch.utils._python_dispatch import TorchDispatchMode

from supermimic.replay_buffer import FieldSpec, ReplayBuffer


def make_buffer(capacity, steps):
  buf = ReplayBuffer(
    {
      "value": FieldSpec(),
      "terminated": FieldSpec(dtype=torch.bool),
      "truncated": FieldSpec(dtype=torch.bool),
    },
    capacity,
    4,
  )
  for step in range(steps):
    buf.insert(
      TensorDict(
        {
          "value": 1000 * torch.arange(4) + step,
          "terminated": torch.ones(4, dtype=torch.bool),
          "truncated": torch.zeros(4, dtype=torch.bool),
        },
        [4],
      )
    )
  return buf


@pytest.mark.parametrize("capacity,steps", [(11, 5), (11, 27)])
def test_single_frames_preserve_wrapped_values_and_environment_balance(capacity, steps):
  torch.manual_seed(37)
  samples = make_buffer(capacity, steps).sample(1, 12000)
  assert samples.batch_size == torch.Size([12000, 1])
  values = samples["value"][:, 0].long()
  envs, ages = values // 1000, values % 1000
  assert ages.min() == max(0, steps - capacity)
  assert ages.max() == steps - 1
  counts = torch.bincount(envs, minlength=4)
  assert counts.min() > 2700 and counts.max() < 3300
  assert samples["terminated"].all()


def test_single_frame_sampling_temporary_size_does_not_scale_with_capacity():
  class Sizes(TorchDispatchMode):
    largest = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
      result = func(*args, **(kwargs or {}))
      leaves, _ = torch.utils._pytree.tree_flatten(result)
      self.largest = max(
        [self.largest, *(x.numel() for x in leaves if isinstance(x, torch.Tensor))]
      )
      return result

  buf = make_buffer(2048, 3)
  mode = Sizes()
  with mode:
    sample = buf.sample(1, 16)
  assert sample.numel() == 16
  assert mode.largest <= 64


def test_empty_buffer_has_a_clear_sampling_error():
  with pytest.raises(RuntimeError, match="no valid sequences"):
    make_buffer(8, 0).sample(1, 2)
