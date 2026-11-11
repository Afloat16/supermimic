from unittest.mock import patch

import torch

from supermimic.utils.logger import Logger


def test_metrics_preserve_weighting_missing_keys_dtypes_and_empty_values():
  log = Logger(None, 1, 10, "cpu")
  log.ep_extras = [
    {"a": torch.tensor([1, 2]), "b": 10, "empty": torch.tensor([])},
    {"a": torch.tensor([[3.0, 6.0]], dtype=torch.float64)},
    {"b": 20},
  ]
  assert log._episode_extras() == {"a": 3.0, "b": 15.0}


def test_long_metric_batches_copy_each_values_tensor_only_once():
  log = Logger(None, 1, 10, "cpu")
  log.ep_extras = [{"reward": torch.tensor([float(i)])} for i in range(1024)]
  with patch("torch.cat", wraps=torch.cat) as concatenate:
    means = log._episode_extras()
  assert means["reward"] == 511.5
  copied = sum(
    sum(t.numel() for t in call.args[0]) for call in concatenate.call_args_list
  )
  assert copied <= 2048
