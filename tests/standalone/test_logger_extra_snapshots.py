import pytest
import torch

from supermimic.utils.logger import Logger


@pytest.mark.parametrize("key", ["episode", "log"])
def test_metrics_capture_each_step_before_buffers_are_reused(key):
  logger = Logger(None, 1, 100, "cpu")
  value = torch.tensor([1.0], requires_grad=True)
  metrics = {"score": value, "count": 1.0}
  logger.process_env_step(
    torch.zeros(1), torch.zeros(1, dtype=torch.bool), {key: metrics}
  )
  with torch.no_grad():
    value.fill_(3.0)
  metrics["count"] = 3.0
  logger.process_env_step(
    torch.zeros(1), torch.zeros(1, dtype=torch.bool), {key: metrics}
  )
  metrics.clear()
  assert logger._episode_extras() == {"score": 2.0, "count": 2.0}
  assert not logger.ep_extras[0]["score"].requires_grad


def test_episode_takes_precedence_over_log():
  logger = Logger(None, 1, 100, "cpu")
  logger.process_env_step(
    torch.zeros(1),
    torch.zeros(1, dtype=torch.bool),
    {"episode": {"score": 2.0}, "log": {"score": 9.0}},
  )
  assert logger._episode_extras() == {"score": 2.0}
