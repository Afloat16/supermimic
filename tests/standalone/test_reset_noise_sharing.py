from types import SimpleNamespace

import torch
from tensordict import TensorDict

from supermimic.envs.motion_tracking_env import MotionTrackingEnv


def env_for_reset():
  env = MotionTrackingEnv.__new__(MotionTrackingEnv)
  env.num_envs, env.history_length, env.device = 3, 4, "cpu"
  env._update_goal = lambda: None
  env._synth_sensor = lambda: torch.tensor([[100.0], [200.0], [300.0]])
  env.representation = SimpleNamespace(
    add_proprio_noise=lambda x: x + torch.arange(x.numel()).reshape(x.shape)
  )
  return env


def test_all_padding_matches_the_single_current_noise_sample():
  env = env_for_reset()
  masks = torch.tensor(
    [
      [True, True, True, False],
      [True, True, False, False],
      [False, False, False, False],
    ]
  )
  batch = TensorDict(
    {
      "motion_time": torch.zeros(3, dtype=torch.long),
      "sensor_window": torch.zeros(3, 4, 1),
      "pad_mask": masks,
    },
    [3],
  )
  env._on_reset(batch)
  history = env._sensor_history()
  torch.testing.assert_close(history[0], torch.full((4, 1), 100.0))
  torch.testing.assert_close(history[1, :3], torch.full((3, 1), 5.0))
  torch.testing.assert_close(history[2, :3, 0], torch.tensor([6.0, 7.0, 8.0]))
  torch.testing.assert_close(history[:, -1, 0], torch.tensor([100.0, 200.0, 300.0]))
