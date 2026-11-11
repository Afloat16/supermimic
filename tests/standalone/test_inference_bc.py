from types import SimpleNamespace

import torch
from tensordict import TensorDict

from supermimic.models import MultiEncoderAE


def make_actor():
  model = MultiEncoderAE(
    2,
    1,
    [
      SimpleNamespace(name="motion", goal_dim=2),
      SimpleNamespace(name="velocity", goal_dim=2),
    ],
    {"motion": 0.5, "velocity": 0.5},
    2,
    (4,),
    0.1,
    1,
  )
  obs = TensorDict(
    {
      "history": torch.randn(6, 1, 2),
      "goal_motion": torch.randn(6, 2),
      "goal_velocity": torch.randn(6, 2),
    },
    [6],
    device="cpu",
  )
  return model, obs


def test_collection_skips_extra_encoder_work_without_changing_actions():
  model, obs = make_actor()
  model.train()
  calls = []
  hooks = [
    encoder.register_forward_hook(lambda *args: calls.append(1))
    for encoder in model.encoders.values()
  ]
  with torch.inference_mode():
    actual = model(obs)
  assert len(calls) == 2
  assert torch.count_nonzero(model.auxiliary_loss()) == 0
  for hook in hooks:
    hook.remove()
  model.eval()
  with torch.no_grad():
    expected = model(obs)
  torch.testing.assert_close(actual, expected)


def test_training_keeps_behavior_cloning_gradients():
  model, obs = make_actor()
  model.train()
  model(obs)
  loss = model.auxiliary_loss().sum()
  assert loss.item() > 0
  loss.backward()
  assert any(
    p.grad is not None and p.grad.abs().sum() > 0
    for p in model.encoders["velocity"].parameters()
  )
  assert all(p.grad is None for p in model.encoders["motion"].parameters())
  assert all(p.grad is None for p in model.decoder.parameters())
