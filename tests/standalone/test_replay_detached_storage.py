import torch
from tensordict import TensorDict

from supermimic.replay_buffer import FieldSpec, ReplayBuffer


def test_replay_does_not_retain_source_graphs_after_wrap():
  buf = ReplayBuffer(
    {
      "state": FieldSpec(2),
      "terminated": FieldSpec(dtype=torch.bool),
      "truncated": FieldSpec(dtype=torch.bool),
    },
    3,
    1,
  )
  leaf = torch.ones(1, 2, requires_grad=True)
  for step in range(7):
    state = leaf * (step + 1)
    td = TensorDict(
      {
        "state": state,
        "terminated": torch.tensor([False]),
        "truncated": torch.tensor([False]),
      },
      [1],
    )
    buf.insert(td)
    assert not buf._storage["state"].requires_grad
    assert buf._storage["state"].grad_fn is None
    assert not buf.sample(1, 8)["state"].requires_grad
  state.sum().backward()
  torch.testing.assert_close(leaf.grad, torch.full_like(leaf, 7.0))
  assert set(buf.sample(1, 256)["state"].flatten().tolist()) == {5.0, 6.0, 7.0}
