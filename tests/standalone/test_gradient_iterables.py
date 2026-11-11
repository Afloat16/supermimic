import pytest
import torch

from supermimic.learners.learner_utils import _optimize_step


@pytest.mark.parametrize("as_generator", [False, True])
def test_norm_and_element_clipping_share_parameters(as_generator):
  params = [torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(1))]
  optimizer = torch.optim.SGD(params, lr=1.0)
  iterable = (p for p in params) if as_generator else params
  norm = _optimize_step(
    [optimizer], sum(p.sum() * 100 for p in params), iterable, 1.0, 0.1
  )
  assert norm > 1.0
  for p in params:
    torch.testing.assert_close(p, torch.full_like(p, 0.9))


def test_generator_still_supports_norm_only():
  p = torch.nn.Parameter(torch.ones(1))
  _optimize_step([torch.optim.SGD([p], lr=1.0)], p.sum() * 10, iter([p]), 0.5)
  torch.testing.assert_close(p, torch.tensor([0.5]))
