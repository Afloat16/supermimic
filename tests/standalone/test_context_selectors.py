from types import SimpleNamespace as NS

import pytest
import torch

from supermimic.sim_context import CoMOffsetSpec, FootFrictionSpec


def env_for(pattern, kind):
  asset = NS(
    geom_names=["foot_extra", "foot"],
    body_names=["world", "pelvis"],
    indexing=NS(geom_ids=torch.tensor([3, 4]), body_ids=torch.tensor([2, 5])),
  )
  asset_cfg = NS(name="robot", geom_names=pattern, body_names=pattern)
  event = NS(
    params={
      "asset_cfg": asset_cfg,
      "ranges": {0: (0.1, 1.0), 1: (-0.1, 0.1), 2: (-0.1, 0.1)},
    }
  )
  return NS(
    cfg=NS(events={kind: event}),
    scene={"robot": asset},
    sim=NS(
      model=NS(
        geom_friction=torch.arange(18).reshape(1, 6, 3).float(),
        body_ipos=torch.arange(18).reshape(1, 6, 3).float(),
      )
    ),
  )


@pytest.mark.parametrize("pattern", ["foot", ["f.*t"], ("foot",)])
def test_friction_uses_the_same_full_name_regex_selection_as_the_event(pattern):
  env = env_for(pattern, "foot_friction")
  spec = FootFrictionSpec()
  spec.bind(env)
  torch.testing.assert_close(spec.read(env), torch.tensor([[12.0]]))


@pytest.mark.parametrize("pattern", ["pelvis", ["pelv.*"], ("pelvis",)])
def test_com_accepts_single_name_and_regex_selectors(pattern):
  env = env_for(pattern, "base_com")
  spec = CoMOffsetSpec()
  spec.bind(env)
  torch.testing.assert_close(spec.read(env), torch.tensor([[15.0, 16.0, 17.0]]))


def test_com_rejects_a_selector_that_matches_multiple_bodies():
  with pytest.raises(ValueError, match="exactly 1"):
    CoMOffsetSpec().bind(env_for(".*", "base_com"))
