import ast
from pathlib import Path

import pytest
import torch


def production_method(path, cls_name, name, namespace):
  source = Path(__file__).resolve().parents[2] / "src/supermimic" / path
  tree = ast.parse(source.read_text())
  cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
  method = next(
    n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
  )
  future = ast.ImportFrom(
    module="__future__", names=[ast.alias(name="annotations")], level=0
  )
  module = ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[]))
  exec(compile(module, str(source), "exec"), namespace)
  return namespace[name]


@pytest.mark.parametrize(
  "step,expected", [(-100, 0), (-1, 0), (0, 0), (1, 1), (2, 2), (100, 2)]
)
def test_motion_deploy_clamps_the_command_and_every_motion_output_together(
  step, expected
):
  forward = production_method(
    "runners/exporter.py", "_MotionDeployWrapper", "forward", {"torch": torch}
  )

  class Wrapper(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self._time_step_total = 3
      self._has_command = True
      self.actor = torch.nn.Identity()
      for name in (
        "_command_table",
        "_joint_pos",
        "_joint_vel",
        "_body_pos_w",
        "_body_quat_w",
        "_body_lin_vel_w",
        "_body_ang_vel_w",
      ):
        self.register_buffer(name, torch.arange(3).float().unsqueeze(1))

  Wrapper.forward = forward
  wrapper = Wrapper()
  for model in (
    wrapper,
    torch.jit.trace(wrapper, (torch.zeros(1, 1), torch.zeros(1, 1))),
  ):
    output = model(torch.zeros(1, 1), torch.tensor([[float(step)]]))
    assert output[0][0, -1] == expected
    assert all(x.item() == expected for x in output[1:])
