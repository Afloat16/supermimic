import ast
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch.nn as nn

from supermimic.sim_context import EncoderBiasSpec, SimContext


def load_production(path, names, namespace):
  source = Path(__file__).resolve().parents[2] / "src/supermimic" / path
  tree = ast.parse(source.read_text())
  nodes = [
    n
    for n in tree.body
    if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
  ]
  future = ast.ImportFrom(
    module="__future__", names=[ast.alias(name="annotations")], level=0
  )
  module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
  exec(compile(module, str(source), "exec"), namespace)
  return namespace


def representation():
  namespace = {"torch": torch, "nn": nn, "ABC": ABC, "abstractmethod": abstractmethod}
  load_production(
    "utils/gencoord_helpers.py",
    {"rep_state_dim_from_num_joints", "rep_state_offsets", "decompose_rep_state"},
    namespace,
  )
  load_production(
    "representations.py", {"StateRepresentation", "GenCoordRepresentation"}, namespace
  )
  return namespace["GenCoordRepresentation"](2, torch.tensor([1.0, 2.0]))


def test_gencoord_sensor_includes_the_real_encoder_bias_and_keeps_gradients():
  rep = representation()
  spec = EncoderBiasSpec()
  spec.dim = 2
  ctx = SimContext([spec], [(-0.1, 0.1), (-0.1, 0.1)])
  env = NS(
    scene={"robot": NS(data=NS(soft_joint_pos_limits=torch.zeros(1, 2, 2)))},
    command_manager=NS(get_term=lambda name: NS(motion=NS(fk=None))),
  )
  rep.bind(env, ctx)
  joints = torch.tensor([[3.0, 5.0, 7.0, 9.0]], requires_grad=True)
  obs = rep.compute_wm_obs(
    torch.zeros(1, 14), torch.zeros(1, 7), joints, torch.tensor([[0.5, -0.5]])
  )
  torch.testing.assert_close(obs[:, 6:8], torch.tensor([[2.05, 2.95]]))
  torch.testing.assert_close(obs[:, 8:10], joints[:, 2:])
  obs.sum().backward()
  torch.testing.assert_close(joints.grad, torch.ones_like(joints))


def test_no_bias_context_preserves_the_unbiased_sensor():
  rep = representation()
  obs = rep.compute_wm_obs(
    torch.zeros(1, 14),
    torch.zeros(1, 7),
    torch.tensor([[3.0, 5.0, 7.0, 9.0]]),
    torch.zeros(1, 0),
  )
  torch.testing.assert_close(obs[:, 6:], torch.tensor([[2.0, 3.0, 7.0, 9.0]]))


def test_default_joint_positions_follow_model_conversion_without_changing_checkpoint_keys():
  rep = representation().double()
  assert rep._default_joint_pos.dtype == torch.float64
  assert "_default_joint_pos" not in rep.state_dict()
