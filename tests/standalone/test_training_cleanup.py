import ast
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS

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


@pytest.mark.parametrize("fail_at", [1, 2])
def test_training_failure_drains_uploads_and_closes_resources(tmp_path, fail_at):
  events = []
  count = 0

  def step():
    nonlocal count
    count += 1
    if count == fail_at:
      raise RuntimeError("training failure")
    return 1

  def upload(paths):
    torch.load(paths[0], weights_only=True)
    events.append("upload")

  run = production_method(
    "runners/supermimic_runner.py",
    "SuperMimicRunner",
    "run",
    {
      "time": time,
      "torch": torch,
      "ThreadPoolExecutor": ThreadPoolExecutor,
      "Ratio": lambda ratio: lambda step: 0,
      "wandb": NS(run=NS(name="test")),
      "log_model_artifact": upload,
    },
  )
  runner = NS(
    cfg=NS(
      max_iterations=3,
      replay_ratio=0,
      log_interval=10,
      save_interval=10,
      wm=NS(num_batches=1, batch_size=1, horizon=2),
    ),
    env=NS(num_envs=1, reset=lambda: None),
    _create_log_dir=lambda: tmp_path,
    _init_wandb=lambda path: None,
    _seed_sim_history=lambda: None,
    _cleanup=lambda: events.append("cleanup"),
    step=step,
    policy_learner=NS(actor=torch.nn.Linear(1, 1)),
    _wm=NS(bundle=lambda: {"a": torch.ones(1)}),
    _representation=NS(actor_obs_mode="rep_state"),
    logger=NS(log=lambda **kw: None),
  )
  with pytest.raises(RuntimeError, match="training failure"):
    run(runner)
  assert events == (["upload", "cleanup"] if fail_at == 2 else ["cleanup"])
