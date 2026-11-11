import ast
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def run_method():
  # Load the production loop without importing the unavailable simulator package.
  source = (
    Path(__file__).resolve().parents[2] / "src/supermimic/runners/supermimic_runner.py"
  )
  tree = ast.parse(source.read_text())
  cls = next(
    n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SuperMimicRunner"
  )
  method = next(
    n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run"
  )
  module = ast.Module(
    body=[
      ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
      ),
      method,
    ],
    type_ignores=[],
  )
  return compile(ast.fix_missing_locations(module), str(source), "exec")


@pytest.mark.parametrize("iterations", [1, 2, 4, 5, 7])
def test_normal_exit_preserves_the_last_completed_step(tmp_path, iterations):
  actor = torch.nn.Linear(1, 1, bias=False)
  critic = torch.nn.Linear(1, 1, bias=False)
  completed = 0
  uploaded = []

  def step():
    nonlocal completed
    completed += 1
    with torch.no_grad():
      actor.weight.fill_(completed)
      critic.weight.fill_(completed)
    return 1

  def upload(paths):
    uploaded.append(torch.load(paths[0], weights_only=True)["step"].item())

  runner = SimpleNamespace(
    cfg=SimpleNamespace(
      max_iterations=iterations,
      replay_ratio=0,
      save_interval=4,
      log_interval=8,
      wm=SimpleNamespace(num_batches=1, batch_size=1, horizon=1),
    ),
    env=SimpleNamespace(num_envs=1, reset=lambda: None),
    _create_log_dir=lambda: tmp_path,
    _init_wandb=lambda path: None,
    _seed_sim_history=lambda: None,
    _cleanup=lambda: None,
    step=step,
    _wm=SimpleNamespace(bundle=lambda: {"step": torch.tensor(completed)}),
    policy_learner=SimpleNamespace(actor=actor, critic=critic),
    logger=SimpleNamespace(log=lambda **kwargs: None),
    _representation=SimpleNamespace(actor_obs_mode="rep_state"),
  )
  namespace = {
    "torch": torch,
    "time": time,
    "ThreadPoolExecutor": ThreadPoolExecutor,
    "Ratio": lambda ratio: lambda step: 0,
    "wandb": SimpleNamespace(run=SimpleNamespace(name="test")),
    "log_model_artifact": upload,
  }
  exec(run_method(), namespace)
  namespace["run"](runner)
  assert uploaded[-1] == iterations
  assert uploaded == sorted(set(uploaded))
  for file in ("actor.pt", "critic.pt"):
    state = torch.load(tmp_path / file, weights_only=True)
    torch.testing.assert_close(state["weight"], torch.full((1, 1), float(iterations)))
