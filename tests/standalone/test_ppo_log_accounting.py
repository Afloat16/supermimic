import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


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


@pytest.mark.parametrize("interval", [1, 3, 10])
def test_logs_account_for_all_collected_steps_and_warmup(interval):
  clock = iter(range(100))
  run = production_method(
    "runners/ppo_runner.py", "PpoRunner", "run", {"time": NS(time=lambda: next(clock))}
  )
  rows = []
  buffer = NS(num_transitions=0, sample=lambda *a: None)

  def collect():
    buffer.num_transitions += 1
    return 10

  runner = NS(
    cfg=NS(
      max_iterations=5,
      log_interval=interval,
      save_interval=100,
      wm=NS(horizon=2, num_batches=1, batch_size=1),
    ),
    _create_log_dir=lambda *a: None,
    _init_wandb=lambda *a, **kw: None,
    _seed_sim_history=lambda: None,
    collect_data=collect,
    replay_buffer=buffer,
    world_model_learner=NS(update=lambda data: {}),
    imagine=lambda: {},
    logger=NS(log=lambda **kw: rows.append(kw)),
    _cleanup=lambda: None,
  )
  run(runner)
  assert sum(row["num_collected_steps"] for row in rows) == 50
  assert sum(row["collect_time"] for row in rows) == 5
  assert sum(row["wm_learn_time"] for row in rows) == 4
  assert sum(row["policy_learn_time"] for row in rows) == 4
  assert rows[-1]["it"] == 4
