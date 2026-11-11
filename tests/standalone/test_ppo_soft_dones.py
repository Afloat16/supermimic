import ast
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict


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


def test_ppo_imagination_preserves_fractional_terminal_probabilities_in_gae():
  namespace = {
    "torch": torch,
    "TensorDict": TensorDict,
    "RolloutStorage": RolloutStorage,
    "PPO": PPO,
    "MLPModel": MLPModel,
  }
  init = production_method(
    "runners/ppo_runner.py", "PpoRunner", "_init_policy", namespace
  )
  imagine = production_method(
    "runners/ppo_runner.py", "PpoRunner", "imagine", namespace
  )
  build = production_method(
    "runners/supermimic_runner.py", "BaseRunner", "_build_mlp", namespace
  )
  obs = TensorDict({"actor": torch.zeros(4, 2), "critic": torch.zeros(4, 2)}, [4])
  probabilities = torch.tensor([0.0, 0.25, 0.75, 1.0])
  cfg = NS(
    actor_hidden_dims=(4,),
    critic_hidden_dims=(4,),
    activation="elu",
    init_noise_std=1.0,
    batch_size=4,
    num_steps_per_env=1,
    num_learning_epochs=1,
    num_mini_batches=1,
    clip_param=0.2,
    gamma=0.9,
    lam=0.95,
    value_loss_coef=1.0,
    entropy_coef=0.01,
    learning_rate=0.01,
    max_grad_norm=1.0,
    desired_kl=0.01,
  )
  runner = NS(
    cfg=cfg,
    device="cpu",
    _num_actions=1,
    _policy_input_dims=lambda: (2, 2),
    _create_wm_env=lambda batch: NS(
      reset=lambda: obs, step=lambda action: (obs, torch.ones(4), probabilities, {})
    ),
  )
  runner._build_mlp = lambda *a, **kw: build(runner, *a, **kw)
  init(runner)
  for parameter in runner.alg.critic.parameters():
    parameter.data.zero_()
  linear = [m for m in runner.alg.critic.modules() if isinstance(m, torch.nn.Linear)][
    -1
  ]
  linear.bias.data.fill_(4.0)
  runner.alg.update = lambda: {}
  imagine(runner)
  torch.testing.assert_close(runner.alg.storage.dones[0, :, 0], probabilities)
  torch.testing.assert_close(
    runner.alg.storage.returns[0, :, 0], torch.tensor([4.6, 3.7, 1.9, 1.0])
  )
