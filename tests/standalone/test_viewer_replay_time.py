import ast
from pathlib import Path
from types import SimpleNamespace as NS

import torch
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


def test_world_model_and_simulator_use_the_same_replay_command_frame():
  class Representation:
    actor_obs_mode = "rep_state"
    num_bodies = 1

    def extract_aux_state(self, state):
      return torch.zeros(1, 7)

    def represent(self, state):
      return torch.zeros(1, 2)

    def compute_actor_sensor(self, *args, **kwargs):
      return torch.zeros(1, 2)

    def compute_wm_goal(self, *args):
      return torch.zeros(1, 2), torch.zeros(1, 2)

    def joint_state(self, state):
      return torch.zeros(1, 2)

    def reconstruct_full_state(self, *args):
      return torch.zeros(1, 13)

  class Actor:
    history_length = 1

    def __init__(self):
      self.goals = []

    def __call__(self, obs):
      self.goals.append(obs["goal_velocity"].clone())
      return torch.zeros(1, 1)

  motion = NS(time_steps=torch.tensor([0]), motion=object())
  obs = {"full_state": torch.zeros(1, 13), "goal": torch.zeros(1, 2)}

  def step(actions):
    motion.time_steps.add_(1)
    return obs, torch.zeros(1), torch.zeros(1, dtype=torch.bool), {}

  def actor_obs(goals, history):
    return TensorDict(
      {"history": history, **{"goal_" + k: v for k, v in goals.items()}}, [1]
    )

  ns = {
    "torch": torch,
    "MaxCoordRepresentation": Representation,
    "build_actor_obs_td": actor_obs,
  }
  predict = production_method(
    "scripts/viewers.py", "SuperMimicViewerMixin", "step_and_predict", ns
  )
  goal = production_method(
    "scripts/viewers.py", "SuperMimicViewerMixin", "_build_actor_goal", ns
  )
  history = production_method(
    "scripts/viewers.py", "SuperMimicViewerMixin", "_advance_history", ns
  )
  actor = Actor()
  viewer = NS(
    _is_paused=False,
    env=NS(
      get_observations=lambda: obs,
      step=step,
      unwrapped=NS(command_manager=NS(get_term=lambda name: motion)),
    ),
    representation=Representation(),
    policy=actor,
    curr_wm_aux=None,
    step_in_rollout=0,
    last_actions=None,
    sim_context=NS(read=lambda env: torch.zeros(1, 1)),
    dynamics=lambda *a: (torch.zeros(1, 7), torch.zeros(1, 2)),
    current_task="velocity",
    task_specs={"velocity": NS(build_inference_goal_obs=lambda raw: raw)},
    _replay_mode={"velocity": True},
    _replay_goal_tables={"velocity": torch.tensor([[10.0], [20.0], [30.0]])},
    _step_count=0,
    _stats_steps=0,
    num_steps_rollout=10,
    pause=lambda: None,
  )

  # The normal initializer supplies this dimensional attribute on Dynamics.
  class Dynamics:
    num_actions = 1

    def __call__(self, *args):
      return torch.zeros(1, 7), torch.zeros(1, 2)

  viewer.dynamics = Dynamics()
  viewer._build_actor_goal = lambda *args: goal(viewer, *args)
  viewer._advance_history = lambda *args: history(viewer, *args)
  assert predict(viewer), getattr(viewer, "_last_error", None)
  assert [x.item() for x in actor.goals] == [10.0, 10.0]
