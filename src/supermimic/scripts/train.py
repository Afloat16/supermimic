"""Script to train World Model using SuperMimicRunner."""

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
import wandb
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

import supermimic.tasks  # noqa: F401
from supermimic.models import Policy, WorldModel
from supermimic.replay_buffer import ReplayBuffer
from supermimic.runners.learner import PolicyLearner, WorldModelLearner
from supermimic.runners.supermimic_runner import SuperMimicRunner


@dataclass
class TrainConfig:
  # Experiment & Environment
  task_id: str = "SuperMimic-WorldModel-G1"
  num_envs: int = 1024
  headless: bool = True
  device: str | None = None
  wandb_project: str = "supermimic"
  wandb_group: str | None = None
  registry_name: str | None = None

  # Training Loop & Buffer
  max_iterations: int = 100000
  buffer_size: int = 1000
  collection_buffer_ratio: float = 0.05

  # World Model Learning
  model_lr: float = 1e-4
  num_batches_model: int = 100
  batch_size_model: int = 2048
  window_size: int = 8

  # Loss Weights (Bryson's Rule)
  nominal_error_root_pos: float = 0.1
  nominal_error_root_ori: float = 0.1
  nominal_error_root_lin_vel: float = 1.0
  nominal_error_root_ang_vel: float = 3.14
  nominal_error_joint_pos: float = 0.1
  nominal_error_joint_vel: float = 3.14

  # Policy Learning
  policy_lr: float = 1e-4
  num_batches_policy: int = 50
  batch_size_policy: int = 1024
  policy_noise: float = 0.1
  horizon: int = 32


def main():
  args = tyro.cli(TrainConfig)
  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Load Environment
  print(f"[INFO] Loading environment: {args.task_id}")
  env_cfg = load_env_cfg(args.task_id, play=True)

  # Check for tracking task and registry
  if args.registry_name:
    registry_name = args.registry_name
    if ":" not in registry_name:
      registry_name = registry_name + ":latest"

    # Update args with normalized name for Runner
    args.registry_name = registry_name

    print(f"[INFO] Downloading motion from registry: {registry_name}")
    api = wandb.Api()
    artifact = api.artifact(registry_name)
    motion_file_path = str(Path(artifact.download()) / "motion.npz")

    env_cfg.commands["motion"].motion_file = motion_file_path
    print(f"[INFO] Using motion file: {motion_file_path}")

  # Populate motion_file in args for PolicyLearner
  args.motion_file = env_cfg.commands["motion"].motion_file

  env_cfg.scene.num_envs = args.num_envs

  if args.headless:
    env_cfg.viewer.eye = (0.0, 0.0, 0.0)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env)

  # Initialize Dimensions
  obs, _ = env.reset()
  state_obs = obs["full_state"] if hasattr(obs, "keys") else obs
  num_state = state_obs.shape[-1]
  goal_obs = obs["goal"] if hasattr(obs, "keys") else obs
  num_goals = goal_obs.shape[-1]
  num_actions = env.num_actions

  # Initialize World Model & Learner
  print(f"[INFO] State Dim: {num_state}, Goal Dim: {num_goals}")
  wm = WorldModel(num_state, num_actions, num_goals).to(device)

  args.dt = env.unwrapped.step_dt
  wm_learner = WorldModelLearner(wm, args, device)

  # Initialize Policy & Learner
  print("[INFO] Training new policy from scratch (State+Goal)...")
  policy_input_dim = num_state + num_goals
  policy = Policy(policy_input_dim, num_actions, normalization=True).to(device)
  policy_learner = PolicyLearner(policy, wm, args, device)

  # Buffer: [state, action, goal, reward, done]
  replay_buffer = ReplayBuffer(
    dim=[num_state, num_actions, num_goals, 1, 1],
    buffer_size=args.buffer_size,
    device=device,
  )

  # Init Runner
  runner = SuperMimicRunner(
    env=env,
    replay_buffer=replay_buffer,
    world_model_learner=wm_learner,
    policy_learner=policy_learner,
    cfg=args,
  )

  runner.run()


if __name__ == "__main__":
  main()
