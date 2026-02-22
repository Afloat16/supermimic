"""Script to train World Model using SuperMimicRunner."""

from dataclasses import dataclass
from pathlib import Path

import mjlab.tasks  # noqa: F401
import torch
import tyro
import wandb
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.torch import configure_torch_backends

import supermimic.tasks  # noqa: F401
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg


@dataclass
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  runner: SuperMimicRunnerCfg

  # Experiment & Environment
  num_envs: int = 1024
  device: str | None = None
  registry_name: str | None = "supermimic/wandb-registry-motions/lafan_run1_subject5"

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id, play=True)
    return TrainConfig(
      env=env_cfg,
      runner=SuperMimicRunnerCfg(),
    )


def main():
  # Parse first argument to choose the task.
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    config=mjlab.TYRO_FLAGS,
  )

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Load Environment
  print(f"[INFO] Loading environment: {chosen_task}")

  # Check for tracking task and registry
  if args.registry_name:
    # Support comma-separated registry names for multi-motion
    registry_names = [n.strip() for n in args.registry_name.split(",")]
    api = wandb.Api()

    motion_files: list[str] = []
    for name in registry_names:
      if ":" not in name:
        name = name + ":latest"
      print(f"[INFO] Downloading motion from registry: {name}")
      artifact = api.artifact(name)
      motion_files.append(str(Path(artifact.download()) / "motion.npz"))

    # Pass list if multiple, single string if one (for backward compatibility)
    motion_file = motion_files if len(motion_files) > 1 else motion_files[0]
    args.env.commands["motion"].motion_file = motion_file
    print(f"[INFO] Loaded {len(motion_files)} motion(s)")

  # Auto-detect representation from task_id if not explicitly set
  if "MaxCoord" in chosen_task and args.runner.representation == "body_frame":
    args.runner.representation = "maxcoord"
    print("[INFO] Auto-detected maxcoord representation from task_id")

  args.env.scene.num_envs = args.num_envs

  env = ManagerBasedRlEnv(cfg=args.env, device=device)
  env = RslRlVecEnvWrapper(env)

  # Init Runner
  runner = SuperMimicRunner(
    env=env,
    cfg=args.runner,
    task_id=chosen_task,
    registry_name=args.registry_name,
    device=device,
  )

  runner.run()


if __name__ == "__main__":
  main()
