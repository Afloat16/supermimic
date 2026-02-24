"""Script to test PPORunner."""

from dataclasses import dataclass

import tyro
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.runners.ppo_runner import PpoRunner, PpoRunnerCfg
from supermimic.scripts.train_utils import (
  create_env,
  download_registry_motions,
  setup_device,
)


@dataclass
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  runner: PpoRunnerCfg

  # Experiment & Environment
  task_id: str = "SuperMimic-WorldModel-G1"
  num_envs: int = 1024
  device: str | None = None
  registry_name: str | None = "supermimic/wandb-registry-motions/lafan_run1_subject5"

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id, play=True)
    return TrainConfig(
      env=env_cfg,
      runner=PpoRunnerCfg(),
      task_id=task_id,
    )


def main():
  chosen_task = TrainConfig.task_id
  default_cfg = TrainConfig.from_task(chosen_task)

  args = tyro.cli(
    TrainConfig,
    default=default_cfg,
    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
  )

  device = setup_device(args.device)
  print(f"[INFO] Loading environment: {args.task_id}")

  motion_file = download_registry_motions(args.registry_name)
  if motion_file is not None:
    args.env.commands["motion"].motion_file = motion_file

  env = create_env(args.env, args.num_envs, device)

  # Init Runner
  runner = PpoRunner(
    env=env,
    cfg=args.runner,
    task_id=args.task_id,
    registry_name=args.registry_name,
    device=device,
  )

  runner.run()


if __name__ == "__main__":
  main()
