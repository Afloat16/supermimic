"""Train a posterior MLP over frozen WM + VAE."""

from __future__ import annotations

from dataclasses import dataclass

import mjlab.tasks  # noqa: F401
import tyro
from mjlab.tasks.registry import list_tasks, load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.runners.posterior_runner import (
  PosteriorRunner,
  PosteriorRunnerCfg,
)
from supermimic.runners.vae_utils import modify_env_cfg_for_vae_prior
from supermimic.scripts.train_utils import (
  create_env,
  download_registry_motions,
  setup_device,
)


@dataclass
class TrainConfig:
  runner: PosteriorRunnerCfg
  num_envs: int = 1024
  device: str | None = None
  registry_name: str | None = "supermimic/wandb-registry-motions/lafan_run1_subject5"


def main() -> None:
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
    default=TrainConfig(runner=PosteriorRunnerCfg()),
    config=mjlab.TYRO_FLAGS,
  )

  device = setup_device(args.device)

  motion_file = download_registry_motions(args.registry_name)
  env_cfg = load_env_cfg(chosen_task, play=False)
  if motion_file is not None:
    env_cfg.commands["motion"].motion_file = motion_file
  modify_env_cfg_for_vae_prior(env_cfg)

  env = create_env(env_cfg, args.num_envs, device)

  runner = PosteriorRunner(
    env=env,
    cfg=args.runner,
    task_id=chosen_task,
    registry_name=args.registry_name,
    device=device,
  )
  runner.run()


if __name__ == "__main__":
  main()
