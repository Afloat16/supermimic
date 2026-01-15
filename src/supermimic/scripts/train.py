"""Script to train World Model using SuperMimicRunner."""

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
import wandb
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.gcs import ensure_default_motion
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

import supermimic.tasks  # noqa: F401
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg


@dataclass
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  runner: SuperMimicRunnerCfg

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
      runner=SuperMimicRunnerCfg(),
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

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Load Environment
  print(f"[INFO] Loading environment: {args.task_id}")

  # Check for tracking task and registry
  if args.registry_name:
    registry_name = args.registry_name
    if ":" not in registry_name:
      registry_name = registry_name + ":latest"
      args.registry_name = registry_name

    print(f"[INFO] Downloading motion from registry: {registry_name}")
    api = wandb.Api()
    artifact = api.artifact(registry_name)
    motion_file_path = str(Path(artifact.download()) / "motion.npz")

    args.env.commands["motion"].motion_file = motion_file_path
    print(f"[INFO] Using motion file: {motion_file_path}")
  else:
    print("[INFO] Using mjlab demo assets.")
    try:
      motion_path = ensure_default_motion()
      args.runner.demo_policy = True

      from mjlab.asset_zoo.robots import G1_ACTION_SCALE
      from mjlab.envs.mdp.actions import JointPositionActionCfg

      # Force JointPositionAction for demo policy compatibility
      args.env.actions["joint_pos"] = JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=G1_ACTION_SCALE,
        use_default_offset=True,
      )

      args.env.commands["motion"].motion_file = motion_path
    except Exception as e:
      print(f"[WARN] Failed to download demo assets: {e}")

  args.env.scene.num_envs = args.num_envs

  env = ManagerBasedRlEnv(cfg=args.env, device=device)
  env = RslRlVecEnvWrapper(env)

  # Init Runner
  runner = SuperMimicRunner(
    env=env,
    cfg=args.runner,
    task_id=args.task_id,
    registry_name=args.registry_name,
    device=device,
  )

  runner.run()


if __name__ == "__main__":
  main()
