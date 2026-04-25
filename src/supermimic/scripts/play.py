"""Script to visualize World Model prediction vs Ground Truth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mjlab.tasks  # noqa: F401
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends
from tensordict import TensorDict

import supermimic.tasks  # noqa: F401
from supermimic.models import Dynamics, VAEActor, load_actor
from supermimic.representations import actor_goal_dim, create_representation
from supermimic.scripts.train_utils import download_artifact_cached
from supermimic.scripts.viewers import (
  SuperMimicNativeViewer,
  SuperMimicViserViewer,
  create_wm_vis_model,
)
from supermimic.utils.helpers import extract_action_scale_offset, unpack_obs_dims


@dataclass
class PlayConfig:
  wandb_path: str
  device: str | None = None
  num_steps_rollout: int = 1000000
  viewer: Literal["auto", "native", "viser"] = "viser"
  use_prior: bool = False


def download_model(
  run: Any, filename: str, target_dir: Path, device: str
) -> dict[str, Any]:
  run.file(filename).download(root=str(target_dir), replace=True)
  path = target_dir / filename
  print(f"[INFO] Loading {filename} from {path}")
  return torch.load(path, map_location=device)


def play(task_id: str, cfg: PlayConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Load Env
  print(f"[INFO] Loading environment: {task_id}")
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1

  if cfg.use_prior:
    for key in ("anchor_pos", "anchor_ori", "ee_body_pos", "motion_end"):
      env_cfg.terminations.pop(key, None)
    env_cfg.commands["motion"].debug_vis = False

  # Connect to WandB and check for motion artifact
  import wandb

  print(f"[INFO] Connecting to WandB run: {cfg.wandb_path}")
  api = wandb.Api()
  run = api.run(cfg.wandb_path)

  # Check for linked motion artifacts (supports multi-motion)
  motion_artifacts = [a for a in run.used_artifacts() if a.type == "motions"]
  if motion_artifacts:
    motion_files = []
    for artifact in motion_artifacts:
      print(f"[INFO] Found linked motion artifact: {artifact.name}")
      motion_files.append(str(Path(download_artifact_cached(artifact)) / "motion.npz"))

    # Pass list if multiple, single string if one
    motion_file = motion_files if len(motion_files) > 1 else motion_files[0]
    command_term = env_cfg.commands["motion"]
    assert isinstance(command_term, MotionCommandCfg)
    command_term.motion_file = motion_file
    print(f"[INFO] Loaded {len(motion_files)} motion(s)")

  motion_names = [a.name.split(":")[0] for a in motion_artifacts]

  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(base_env)

  obs, _ = env.reset()
  num_state, num_goals = unpack_obs_dims(obs)
  num_actions = env.num_actions

  # Check for reward terms
  reward_manager = env.unwrapped.reward_manager
  reward_names = getattr(reward_manager, "active_terms", ["reward"])
  num_rewards = len(reward_names)
  print(f"[INFO] Reward Dim: {num_rewards} ({reward_names})")

  repr_name = "maxcoord" if "MaxCoord" in task_id else "gencoord"
  representation = create_representation(
    repr_name,
    num_state,
    num_actions,
    num_goals,
    num_rewards,
    env.unwrapped.step_dt,
  ).to(device)

  print(
    f"[INFO] Representation: full={representation.full_state_dim}, "
    f"rep={representation.rep_state_dim}, pred={representation.prediction_dim}"
  )

  action_scale, action_offset = extract_action_scale_offset(env)
  dynamics = Dynamics(representation, num_actions, action_scale, action_offset).to(
    device
  )

  download_dir = Path("logs") / task_id / run.id / "wandb_download"
  download_dir.mkdir(parents=True, exist_ok=True)

  bundle = download_model(run, "world_model.pt", download_dir, device)
  dynamics.load_state_dict(bundle["dynamics"])
  representation.load_state_dict(bundle["representation"])
  dynamics.eval()
  representation.eval()

  # Load Actor
  actor_state = download_model(run, "actor.pt", download_dir, device)
  actor = load_actor(
    actor_state,
    representation.rep_state_dim,
    actor_goal_dim(num_goals),
    num_actions,
    device,
  )

  if isinstance(actor, VAEActor) and cfg.use_prior:
    print("[INFO] VAE: using prior encoder (no target conditioning)")
    _vae = actor

    def actor(obs_td: TensorDict) -> torch.Tensor:
      x = obs_td["actor"]
      obs = x[..., : _vae.obs_dim]
      z = _vae.encode(obs)
      return _vae.decode(z, obs)

  # Setup WM Vis Model
  wm_vis_model = create_wm_vis_model(base_env.sim.mj_model)

  # Viewer Selection
  import os

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = cfg.viewer

  print(f"[INFO] Starting viewer: {resolved_viewer}")

  if resolved_viewer == "native":
    viewer = SuperMimicNativeViewer(
      env=env,
      actor=actor,
      dynamics=dynamics,
      representation=representation,
      wm_vis_model=wm_vis_model,
      num_steps_rollout=cfg.num_steps_rollout,
    )
    viewer.run()

  elif resolved_viewer == "viser":
    viewer = SuperMimicViserViewer(
      env=env,
      actor=actor,
      dynamics=dynamics,
      representation=representation,
      wm_vis_model=wm_vis_model,
      num_steps_rollout=cfg.num_steps_rollout,
      motion_names=motion_names,
    )
    viewer.run()

  env.close()


def main() -> None:
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    config=mjlab.TYRO_FLAGS,
  )
  play(chosen_task, args)


if __name__ == "__main__":
  main()
