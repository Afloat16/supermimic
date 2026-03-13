"""Script to visualize World Model prediction vs Ground Truth."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mjlab.tasks  # noqa: F401
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.models import MLPModel
from tensordict import TensorDict

import supermimic.tasks  # noqa: F401
from supermimic.models import WorldModel
from supermimic.representations import create_representation
from supermimic.scripts.train_utils import download_artifact_cached
from supermimic.scripts.viewers import (
  SuperMimicNativeViewer,
  SuperMimicViserViewer,
  create_wm_vis_model,
)
from supermimic.utils.helpers import unpack_obs_dims


@dataclass
class PlayConfig:
  wandb_path: str
  device: str | None = None
  num_steps_rollout: int = 1000000
  viewer: Literal["auto", "native", "viser"] = "viser"
  history: int = 1


def download_model(run, filename, target_dir, device):
  run.file(filename).download(root=str(target_dir), replace=True)
  path = target_dir / filename
  print(f"[INFO] Loading {filename} from {path}")
  return torch.load(path, map_location=device)


def play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Load Env
  print(f"[INFO] Loading environment: {task_id}")
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1

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

  # Create representation based on task_id
  repr_name = "maxcoord" if "MaxCoord" in task_id else "body_frame"
  representation = create_representation(repr_name, num_state)

  print(
    f"[INFO] Representation: full={representation.full_state_dim}, "
    f"rep={representation.rep_state_dim}, pred={representation.prediction_dim}"
  )

  # Load World Model
  wm = WorldModel(
    representation,
    num_actions,
    num_goals,
    num_rewards,
    dt=env.unwrapped.step_dt,
    history=cfg.history,
  ).to(device)

  # Download from WandB
  download_dir = Path("logs") / task_id / run.id / "wandb_download"
  download_dir.mkdir(parents=True, exist_ok=True)

  # Download World Model
  wm_state = download_model(run, "world_model.pt", download_dir, device)
  wm.load_state_dict(wm_state, strict=False)
  wm.eval()

  # Load Actor
  goal_extra = representation.goal_dim_split()
  actor_input_dim = wm.num_rep_state + num_goals - goal_extra

  actor_obs = TensorDict(
    {"actor": torch.zeros(1, actor_input_dim, device=device)},
    batch_size=[1],
    device=device,
  )
  actor = MLPModel(
    obs=actor_obs,
    obs_groups={"actor": ["actor"]},
    obs_set="actor",
    output_dim=num_actions,
    hidden_dims=[512, 256, 128],
    obs_normalization=True,
  ).to(device)

  # Download Actor from WandB
  try:
    actor_state = download_model(run, "actor.pt", download_dir, device)
    print("[INFO] Loaded actor.pt")
  except Exception:
    print("[WARN] actor.pt not found, trying policy.pt")
    actor_state = download_model(run, "policy.pt", download_dir, device)

  # Handle key mapping from old Policy to new MLPModel
  new_actor_state = {}
  for k, v in actor_state.items():
    if k.startswith("actor."):
      new_key = "mlp." + k[len("actor.") :]
      new_actor_state[new_key] = v
    elif k.startswith("actor_obs_normalizer."):
      new_key = "obs_normalizer." + k[len("actor_obs_normalizer.") :]
      new_actor_state[new_key] = v
    elif k.startswith("mlp.") or k.startswith("obs_normalizer."):
      # Already in new format
      new_actor_state[k] = v

  actor.load_state_dict(new_actor_state, strict=False)

  actor.eval()

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
      wm=wm,
      wm_vis_model=wm_vis_model,
      num_steps_rollout=cfg.num_steps_rollout,
      history=cfg.history,
    )
    viewer.run()

  elif resolved_viewer == "viser":
    viewer = SuperMimicViserViewer(
      env=env,
      actor=actor,
      wm=wm,
      wm_vis_model=wm_vis_model,
      num_steps_rollout=cfg.num_steps_rollout,
      history=cfg.history,
    )
    viewer.run()

  env.close()


def main():
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
