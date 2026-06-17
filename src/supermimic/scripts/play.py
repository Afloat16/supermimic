"""Play a SuperMimic multi-encoder actor against the sim."""

from __future__ import annotations

import glob
import os
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

import supermimic.tasks  # noqa: F401
from supermimic.models import Dynamics, MultiEncoderAE
from supermimic.representations import create_representation
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg
from supermimic.scripts.viewers import (
  SuperMimicNativeViewer,
  SuperMimicViserViewer,
  create_wm_vis_model,
)
from supermimic.sim_context import SimContext
from supermimic.tasks.task_spec import (
  TasksCfg,
  VelocityTaskCfg,
  resolve_task_specs,
)
from supermimic.utils.helpers import extract_action_scale_offset, unpack_obs_dims
from supermimic.utils.wandb_artifacts import (
  download_artifact_cached,
  download_model_dir,
)


@dataclass
class PlayConfig:
  wandb_path: str
  motion_file: str | None = None
  initial_task: Literal["motion", "velocity"] = "motion"
  """Task selected when the viewer opens. Switch live via the GUI dropdown."""
  device: str | None = None
  num_steps_rollout: int = 1000000
  viewer: Literal["auto", "native", "viser"] = "viser"


def _resolve_tasks_cfg(runner_cfg: dict[str, Any]) -> TasksCfg:
  """Rebuild TasksCfg from a wandb-serialized runner config.

  JSON deserialization renders tuples as lists; coerce range fields back.
  Missing fields fall through to each task's dataclass defaults.
  """
  tuple_fields = {"lin_vel_x_range", "lin_vel_y_range", "ang_vel_z_range"}
  vel_raw = runner_cfg["tasks"].get("velocity", {})
  vel_kwargs = {k: tuple(v) if k in tuple_fields else v for k, v in vel_raw.items()}
  return TasksCfg(velocity=VelocityTaskCfg(**vel_kwargs))


def play(task_id: str, cfg: PlayConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  print(f"[INFO] Loading environment: {task_id}")
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.observations["actor"].enable_corruption = True

  import wandb

  print(f"[INFO] Connecting to WandB run: {cfg.wandb_path}")
  run = wandb.Api().run(cfg.wandb_path)
  runner_cfg = run.config.get("runner_cfg", {}) or {}
  default_cfg = SuperMimicRunnerCfg()

  motion_artifacts = [a for a in run.used_artifacts() if a.type == "motions"]
  command_term = env_cfg.commands["motion"]
  assert isinstance(command_term, MotionCommandCfg)
  motion_names: list[str] = []
  if cfg.motion_file is not None:
    expanded = os.path.expanduser(cfg.motion_file)
    if any(c in expanded for c in "*?[]"):
      files = sorted(glob.glob(expanded))
    else:
      files = [expanded]
    if not files:
      raise ValueError(f"No motion files matched: {cfg.motion_file}")
    command_term.motion_file = files
    motion_names = [Path(f).stem for f in files]
    print(f"[INFO] Using {len(files)} local motion(s)")
  elif motion_artifacts:
    motion_files = []
    for artifact in motion_artifacts:
      print(f"[INFO] Found linked motion artifact: {artifact.name}")
      motion_files.append(str(download_artifact_cached(artifact) / "motion.npz"))
    command_term.motion_file = (
      motion_files if len(motion_files) > 1 else motion_files[0]
    )
    motion_names = [a.name.split(":")[0] for a in motion_artifacts]
    print(f"[INFO] Loaded {len(motion_files)} motion(s)")

  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))

  obs, _ = env.reset()
  num_state, num_goals = unpack_obs_dims(obs)
  num_actions = env.num_actions

  reward_manager = env.unwrapped.reward_manager
  reward_names = getattr(reward_manager, "active_terms", ["reward"])
  num_rewards = len(reward_names)
  print(f"[INFO] Reward Dim: {num_rewards} ({reward_names})")

  repr_name = "maxcoord" if "MaxCoord" in task_id else "gencoord"
  default_joint_pos = env.unwrapped.scene["robot"].data.default_joint_pos[0].to(device)
  has_state_estimation = (
    "base_lin_vel" in env.unwrapped.observation_manager.active_terms["actor"]
  )
  actor_obs_mode: str = runner_cfg["actor_obs"]

  sim_context = SimContext.from_env(env.unwrapped, device)
  sim_ctx_dim = sim_context.output_dim
  print(
    f"[INFO] Sim Context: {sim_ctx_dim} dim ({[s.name for s in sim_context.specs]})"
  )

  representation = create_representation(
    repr_name,
    num_state,
    num_actions,
    num_goals,
    num_rewards,
    sim_ctx_dim,
    env.unwrapped.step_dt,
    default_joint_pos,
    has_state_estimation,
    actor_obs_mode=actor_obs_mode,
    reward_hidden_dims=default_cfg.reward_hidden_dims,
    term_hidden_dims=default_cfg.term_hidden_dims,
    obs_hidden_dims=default_cfg.obs_hidden_dims,
  ).to(device)

  print(
    f"[INFO] Representation: full={representation.full_state_dim}, "
    f"rep={representation.rep_state_dim}, pred={representation.prediction_dim}"
  )

  action_scale, action_offset = extract_action_scale_offset(env)
  dynamics = Dynamics(
    representation,
    num_actions,
    sim_ctx_dim,
    action_scale,
    action_offset,
    hidden_dims=default_cfg.dynamics_hidden_dims,
  ).to(device)

  model_dir = download_model_dir(run)
  bundle = torch.load(model_dir / "world_model.pt", map_location=device)
  dynamics.load_state_dict(bundle["dynamics"])
  representation.load_state_dict(bundle["representation"])
  dynamics.eval()
  representation.eval()

  tasks_cfg = _resolve_tasks_cfg(runner_cfg)
  task_specs = resolve_task_specs(tasks_cfg, representation, num_goals)
  task_names = [ts.name for ts in task_specs]
  if cfg.initial_task not in task_names:
    raise ValueError(
      f"--initial-task {cfg.initial_task} not in trained task set {task_names}"
    )

  obs_dim = (
    representation.rep_state_dim
    if actor_obs_mode == "rep_state"
    else representation.proprio_obs_dim(num_actions)
  )
  actor = MultiEncoderAE(
    obs_dim=obs_dim,
    action_dim=num_actions,
    task_specs=task_specs,
    slot_mix={cfg.initial_task: 1.0},
    latent_dim=runner_cfg["latent_dim"],
    hidden_dims=default_cfg.actor_hidden_dims,
    latent_bc_weight=default_cfg.latent_bc_weight,
    history_length=runner_cfg["history_length"],
  ).to(device)
  actor_state = torch.load(model_dir / "actor.pt", map_location=device)
  actor.load_state_dict(actor_state)
  actor.eval()

  wm_vis_model = create_wm_vis_model(env.unwrapped.sim.mj_model)

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved = "native" if has_display else "viser"
  else:
    resolved = cfg.viewer
  print(f"[INFO] Starting viewer: {resolved}")

  if resolved == "native":
    SuperMimicNativeViewer(
      env=env,
      actor=actor,
      dynamics=dynamics,
      representation=representation,
      wm_vis_model=wm_vis_model,
      num_steps_rollout=cfg.num_steps_rollout,
      sim_context=sim_context,
      task_specs=task_specs,
      initial_task=cfg.initial_task,
    ).run()
  else:
    SuperMimicViserViewer(
      env=env,
      actor=actor,
      dynamics=dynamics,
      representation=representation,
      wm_vis_model=wm_vis_model,
      num_steps_rollout=cfg.num_steps_rollout,
      motion_names=motion_names,
      sim_context=sim_context,
      task_specs=task_specs,
      initial_task=cfg.initial_task,
    ).run()

  env.close()


def main() -> None:
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(PlayConfig, args=remaining_args, config=mjlab.TYRO_FLAGS)
  play(chosen_task, args)


if __name__ == "__main__":
  main()
