"""Script to benchmark Speed of Simulation and World Model."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import tyro
import wandb
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

import supermimic.tasks  # noqa: F401
from supermimic.models import WorldModel
from supermimic.representations import create_representation
from supermimic.scripts.train_utils import download_artifact_cached
from supermimic.utils.helpers import unpack_obs_dims


@dataclass
class BenchmarkConfig:
  wandb_path: str
  task_id: str = "SuperMimic-WorldModel-G1"
  device: str | None = None

  # Benchmark settings
  num_envs_list: list[int] = field(
    default_factory=lambda: [64, 128, 256, 512, 1024, 2048, 4096]
  )
  steps: int = 500
  history: int = 8

  use_policy: bool = True
  inference_mode: bool = False


def download_model(
  run: Any, filename: str, target_dir: Path, device: str
) -> dict[str, Any] | None:
  try:
    print(f"[INFO] Downloading {filename}...")
    run.file(filename).download(root=str(target_dir), replace=True)
    path = target_dir / filename
    print(f"[INFO] Loading {filename} from {path}")
    return torch.load(path, map_location=device)
  except Exception as e:
    print(f"[WARNING] Could not download/load {filename}: {e}")
    return None


def benchmark(cfg: BenchmarkConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  # Connect to WandB
  print(f"[INFO] Connecting to WandB run: {cfg.wandb_path}")
  api = wandb.Api()
  run = api.run(cfg.wandb_path)

  download_dir = Path("logs") / cfg.task_id / run.id / "wandb_download"
  download_dir.mkdir(parents=True, exist_ok=True)

  # Prepare to store results
  results = []

  # Load Reference World Model Weights first to reuse
  wm_state_dict = download_model(run, "world_model.pt", download_dir, device)
  if wm_state_dict is None:
    print("No world model found.")
    return

  policy_state_dict = None
  if cfg.use_policy:
    policy_state_dict = download_model(run, "policy.pt", download_dir, device)
    if policy_state_dict is None:
      print("No policy found, cannot benchmark with policy.")
      return

  # Check for linked motion artifacts (same as play.py)
  motion_file_path = None

  # run.used_artifacts() might be slow, do it once
  motion_artifact = next((a for a in run.used_artifacts() if a.type == "motions"), None)
  if motion_artifact:
    motion_file_path = str(
      Path(download_artifact_cached(motion_artifact)) / "motion.npz"
    )

  for n_envs in cfg.num_envs_list:
    print(f"\n{'=' * 40}")
    print(f"Benchmarking num_envs = {n_envs}")
    print(f"{'=' * 40}")

    # ---------------------------
    # 1. Environment / Simulation
    # ---------------------------
    try:
      env_cfg = load_env_cfg(cfg.task_id, play=True)
      env_cfg.scene.num_envs = n_envs
      # Disable viewer explicitly just in case, though we are not instantiating one

      # Check for linked motion artifacts (same as play.py)
      if motion_file_path:
        env_cfg.commands["motion"].motion_file = motion_file_path

      base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
      env = RslRlVecEnvWrapper(base_env)

      # Warmup and get obs
      obs, _ = env.reset()

      # Extract dims immediately
      num_state, num_goals = unpack_obs_dims(obs)

      action_dim = env.num_actions
      step_dt = env.unwrapped.step_dt

      # Create representation based on task_id
      repr_name = "maxcoord" if "MaxCoord" in cfg.task_id else "body_frame"
      representation = create_representation(repr_name, num_state)

      policy = None
      if cfg.use_policy:
        from rsl_rl.models import MLPModel
        from tensordict import TensorDict

        goal_extra = representation.goal_dim_split()
        policy_input_dim = representation.rep_state_dim + num_goals - goal_extra
        policy_obs = TensorDict(
          {"actor": torch.zeros(1, policy_input_dim, device=device)},
          batch_size=[1],
          device=device,
        )
        policy = MLPModel(
          obs=policy_obs,
          obs_groups={"actor": ["actor"]},
          obs_set="actor",
          output_dim=action_dim,
          hidden_dims=[512, 256, 128],
          obs_normalization=True,
        ).to(device)
        policy.load_state_dict(policy_state_dict, strict=False)
        policy.eval()
      else:
        actions = torch.rand((n_envs, action_dim), device=device) * 2 - 1

      # Sync CUDA
      if "cuda" in device:
        torch.cuda.synchronize()

      start_time = time.time()
      for _ in range(cfg.steps):
        with torch.inference_mode():
          if policy is not None:
            curr_gt_state = obs["full_state"]
            curr_goal = obs["goal"]

            actor_goal = representation.extract_actor_goal(curr_goal)
            policy_input = representation.construct_policy_input(
              curr_gt_state, actor_goal
            )
            actor_obs = TensorDict(
              {"actor": policy_input},
              batch_size=policy_input.shape[:-1],
              device=device,
            )
            actions = policy(actor_obs)

          obs, _, _, _ = env.step(actions)

      if "cuda" in device:
        torch.cuda.synchronize()
      end_time = time.time()

      dur = end_time - start_time
      sim_sps = (cfg.steps * n_envs) / dur
      label = "Sim + Policy" if cfg.use_policy else "Simulation"
      print(
        f"{label}: {sim_sps:.2f} steps/s (Total time: {dur:.2f}s for {cfg.steps} steps)"
      )

      reward_manager = env.unwrapped.reward_manager
      reward_names = getattr(reward_manager, "active_terms", ["reward"])
      num_rewards = len(reward_names)

      env.close()
      del env
      del base_env
      if "cuda" in device:
        torch.cuda.empty_cache()

    except Exception as e:
      print(f"Simulation Benchmark Failed for {n_envs}: {e}")
      sim_sps = 0.0
      num_state = 0  # Placeholder if failed
      num_goals = 0
      num_rewards = 0
      step_dt = 0.02

    # ---------------------------
    # 2. World Model Inference
    # ---------------------------
    try:
      # Re-use representation from sim benchmark (or create if sim failed)
      if num_state == 0:
        continue
      repr_name = "maxcoord" if "MaxCoord" in cfg.task_id else "body_frame"
      wm_repr = create_representation(repr_name, num_state)
      wm = WorldModel(
        wm_repr, action_dim, num_goals, num_rewards, dt=step_dt, history=cfg.history
      ).to(device)
      wm.load_state_dict(wm_state_dict, strict=False)
      wm.eval()

      # Create Dummy Inputs
      # WM transition: rep_state_hist, action, last_action, goal, aux_state
      batch_size = n_envs
      dummy_rep_hist = torch.randn(
        (batch_size, cfg.history, wm.num_rep_state), device=device
      )
      dummy_action = torch.randn((batch_size, action_dim), device=device)
      dummy_last_action = torch.randn((batch_size, action_dim), device=device)
      dummy_goal = torch.randn((batch_size, num_goals), device=device)
      dummy_aux = torch.randn((batch_size, 7), device=device)

      # Warmup
      for _ in range(10):
        wm(dummy_rep_hist, dummy_action, dummy_last_action, dummy_goal, dummy_aux)

      if "cuda" in device:
        torch.cuda.synchronize()

      start_time = time.time()
      for _ in range(cfg.steps):
        with torch.inference_mode(cfg.inference_mode):
          wm(
            dummy_rep_hist,
            dummy_action,
            dummy_last_action,
            dummy_goal,
            dummy_aux,
          )

      if "cuda" in device:
        torch.cuda.synchronize()
      end_time = time.time()

      dur = end_time - start_time
      wm_sps = (cfg.steps * n_envs) / dur
      mode_str = " (Inference Mode)" if cfg.inference_mode else ""
      print(
        f"WorldModel{mode_str}: {wm_sps:.2f} steps/s (Total time: {dur:.2f}s for {cfg.steps} steps)"
      )

      del wm
      if "cuda" in device:
        torch.cuda.empty_cache()

    except Exception as e:
      print(f"WorldModel Benchmark Failed for {n_envs}: {e}")
      import traceback

      traceback.print_exc()
      wm_sps = 0.0

    results.append({"num_envs": n_envs, "sim_sps": sim_sps, "wm_sps": wm_sps})

  # Print Summary Table
  print("\n" + "=" * 80)
  print(f"{'Num Envs':<15} | {'Sim SPS':<20} | {'WM SPS':<20} | {'Ratio (WM/Sim)':<20}")
  print("-" * 80)
  for r in results:
    ratio = r["wm_sps"] / r["sim_sps"] if r["sim_sps"] > 0 else 0.0
    print(
      f"{r['num_envs']:<15} | {r['sim_sps']:<20.2f} | {r['wm_sps']:<20.2f} | {ratio:<20.2f}"
    )
  print("=" * 80 + "\n")


def main() -> None:
  args = tyro.cli(BenchmarkConfig)
  benchmark(args)


if __name__ == "__main__":
  main()
