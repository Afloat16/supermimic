"""Script to benchmark Speed of Simulation and World Model."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import tyro
import wandb
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from tensordict import TensorDict

import supermimic.tasks  # noqa: F401
from supermimic.models import Dynamics, MultiEncoderAE
from supermimic.representations import (
  GenCoordRepresentation,
  MaxCoordRepresentation,
  create_representation,
  extract_actor_goal,
)
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg
from supermimic.sim_context import SimContext
from supermimic.tasks.task_spec import make_motion_task
from supermimic.utils.gencoord_helpers import infer_num_joints_from_state_dim
from supermimic.utils.helpers import unpack_obs_dims
from supermimic.utils.wandb_artifacts import (
  download_artifact_cached,
  download_model_dir,
)


@dataclass
class BenchmarkConfig:
  wandb_path: str
  task_id: str = "SuperMimic-GenCoord-G1"
  device: str | None = None

  num_envs_list: list[int] = field(
    default_factory=lambda: [64, 128, 256, 512, 1024, 2048, 4096]
  )
  steps: int = 500

  use_policy: bool = True
  inference_mode: bool = False


def benchmark(cfg: BenchmarkConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  print(f"[INFO] Connecting to WandB run: {cfg.wandb_path}")
  run = wandb.Api().run(cfg.wandb_path)

  model_dir = download_model_dir(run)
  bundle = torch.load(model_dir / "world_model.pt", map_location=device)
  policy_state_dict = (
    torch.load(model_dir / "actor.pt", map_location=device) if cfg.use_policy else None
  )

  results = []

  # Resolve the run's linked motion clip once; reused for every num_envs.
  motion_file_path = None
  motion_artifact = next((a for a in run.used_artifacts() if a.type == "motions"), None)
  if motion_artifact:
    motion_file_path = str(download_artifact_cached(motion_artifact) / "motion.npz")

  default_cfg = SuperMimicRunnerCfg()
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

      repr_name = "maxcoord" if "MaxCoord" in cfg.task_id else "gencoord"
      default_joint_pos = (
        env.unwrapped.scene["robot"].data.default_joint_pos[0].to(device)
      )
      has_state_estimation = (
        "base_lin_vel" in env.unwrapped.observation_manager.active_terms["actor"]
      )
      if repr_name == "gencoord":
        representation = GenCoordRepresentation(
          infer_num_joints_from_state_dim(num_state), default_joint_pos
        )
      else:
        representation = MaxCoordRepresentation(
          num_state // 13, default_joint_pos.shape[0]
        )
      representation.has_state_estimation = has_state_estimation

      policy = None
      if cfg.use_policy:
        assert policy_state_dict is not None  # use_policy gated the load above
        motion_task = make_motion_task(representation, num_goals)
        policy = MultiEncoderAE(
          obs_dim=representation.rep_state_dim,
          action_dim=action_dim,
          task_specs=[motion_task],
          slot_mix={"motion": 1.0},
          latent_dim=default_cfg.latent_dim,
          hidden_dims=default_cfg.actor_hidden_dims,
          latent_bc_weight=default_cfg.latent_bc_weight,
          history_length=default_cfg.history_length,
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

            sensor = representation.represent(curr_gt_state)
            actor_goal = extract_actor_goal(curr_goal)
            actor_obs = TensorDict(
              {"history": sensor.unsqueeze(1), "goal_motion": actor_goal},
              batch_size=sensor.shape[:-1],
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

      sim_context = SimContext.from_env(env.unwrapped, device)
      sim_ctx_dim = sim_context.output_dim

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
      repr_name = "maxcoord" if "MaxCoord" in cfg.task_id else "gencoord"
      representation = create_representation(
        repr_name,
        num_state,
        action_dim,
        num_goals,
        num_rewards,
        sim_ctx_dim,
        step_dt,
        default_joint_pos,
        has_state_estimation,
        actor_obs_mode="rep_state",
        reward_hidden_dims=default_cfg.reward_hidden_dims,
        term_hidden_dims=default_cfg.term_hidden_dims,
        obs_hidden_dims=default_cfg.obs_hidden_dims,
      ).to(device)
      # action_scale / action_offset are placeholders here — load_state_dict
      # overwrites them with the checkpoint's registered buffers on Dynamics.
      dynamics = Dynamics(
        representation,
        action_dim,
        sim_ctx_dim,
        torch.zeros(action_dim),
        torch.zeros(action_dim),
        hidden_dims=default_cfg.dynamics_hidden_dims,
      ).to(device)
      dynamics.load_state_dict(bundle["dynamics"])
      representation.load_state_dict(bundle["representation"])
      dynamics.eval()
      representation.eval()

      # Create Dummy Inputs
      batch_size = n_envs
      dummy_rep_state = torch.randn(
        (batch_size, representation.rep_state_dim), device=device
      )
      dummy_action = torch.randn((batch_size, action_dim), device=device)
      dummy_goal = torch.randn((batch_size, num_goals), device=device)
      dummy_aux = torch.randn((batch_size, 7), device=device)
      dummy_sim_ctx = torch.randn((batch_size, sim_ctx_dim), device=device)

      # Warmup
      for _ in range(10):
        dynamics(dummy_rep_state, dummy_action, dummy_aux, dummy_sim_ctx)
        representation.predict_termination(dummy_rep_state, dummy_action, dummy_goal)
        representation.predict_reward(
          dummy_rep_state, dummy_action, dummy_action, dummy_goal
        )

      if "cuda" in device:
        torch.cuda.synchronize()

      start_time = time.time()
      for _ in range(cfg.steps):
        with torch.inference_mode(cfg.inference_mode):
          dynamics(dummy_rep_state, dummy_action, dummy_aux, dummy_sim_ctx)
          representation.predict_termination(dummy_rep_state, dummy_action, dummy_goal)
          representation.predict_reward(
            dummy_rep_state, dummy_action, dummy_action, dummy_goal
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

      del dynamics, representation
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
