"""Script to train World Model using SuperMimicRunner."""

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

import supermimic.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.gcs import ensure_default_checkpoint
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from supermimic.models import Policy, WorldModel
from supermimic.replay_buffer import ReplayBuffer
from supermimic.runners.learner import WorldModelLearner, PolicyLearner
from supermimic.runners.supermimic_runner import SuperMimicRunner


@dataclass
class TrainConfig:
    # Task arguments
    task_id: str = "SuperMimic-WorldModel-G1"

    # Environment arguments
    num_envs: int = 512  # Large batch for data collection
    device: str | None = None

    # Data collection arguments
    buffer_size: int = 1000

    # Visualization
    headless: bool = False

    # Training
    epochs: int = 100000
    num_mini_batch: int = 10
    mini_batch_size: int = 8192
    window_size: int = 25

    # Nominal errors for Bryson's Rule (Loss Weights)
    nominal_error_root_pos: float = 0.1
    nominal_error_root_ori: float = 0.1
    nominal_error_root_lin_vel: float = 1.0
    nominal_error_root_ang_vel: float = 3.14
    nominal_error_joint_pos: float = 0.1
    nominal_error_joint_vel: float = 3.14

    # WandB
    wandb_project: str = "supermimic"
    wandb_group: str | None = None


def main():
    args = tyro.cli(TrainConfig)
    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Environment
    print(f"[INFO] Loading environment: {args.task_id}")
    env_cfg = load_env_cfg(args.task_id, play=True)
    env_cfg.scene.num_envs = args.num_envs
    if args.headless:
        env_cfg.viewer.eye = (0.0, 0.0, 0.0)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env)

    # Load Policy
    print("[INFO] Loading policy from demo checkpoint...")
    resume_path = Path(ensure_default_checkpoint())

    # Initialize Policy
    obs, _ = env.reset()
    policy_obs = obs["policy"] if hasattr(obs, "keys") else obs
    num_obs = policy_obs.shape[-1]
    num_actions = env.num_actions

    policy = Policy(num_obs, num_actions, normalization=True).to(device)
    policy.load_state_dict_from_rsl_rl_ckpt(str(resume_path), device)

    policy_learner = PolicyLearner(policy)

    # Initialize World Model & Buffer
    state_obs = obs["full_state"] if hasattr(obs, "keys") else obs
    num_state = state_obs.shape[-1]
    goal_obs = obs["goal"] if hasattr(obs, "keys") else obs
    num_goals = goal_obs.shape[-1]
    print(f"[INFO] Policy Dim: {num_obs}, State Dim: {num_state}, Goal Dim: {num_goals}")

    wm = WorldModel(num_state, num_actions, num_goals).to(device)
    # Inject dt into config dynamically
    args.dt = env.unwrapped.step_dt
    wm_learner = WorldModelLearner(wm, args)

    # Buffer: [state, action, next_state, reward, goal, done]
    replay_buffer = ReplayBuffer(dim=[num_state, num_actions, num_state, 1, num_goals, 1],
                                 buffer_size=args.buffer_size, device=device)

    # Init Runner
    runner = SuperMimicRunner(
        env=env,
        policy_learner=policy_learner,
        world_model_learner=wm_learner,
        replay_buffer=replay_buffer,
        cfg=args
    )

    runner.run()


if __name__ == "__main__":
    main()
