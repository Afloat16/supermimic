"""Script to train World Model using SuperMimicRunner."""

from dataclasses import dataclass

import torch
import tyro

import supermimic.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from supermimic.models import Policy, WorldModel
from supermimic.replay_buffer import ReplayBuffer
from supermimic.runners.learner import WorldModelLearner, PolicyLearner
from supermimic.runners.supermimic_runner import SuperMimicRunner


@dataclass
class TrainConfig:
    # Task & Environment
    task_id: str = "SuperMimic-WorldModel-G1"
    num_envs: int = 512
    headless: bool = True
    device: str | None = None

    # Training Loop
    buffer_size: int = 1000
    max_iterations: int = 10000
    num_steps_model: int = 50
    num_steps_policy: int = 50

    # World Model
    model_lr: float = 1e-4
    num_mini_batch: int = 10
    mini_batch_size: int = 2048
    window_size: int = 25

    # Policy
    policy_lr: float = 1e-4
    policy_noise: float = 0.1
    horizon: int = 50

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
    env_cfg.episode_length_s = env_cfg.sim.mujoco.timestep * env_cfg.decimation * (args.window_size + 1)

    if args.headless:
        env_cfg.viewer.eye = (0.0, 0.0, 0.0)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env)

    # Initialize Dimensions
    obs, _ = env.reset()
    state_obs = obs["full_state"] if hasattr(obs, "keys") else obs
    num_state = state_obs.shape[-1]
    goal_obs = obs["goal"] if hasattr(obs, "keys") else obs
    num_goals = goal_obs.shape[-1]
    num_actions = env.num_actions

    # Initialize World Model & Learner
    print(f"[INFO] State Dim: {num_state}, Goal Dim: {num_goals}")
    wm = WorldModel(num_state, num_actions, num_goals).to(device)

    args.dt = env.unwrapped.step_dt
    wm_learner = WorldModelLearner(wm, args)

    # Initialize Policy & Learner
    print("[INFO] Training new policy from scratch (State+Goal)...")
    policy_input_dim = num_state + num_goals
    policy = Policy(policy_input_dim, num_actions, normalization=True).to(device)
    policy_learner = PolicyLearner(policy, wm, args)

    # Buffer: [state, action, goal, reward, done]
    replay_buffer = ReplayBuffer(dim=[num_state, num_actions, num_goals, 1, 1],
                                 buffer_size=args.buffer_size, device=device)

    # Init Runner
    runner = SuperMimicRunner(
        env=env,
        replay_buffer=replay_buffer,
        world_model_learner=wm_learner,
        policy_learner=policy_learner,
        cfg=args
    )

    runner.run()


if __name__ == "__main__":
    main()
