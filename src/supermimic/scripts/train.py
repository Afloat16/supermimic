"""Script to collect data for SuperTrack world model."""

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

import supermimic.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.gcs import ensure_default_checkpoint, ensure_default_motion
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer
from supermimic.policy import ActorMLP
from supermimic.replay_buffer import ReplayBuffer
from supermimic.world_model import WorldModel


@dataclass
class TrainConfig:
    # Task arguments
    task_id: str = "SuperMimic-WorldModel-G1"

    # Environment arguments
    num_envs: int = 4096  # Large batch for data collection
    device: str | None = None

    # Data collection arguments
    buffer_size: int = 1000000
    collection_steps: int = 1000  # Number of steps to collect per env

    # Visualization
    headless: bool = False


def collect_data(cfg: TrainConfig):
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Environment
    print(f"[INFO] Loading environment: {cfg.task_id}")
    env_cfg = load_env_cfg(cfg.task_id, play=True)
    env_cfg.scene.num_envs = cfg.num_envs
    if cfg.headless:
        env_cfg.viewer.eye = (0.0, 0.0, 0.0)

    # Configure Tracking Task
    if (env_cfg.commands is not None and "motion" in env_cfg.commands
            and isinstance(env_cfg.commands["motion"], MotionCommandCfg)):
        print("[INFO] Tracking task detected. Using default demo motion.")
        motion_cmd = env_cfg.commands["motion"]
        motion_cmd.sampling_mode = "uniform"
        motion_cmd.motion_file = ensure_default_motion()

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

    policy = ActorMLP(num_obs, num_actions).to(device)
    policy.load_state_dict_from_rsl_rl_ckpt(str(resume_path), device)
    policy.eval()

    # Initialize World Model & Buffer
    state_obs = obs["full_state"] if hasattr(obs, "keys") else obs
    num_state = state_obs.shape[-1]
    print(f"[INFO] Policy Dim: {num_obs}, State Dim: {num_state}")

    wm = WorldModel(num_state, num_actions).to(device)
    optimizer = torch.optim.Adam(wm.parameters(), lr=1e-3)

    # Buffer: [state, action, next_state, done]
    replay_buffer = ReplayBuffer(dim=[num_state, num_actions, num_state, 1],
                                 buffer_size=cfg.buffer_size, device=device)

    # Collection Loop
    print(f"[INFO] Collecting {cfg.collection_steps} steps...")

    viewer = None
    if not cfg.headless:
        viewer = NativeMujocoViewer(env, policy)
        viewer.setup()

    def get_obs_vec(o):
        return (o["policy"], o["full_state"]) if hasattr(o, "keys") else (o, o)

    replay_buffer.step = 0
    replay_buffer.num_transitions = 0

    with torch.inference_mode():
        for step in range(cfg.collection_steps):
            curr_policy, curr_state = get_obs_vec(obs)
            action = policy(curr_policy)
            next_obs, _, done, _ = env.step(action)
            _, next_state = get_obs_vec(next_obs)

            if viewer and not viewer.is_running():
                break
            if viewer:
                viewer.sync_env_to_viewer()

            replay_buffer.insert([
                curr_state.unsqueeze(1),
                action.unsqueeze(1),
                next_state.unsqueeze(1),
                done.unsqueeze(1).unsqueeze(2).float()
            ])
            obs = next_obs

            if step % 100 == 0:
                print(f"Step {step}/{cfg.collection_steps} | Buffered: {replay_buffer.num_transitions}")
            if replay_buffer.num_transitions >= cfg.buffer_size:
                break

    # Train
    print("[INFO] Training World Model...")
    train_world_model(wm, optimizer, replay_buffer)

    if viewer: viewer.close()
    env.close()


def train_world_model(
        model: WorldModel,
        optimizer: torch.optim.Optimizer,
        replay_buffer: ReplayBuffer,
        epochs: int = 500,
        batch_size: int = 32,
        window_size: int = 8
):
    """Train World Model auto-regressively using replayed sequences."""
    model.train()
    criterion = torch.nn.MSELoss()

    if replay_buffer.num_transitions < window_size:
        print("[WARN] Insufficient data for training.")
        return

    print(f"[INFO] Training on {replay_buffer.num_transitions} transitions (Window: {window_size})")

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        num_mini_batch = max(1, int(replay_buffer.num_transitions / batch_size))

        # Generator yields [states, actions, next_states, dones]
        data_loader = replay_buffer.mini_batch_generator(
            sequence_length=window_size + 1,
            num_mini_batch=num_mini_batch,
            mini_batch_size=batch_size
        )

        for batch in data_loader:
            states, actions = batch[0], batch[1]
            curr_pred = states[:, 0, :]  # P_0 = S_0
            loss = 0

            # Auto-regressive loop
            for t in range(window_size):
                curr_pred = model(curr_pred, actions[:, t, :])
                loss += criterion(curr_pred, states[:, t + 1, :])  # Compare P_{t+1} vs S_{t+1}

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        print(f"Epoch {epoch + 1}/{epochs} | Avg Loss: {total_loss / num_batches:.6f}")


def main():
    args = tyro.cli(TrainConfig)
    collect_data(args)


if __name__ == "__main__":
    main()
