"""Script to collect data for SuperTrack world model."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import tyro

import supermimic.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.gcs import ensure_default_checkpoint
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer
from supermimic.policy import Mlp
from supermimic.replay_buffer import ReplayBuffer
from supermimic.world_model import WorldModel


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

    # Nominal errors for Bryson's Rule (Loss Weights)
    nominal_error_root_pos: float = 0.1
    nominal_error_root_ori: float = 0.1
    nominal_error_root_lin_vel: float = 1.0
    nominal_error_root_ang_vel: float = 3.14
    nominal_error_joint_pos: float = 0.1
    nominal_error_joint_vel: float = 3.14


def collect_data(cfg: TrainConfig):
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Environment
    print(f"[INFO] Loading environment: {cfg.task_id}")
    env_cfg = load_env_cfg(cfg.task_id, play=True)
    env_cfg.scene.num_envs = cfg.num_envs
    if cfg.headless:
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

    policy = Mlp(num_obs, num_actions, normalization=True).to(device)
    policy.load_state_dict_from_rsl_rl_ckpt(str(resume_path), device)
    policy.eval()

    # Initialize World Model & Buffer
    state_obs = obs["full_state"] if hasattr(obs, "keys") else obs
    num_state = state_obs.shape[-1]
    print(f"[INFO] Policy Dim: {num_obs}, State Dim: {num_state}")

    wm = WorldModel(num_state, num_actions).to(device)
    optimizer = torch.optim.Adam(wm.parameters(), lr=1e-5)

    # Buffer: [state, action, next_state, done]
    replay_buffer = ReplayBuffer(dim=[num_state, num_actions, num_state, 1],
                                 buffer_size=cfg.buffer_size, device=device)

    # Collection Loop
    print(f"[INFO] Collecting {cfg.buffer_size} steps...")

    viewer = None
    if not cfg.headless:
        viewer = NativeMujocoViewer(env, policy)
        viewer.setup()

    def get_obs_vec(o):
        return o["policy"], o["full_state"]

    replay_buffer.step = 0
    replay_buffer.num_transitions = 0

    with torch.inference_mode():
        for step in range(cfg.buffer_size):
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
                print(f"Step {step}/{cfg.buffer_size}")
            if replay_buffer.num_transitions >= cfg.buffer_size:
                break

    # Compute Normalization Stats from Buffer
    print("[INFO] Computing normalization statistics...")
    valid_steps = replay_buffer.num_transitions

    # Extract and Flatten (NumEnvs * ValidSteps, Dim)
    all_states = replay_buffer.replay_buf[0][:, :valid_steps, :].reshape(-1, num_state)
    all_actions = replay_buffer.replay_buf[1][:, :valid_steps, :].reshape(-1, num_actions)

    all_inputs = torch.cat([all_states, all_actions], dim=-1)

    input_mean = all_inputs.mean(dim=0)
    input_std = all_inputs.std(dim=0)

    # Copy to World Model
    wm.input_mean.copy_(input_mean)
    wm.input_std.copy_(input_std)

    # Train
    log_dir = Path("logs") / cfg.task_id / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging to {log_dir}")

    print("[INFO] Training World Model...")
    train_world_model(wm, optimizer, replay_buffer, cfg, log_dir)

    if viewer: viewer.close()
    env.close()


def train_world_model(
        model: WorldModel,
        optimizer: torch.optim.Optimizer,
        replay_buffer: ReplayBuffer,
        cfg: TrainConfig,
        log_dir: Path,
        batch_size: int = 256,
        window_size: int = 8
):
    """Train World Model auto-regressively using replayed sequences."""
    model.train()
    criterion = torch.nn.MSELoss()

    if replay_buffer.num_transitions < window_size:
        print("[WARN] Insufficient data for training.")
        return

    print(f"[INFO] Training on {replay_buffer.num_transitions} transitions (Window: {window_size})")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(log_dir))

    # Weights (Bryson's Rule: 1 / nominal_error^2)
    w_root_pos = 1.0 / (cfg.nominal_error_root_pos ** 2)
    w_root_ori = 1.0 / (cfg.nominal_error_root_ori ** 2)
    w_root_lin = 1.0 / (cfg.nominal_error_root_lin_vel ** 2)
    w_root_ang = 1.0 / (cfg.nominal_error_root_ang_vel ** 2)
    w_joint_pos = 1.0 / (cfg.nominal_error_joint_pos ** 2)
    w_joint_vel = 1.0 / (cfg.nominal_error_joint_vel ** 2)

    for epoch in range(cfg.epochs):
        total_loss = 0
        num_batches = 0
        num_mini_batch = max(1, int(replay_buffer.num_transitions / batch_size))

        data_loader = replay_buffer.mini_batch_generator(
            sequence_length=window_size + 1,
            num_mini_batch=num_mini_batch,
            mini_batch_size=batch_size
        )

        for batch in data_loader:
            states, actions = batch[0], batch[1]
            curr_pred = states[:, 0, :]  # P_0 = S_0
            loss = 0

            # Identify indices
            num_state = states.shape[-1]
            num_joints = (num_state - 15) // 2

            idx_rp = 3
            idx_ro = 9
            idx_rl = 12
            idx_ra = 15
            idx_qp = 15 + num_joints
            idx_qv = 15 + 2 * num_joints

            # Auto-regressive loop
            for t in range(window_size):
                curr_pred = curr_pred + model(curr_pred, actions[:, t, :])
                target = states[:, t + 1, :]

                # Weighted MSE Loss
                loss += w_root_pos * criterion(curr_pred[:, :idx_rp], target[:, :idx_rp])
                loss += w_root_ori * criterion(curr_pred[:, idx_rp:idx_ro], target[:, idx_rp:idx_ro])
                loss += w_root_lin * criterion(curr_pred[:, idx_ro:idx_rl], target[:, idx_ro:idx_rl])
                loss += w_root_ang * criterion(curr_pred[:, idx_rl:idx_ra], target[:, idx_rl:idx_ra])
                loss += w_joint_pos * criterion(curr_pred[:, idx_ra:idx_qp], target[:, idx_ra:idx_qp])
                loss += w_joint_vel * criterion(curr_pred[:, idx_qp:idx_qv], target[:, idx_qp:idx_qv])
            loss = loss / window_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{cfg.epochs} | Avg Loss: {avg_loss:.6f}")

        # Log to TensorBoard
        writer.add_scalar("Loss/train", avg_loss, epoch)

        # Save Checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            # Save to log dir
            torch.save(model.state_dict(), log_dir / f"world_model_{epoch + 1}.pt")
            torch.save(model.state_dict(), log_dir / "world_model.pt")

    writer.close()


def main():
    args = tyro.cli(TrainConfig)
    collect_data(args)


if __name__ == "__main__":
    main()
