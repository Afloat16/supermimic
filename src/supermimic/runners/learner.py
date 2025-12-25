from pathlib import Path
from typing import Any

import torch
import torch.nn as nn



class PolicyLearner:
    """Wrapper for Policy (Actor of RL). Currently static/eval mode."""

    def __init__(self, policy: nn.Module):
        self.policy = policy
        self.policy.eval()  # Static

    def get_action(self, obs):
        return self.policy(obs)


class WorldModelLearner:
    """Trainer for World Model (Supervised/Self-Supervised)."""

    def __init__(self, model: nn.Module, cfg: Any):
        self.model = model
        self.cfg = cfg
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Hardcoded in train.py
        self.criterion = nn.MSELoss()

    def compute_stats(self, replay_buffer):
        print("[INFO] Computing normalization statistics...")
        valid_steps = replay_buffer.num_transitions
        num_state = self.model.net[0].in_features - self.model.net[0].out_features  # Trick or pass dim?

        all_states = replay_buffer.replay_buf[0][:, :valid_steps, :]
        all_actions = replay_buffer.replay_buf[1][:, :valid_steps, :]

        # Reshape
        # We need to flatten E and T.
        all_states = all_states.reshape(-1, all_states.shape[-1])
        all_actions = all_actions.reshape(-1, all_actions.shape[-1])

        all_inputs = torch.cat([all_states, all_actions], dim=-1)

        input_mean = all_inputs.mean(dim=0)
        input_std = all_inputs.std(dim=0)

        self.model.input_mean.copy_(input_mean)
        self.model.input_std.copy_(input_std)
        print(f"[INFO] Norm Stats -- Mean: {input_mean.mean():.4f}, Std: {input_std.mean():.4f}")

    def update(self, replay_buffer, log_dir: Path):
        self.model.train()
        window_size = self.cfg.window_size
        batch_size = self.cfg.batch_size

        if replay_buffer.num_transitions < window_size:
            print("[WARN] Insufficient data for training.")
            return

        print(f"[INFO] Training on {replay_buffer.num_transitions} transitions (Window: {window_size})")

        import wandb
        wandb.init(
            project=self.cfg.wandb_project,
            group=self.cfg.wandb_group,
            name=f"{self.cfg.task_id}-{log_dir.name}",
            config=self.cfg,  # wandb handles dataclasses automatically
            dir=str(log_dir) # Set dir so wandb files go there? No, explicit dir often better.
        )

        # Weights (Bryson's Rule)
        # Assuming cfg has these fields.
        w_root_pos = 1.0 / (self.cfg.nominal_error_root_pos ** 2)
        w_root_ori = 1.0 / (self.cfg.nominal_error_root_ori ** 2)
        w_root_lin = 1.0 / (self.cfg.nominal_error_root_lin_vel ** 2)
        w_root_ang = 1.0 / (self.cfg.nominal_error_root_ang_vel ** 2)
        w_joint_pos = 1.0 / (self.cfg.nominal_error_joint_pos ** 2)
        w_joint_vel = 1.0 / (self.cfg.nominal_error_joint_vel ** 2)

        for epoch in range(self.cfg.epochs):
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
                curr_pred = states[:, 0, :]
                loss = 0

                # Indices logic
                num_state = states.shape[-1]
                num_joints = (num_state - 15) // 2
                idx_rp = 3
                idx_ro = 9
                idx_rl = 12
                idx_ra = 15
                idx_qp = 15 + num_joints
                idx_qv = 15 + 2 * num_joints

                for t in range(window_size):
                    # Residual Delta Prediction
                    curr_pred = curr_pred + self.model(curr_pred, actions[:, t, :])
                    target = states[:, t + 1, :]

                    loss += w_root_pos * self.criterion(curr_pred[:, :idx_rp], target[:, :idx_rp])
                    loss += w_root_ori * self.criterion(curr_pred[:, idx_rp:idx_ro], target[:, idx_rp:idx_ro])
                    loss += w_root_lin * self.criterion(curr_pred[:, idx_ro:idx_rl], target[:, idx_ro:idx_rl])
                    loss += w_root_ang * self.criterion(curr_pred[:, idx_rl:idx_ra], target[:, idx_rl:idx_ra])
                    loss += w_joint_pos * self.criterion(curr_pred[:, idx_ra:idx_qp], target[:, idx_ra:idx_qp])
                    loss += w_joint_vel * self.criterion(curr_pred[:, idx_qp:idx_qv], target[:, idx_qp:idx_qv])

                loss = loss / window_size

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{self.cfg.epochs} | Avg Loss: {avg_loss:.6f}")
            wandb.log({"Loss/train": avg_loss})

            if (epoch + 1) % 100 == 0:
                torch.save(self.model.state_dict(), log_dir / "world_model.pt")
                wandb.save(str(log_dir / "world_model.pt"), base_path=str(log_dir), policy="now")

        wandb.finish()
