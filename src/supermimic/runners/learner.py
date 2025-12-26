from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import wandb


class PolicyLearner:
    """Wrapper for Policy (Actor of RL). Currently static/eval mode."""

    def __init__(self, policy: nn.Module):
        self.policy = policy
        self.policy.eval()  # Static

    def get_action(self, obs):
        action = self.policy(obs)
        noise = torch.randn_like(action) * 0.2
        return action + noise


class WorldModelLearner:
    """Trainer for World Model (Supervised)."""

    def __init__(self, model: nn.Module, cfg: Any):
        self.model = model
        self.cfg = cfg
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Pre-calculate slice indices based on model input size
        self.dim = model.net[-1].out_features - 1  # Output is (State Delta + Reward)
        self.num_joints = (self.dim - 15) // 2
        self.slices = {
            'root_pos': slice(0, 3), 'root_ori': slice(3, 9),
            'root_lin_vel': slice(9, 12), 'root_ang_vel': slice(12, 15),
            'joint_pos': slice(15, 15 + self.num_joints),
            'joint_vel': slice(15 + self.num_joints, None)
        }

    def compute_stats(self, replay_buffer):
        print("[INFO] Computing normalization statistics...")
        # Flatten (Batch, Time, Dim) -> (N, Dim) directly
        states = replay_buffer.replay_buf[0][:, :replay_buffer.num_transitions].reshape(-1, self.dim)
        actions = replay_buffer.replay_buf[1][:, :replay_buffer.num_transitions].reshape(-1, replay_buffer.replay_buf[
            1].shape[-1])
        goals = replay_buffer.replay_buf[4][:, :replay_buffer.num_transitions].reshape(-1, replay_buffer.replay_buf[
            4].shape[-1])

        # Mask root position for stats
        states_masked = states.clone()
        states_masked[:, :3] = 0.0

        inputs = torch.cat([states_masked, actions, goals], dim=-1)
        self.model.input_mean.copy_(inputs.mean(dim=0))
        self.model.input_std.copy_(inputs.std(dim=0))
        print(f"[INFO] Stats -- Mean: {inputs.mean():.4f}, Std: {inputs.std():.4f}")

    def _get_loss_weights(self, device):
        """Creates a weight tensor matching state dimension (Broadcasting ready)."""
        W = torch.ones(self.dim, device=device)
        c = self.cfg
        weights = {
            'root_pos': 1.0 / c.nominal_error_root_pos ** 2,
            'root_ori': 1.0 / c.nominal_error_root_ori ** 2,
            'root_lin_vel': 1.0 / c.nominal_error_root_lin_vel ** 2,
            'root_ang_vel': 1.0 / c.nominal_error_root_ang_vel ** 2,
            'joint_pos': 1.0 / c.nominal_error_joint_pos ** 2,
            'joint_vel': 1.0 / c.nominal_error_joint_vel ** 2,
        }
        for name, sl in self.slices.items():
            W[sl] = weights[name]
        return W

    def update(self, replay_buffer, log_dir: Path):
        if replay_buffer.num_transitions < self.cfg.window_size:
            print("[WARN] Insufficient data.")
            return

        self.model.train()
        wandb.init(project=self.cfg.wandb_project, group=self.cfg.wandb_group,
                   name=f"{self.cfg.task_id}-{log_dir.name}", config=self.cfg)

        device = next(self.model.parameters()).device
        loss_weights = self._get_loss_weights(device)

        for epoch in range(self.cfg.epochs):
            loader = replay_buffer.mini_batch_generator(
                self.cfg.window_size + 1,
                self.cfg.num_mini_batch,
                self.cfg.mini_batch_size)
            epoch_loss = 0
            epoch_loss_reward = 0
            epoch_metrics = {k: 0.0 for k in self.slices.keys()}
            steps = 0

            for states, actions, _, rewards, goals, _ in loader:
                # 1. Unroll Dynamics (Auto-regressive)
                curr_state = states[:, 0, :]
                preds = []
                pred_rewards = []

                for t in range(self.cfg.window_size):
                    # Output is [State Delta, Reward]
                    out = self.model(curr_state, actions[:, t, :], goals[:, t, :])
                    delta = out[..., :-1]
                    pred_r = out[..., -1:]

                    curr_state = curr_state + delta
                    preds.append(curr_state)
                    pred_rewards.append(pred_r)

                # Stack to (Batch, Window, Dim)
                preds = torch.stack(preds, dim=1)
                pred_rewards = torch.stack(pred_rewards, dim=1)

                targets = states[:, 1:, :]
                target_rewards = rewards[:, :self.cfg.window_size, :]

                # 2. Vectorized Loss Calculation
                sq_err = (preds - targets) ** 2
                loss_state = (sq_err * loss_weights).mean()
                # Reward Loss (Scale by 1/dt^2 to match reward rate error)
                loss_reward = (pred_rewards - target_rewards).pow(2).mean() / (self.cfg.dt ** 2)

                loss = loss_state  # + loss_reward

                # 3. Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 4. Logging
                epoch_loss += loss.item()
                epoch_loss_reward += loss_reward.item()

                with torch.no_grad():
                    for name, sl in self.slices.items():
                        epoch_metrics[name] += sq_err[..., sl].mean().item()
                steps += 1

            # Averages
            avg_loss = epoch_loss / steps
            avg_loss_reward = epoch_loss_reward / steps

            log_dict = {
                "Loss/state": avg_loss,
                "Loss/reward": avg_loss_reward
            }
            for k, v in epoch_metrics.items():
                log_dict[f"Loss/{k}"] = v / steps

            print(f"Epoch {epoch + 1} | Loss: {avg_loss:.6f}")
            wandb.log(log_dict)

            if (epoch + 1) % 100 == 0:
                torch.save(self.model.state_dict(), log_dir / "world_model.pt")
                wandb.save(str(log_dir / "world_model.pt"), base_path=str(log_dir), policy="now")

        wandb.finish()
