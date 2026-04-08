"""World model learner (supervised training)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn

from supermimic.learners.learner_utils import _optimize_step


@dataclass
class WorldModelLearnerCfg:
  lr: float = 1e-3
  num_batches: int = 20
  batch_size: int = 2048
  horizon: int = 8

  # Loss Weights (Bryson's Rule) — body-frame representation
  nominal_error_root_pos_z: float = 0.1
  nominal_error_root_gravity_b: float = 0.1
  nominal_error_root_lin_vel: float = 1.0
  nominal_error_root_ang_vel: float = 3.14
  nominal_error_joint_pos: float = 0.1
  nominal_error_joint_vel: float = 3.14

  # Loss Weights — maxcoord representation
  nominal_error_body_pos: float = 0.05
  nominal_error_body_rot: float = 0.1
  nominal_error_body_vel: float = 1.0
  nominal_error_body_avel: float = 3.14
  nominal_error_root_height: float = 0.1
  nominal_error_up_dir: float = 0.1


class WorldModelLearner:
  """Trainer for World Model (Supervised)."""

  def __init__(
    self,
    model: nn.Module,
    cfg: WorldModelLearnerCfg,
    dt: float,
    device: str = "cpu",
    reward_names: list[str] | None = None,
  ) -> None:
    self.model = model
    self.cfg = cfg
    self.device = device
    self.dt = dt
    self.reward_names = reward_names or ["reward"]
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    representation = self.model.representation
    self.slices = representation.loss_slices()
    self.loss_weights = representation.loss_weights(cfg, device)
    self._predict_reward = representation.has_learned_reward

    self._compiled_rollout = torch.compile(self._rollout, mode="reduce-overhead")
    self._compiled_compute_losses = torch.compile(self._compute_losses, mode="default")

  def update_normalization(
    self, states: torch.Tensor, actions: torch.Tensor, goals: torch.Tensor
  ) -> None:
    representation = self.model.representation
    rep_states = representation.represent(states)
    rep_states_flat = rep_states.reshape(-1, rep_states.shape[-1])

    actions_flat = actions.reshape(-1, actions.shape[-1])

    # The policy learner's _prepare_update() sets wm.eval(), which puts the
    # normalizers in eval mode. Toggle to train so update() is not a no-op.
    for n in self.model.normalizers():
      n.train()
    self.model.state_normalizer.update(rep_states_flat)
    self.model.action_normalizer.update(actions_flat)
    self.model.goal_normalizer.update(goals)
    for n in self.model.normalizers():
      n.eval()

  def update(self, data: Iterable[tuple[torch.Tensor, ...]]) -> dict[str, float]:
    self.model.train()
    for p in self.model.parameters():
      p.requires_grad_(True)

    device = next(self.model.parameters()).device
    representation = self.model.representation

    epoch_loss_total = torch.zeros(1, device=device)
    epoch_loss_state = torch.zeros(1, device=device)
    epoch_loss_reward = torch.zeros(1, device=device)
    epoch_loss_term = torch.zeros(1, device=device)
    _metric_slice_list = list(self.slices.items())
    epoch_metrics_gpu = torch.zeros(len(_metric_slice_list), device=device)
    epoch_reward_mse = torch.zeros(len(self.reward_names), device=device)

    for full_states, actions, goals, rewards, _, dones in data:
      # Window length is exactly `horizon`. Rollout produces `horizon` predictions
      # but the last one (state[horizon]) has no GT in the window, so it is dropped.
      all_rep_states = representation.represent(full_states)

      init_state = all_rep_states[:, 0, :]
      init_aux = representation.extract_aux_state(full_states[:, 0, :])

      pred_states_all, pred_rewards_all, pred_dones_all = self._compiled_rollout(
        init_state,
        actions,
        goals,
        init_aux,
      )

      # Drop the trailing state prediction whose GT lies outside the window.
      pred_states = pred_states_all[:, : self.cfg.horizon - 1, :]
      gt_states = all_rep_states[:, 1 : self.cfg.horizon, :]
      gt_rewards = rewards
      gt_dones = dones

      # 2. Vectorized Loss Calculation (always FP32, compiled for kernel fusion)
      loss, loss_state, loss_reward, loss_term, sq_err, reward_mse = (
        self._compiled_compute_losses(
          pred_states,
          pred_rewards_all,
          pred_dones_all,
          gt_states,
          gt_rewards,
          gt_dones,
        )
      )

      # 3. Backprop
      _optimize_step(self.optimizer, loss, self.model.parameters(), max_norm=1.0)

      # 4. Logging (GPU accumulation — no sync here, .item() deferred to after loop)
      epoch_loss_total += loss.detach()
      epoch_loss_state += loss_state.detach()
      epoch_loss_reward += loss_reward.detach()
      epoch_loss_term += loss_term.detach()

      with torch.no_grad():
        for i, (_, sl) in enumerate(_metric_slice_list):
          epoch_metrics_gpu[i] += sq_err[..., sl].mean()

        if reward_mse.shape[0] == len(self.reward_names):
          epoch_reward_mse += reward_mse.detach()

    # Averages — single .item() sync per scalar after all batches are done
    n = self.cfg.num_batches
    log_dict = {
      "Loss/total": epoch_loss_total.item() / n,
      "Loss/state": epoch_loss_state.item() / n,
      "Loss/reward": epoch_loss_reward.item() / n,
      "Loss/termination": epoch_loss_term.item() / n,
    }
    avg_metrics_gpu = epoch_metrics_gpu / n
    for i, (name, _) in enumerate(_metric_slice_list):
      log_dict[f"Loss/{name}"] = avg_metrics_gpu[i].item()

    avg_reward_mse = epoch_reward_mse / self.cfg.num_batches
    for i, name in enumerate(self.reward_names):
      log_dict[f"Loss/reward_{name}"] = avg_reward_mse[i].item()

    return log_dict

  def _rollout(
    self,
    init_state: torch.Tensor,
    actions: torch.Tensor,
    goals: torch.Tensor,
    init_aux: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Autoregressive rollout for `horizon` steps using the world model."""
    horizon = actions.shape[1]
    curr_state = init_state
    curr_aux = init_aux

    pred_states: list[torch.Tensor] = []
    pred_rewards: list[torch.Tensor] = []
    pred_dones: list[torch.Tensor] = []

    for k in range(horizon):
      curr_a = actions[:, k, :]
      last_a = actions[:, k - 1, :] if k > 0 else torch.zeros_like(curr_a)
      curr_g = goals[:, k, :]

      curr_aux, next_state, pred_r, pred_d = self.model(
        curr_state, curr_a, last_a, curr_g, curr_aux, True
      )

      curr_state = next_state

      pred_states.append(next_state)
      pred_rewards.append(pred_r)
      pred_dones.append(pred_d)

    return (
      torch.stack(pred_states, dim=1),
      torch.stack(pred_rewards, dim=1),
      torch.stack(pred_dones, dim=1),
    )

  def _compute_losses(
    self,
    pred_states: torch.Tensor,
    pred_rewards: torch.Tensor,
    pred_dones: torch.Tensor,
    gt_states: torch.Tensor,
    gt_rewards: torch.Tensor,
    gt_dones: torch.Tensor,
  ) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
  ]:
    # Mean Squared Error on State
    sq_err = (pred_states - gt_states) ** 2

    # Weighted Mean on body-state channels.
    loss_state = (sq_err * self.loss_weights).mean()

    # Reward Loss (skip when reward is computed analytically, not learned)
    if self._predict_reward:
      reward_mse = (pred_rewards - gt_rewards).pow(2)
      reward_mse_per_id = reward_mse.mean(dim=(0, 1))
      loss_reward = reward_mse_per_id.sum()
    else:
      loss_reward = torch.zeros(1, device=pred_states.device).squeeze()
      reward_mse_per_id = torch.zeros(pred_rewards.shape[-1], device=pred_states.device)

    # Termination Loss — float masks instead of bool indexing (avoids CPU-GPU sync).
    # Timeouts (gt_dones==2) masked out; terminations (gt_dones==1) are positives.
    valid_mask = (gt_dones != 2.0).float()
    gt_d_binary = (gt_dones == 1.0).float()

    num_pos = gt_d_binary.sum()
    num_valid = valid_mask.sum()
    pos_weight = torch.clamp((num_valid - num_pos) / (num_pos + 1e-6), max=50.0)

    ew = valid_mask * (1.0 + (pos_weight - 1.0) * gt_d_binary)
    bce_raw = nn.functional.binary_cross_entropy_with_logits(
      pred_dones, gt_d_binary, reduction="none"
    )
    loss_term = (bce_raw * ew).sum() / (num_valid + 1e-6)

    loss = loss_state + loss_reward + loss_term

    return loss, loss_state, loss_reward, loss_term, sq_err, reward_mse_per_id
