"""World model learner (supervised training)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn

from supermimic.learners.learner_utils import _optimize_step
from supermimic.utils.helpers import shift_history


@dataclass
class WorldModelLearnerCfg:
  lr: float = 1e-3
  num_batches: int = 20
  batch_size: int = 2048
  horizon: int = 8
  history: int = 1

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
    self.history = cfg.history
    self.reward_names = reward_names or ["reward"]
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    representation = self.model.representation
    self.body_state_dim = representation.rep_state_dim
    self.slices = representation.loss_slices()
    self.loss_weights = representation.loss_weights(cfg, device)
    self._predict_reward = representation.has_learned_reward

    self._compiled_rollout = torch.compile(self._rollout, mode="reduce-overhead")
    self._compiled_compute_losses = torch.compile(self._compute_losses, mode="default")

  def update_normalization(
    self, states: torch.Tensor, actions: torch.Tensor, goals: torch.Tensor
  ) -> None:
    representation = self.model.representation
    body_states = representation.represent(states)
    body_states_flat = body_states.reshape(-1, body_states.shape[-1])
    body_states_flat = body_states_flat.repeat(1, self.history)

    actions_flat = actions.reshape(-1, actions.shape[-1])

    # The policy learner's _prepare_update() sets wm.eval(), which puts the
    # normalizers in eval mode. Toggle to train so update() is not a no-op.
    for n in self.model.normalizers():
      n.train()
    self.model.state_normalizer.update(body_states_flat)
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
      # history=1: start_idx is always 0 (avoid GPU randint + .item() sync).
      if self.cfg.history == 1:
        start_idx = 0
      else:
        start_idx = int(torch.randint(0, self.cfg.history, (1,)).item())
      full_len = full_states.shape[1]

      all_body_states = representation.represent(full_states)

      # Build initial history tensor (padded if needed)
      hist_context = all_body_states[:, : start_idx + 1]
      if hist_context.shape[1] < self.history:
        pad_len = self.history - hist_context.shape[1]
        pad = hist_context[:, 0:1, :].expand(-1, pad_len, -1)
        init_hist = torch.cat([pad, hist_context], dim=1)
      else:
        init_hist = hist_context[:, -self.history :, :]

      # Slice actions/goals for this rollout window
      roll_actions = actions[:, start_idx : start_idx + self.cfg.horizon, :]
      roll_goals = goals[:, start_idx : start_idx + self.cfg.horizon, :]

      init_aux = representation.extract_aux_state(full_states[:, start_idx, :])
      pred_states_all, pred_rewards_all, pred_dones_all = self._compiled_rollout(
        init_hist,
        roll_actions,
        roll_goals,
        init_aux,
      )

      # Only supervise states where GT exists
      state_steps = min(self.cfg.horizon, full_len - start_idx - 1)
      pred_states = pred_states_all[:, :state_steps, :]

      gt_states = all_body_states[:, start_idx + 1 : start_idx + 1 + state_steps, :]
      gt_rewards = rewards[:, start_idx : start_idx + self.cfg.horizon]
      gt_dones = dones[:, start_idx : start_idx + self.cfg.horizon]

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
    init_hist: torch.Tensor,
    actions: torch.Tensor,
    goals: torch.Tensor,
    init_aux: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Autoregressive rollout for `horizon` steps using the world model."""
    horizon = actions.shape[1]
    curr_body_hist = init_hist
    curr_aux = init_aux

    pred_states: list[torch.Tensor] = []
    pred_rewards: list[torch.Tensor] = []
    pred_dones: list[torch.Tensor] = []

    for k in range(horizon):
      curr_a = actions[:, k, :]
      last_a = actions[:, k - 1, :] if k > 0 else torch.zeros_like(curr_a)
      curr_g = goals[:, k, :]

      curr_aux, next_body_state, pred_r, pred_d = self.model(
        curr_body_hist, curr_a, last_a, curr_g, curr_aux, True
      )

      curr_body_hist = shift_history(curr_body_hist, next_body_state)

      pred_states.append(next_body_state)
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
