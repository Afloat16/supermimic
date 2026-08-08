"""World model learner (supervised training)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from tensordict import TensorDict

from supermimic.learners.learner_utils import _optimize_step
from supermimic.representations import (
  StateRepresentation as _BaseStateRepresentation,
)

if TYPE_CHECKING:
  from supermimic.models import Dynamics
  from supermimic.representations import StateRepresentation
  from supermimic.sim_context import SimContext


@dataclass
class WorldModelLearnerCfg:
  lr: float = 2e-4
  num_batches: int = 10
  batch_size: int = 2048
  horizon: int = 8

  # Loss Weights (Bryson's Rule) — integrated root pose, both representations
  nominal_error_root_pose_pos: float = 0.01
  nominal_error_root_pose_ori: float = 0.02

  # Loss Weights — gencoord representation
  nominal_error_joint_pos: float = 0.1
  nominal_error_joint_vel: float = 3.14

  # Loss Weights — maxcoord representation
  nominal_error_body_pos: float = 0.05
  nominal_error_body_rot: float = 0.1
  nominal_error_body_vel: float = 1.0
  nominal_error_body_avel: float = 3.14


class WorldModelLearner:
  """Trainer for World Model (Supervised)."""

  def __init__(
    self,
    dynamics: Dynamics,
    representation: StateRepresentation,
    cfg: WorldModelLearnerCfg,
    sim_context: SimContext,
    device: str = "cpu",
    reward_names: list[str] | None = None,
  ) -> None:
    self.dynamics = dynamics
    self.representation = representation
    self.cfg = cfg
    self.device = device
    self.reward_names = reward_names or ["reward"]
    self.sim_context = sim_context

    self._all_params = (
      list(dynamics.parameters())
      + list(representation.parameters())
      + list(sim_context.parameters())
    )
    self.optimizer = torch.optim.Adam(self._all_params, lr=self.cfg.lr)

    # Bryson's rule: each named channel group is weighted 1 / nominal_error^2.
    self.loss_group_weights = {
      name: 1.0 / getattr(cfg, attr) ** 2
      for name, attr in representation.LOSS_WEIGHT_CONFIG.items()
    }

    self._predict_reward = (
      type(representation).predict_reward is not _BaseStateRepresentation.predict_reward
    )

    self._compiled_rollout = torch.compile(self._rollout, mode="reduce-overhead")
    self._compiled_compute_losses = torch.compile(
      self._compute_losses, mode="reduce-overhead"
    )
    self._compiled_represent = torch.compile(self.representation.represent)

  def update_normalization(
    self, states: torch.Tensor, actions: torch.Tensor, goals: torch.Tensor
  ) -> None:
    """Update state/action/goal normalizers from fresh collected data.

    ``actions`` are raw policy outputs (what the replay buffer stores).
    Representation normalizers see raw (rew/term heads consume raw);
    dynamics' action normalizer sees proc (matching what ``forward``
    feeds the MLP after ``Dynamics.proc``).
    """
    rep_states = self.representation.represent(states)
    rep_states_flat = rep_states.reshape(-1, rep_states.shape[-1])
    actions_flat = actions.reshape(-1, actions.shape[-1])

    # EmpiricalNormalization.update is a no-op in eval mode; toggle to train
    # for the updates and restore afterward.
    all_normalizers = self.dynamics.normalizers() + self.representation.normalizers()
    for n in all_normalizers:
      n.train()

    self.dynamics.state_normalizer.update(rep_states_flat)
    self.dynamics.action_normalizer.update(self.dynamics.proc(actions_flat))
    self.representation.state_normalizer.update(rep_states_flat)
    self.representation.action_normalizer.update(actions_flat)
    self.representation.goal_normalizer.update(goals)

    for n in all_normalizers:
      n.eval()

  def update(self, data: Iterable[TensorDict]) -> dict[str, float]:
    self.dynamics.train()
    self.representation.train()
    for p in self._all_params:
      p.requires_grad_(True)

    device = next(self.dynamics.parameters()).device
    representation = self.representation

    epoch_loss_total = torch.zeros(1, device=device)
    epoch_loss_state = torch.zeros(1, device=device)
    epoch_loss_reward = torch.zeros(1, device=device)
    epoch_loss_term = torch.zeros(1, device=device)
    epoch_loss_obs = torch.zeros(1, device=device)
    obs_head_slice = _proprio_obs_head_slice(representation)
    pred_steps = slice(0, self.cfg.horizon - 1)
    gt_steps = slice(1, self.cfg.horizon)
    epoch_metrics_gpu = torch.zeros(len(self.loss_group_weights), device=device)
    epoch_reward_mse = torch.zeros(len(self.reward_names), device=device)

    for td in data:
      full_states = td["state"]
      actions = td["action"]
      goals = td["goal"]
      gt_rewards = td["reward"]
      gt_terminated = td["terminated"]
      gt_truncated = td["truncated"]

      all_rep_states = self._compiled_represent(full_states)

      all_aux_states = representation.extract_aux_state(full_states)
      init_rep_state = all_rep_states[:, 0, :]
      init_aux = all_aux_states[:, 0, :]

      sim_ctx = self.sim_context.encode_batch(td, step_idx=0)

      pred_rep_states, pred_aux_states, pred_rewards, term_logits, pred_obs = (
        self._compiled_rollout(init_rep_state, actions, goals, init_aux, sim_ctx)
      )

      # 2. Vectorized Loss Calculation (always FP32, compiled for kernel fusion)
      loss, loss_state, loss_reward, loss_term, group_mse, reward_mse = (
        self._compiled_compute_losses(
          pred_rep_states[:, pred_steps],
          pred_aux_states[:, pred_steps],
          pred_rewards,
          term_logits,
          all_rep_states[:, gt_steps],
          all_aux_states[:, gt_steps],
          gt_rewards,
          gt_terminated,
          gt_truncated,
        )
      )

      if representation.obs_head is not None:
        target_wm_obs = td["actor_obs"][..., : representation.wm_obs_dim]
        loss_obs = (
          (pred_obs[..., obs_head_slice] - target_wm_obs[..., obs_head_slice])
          .pow(2)
          .mean()
        )
        loss = loss + loss_obs
      else:
        loss_obs = torch.zeros((), device=device)

      # 3. Backprop
      _optimize_step(self.optimizer, loss, self._all_params, max_norm=1.0)

      # 4. Logging (GPU accumulation — no sync here, .item() deferred to after loop)
      epoch_loss_total += loss.detach()
      epoch_loss_state += loss_state.detach()
      epoch_loss_reward += loss_reward.detach()
      epoch_loss_term += loss_term.detach()
      epoch_loss_obs += loss_obs.detach()

      with torch.no_grad():
        epoch_metrics_gpu += group_mse

        if reward_mse.shape[0] == len(self.reward_names):
          epoch_reward_mse += reward_mse.detach()

    # Averages — single .item() sync per scalar after all batches are done
    n = self.cfg.num_batches
    log_dict = {
      "Loss/total": epoch_loss_total.item() / n,
      "Loss/state": epoch_loss_state.item() / n,
      "Loss/reward": epoch_loss_reward.item() / n,
      "Loss/termination": epoch_loss_term.item() / n,
      "Loss/obs": epoch_loss_obs.item() / n,
    }
    avg_metrics_gpu = epoch_metrics_gpu / n
    for i, name in enumerate(self.loss_group_weights):
      log_dict[f"Loss/{name}"] = avg_metrics_gpu[i].item()

    avg_reward_mse = epoch_reward_mse / self.cfg.num_batches
    for i, name in enumerate(self.reward_names):
      log_dict[f"Loss/reward_{name}"] = avg_reward_mse[i].item()

    return log_dict

  def _rollout(
    self,
    init_rep_state: torch.Tensor,
    actions: torch.Tensor,
    goals: torch.Tensor,
    init_aux: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Autoregressive rollout for `horizon` steps.

    Reward/term consume the detached post-step rep_state so their losses
    don't backprop through the autoregressive chain into earlier dynamics
    steps (parameter sets are already disjoint, so this only blocks the
    activation path). ``compute_wm_obs`` is evaluated on the state/aux
    going INTO each step with all inputs detached, so the obs MSE trains
    only the obs head and not the upstream dynamics.

    Returns:
      (pred_rep_states, pred_aux_states, pred_rewards, pred_dones, pred_obs),
      stacked on dim 1.
    """
    horizon = actions.shape[1]
    curr_rep_state = init_rep_state
    curr_aux = init_aux
    train_obs_head = self.representation.obs_head is not None

    pred_rep_states: list[torch.Tensor] = []
    pred_aux_states: list[torch.Tensor] = []
    pred_rewards: list[torch.Tensor] = []
    term_logits: list[torch.Tensor] = []
    pred_obs: list[torch.Tensor] = []

    for k in range(horizon):
      curr_a = actions[:, k, :]
      last_a = actions[:, k - 1, :] if k > 0 else torch.zeros_like(curr_a)
      curr_g = goals[:, k, :]

      if train_obs_head:
        pred_obs.append(
          self.representation.compute_wm_obs(
            curr_rep_state.detach(),
            curr_aux.detach(),
            sim_ctx,
          )
        )

      next_aux, next_rep_state = self.dynamics(
        curr_rep_state, curr_a, curr_aux, sim_ctx
      )
      next_rep_state_det = next_rep_state.detach()

      if self._predict_reward:
        pred_r = self.representation.predict_reward(
          next_rep_state_det, curr_a, last_a, curr_g
        )
      else:
        pred_r = curr_rep_state.new_zeros(
          curr_rep_state.shape[0], len(self.reward_names)
        )
      term_logit = self.representation.predict_termination(
        next_rep_state_det, curr_a, curr_g
      )

      curr_rep_state = next_rep_state
      curr_aux = next_aux

      pred_rep_states.append(next_rep_state)
      pred_aux_states.append(next_aux)
      pred_rewards.append(pred_r)
      term_logits.append(term_logit)

    pred_obs_out = (
      torch.stack(pred_obs, dim=1) if train_obs_head else init_rep_state.new_zeros(0)
    )
    return (
      torch.stack(pred_rep_states, dim=1),
      torch.stack(pred_aux_states, dim=1),
      torch.stack(pred_rewards, dim=1),
      torch.stack(term_logits, dim=1),
      pred_obs_out,
    )

  def _compute_losses(
    self,
    pred_rep_states: torch.Tensor,
    pred_aux_states: torch.Tensor,
    pred_rewards: torch.Tensor,
    term_logits: torch.Tensor,
    gt_rep_states: torch.Tensor,
    gt_aux_states: torch.Tensor,
    gt_rewards: torch.Tensor,
    gt_terminated: torch.Tensor,
    gt_truncated: torch.Tensor,
  ) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
  ]:
    # Squared error per named channel group. Bryson's rule scales each group by
    # its weight; the state loss is the plain mean over every channel.
    loss_blocks = self.representation.loss_blocks
    pred_blocks = loss_blocks(pred_rep_states, pred_aux_states)
    gt_blocks = loss_blocks(gt_rep_states, gt_aux_states)
    sq_errs = {name: (pred_blocks[name] - gt_blocks[name]) ** 2 for name in pred_blocks}
    widths = torch.tensor(
      [sq.shape[-1] for sq in sq_errs.values()], device=pred_rep_states.device
    )
    group_mse = torch.stack([sq.mean() for sq in sq_errs.values()])
    weights = torch.tensor(
      [self.loss_group_weights[name] for name in sq_errs], device=group_mse.device
    )
    loss_state = (group_mse * weights * widths).sum() / widths.sum()

    # Reward Loss (skip when reward is computed analytically, not learned)
    if self._predict_reward:
      reward_mse = (pred_rewards - gt_rewards).pow(2)
      reward_mse_per_id = reward_mse.mean(dim=(0, 1))
      loss_reward = reward_mse_per_id.sum()
    else:
      loss_reward = torch.zeros(1, device=pred_rep_states.device).squeeze()
      reward_mse_per_id = torch.zeros(
        pred_rewards.shape[-1], device=pred_rep_states.device
      )

    # Termination Loss — truncations masked out; terminations are positives.
    term_logits = term_logits.squeeze(-1)
    num_pos = gt_terminated.float().sum()
    num_valid = (~gt_truncated).float().sum()
    pos_weight = torch.clamp((num_valid - num_pos) / (num_pos + 1e-6), max=50.0)

    ew = (~gt_truncated).float() * (1.0 + (pos_weight - 1.0) * gt_terminated.float())
    bce_raw = nn.functional.binary_cross_entropy_with_logits(
      term_logits, gt_terminated.float(), reduction="none"
    )
    loss_term = (bce_raw * ew).sum() / (num_valid + 1e-6)

    loss = loss_state + loss_reward + loss_term

    return loss, loss_state, loss_reward, loss_term, group_mse, reward_mse_per_id


def _proprio_obs_head_slice(representation: _BaseStateRepresentation) -> slice:
  offset = representation.wm_obs_dim - 2 * representation.num_joints
  return slice(offset, offset + 2 * representation.num_joints)
