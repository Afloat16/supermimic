"""World model learner (supervised training)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from supermimic.learners.learner_utils import (
  CapturedUpdate,
  _optimize_step,
  graph_state,
)
from supermimic.representations import (
  StateRepresentation as _BaseStateRepresentation,
)

if TYPE_CHECKING:
  from mjlab.tasks.tracking.mdp.commands import MotionLoader

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
    motion: MotionLoader,
    anchor_index: int,
    device: str = "cpu",
    reward_names: list[str] | None = None,
  ) -> None:
    self.dynamics = dynamics
    self.representation = representation
    self.cfg = cfg
    self.device = device
    self.reward_names = reward_names or ["reward"]
    self.sim_context = sim_context
    self.motion = motion
    self._anchor_index = anchor_index

    self._all_params = (
      list(dynamics.parameters())
      + list(representation.parameters())
      + list(sim_context.parameters())
    )
    self.optimizer = torch.optim.Adam(
      self._all_params,
      lr=torch.tensor(self.cfg.lr, device=self.device),
      fused=True,
      capturable=True,
    )

    # Bryson's rule: each named channel group is weighted 1 / nominal_error^2.
    self.loss_group_weights = {
      name: 1.0 / getattr(cfg, attr) ** 2
      for name, attr in representation.LOSS_WEIGHT_CONFIG.items()
    }

    self._predict_reward = (
      type(representation).predict_reward is not _BaseStateRepresentation.predict_reward
    )
    self._batch_keys = ["state", "action", *(spec.name for spec in sim_context.specs)]
    if self._predict_reward:
      self._batch_keys += ["motion_time", "reward"]

    self._compiled_rollout = torch.compile(self._rollout)
    self._compiled_compute_losses = torch.compile(self._compute_losses)
    self._compiled_represent = torch.compile(self.representation.represent)
    self._captured_update = CapturedUpdate(
      self._update_from,
      lambda: graph_state([dynamics, representation, sim_context], [self.optimizer]),
    )

  def update_normalization(
    self, states: torch.Tensor, actions: torch.Tensor, goals: torch.Tensor
  ) -> None:
    """Update state/action/goal normalizers from fresh collected data.

    ``actions`` are raw policy outputs (what the replay buffer stores).
    Representation normalizers see raw (the reward head consumes raw);
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

  def update(self, data: Iterable[TensorDict]) -> dict[str, torch.Tensor]:
    self.dynamics.train()
    self.representation.train()
    for p in self._all_params:
      p.requires_grad_(True)
    batches = torch.stack([td.select(*self._batch_keys) for td in data])
    return self._captured_update(batches)

  def _update_from(self, batches: TensorDict) -> dict[str, torch.Tensor]:
    representation = self.representation

    pred_steps = slice(0, self.cfg.horizon - 1)
    gt_steps = slice(1, self.cfg.horizon)
    stats = []

    for td in batches.unbind(0):
      full_states = td["state"]
      actions = td["action"]
      gt_rewards = td["reward"] if self._predict_reward else None

      all_rep_states = self._compiled_represent(full_states)

      all_aux_states = representation.extract_aux_state(full_states)
      goals = None
      if self._predict_reward:
        goals, _ = representation.compute_wm_goal(
          self.motion,
          td["motion_time"].flatten(),
          all_aux_states.flatten(0, 1),
          self._anchor_index,
        )
        goals = goals.unflatten(0, td.batch_size)
      init_rep_state = all_rep_states[:, 0, :]
      init_aux = all_aux_states[:, 0, :]

      sim_ctx = self.sim_context.encode_batch(td, step_idx=0)

      pred_rep_states, pred_aux_states, pred_rewards = self._compiled_rollout(
        init_rep_state, actions, goals, init_aux, sim_ctx
      )

      # 2. Vectorized Loss Calculation (always FP32, compiled for kernel fusion)
      loss, loss_state, loss_reward, group_mse, reward_mse = (
        self._compiled_compute_losses(
          pred_rep_states[:, pred_steps],
          pred_aux_states[:, pred_steps],
          pred_rewards,
          all_rep_states[:, gt_steps],
          all_aux_states[:, gt_steps],
          gt_rewards,
        )
      )

      # 3. Backprop
      _optimize_step(self.optimizer, loss, self._all_params, max_norm=1.0)

      # 4. Logging
      scalars = torch.stack([loss, loss_state, loss_reward])
      stats.append(torch.cat([scalars, group_mse, reward_mse]).detach())

    names = ["total", "state", "reward", *self.loss_group_weights]
    names += [f"reward_{name}" for name in self.reward_names]
    means = torch.stack(stats).mean(0)
    return {f"Loss/{name}": m for name, m in zip(names, means, strict=True)}

  def _rollout(
    self,
    init_rep_state: torch.Tensor,
    actions: torch.Tensor,
    goals: torch.Tensor | None,
    init_aux: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Autoregressive rollout over the window's actions.

    The reward head consumes the detached post-step rep_state so its loss
    doesn't backprop through the autoregressive chain into earlier dynamics
    steps (parameter sets are already disjoint, so this only blocks the
    activation path). Without a reward head ``goals`` is None, and the
    prediction from the last action is never scored, so the rollout stops one
    step short.

    Returns:
      (pred_rep_states, pred_aux_states, pred_rewards), stacked on dim 1.
    """
    horizon = actions.shape[1]
    steps = horizon if goals is not None else horizon - 1
    curr_rep_state = init_rep_state
    curr_aux = init_aux

    pred_rep_states: list[torch.Tensor] = []
    pred_aux_states: list[torch.Tensor] = []
    pred_rewards: list[torch.Tensor] = []

    for k in range(steps):
      curr_a = actions[:, k, :]
      last_a = actions[:, k - 1, :] if k > 0 else torch.zeros_like(curr_a)

      next_aux, next_rep_state = self.dynamics(
        curr_rep_state, curr_a, curr_aux, sim_ctx
      )
      next_rep_state_det = next_rep_state.detach()

      if goals is not None:
        pred_r = self.representation.predict_reward(
          next_rep_state_det, curr_a, last_a, goals[:, k, :]
        )
      else:
        pred_r = curr_rep_state.new_zeros(
          curr_rep_state.shape[0], len(self.reward_names)
        )

      curr_rep_state = next_rep_state
      curr_aux = next_aux

      pred_rep_states.append(next_rep_state)
      pred_aux_states.append(next_aux)
      pred_rewards.append(pred_r)

    return (
      torch.stack(pred_rep_states, dim=1),
      torch.stack(pred_aux_states, dim=1),
      torch.stack(pred_rewards, dim=1),
    )

  def _compute_losses(
    self,
    pred_rep_states: torch.Tensor,
    pred_aux_states: torch.Tensor,
    pred_rewards: torch.Tensor,
    gt_rep_states: torch.Tensor,
    gt_aux_states: torch.Tensor,
    gt_rewards: torch.Tensor | None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Squared error per named channel group. Bryson's rule scales each group by
    # its weight; the state loss is the plain mean over every channel.
    loss_blocks = self.representation.loss_blocks
    pred_blocks = loss_blocks(pred_rep_states, pred_aux_states)
    gt_blocks = loss_blocks(gt_rep_states, gt_aux_states)
    sq_errs = {name: (pred_blocks[name] - gt_blocks[name]) ** 2 for name in pred_blocks}
    group_mse = torch.stack([sq.mean() for sq in sq_errs.values()])
    # Filled on the device: a captured update cannot copy from the host.
    widths = torch.stack(
      [group_mse.new_full((), sq.shape[-1]) for sq in sq_errs.values()]
    )
    weights = torch.stack(
      [group_mse.new_full((), self.loss_group_weights[name]) for name in sq_errs]
    )
    loss_state = (group_mse * weights * widths).sum() / widths.sum()

    # Reward Loss (skip when reward is computed analytically, not learned)
    if gt_rewards is not None:
      reward_mse = (pred_rewards - gt_rewards).pow(2)
      reward_mse_per_id = reward_mse.mean(dim=(0, 1))
      loss_reward = reward_mse_per_id.sum()
    else:
      loss_reward = torch.zeros(1, device=pred_rep_states.device).squeeze()
      reward_mse_per_id = torch.zeros(
        pred_rewards.shape[-1], device=pred_rep_states.device
      )

    loss = loss_state + loss_reward

    return loss, loss_state, loss_reward, group_mse, reward_mse_per_id
