from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated

import torch
import torch.nn as nn
import tyro
import wandb
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from supermimic.envs.motion_tracking_env import MotionTrackingEnv
from supermimic.learners.learner_utils import load_wandb_checkpoint
from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
  BpttPolicyLearner,
  BpttPolicyLearnerCfg,
)
from supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
from supermimic.models import Dynamics, VAEActor
from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.representations import (
  actor_goal_dim,
  create_representation,
  extract_actor_goal,
)
from supermimic.utils.helpers import extract_action_scale_offset, unpack_obs_dims
from supermimic.utils.logger import Logger


@dataclass
class BaseRunnerCfg:
  """Shared configuration fields for all runners."""

  wandb_project: str = "supermimic"
  run_name: str | None = None
  wandb_run_path: str | None = None
  headless: bool = True
  buffer_size: int = 2000
  representation: str = "gencoord"  # "gencoord" or "maxcoord"
  log_interval: int = 10
  save_interval: int = 1000
  wm: WorldModelLearnerCfg = field(default_factory=WorldModelLearnerCfg)


BpttCfg = Annotated[BpttPolicyLearnerCfg, tyro.conf.subcommand(name="bptt")]
ActorCriticCfg = Annotated[
  ActorCriticPolicyLearnerCfg, tyro.conf.subcommand(name="actor-critic")
]


@dataclass
class SuperMimicRunnerCfg(BaseRunnerCfg):
  max_iterations: int = 1000000
  replay_ratio: int = 128
  actor_type: str = "mlp"  # "mlp" or "vae"
  latent_dim: int = 64
  fixed_std: float = 0.3
  kl_weight: float = 5e-5
  policy: BpttCfg | ActorCriticCfg = field(default_factory=BpttPolicyLearnerCfg)


class Ratio:
  def __init__(self, ratio: float) -> None:
    self._ratio = ratio
    self._prev: int | None = None

  def __call__(self, step: int) -> int:
    step = int(step)
    if self._ratio == 0:
      return 0
    if self._ratio < 0:
      return 1
    if self._prev is None:
      self._prev = step
      return 1
    repeats = int((step - self._prev) * self._ratio)
    self._prev += repeats / self._ratio
    return repeats


class BaseRunner[CfgT: BaseRunnerCfg]:
  """Base class with shared initialization for all runners.

  Handles: env dimension extraction, representation creation, world model
  and learner setup, replay buffer, action manager, and logger.

  Subclasses implement ``_init_policy()`` for their specific policy setup.
  """

  cfg: CfgT

  def __init__(
    self,
    env,
    cfg: CfgT,
    task_id: str,
    registry_name: str | None = None,
    device: str = "cuda:0",
  ):
    self.env = env
    self.cfg = cfg
    self.task_id = task_id
    self.registry_name = registry_name
    self.device = device

    self._init_common()
    self._init_policy()

    self.logger = Logger(
      cfg=asdict(cfg),
      num_envs=env.num_envs,
      device=device,
    )

  def _init_common(self) -> None:
    """Initialize components shared by all runners."""
    # Dimensions
    obs, _ = self.env.reset()
    num_full_state, num_goals = unpack_obs_dims(obs)
    num_actions = self.env.num_actions
    dt = self.env.unwrapped.step_dt

    print(f"[INFO] State Dim: {num_full_state}, Goal Dim: {num_goals}")

    # Resolve reward dim before building the representation so the reward
    # head gets the correct output dim.
    self.reward_manager = self.env.unwrapped.reward_manager
    self.reward_names = getattr(self.reward_manager, "active_terms", ["reward"])
    num_rewards = len(self.reward_names)
    print(f"[INFO] Reward Dim: {num_rewards} ({self.reward_names})")

    representation = create_representation(
      self.cfg.representation,
      num_full_state,
      num_actions,
      num_goals,
      num_rewards,
      dt,
    ).to(self.device)
    self._representation = representation

    print(
      f"[INFO] Representation: {self.cfg.representation} "
      f"(full={representation.full_state_dim}, rep={representation.rep_state_dim}, "
      f"pred={representation.prediction_dim})"
    )

    action_scale, action_offset = extract_action_scale_offset(self.env)
    dynamics = Dynamics(representation, num_actions, action_scale, action_offset).to(
      self.device
    )

    self.world_model_learner = WorldModelLearner(
      dynamics,
      representation,
      self.cfg.wm,
      device=self.device,
      reward_names=self.reward_names,
    )

    if self.cfg.wandb_run_path is not None:
      load_wandb_checkpoint(
        self.cfg.wandb_run_path, self.task_id, dynamics, representation, self.device
      )

    # Action manager — reads ``action_manager.action`` (raw, post-clamp) for the
    # replay buffer so stored actions match what mjlab shows the policy next step.
    self.action_manager = self.env.unwrapped.action_manager

    self.replay_buffer = ReplayBuffer(
      schema=self._buffer_schema(num_full_state, num_actions, num_goals, num_rewards),
      buffer_size=self.cfg.buffer_size,
      num_envs=self.env.num_envs,
      device=self.device,
    )

    self._pending_norm_updates: list[
      tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = []

    # Expose dims for subclass policy init
    self._num_full_state = num_full_state
    self._num_goals = num_goals
    self._num_actions = num_actions
    self._num_rewards = num_rewards
    self._dynamics = dynamics

  def _policy_input_dims(self) -> tuple[int, int]:
    """Compute actor and critic input dimensions from the representation."""
    actor_dim = self._representation.rep_state_dim + actor_goal_dim(self._num_goals)
    critic_dim = self._representation.rep_state_dim + self._num_goals
    return actor_dim, critic_dim

  def _build_mlp(
    self,
    name: str,
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...] = (512, 256, 128),
    **kwargs: object,
  ) -> nn.Module:
    """Build an MLPModel with obs_normalization enabled.

    Args:
      name: Observation set name (e.g. "actor", "critic").
      input_dim: Input feature dimension.
      output_dim: Output dimension.
      hidden_dims: MLP hidden layer sizes.
      **kwargs: Extra keyword arguments forwarded to :class:`MLPModel`.
    """
    obs = TensorDict(
      {name: torch.zeros(1, input_dim, device=self.device)},
      batch_size=[1],
      device=self.device,
    )
    return MLPModel(
      obs=obs,
      obs_groups={name: [name]},
      obs_set=name,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      obs_normalization=True,
      **kwargs,
    ).to(self.device)

  def _create_wm_env(
    self, motion_file: str, batch_size: int, horizon: int
  ) -> MotionTrackingEnv:
    """Create a :class:`MotionTrackingEnv` bound to this runner's world model and buffer."""
    wm_env = MotionTrackingEnv(
      dynamics=self._dynamics,
      representation=self._representation,
      motion_file=motion_file,
      env=self.env,
      batch_size=batch_size,
      horizon=horizon,
      replay_buffer=self.replay_buffer,
      device=self.device,
    )
    if torch.cuda.is_available():
      wm_env._wm_step = torch.compile(wm_env._wm_step, mode="reduce-overhead")
    return wm_env

  def _actor_obs(
    self, curr_states: torch.Tensor, curr_goals: torch.Tensor
  ) -> TensorDict:
    """Build the actor observation TensorDict from full states and goals."""
    actor_goal = extract_actor_goal(curr_goals)
    inputs = self._representation.construct_policy_input(curr_states, actor_goal)
    return TensorDict(
      {"actor": inputs}, batch_size=inputs.shape[:-1], device=self.device
    )

  def _buffer_schema(
    self,
    num_full_state: int,
    num_actions: int,
    num_goals: int,
    num_rewards: int,
  ) -> dict[str, FieldSpec]:
    """Override to customize the replay buffer schema (e.g. raw task goals)."""
    return {
      "state": FieldSpec(num_full_state),
      "action": FieldSpec(num_actions),
      "goal": FieldSpec(num_goals),
      "reward": FieldSpec(num_rewards),
      "motion_time": FieldSpec(dtype=torch.long),
      "terminated": FieldSpec(dtype=torch.bool),
      "truncated": FieldSpec(dtype=torch.bool),
    }

  def _init_policy(self) -> None:
    """Subclasses implement this to create their policy learner."""
    raise NotImplementedError

  def _step_and_store(
    self, actions: torch.Tensor, curr_states: torch.Tensor, curr_goals: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Step the environment, process the transition, and store in the replay buffer.

    Must be called inside ``torch.inference_mode()``. Queues a normalizer
    update with the raw actions; flush via ``_flush_normalization``.

    Returns:
      (next_obs, rewards, dones, extras) from the environment step.
    """
    motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
    motion_time = motion_cmd.time_steps.clone()

    next_obs, rewards, dones, extras = self.env.step(actions)
    # Raw post-clamp action — matches mjlab's ``obs["actor"]`` last_action channel
    # so WM train/deploy see the same distribution. Dynamics applies scale/offset
    # internally; heads (reward, termination) consume raw directly.
    raw_actions = self.action_manager.action.clone()

    reward_vec = self.reward_manager._step_reward.to(self.device).clone()
    if reward_vec.dim() == 1:
      reward_vec = reward_vec.unsqueeze(1)

    truncated = extras.get(
      "time_outs",
      torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device),
    )
    terminated = dones.bool() & ~truncated

    self.replay_buffer.insert(
      TensorDict(
        {
          "state": curr_states,
          "action": raw_actions,
          "goal": curr_goals,
          "reward": reward_vec,
          "motion_time": motion_time,
          "terminated": terminated,
          "truncated": truncated,
        },
        batch_size=[self.env.num_envs],
      )
    )
    self._pending_norm_updates.append(
      (curr_states.clone(), raw_actions.clone(), curr_goals.clone())
    )

    return next_obs, rewards, dones, extras

  def _flush_normalization(self) -> None:
    """Apply queued normalizer updates. Call outside ``inference_mode()``."""
    states, actions, goals = zip(*self._pending_norm_updates, strict=True)
    states = torch.cat(states)
    actions = torch.cat(actions)
    goals = torch.cat(goals)
    self.world_model_learner.update_normalization(states, actions, goals)
    self._update_actor_normalization(states, goals)
    self._pending_norm_updates.clear()

  def _update_actor_normalization(
    self, states: torch.Tensor, goals: torch.Tensor
  ) -> None:
    """Subclasses override to update the actor obs normalizer from sim data."""
    raise NotImplementedError

  def _create_log_dir(self, *extra_parts: str) -> Path:
    """Create and return a timestamped log directory.

    Args:
      *extra_parts: Optional path components inserted between task_id
        and the timestamp (e.g. "ppo").
    """
    log_dir = (
      Path("logs")
      / self.task_id
      / Path(*extra_parts)
      / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
      if extra_parts
      else Path("logs") / self.task_id / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging to {log_dir}")
    self.logger.log_dir = str(log_dir)
    return log_dir

  def _init_wandb(self, log_dir: Path, name_suffix: str = "") -> None:
    """Initialize W&B run, link artifacts, and wire up the logger.

    Args:
      log_dir: Directory for log files.
      name_suffix: Optional suffix for auto-generated run name (e.g. "-ppo").
    """
    wandb_config = {
      "env_cfg": asdict(self.env.cfg),
      "runner_cfg": asdict(self.cfg),
    }

    default_name = f"{self.task_id}{name_suffix}-{log_dir.name}"
    wandb.init(
      project=self.cfg.wandb_project,
      name=self.cfg.run_name or default_name,
      config=wandb_config,
    )
    self.logger.writer = wandb

    if self.registry_name:
      api = wandb.Api()
      for name in self.registry_name.split(","):
        name = name.strip()
        if ":" not in name:
          name = name + ":latest"
        print(f"[INFO] Linking registry artifact: {name}")
        artifact = api.artifact(name)
        wandb.run.use_artifact(artifact)

  def _cleanup(self) -> None:
    """Close environment and finish W&B run."""
    env_close_fn = getattr(self.env, "close", None)
    if env_close_fn:
      env_close_fn()
    wandb.finish()


class SuperMimicRunner(BaseRunner[SuperMimicRunnerCfg]):
  def _init_policy(self) -> None:
    actor_input_dim, critic_input_dim = self._policy_input_dims()

    if self.cfg.actor_type == "vae":
      actor = VAEActor(
        obs_dim=self._representation.rep_state_dim,
        goal_dim=actor_goal_dim(self._num_goals),
        action_dim=self._num_actions,
        latent_dim=self.cfg.latent_dim,
        fixed_std=self.cfg.fixed_std,
        kl_weight=self.cfg.kl_weight,
      ).to(self.device)
    else:
      actor = self._build_mlp("actor", actor_input_dim, self._num_actions)
    critic = self._build_mlp("critic", critic_input_dim, 1)
    policy_learner_cfg = self.cfg.policy
    policy_learner_cfg.motion_file = self.env.unwrapped.cfg.commands[
      "motion"
    ].motion_file

    self.wm_env = self._create_wm_env(
      policy_learner_cfg.motion_file,
      policy_learner_cfg.batch_size,
      policy_learner_cfg.horizon,
    )

    if isinstance(policy_learner_cfg, ActorCriticPolicyLearnerCfg):
      self.policy_learner = ActorCriticPolicyLearner(
        actor, critic, self.wm_env, policy_learner_cfg, device=self.device
      )
    else:
      self.policy_learner = BpttPolicyLearner(
        actor, self.wm_env, policy_learner_cfg, device=self.device
      )

  def step(self) -> int:
    with torch.inference_mode():
      obs = self.env.get_observations()
      curr_states, curr_goals = obs["full_state"], obs["goal"]
      actions = self.policy_learner.actor(self._actor_obs(curr_states, curr_goals))

      noise = torch.randn_like(actions) * self.policy_learner.cfg.noise
      actions = actions + noise

      _, rewards, dones, extras = self._step_and_store(actions, curr_states, curr_goals)
      self.logger.process_env_step(rewards, dones, extras)

    self._flush_normalization()

    return self.env.num_envs

  def _update_actor_normalization(
    self, states: torch.Tensor, goals: torch.Tensor
  ) -> None:
    # EmpiricalNormalization.update is a no-op in eval mode.
    self.policy_learner.actor.train()
    self.policy_learner.actor.update_normalization(self._actor_obs(states, goals))

  def run(self) -> None:
    log_dir = self._create_log_dir()
    self._init_wandb(log_dir)

    self.env.reset()

    print(f"[INFO] Starting training loop for {self.cfg.max_iterations} iterations...")

    should_train = Ratio(
      self.env.num_envs
      / self.cfg.wm.num_batches
      / self.cfg.wm.batch_size
      / self.cfg.wm.horizon
      * self.cfg.replay_ratio
    )

    num_collected_steps = 0
    start_time = time.time()
    wm_learn_time = 0.0
    policy_learn_time = 0.0
    loss_dict = {}
    log_count = 0

    for step in range(self.cfg.max_iterations):
      # 1. Collect Data
      num_collected_steps += self.step()

      # 2. Train World Model & Policy
      for _ in range(should_train(step)):
        if self.replay_buffer.num_transitions < self.cfg.wm.horizon:
          continue

        data = (
          self.replay_buffer.sample(self.cfg.wm.horizon, self.cfg.wm.batch_size)
          for _ in range(self.cfg.wm.num_batches)
        )

        # Train World Model
        start_wm_time = time.time()
        wm_logs = self.world_model_learner.update(data)
        wm_learn_time += time.time() - start_wm_time

        # Train Policy
        start_policy_time = time.time()
        policy_logs = self.policy_learner.update()
        policy_learn_time += time.time() - start_policy_time

        # Accumulate logs
        loss_dict.update(wm_logs)
        loss_dict.update(policy_logs)

      # 3. Logging & Saving
      if step % self.cfg.log_interval == 0:
        elapsed_time = time.time() - start_time
        total_train_time = wm_learn_time + policy_learn_time
        collect_time = elapsed_time - total_train_time

        self.logger.log(
          it=log_count,
          total_it=self.cfg.max_iterations // self.cfg.log_interval,
          collect_time=collect_time,
          wm_learn_time=wm_learn_time,
          policy_learn_time=policy_learn_time,
          loss_dict=loss_dict,
          num_collected_steps=num_collected_steps,
        )

        log_count += 1
        # Reset accumulators
        start_time = time.time()
        wm_learn_time = 0
        policy_learn_time = 0
        num_collected_steps = 0

      if step % self.cfg.save_interval == 0:
        torch.save(
          {
            "dynamics": self._dynamics.state_dict(),
            "representation": self._representation.state_dict(),
          },
          log_dir / "world_model.pt",
        )
        torch.save(self.policy_learner.actor.state_dict(), log_dir / "actor.pt")
        wandb.save(
          str(log_dir / "world_model.pt"), base_path=str(log_dir), policy="now"
        )
        wandb.save(str(log_dir / "actor.pt"), base_path=str(log_dir), policy="now")

    self._cleanup()
