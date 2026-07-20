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
from supermimic.envs.multi_task_world_model_env import MultiTaskWorldModelEnv
from supermimic.learners.learner_utils import load_wandb_checkpoint
from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
  BpttPolicyLearner,
  BpttPolicyLearnerCfg,
)
from supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
from supermimic.models import Dynamics, MultiEncoderAE
from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.representations import create_representation
from supermimic.runners.exporter import export_actor_as_onnx
from supermimic.sim_context import SimContext
from supermimic.tasks.task_spec import (
  TasksCfg,
  TaskSpec,
  build_actor_obs_td,
  resolve_task_specs,
)
from supermimic.utils.helpers import (
  extract_action_scale_offset,
  unpack_obs_dims,
)
from supermimic.utils.logger import Logger
from supermimic.utils.wandb_artifacts import log_model_artifact


@dataclass
class BaseRunnerCfg:
  """Shared configuration fields for all runners."""

  wandb_project: str = "supermimic"
  run_name: str | None = None
  wandb_run_path: str | None = None
  headless: bool = True
  buffer_size: int = 1000
  representation: str = "gencoord"  # "gencoord" or "maxcoord"
  actor_obs: str = "rep_state"  # "rep_state" or "proprio"
  log_interval: int = 10
  save_interval: int = 1000
  history_length: int = 1
  dynamics_hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
  reward_hidden_dims: tuple[int, ...] = (256, 128)
  term_hidden_dims: tuple[int, ...] = (256, 128)
  obs_hidden_dims: tuple[int, ...] = (512, 512)
  wm: WorldModelLearnerCfg = field(default_factory=WorldModelLearnerCfg)


BpttCfg = Annotated[BpttPolicyLearnerCfg, tyro.conf.subcommand(name="bptt")]
ActorCriticCfg = Annotated[
  ActorCriticPolicyLearnerCfg, tyro.conf.subcommand(name="actor-critic")
]


@dataclass
class SuperMimicRunnerCfg(BaseRunnerCfg):
  max_iterations: int = 1000000
  replay_ratio: int = 16
  latent_dim: int = 32
  actor_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)
  critic_hidden_dims: tuple[int, ...] = (256, 128)
  latent_bc_weight: float = 1e-1
  """Weight on the latent BC loss pulling non-anchor encoders toward the
  anchor encoder's latent on the same state. 0 disables."""
  tasks: TasksCfg = field(default_factory=TasksCfg)
  """Per-task hyperparameters."""
  slot_mix: dict[str, float] = field(
    default_factory=lambda: {"motion": 1.0, "velocity": 0.0}
  )
  """Fractional slot allocation per task. Must sum to 1.0."""
  policy: BpttCfg | ActorCriticCfg = field(default_factory=ActorCriticCfg)


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
  setup, replay buffer, action manager, and logger. Subclasses implement
  ``_init_policy()`` for their specific policy setup.
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
      max_episode_length=env.max_episode_length,
      device=device,
    )

  def _init_common(self) -> None:
    obs, _ = self.env.reset()
    num_full_state, num_goals = unpack_obs_dims(obs)
    num_actions = self.env.num_actions
    dt = self.env.unwrapped.step_dt

    print(f"[INFO] State Dim: {num_full_state}, Goal Dim: {num_goals}")

    self.reward_manager = self.env.unwrapped.reward_manager
    self.reward_names = getattr(self.reward_manager, "active_terms", ["reward"])
    num_rewards = len(self.reward_names)
    print(f"[INFO] Reward Dim: {num_rewards} ({self.reward_names})")

    self._sim_context = SimContext.from_env(self.env.unwrapped, self.device)
    sim_ctx_dim = self._sim_context.output_dim
    print(
      f"[INFO] Sim Context: {sim_ctx_dim} dim "
      f"({[s.name for s in self._sim_context.specs]})"
    )

    default_joint_pos = (
      self.env.unwrapped.scene["robot"].data.default_joint_pos[0].to(self.device)
    )
    has_state_estimation = (
      "base_lin_vel" in self.env.unwrapped.observation_manager.active_terms["actor"]
    )
    representation = create_representation(
      self.cfg.representation,
      num_full_state,
      num_actions,
      num_goals,
      num_rewards,
      sim_ctx_dim,
      dt,
      default_joint_pos,
      has_state_estimation,
      actor_obs_mode=self.cfg.actor_obs,
      reward_hidden_dims=self.cfg.reward_hidden_dims,
      term_hidden_dims=self.cfg.term_hidden_dims,
      obs_hidden_dims=self.cfg.obs_hidden_dims,
    ).to(self.device)
    self._representation = representation

    if self.cfg.history_length > 1 and representation.obs_head is None:
      raise RuntimeError(
        "history_length > 1 needs proprio actor obs (--runner.actor-obs proprio)"
      )

    print(
      f"[INFO] Representation: {self.cfg.representation} "
      f"(full={representation.full_state_dim}, rep={representation.rep_state_dim}, "
      f"pred={representation.prediction_dim})"
    )

    action_scale, action_offset = extract_action_scale_offset(self.env)
    dynamics = Dynamics(
      representation,
      num_actions,
      sim_ctx_dim,
      action_scale,
      action_offset,
      hidden_dims=self.cfg.dynamics_hidden_dims,
    ).to(self.device)

    if self.cfg.wandb_run_path is not None:
      load_wandb_checkpoint(
        self.cfg.wandb_run_path, dynamics, representation, self.device
      )

    self.action_manager = self.env.unwrapped.action_manager

    self._num_full_state = num_full_state
    self._num_goals = num_goals
    self._num_actions = num_actions
    self._num_rewards = num_rewards
    self._dynamics = dynamics

    self.replay_buffer = ReplayBuffer(
      schema=self._buffer_schema(num_full_state, num_actions, num_rewards),
      buffer_size=self.cfg.buffer_size,
      num_envs=self.env.num_envs,
      device=self.device,
    )

    self._pending_norm_updates: list[tuple] = []
    self._sim_history: torch.Tensor | None = None

  def _seed_sim_history(self) -> None:
    """Seed the real-sim obs-history ring with the noised current sensor."""
    if self.cfg.history_length == 1:
      return
    seed = self._representation.add_proprio_noise(
      self.env.get_observations()["actor"].to(self.device)
    )
    self._sim_history = seed.unsqueeze(1).repeat(1, self.cfg.history_length, 1)

  def _collection_history(self, sensor: torch.Tensor) -> torch.Tensor:
    """The ``(B, history_length, sensor_dim)`` window for a real-sim actor step.
    history_length > 1 needs ``_seed_sim_history()`` first."""
    if self.cfg.history_length == 1:
      return sensor.unsqueeze(1)
    assert self._sim_history is not None
    return self._sim_history

  def _advance_sim_history(self, next_actor: torch.Tensor, dones: torch.Tensor) -> None:
    """Append the noised next sensor; rows whose episode just reset repeat that
    single realization, matching the warm-start padding."""
    if self.cfg.history_length == 1:
      return
    assert self._sim_history is not None
    next_sensor = self._representation.add_proprio_noise(next_actor)
    self._sim_history = torch.cat(
      [self._sim_history[:, 1:], next_sensor.unsqueeze(1)], dim=1
    )
    done_rows = dones.bool()
    if done_rows.any():
      self._sim_history[done_rows] = (
        next_sensor[done_rows].unsqueeze(1).repeat(1, self.cfg.history_length, 1)
      )

  def _policy_input_dims(self) -> tuple[int, int]:
    """Compute actor and critic input dimensions from the representation."""
    actor_dim = self._representation.actor_obs_dim(
      self._num_actions, self._num_goals, self.cfg.history_length
    )
    critic_dim = (
      self._representation.rep_state_dim
      + self._num_goals
      + self._sim_context.output_dim
    )
    return actor_dim, critic_dim

  def _build_mlp(
    self,
    name: str,
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...],
    obs_normalization: bool = True,
    **kwargs: object,
  ) -> nn.Module:
    """Build an MLPModel (input obs normalization on by default)."""
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
      obs_normalization=obs_normalization,
      **kwargs,
    ).to(self.device)

  def _create_wm_env(self, batch_size: int, horizon: int) -> MotionTrackingEnv:
    """Create a :class:`MotionTrackingEnv` bound to this runner's world model and buffer."""
    return MotionTrackingEnv(
      dynamics=self._dynamics,
      representation=self._representation,
      env=self.env,
      batch_size=batch_size,
      horizon=horizon,
      replay_buffer=self.replay_buffer,
      sim_context=self._sim_context,
      history_length=self.cfg.history_length,
      device=self.device,
    )

  def _buffer_schema(
    self,
    num_full_state: int,
    num_actions: int,
    num_rewards: int,
  ) -> dict[str, FieldSpec]:
    """Override to customize the replay buffer schema (e.g. extra fields)."""
    schema = {
      "state": FieldSpec(num_full_state),
      "action": FieldSpec(num_actions),
      "prev_action": FieldSpec(num_actions),
      "goal": FieldSpec(self._num_goals),
      "reward": FieldSpec(num_rewards),
      "motion_time": FieldSpec(dtype=torch.long),
      "terminated": FieldSpec(dtype=torch.bool),
      "truncated": FieldSpec(dtype=torch.bool),
    }
    for spec in self._sim_context.specs:
      schema[spec.name] = FieldSpec(spec.dim)
    return schema

  def _init_policy(self) -> None:
    """Subclasses implement this to create their policy learner."""
    raise NotImplementedError

  def _step_and_store(
    self,
    actions: torch.Tensor,
    curr_states: torch.Tensor,
    curr_goals: torch.Tensor,
    actor_td: TensorDict | None = None,
    extra_fields: dict[str, torch.Tensor] | None = None,
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Step the env and write the transition to the replay buffer.

    When given, ``actor_td`` is queued for the actor-normalizer update.
    """
    motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
    motion_times = motion_cmd.time_steps.clone()

    raw_actions = actions.detach().clone()
    prev_action = self.action_manager.action.clone()

    next_obs, rewards, dones, extras = self.env.step(actions)

    reward_vec = self.reward_manager._step_reward.to(self.device).clone()
    if reward_vec.dim() == 1:
      reward_vec = reward_vec.unsqueeze(1)

    truncated = extras.get(
      "time_outs",
      torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device),
    )
    terminated = dones.bool() & ~truncated

    fields: dict[str, torch.Tensor] = {
      "state": curr_states,
      "action": raw_actions,
      "prev_action": prev_action,
      "goal": curr_goals,
      "reward": reward_vec,
      "motion_time": motion_times,
      "terminated": terminated,
      "truncated": truncated,
    }
    if extra_fields:
      fields.update(extra_fields)
    self.replay_buffer.insert(TensorDict(fields, batch_size=[self.env.num_envs]))

    if actor_td is not None:
      self._pending_norm_updates.append(
        (
          curr_states.clone(),
          raw_actions.clone(),
          curr_goals.clone(),
          actor_td.detach().clone(),
        )
      )
    return next_obs, rewards, dones, extras

  def _flush_normalization(self) -> None:
    """Apply queued normalizer updates. Call outside ``inference_mode()``."""
    if not self._pending_norm_updates:
      return
    states, actions, goals, actor_inputs = zip(*self._pending_norm_updates, strict=True)
    self.world_model_learner.update_normalization(
      torch.cat(states), torch.cat(actions), torch.cat(goals)
    )
    self._update_actor_normalization(actor_inputs)
    self._pending_norm_updates.clear()

  def _update_actor_normalization(self, actor_inputs: tuple[TensorDict, ...]) -> None:
    """Subclasses override to update the actor obs normalizer."""
    raise NotImplementedError

  def _create_log_dir(self, *extra_parts: str) -> Path:
    """Create and return a timestamped log directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path("logs", self.task_id, *extra_parts, timestamp)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging to {log_dir}")
    self.logger.log_dir = str(log_dir)
    return log_dir

  def _init_wandb(self, log_dir: Path, name_suffix: str = "") -> None:
    """Initialize W&B run, link artifacts, and wire up the logger."""
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
  """Trains WM + multi-encoder actor end-to-end via BPTT/AC.

  Slots are contiguous blocks of the env batch; each block routes through its
  task's encoder. Motion-tracking reward is the only reward path; tasks differ
  only in what goal modality their encoder consumes.
  """

  def _buffer_schema(
    self, num_full_state: int, num_actions: int, num_rewards: int
  ) -> dict[str, FieldSpec]:
    schema = super()._buffer_schema(num_full_state, num_actions, num_rewards)
    if self._representation.obs_head is not None:
      schema["actor_obs"] = FieldSpec(self._representation.proprio_obs_dim(num_actions))
    return schema

  def _init_policy(self) -> None:
    self._task_specs: list[TaskSpec] = resolve_task_specs(
      self.cfg.tasks, self._representation, self._num_goals
    )
    self._active_task_specs = [
      ts for ts in self._task_specs if self.cfg.slot_mix.get(ts.name, 0.0) > 0.0
    ]

    motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
    self.motion = motion_cmd.motion
    self._anchor_index = motion_cmd.motion_anchor_body_index

    obs_dim = (
      self._representation.rep_state_dim
      if self.cfg.actor_obs == "rep_state"
      else self._representation.proprio_obs_dim(self._num_actions)
    )

    actor = MultiEncoderAE(
      obs_dim=obs_dim,
      action_dim=self._num_actions,
      task_specs=self._task_specs,
      slot_mix=self.cfg.slot_mix,
      latent_dim=self.cfg.latent_dim,
      hidden_dims=self.cfg.actor_hidden_dims,
      latent_bc_weight=self.cfg.latent_bc_weight,
      history_length=self.cfg.history_length,
    ).to(self.device)

    print("[INFO] Sim slot ranges:")
    for ts, (lo, hi) in zip(
      self._task_specs, actor.slot_ranges(self.env.num_envs), strict=True
    ):
      print(f"  {ts.name}: [{lo}, {hi}) — goal_dim={ts.goal_dim}")
    _, critic_input_dim = self._policy_input_dims()
    critic = self._build_mlp(
      "critic",
      critic_input_dim,
      output_dim=1,
      hidden_dims=self.cfg.critic_hidden_dims,
      obs_normalization=False,
    )

    policy_cfg = self.cfg.policy

    self.wm_env = MultiTaskWorldModelEnv(
      dynamics=self._dynamics,
      representation=self._representation,
      env=self.env,
      batch_size=policy_cfg.batch_size,
      horizon=policy_cfg.horizon,
      replay_buffer=self.replay_buffer,
      sim_context=self._sim_context,
      task_specs=self._active_task_specs,
      history_length=self.cfg.history_length,
      device=self.device,
    )
    if torch.cuda.is_available():
      self.wm_env._wm_step = torch.compile(self.wm_env._wm_step, mode="reduce-overhead")

    self.world_model_learner = WorldModelLearner(
      self._dynamics,
      self._representation,
      self.cfg.wm,
      self._sim_context,
      device=self.device,
      reward_names=self.reward_names,
    )

    if isinstance(policy_cfg, ActorCriticPolicyLearnerCfg):
      self.policy_learner = ActorCriticPolicyLearner(
        actor, critic, self.wm_env, policy_cfg, device=self.device
      )
    else:
      self.policy_learner = BpttPolicyLearner(
        actor, self.wm_env, policy_cfg, device=self.device
      )

  def _build_sim_obs_td(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    sim_actor_obs: torch.Tensor,
    motion_times: torch.Tensor,
  ) -> TensorDict:
    proprio = self.cfg.actor_obs == "proprio"
    sensor = self._representation.compute_actor_sensor(
      rep_state,
      self._num_actions,
      self._sim_context.read(self.env.unwrapped),
      aux_state=aux_state,
      sim_actor_obs=sim_actor_obs if proprio else None,
      noisy=proprio,
    )
    goals = {
      ts.name: ts.extract_training_goal(
        self.motion, motion_times, aux_state, self._anchor_index
      )
      for ts in self._active_task_specs
    }
    return build_actor_obs_td(goals, self._collection_history(sensor))

  def step(self) -> int:
    with torch.inference_mode():
      obs = self.env.get_observations()
      curr_states = obs["full_state"].to(self.device)
      curr_goals = obs["goal"].to(self.device)
      sim_actor_obs = obs["actor"].to(self.device)

      motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
      motion_times = motion_cmd.time_steps.to(self.device)

      rep_state = self._representation.represent(curr_states)
      aux_state = self._representation.extract_aux_state(curr_states)
      actor_td = self._build_sim_obs_td(
        rep_state, aux_state, sim_actor_obs, motion_times
      )

      actions = self.policy_learner.actor(actor_td)
      actions = actions + torch.randn_like(actions) * self.policy_learner.cfg.noise

      extra_fields = self._sim_context.read_raw(self.env.unwrapped)
      if self._representation.obs_head is not None:
        extra_fields["actor_obs"] = sim_actor_obs

      next_obs, rewards, dones, extras = self._step_and_store(
        actions,
        curr_states,
        curr_goals,
        actor_td=actor_td,
        extra_fields=extra_fields,
      )
      self._advance_sim_history(next_obs["actor"].to(self.device), dones)
      terminated = self.env.unwrapped.termination_manager.terminated
      self.logger.process_env_step(rewards, terminated, extras)

    self._flush_normalization()
    return self.env.num_envs

  def _update_actor_normalization(self, actor_inputs: tuple[TensorDict, ...]) -> None:
    self.policy_learner.actor.train()
    for actor_td in actor_inputs:
      self.policy_learner.actor.update_normalization(actor_td)

  def run(self) -> None:
    log_dir = self._create_log_dir()
    self._init_wandb(log_dir)

    self.env.reset()
    self._seed_sim_history()

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
    loss_dict: dict[str, float] = {}
    log_count = 0

    for step in range(self.cfg.max_iterations):
      num_collected_steps += self.step()

      for _ in range(should_train(step)):
        if self.replay_buffer.num_transitions < self.cfg.wm.horizon:
          continue

        data = (
          self.replay_buffer.sample(self.cfg.wm.horizon, self.cfg.wm.batch_size)
          for _ in range(self.cfg.wm.num_batches)
        )

        start_wm_time = time.time()
        wm_logs = self.world_model_learner.update(data)
        wm_learn_time += time.time() - start_wm_time

        start_policy_time = time.time()
        policy_logs = self.policy_learner.update()
        policy_learn_time += time.time() - start_policy_time

        loss_dict.update(wm_logs)
        loss_dict.update(policy_logs)

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
        start_time = time.time()
        wm_learn_time = 0
        policy_learn_time = 0
        num_collected_steps = 0

      if step % self.cfg.save_interval == 0:
        assert wandb.run is not None  # _init_wandb ran before the loop
        world_model_path = str(log_dir / "world_model.pt")
        actor_path = str(log_dir / "actor.pt")
        torch.save(
          {
            "dynamics": self._dynamics.state_dict(),
            "representation": self._representation.state_dict(),
          },
          world_model_path,
        )
        torch.save(self.policy_learner.actor.state_dict(), actor_path)
        checkpoint_files = [world_model_path, actor_path]
        # Skip ONNX export on huge datasets.
        if (
          self._representation.actor_obs_mode == "proprio"
          and self.motion.time_step_total <= 15_000
        ):
          checkpoint_files += export_actor_as_onnx(
            actor=self.policy_learner.actor,
            env=self.env.unwrapped,
            path=str(log_dir),
            run_name=wandb.run.name,
            term_widths=self._representation.proprio_term_widths(self._num_actions),
          )
        log_model_artifact(checkpoint_files)

    self._cleanup()
