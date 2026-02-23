import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import wandb
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from supermimic.models import WorldModel
from supermimic.replay_buffer import ReplayBuffer
from supermimic.representations import create_representation
from supermimic.runners.learner import (
  ActorCriticPolicyLearner,
  BpttPolicyLearner,
  PolicyLearnerCfg,
  WorldModelLearner,
  WorldModelLearnerCfg,
  load_wandb_checkpoint,
)
from supermimic.runners.world_model_env import WorldModelEnv
from supermimic.utils.helpers import unpack_obs_dims
from supermimic.utils.logger import Logger


@dataclass
class SuperMimicRunnerCfg:
  wandb_project: str = "supermimic"
  run_name: str | None = None
  wandb_run_path: str | None = None
  headless: bool = True

  max_iterations: int = 1000000
  buffer_size: int = 2000
  replay_ratio: int = 128
  log_interval: int = 10
  save_interval: int = 1000

  policy_algorithm: str = "bptt"  # "bptt" or "actor_critic"
  representation: str = "body_frame"  # "body_frame" or "maxcoord"

  wm: WorldModelLearnerCfg = field(default_factory=WorldModelLearnerCfg)
  policy: PolicyLearnerCfg = field(default_factory=PolicyLearnerCfg)


class Ratio:
  def __init__(self, ratio):
    self._ratio = ratio
    self._prev = None

  def __call__(self, step):
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


class SuperMimicRunner:
  def __init__(
    self,
    env,
    cfg: SuperMimicRunnerCfg,
    task_id: str,
    registry_name: str | None = None,
    device: str = "cuda:0",
  ):
    self.env = env
    self.cfg = cfg
    self.task_id = task_id
    self.registry_name = registry_name
    self.device = device
    self._init_learners()

    self.logger = Logger(
      cfg=asdict(cfg),
      num_envs=env.num_envs,
      device=device,
    )

  def _init_learners(self):
    # Initialize Dimensions
    obs, _ = self.env.reset()
    num_full_state, num_goals = unpack_obs_dims(obs)
    num_actions = self.env.num_actions

    print(f"[INFO] State Dim: {num_full_state}, Goal Dim: {num_goals}")

    dt = self.env.unwrapped.step_dt

    # Create representation
    representation = create_representation(self.cfg.representation, num_full_state)

    print(
      f"[INFO] Representation: {self.cfg.representation} "
      f"(full={representation.full_state_dim}, rep={representation.rep_state_dim}, "
      f"pred={representation.prediction_dim})"
    )

    # Initialize World Model & Learner
    history = self.cfg.wm.history

    # Check for reward terms
    self.reward_manager = self.env.unwrapped.reward_manager
    self.reward_names = getattr(self.reward_manager, "active_terms", ["reward"])
    num_rewards = len(self.reward_names)
    print(f"[INFO] Reward Dim: {num_rewards} ({self.reward_names})")

    wm = WorldModel(
      representation, num_actions, num_goals, num_rewards, dt=dt, history=history
    ).to(self.device)

    # Compile the hot-path transition method so both WM training and policy
    # rollouts benefit.  Called 64× per policy batch from wm_env.step();
    # reduce-overhead uses CUDA Graphs to amortise dispatch overhead.
    if torch.cuda.is_available():
      wm.transition = torch.compile(wm.transition, mode="reduce-overhead")

    self.world_model_learner = WorldModelLearner(
      wm,
      self.cfg.wm,
      dt=dt,
      device=self.device,
      reward_names=self.reward_names,
    )

    if self.cfg.wandb_run_path is not None:
      load_wandb_checkpoint(self.cfg.wandb_run_path, self.task_id, wm, self.device)

    # Initialize Policy & Learner
    goal_extra = representation.goal_dim_split()
    actor_input_dim = representation.rep_state_dim + num_goals - goal_extra
    critic_input_dim = representation.rep_state_dim + num_goals

    def _build_mlp(name, input_dim, output_dim):
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
        hidden_dims=[512, 256, 128],
        obs_normalization=True,
      ).to(self.device)

    actor = _build_mlp("actor", actor_input_dim, num_actions)
    critic = _build_mlp("critic", critic_input_dim, 1)
    policy_learner_cfg = self.cfg.policy
    policy_learner_cfg.motion_file = self.env.unwrapped.cfg.commands[
      "motion"
    ].motion_file

    # Extract Action Scale & Offset
    self.action_manager = self.env.unwrapped.action_manager
    term_name = self.action_manager.active_terms[0]
    self.action_term = self.action_manager.get_term(term_name)

    # Buffer: [state, action, goal, reward, motion_time, done]
    self.replay_buffer = ReplayBuffer(
      dim=[num_full_state, num_actions, num_goals, num_rewards, 1, 1],
      buffer_size=self.cfg.buffer_size,
      device=self.device,
    )
    self.replay_buffer.initialize(self.env.num_envs)

    self.wm_env = WorldModelEnv(
      world_model=wm,
      motion_file=policy_learner_cfg.motion_file,
      env=self.env,
      batch_size=policy_learner_cfg.batch_size,
      horizon=policy_learner_cfg.horizon,
      replay_buffer=self.replay_buffer,
      device=self.device,
    )

    if self.cfg.policy_algorithm == "bptt":
      self.policy_learner = BpttPolicyLearner(
        actor,
        self.wm_env,
        policy_learner_cfg,
        dt=dt,
        device=self.device,
      )
    elif self.cfg.policy_algorithm == "actor_critic":
      self.policy_learner = ActorCriticPolicyLearner(
        actor,
        critic,
        self.wm_env,
        policy_learner_cfg,
        dt=dt,
        device=self.device,
      )
    else:
      raise ValueError(
        f"Unknown policy_algorithm: {self.cfg.policy_algorithm}. "
        "Must be 'bptt' or 'actor_critic'."
      )

    self._representation = representation

  def step(self):
    with torch.inference_mode():
      obs = self.env.get_observations()
      curr_states, curr_goals = obs["full_state"], obs["goal"]
      policy_goal = self._representation.extract_actor_goal(curr_goals)
      inputs = self._representation.construct_policy_input(curr_states, policy_goal)

      actor_inputs = TensorDict(
        {"actor": inputs}, batch_size=inputs.shape[:-1], device=self.device
      )
      actions = self.policy_learner.actor(actor_inputs)

      noise = torch.randn_like(actions) * self.policy_learner.cfg.noise
      actions = actions + noise

      motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
      motion_time = motion_cmd.time_steps.clone()

      next_obs, rewards, dones, extras = self.env.step(actions)
      proc_actions = self.action_term._processed_actions

      # Get reward vector from reward manager
      reward_vec = self.reward_manager._step_reward.to(self.device).clone()
      if reward_vec.dim() == 1:
        reward_vec = reward_vec.unsqueeze(1)

      # Handle 3-state done: 0=not done, 1=terminated, 2=timeout
      dones_float = dones.float()
      if "time_outs" in extras:
        time_outs = extras["time_outs"]
        dones_float[time_outs] = 2.0

      self.replay_buffer.insert_transition(
        curr_states, proc_actions, curr_goals, reward_vec, motion_time, dones_float
      )
      self.logger.process_env_step(rewards, dones, extras)

    # Update normalization outside inference_mode for torch.compile compatibility.
    self.world_model_learner.update_normalization(curr_states, actions, curr_goals)

    return self.env.num_envs

  def run(self):
    log_dir = Path("logs") / self.task_id / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging to {log_dir}")

    # Update logger with log_dir
    self.logger.log_dir = str(log_dir)

    wandb_config = {
      "env_cfg": asdict(self.env.cfg),
      "runner_cfg": asdict(self.cfg),
    }

    wandb.init(
      project=self.cfg.wandb_project,
      name=self.cfg.run_name or f"{self.task_id}-{log_dir.name}",
      config=wandb_config,
    )

    # Set logger writer to wandb module
    self.logger.writer = wandb

    if self.registry_name:
      # Support comma-separated registry names for multi-motion
      api = wandb.Api()
      for name in self.registry_name.split(","):
        name = name.strip()
        if ":" not in name:
          name = name + ":latest"
        print(f"[INFO] Linking registry artifact: {name}")
        artifact = api.artifact(name)
        wandb.run.use_artifact(artifact)

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
    wm_learn_time = 0
    policy_learn_time = 0
    loss_dict = {}
    log_count = 0

    for step in range(self.cfg.max_iterations):
      # 1. Collect Data
      num_collected_steps += self.step()

      # 2. Train World Model & Policy
      for _ in range(should_train(step)):
        full_len = self.cfg.wm.history - 1 + self.cfg.wm.horizon
        if self.replay_buffer.num_transitions < full_len:
          continue

        data = self.replay_buffer.mini_batch_generator(
          full_len, self.cfg.wm.num_batches, self.cfg.wm.batch_size
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
          self.world_model_learner.model.state_dict(), log_dir / "world_model.pt"
        )
        torch.save(self.policy_learner.actor.state_dict(), log_dir / "actor.pt")
        wandb.save(
          str(log_dir / "world_model.pt"), base_path=str(log_dir), policy="now"
        )
        wandb.save(str(log_dir / "actor.pt"), base_path=str(log_dir), policy="now")

    env_close_fn = getattr(self.env, "close", None)
    if env_close_fn:
      env_close_fn()

    wandb.finish()
