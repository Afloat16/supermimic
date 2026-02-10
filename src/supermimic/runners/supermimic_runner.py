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
from supermimic.runners.learner import (
  ActorCriticPolicyLearner,
  BpttPolicyLearner,
  PolicyLearnerCfg,
  WorldModelLearner,
  WorldModelLearnerCfg,
)
from supermimic.runners.world_model_env import WorldModelEnv
from supermimic.utils.helpers import BodyFrameStateLayout, construct_policy_input
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
    state_obs = obs["full_state"] if hasattr(obs, "keys") else obs
    num_full_state = state_obs.shape[-1]
    goal_obs = obs["goal"] if hasattr(obs, "keys") else obs
    num_goals = goal_obs.shape[-1]
    num_actions = self.env.num_actions

    print(f"[INFO] State Dim: {num_full_state}, Goal Dim: {num_goals}")

    dt = self.env.unwrapped.step_dt
    num_joints = (num_full_state - 13) // 2
    self.layout = BodyFrameStateLayout(num_joints)
    body_state_dim = self.layout.dim

    # Initialize World Model & Learner
    history = self.cfg.wm.history

    # Check for reward terms
    self.reward_manager = self.env.unwrapped.reward_manager
    self.reward_names = getattr(self.reward_manager, "active_terms", ["reward"])
    num_rewards = len(self.reward_names)
    print(f"[INFO] Reward Dim: {num_rewards} ({self.reward_names})")

    wm = WorldModel(body_state_dim, num_actions, num_goals, num_rewards, history).to(
      self.device
    )
    self.world_model_learner = WorldModelLearner(
      wm,
      self.cfg.wm,
      dt=dt,
      layout=self.layout,
      device=self.device,
      reward_names=self.reward_names,
    )

    if self.cfg.wandb_run_path is not None:
      print(f"[INFO] Loading World Model from W&B: {self.cfg.wandb_run_path}")

      run_path = self.cfg.wandb_run_path
      run_id = run_path.split("/")[-1]
      log_root_path = Path("logs") / self.task_id
      download_dir = log_root_path / "wandb_checkpoints" / run_id
      checkpoint_path = download_dir / "world_model.pt"

      if not checkpoint_path.exists():
        print(f"[INFO] Downloading world_model.pt from {run_path}...")
        api = wandb.Api()
        run = api.run(run_path)
        run.file("world_model.pt").download(root=str(download_dir), replace=True)

      state_dict = torch.load(checkpoint_path, map_location=self.device)
      wm.load_state_dict(state_dict)
      print(f"[INFO] World Model loaded from {checkpoint_path}")

    # Initialize Policy & Learner
    actor_input_dim = body_state_dim + num_goals - 3

    # Policy (Actor)
    actor_obs = TensorDict(
      {"actor": torch.zeros(1, actor_input_dim, device=self.device)},
      batch_size=[1],
      device=self.device,
    )
    actor = MLPModel(
      obs=actor_obs,
      obs_groups={"actor": ["actor"]},
      obs_set="actor",
      output_dim=num_actions,
      hidden_dims=[512, 256, 128],
      obs_normalization=True,
    ).to(self.device)

    # Critic
    critic_obs = TensorDict(
      {"critic": torch.zeros(1, actor_input_dim + 3, device=self.device)},
      batch_size=[1],
      device=self.device,
    )
    critic = MLPModel(
      obs=critic_obs,
      obs_groups={"critic": ["critic"]},
      obs_set="critic",
      output_dim=1,
      hidden_dims=[512, 256, 128],
      obs_normalization=True,
    ).to(self.device)
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

    self.wm_env = WorldModelEnv(
      world_model=wm,
      motion_file=policy_learner_cfg.motion_file,
      env=self.env,
      batch_size=policy_learner_cfg.batch_size,
      horizon=policy_learner_cfg.horizon,
      layout=self.layout,
      replay_buffer=self.replay_buffer,
      device=self.device,
    )

    if self.cfg.policy_algorithm == "bptt":
      self.policy_learner = BpttPolicyLearner(
        actor,
        self.wm_env,
        policy_learner_cfg,
        dt=dt,
        layout=self.layout,
        device=self.device,
      )
    elif self.cfg.policy_algorithm == "actor_critic":
      self.policy_learner = ActorCriticPolicyLearner(
        actor,
        critic,
        self.wm_env,
        policy_learner_cfg,
        dt=dt,
        layout=self.layout,
        device=self.device,
      )
    else:
      raise ValueError(
        f"Unknown policy_algorithm: {self.cfg.policy_algorithm}. "
        "Must be 'bptt' or 'actor_critic'."
      )

  def step(self):
    with torch.inference_mode():
      obs = self.env.get_observations()
      curr_states, curr_goals = obs["full_state"], obs["goal"]
      policy_goal = torch.cat([curr_goals[..., :-9], curr_goals[..., -6:]], dim=-1)
      inputs = construct_policy_input(curr_states, policy_goal, self.layout)

      actor_inputs = TensorDict(
        {"actor": inputs}, batch_size=inputs.shape[:-1], device=self.device
      )
      actions = self.policy_learner.actor(actor_inputs)

      noise = torch.randn_like(actions) * self.policy_learner.cfg.noise
      actions = actions + noise
      next_obs, rewards, dones, extras = self.env.step(actions)
      proc_actions = self.action_term._processed_actions

      # Get reward vector from reward manager
      # Note: env.step returns sum of rewards, we want the vector
      # We assume reward_manager._step_reward is populated after step()
      reward_vec = self.reward_manager._step_reward.to(self.device).clone()
      # If reward_vec is (N, D), we need (N, 1, D)
      if reward_vec.dim() == 1:
        reward_vec = reward_vec.unsqueeze(1)  # Handle single env case

      # Handle 3-state done: 0=not done, 1=terminated, 2=timeout
      dones_float = dones.float()
      if "time_outs" in extras:
        time_outs = extras["time_outs"]
        dones_float[time_outs] = 2.0

      # Get motion time from command manager
      motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
      motion_time = motion_cmd.time_steps.clone()

      self.replay_buffer.insert(
        [
          curr_states.unsqueeze(1),
          proc_actions.unsqueeze(1),
          curr_goals.unsqueeze(1),
          reward_vec.unsqueeze(1).float(),
          motion_time.unsqueeze(1).unsqueeze(2).float(),
          dones_float.unsqueeze(1).unsqueeze(2),
        ]
      )
      self.world_model_learner.update_normalization(curr_states, actions, curr_goals)
      self.logger.process_env_step(rewards, dones, extras)

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

    self.replay_buffer.step = 0
    self.replay_buffer.num_transitions = 0

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
