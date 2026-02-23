import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import wandb
from mjlab.tasks.tracking.rl.exporter import attach_onnx_metadata
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from supermimic.models import WorldModel
from supermimic.replay_buffer import ReplayBuffer
from supermimic.runners.exporter import export_motion_policy_as_onnx
from supermimic.runners.learner import (
  WorldModelLearner,
  WorldModelLearnerCfg,
  load_wandb_checkpoint,
)
from supermimic.runners.world_model_env import WorldModelEnv
from supermimic.utils.helpers import construct_policy_input, unpack_obs_dims
from supermimic.utils.logger import Logger


@dataclass
class PpoRunnerCfg:
  wandb_project: str = "supermimic"
  run_name: str | None = None
  wandb_run_path: str | None = None
  headless: bool = True

  max_iterations: int = 25000
  # PPO Parameters
  num_steps_per_env: int = 32
  learning_iterations_per_step: int = 5
  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  clip_param: float = 0.2
  gamma: float = 0.99
  lam: float = 0.95
  desired_kl: float = 0.01
  max_grad_norm: float = 1.0
  learning_rate: float = 1e-3
  value_loss_coef: float = 1.0
  entropy_coef: float = 0.005
  batch_size: int = 4096

  # World Model Parameters
  buffer_size: int = 1000
  buffer_ratio: float = 0.01

  wm: WorldModelLearnerCfg = field(default_factory=WorldModelLearnerCfg)

  # Policy Architecture
  actor_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
  critic_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
  activation: str = "elu"
  init_noise_std: float = 1.0


class PpoRunner:
  def __init__(
    self,
    env,
    cfg: PpoRunnerCfg,
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

    self.dt = self.env.unwrapped.step_dt

    # Initialize World Model & Learner
    history = self.cfg.wm.history

    # Check for reward terms
    self.reward_manager = self.env.unwrapped.reward_manager
    self.reward_names = getattr(self.reward_manager, "active_terms", ["reward"])
    num_rewards = len(self.reward_names)

    wm = WorldModel(
      num_full_state, num_actions, num_goals, num_rewards, dt=self.dt, history=history
    ).to(self.device)
    self.world_model_learner = WorldModelLearner(
      wm,
      self.cfg.wm,
      dt=self.dt,
      device=self.device,
      reward_names=self.reward_names,
    )

    if self.cfg.wandb_run_path is not None:
      load_wandb_checkpoint(self.cfg.wandb_run_path, self.task_id, wm, self.device)

    # Initialize Motion Sampler (Match PolicyLearner)
    motion_file = self.env.unwrapped.cfg.commands["motion"].motion_file

    # Calculate policy input dimension
    actor_input_dim = wm.num_rep_state + num_goals - 3
    critic_input_dim = wm.num_rep_state + num_goals

    print(
      f"[INFO] Actor Input Dim: {actor_input_dim}, Critic Input Dim: {critic_input_dim}"
    )

    # Create Actor and Critic MLPModels
    obs_tensors = TensorDict(
      {
        "actor": torch.zeros(self.env.num_envs, actor_input_dim, device=self.device),
        "critic": torch.zeros(self.env.num_envs, critic_input_dim, device=self.device),
      },
      batch_size=[self.env.num_envs],
      device=self.device,
    )
    obs_groups = {"actor": ["actor"], "critic": ["critic"]}

    actor_model = MLPModel(
      obs=obs_tensors,
      obs_groups=obs_groups,
      obs_set="actor",
      output_dim=num_actions,
      hidden_dims=self.cfg.actor_hidden_dims,
      activation=self.cfg.activation,
      obs_normalization=True,
      stochastic=True,
      init_noise_std=self.cfg.init_noise_std,
    ).to(self.device)

    critic_model = MLPModel(
      obs=obs_tensors,
      obs_groups=obs_groups,
      obs_set="critic",
      output_dim=1,
      hidden_dims=self.cfg.critic_hidden_dims,
      activation=self.cfg.activation,
      obs_normalization=True,
      stochastic=False,
    ).to(self.device)

    # Initialize PPO Storage (Imagination)
    # This is the ONLY storage used for training (on imagined data).
    # Real data is only used for the World Model via ReplayBuffer.
    obs_tensors_imagination = TensorDict(
      {
        "actor": torch.zeros(self.cfg.batch_size, actor_input_dim, device=self.device),
        "critic": torch.zeros(
          self.cfg.batch_size, critic_input_dim, device=self.device
        ),
      },
      batch_size=[self.cfg.batch_size],
      device=self.device,
    )

    storage = RolloutStorage(
      "rl",
      self.cfg.batch_size,
      self.cfg.num_steps_per_env,
      obs_tensors_imagination,
      [num_actions],
      device=self.device,
    )

    # Initialize PPO
    self.alg = PPO(
      actor=actor_model,
      critic=critic_model,
      storage=storage,
      device=self.device,
      num_learning_epochs=self.cfg.num_learning_epochs,
      num_mini_batches=self.cfg.num_mini_batches,
      clip_param=self.cfg.clip_param,
      gamma=self.cfg.gamma,
      lam=self.cfg.lam,
      value_loss_coef=self.cfg.value_loss_coef,
      entropy_coef=self.cfg.entropy_coef,
      learning_rate=self.cfg.learning_rate,
      max_grad_norm=self.cfg.max_grad_norm,
      desired_kl=self.cfg.desired_kl,
    )

    # Buffer for World Model: [state, action, goal, reward, done]
    self.replay_buffer = ReplayBuffer(
      dim=[num_full_state, num_actions, num_goals, num_rewards, 1, 1],
      buffer_size=self.cfg.buffer_size,
      device=self.device,
    )
    self.replay_buffer.initialize(self.env.num_envs)

    # Action Processing
    self.action_manager = self.env.unwrapped.action_manager
    # Assume only one action term
    term_name = self.action_manager.active_terms[0]
    self.action_term = self.action_manager.get_term(term_name)

    self.action_scale = self.action_term.scale.to(self.device).float()
    if self.action_scale.dim() > 1:
      self.action_scale = self.action_scale[0:1]

    self.wm_env = WorldModelEnv(
      world_model=wm,
      motion_file=motion_file,
      env=self.env,
      batch_size=self.cfg.batch_size,
      horizon=self.cfg.num_steps_per_env,
      replay_buffer=self.replay_buffer,
      device=self.device,
    )

  def imagine(self):
    """Unroll World Model to generate imagined trajectories for PPO training."""
    self.alg.eval_mode()
    self.alg.storage.clear()

    # Unroll
    with torch.inference_mode():
      obs = self.wm_env.reset()
      for _ in range(self.cfg.num_steps_per_env):
        actions = self.alg.act(obs)

        next_obs, rewards, dones, infos = self.wm_env.step(actions)
        self.alg.process_env_step(next_obs, rewards, dones, infos)

        obs = next_obs

    # Post-Rollout Update
    self.alg.compute_returns(obs)
    ppo_logs = self.alg.update()

    return ppo_logs

  def collect_data(self, num_steps):
    """Collects data from the environment."""
    steps_collected = 0
    obs = self.env.get_observations()
    norm_updates = []

    with torch.inference_mode():
      for _ in range(num_steps):
        steps_collected += 1
        curr_states, curr_goals = obs["full_state"], obs["goal"]
        actor_goal = torch.cat([curr_goals[..., :-9], curr_goals[..., -6:]], dim=-1)
        inputs = construct_policy_input(curr_states, actor_goal)

        obs_td = TensorDict(
          {"actor": inputs}, batch_size=[inputs.shape[0]], device=self.device
        )
        actions = self.alg.actor(obs_td)

        next_obs, rewards, dones, extras = self.env.step(actions)
        proc_actions = self.action_term._processed_actions

        reward_vec = self.reward_manager._step_reward.to(self.device).clone()
        if reward_vec.dim() == 1:
          reward_vec = reward_vec.unsqueeze(1)

        dones_float = dones.float()
        if "time_outs" in extras:
          dones_float[extras["time_outs"]] = 2.0

        motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
        motion_time = motion_cmd.time_steps.clone()

        self.replay_buffer.insert_transition(
          curr_states, proc_actions, curr_goals, reward_vec, motion_time, dones_float
        )
        norm_updates.append((curr_states.clone(), actions.clone(), curr_goals.clone()))
        self.logger.process_env_step(rewards, dones, extras)
        obs = next_obs

    # Update normalization outside inference_mode for torch.compile compatibility.
    for states, acts, goals in norm_updates:
      self.world_model_learner.update_normalization(states, acts, goals)

    return steps_collected * self.env.num_envs

  def run(self):
    log_dir = (
      Path("logs") / self.task_id / "ppo" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging to {log_dir}")
    self.logger.log_dir = str(log_dir)

    wandb_config = {
      "env_cfg": asdict(self.env.cfg),
      "runner_cfg": asdict(self.cfg),
    }

    wandb.init(
      project=self.cfg.wandb_project,
      name=self.cfg.run_name or f"{self.task_id}-ppo-{log_dir.name}",
      config=wandb_config,
    )
    self.logger.writer = wandb

    if self.registry_name:
      print(f"[INFO] Linking registry artifact: {self.registry_name}")
      artifact = wandb.Api().artifact(self.registry_name)
      wandb.run.use_artifact(artifact)

    # Reset Env
    obs, _ = self.env.reset()
    self.alg.storage.clear()

    start_it = 0
    total_it = self.cfg.max_iterations

    print(f"[INFO] Starting MBPO training loop for {total_it} iterations...")

    for iteration in range(start_it, total_it):
      start_time = time.time()

      # --- 1. Real World Data Collection (Sim) ---
      num_collected_steps = self.collect_data(
        max(1, int(self.cfg.buffer_size * self.cfg.buffer_ratio))
      )
      collect_time = time.time() - start_time

      full_len = self.cfg.wm.history - 1 + self.cfg.wm.horizon
      if self.replay_buffer.num_transitions < full_len:
        print("[WARN] Insufficient data. Skipping world model and policy update.")
        continue
      data = self.replay_buffer.mini_batch_generator(
        full_len, self.cfg.wm.num_batches, self.cfg.wm.batch_size
      )

      # --- 2. Train World Model ---
      start_wm_time = time.time()
      wm_logs = self.world_model_learner.update(data)
      wm_learn_time = time.time() - start_wm_time

      # --- 3. Train Policy (Imagination) ---
      start_policy_time = time.time()

      # Imagine and PPO Update
      ppo_logs = self.imagine()
      # Prefix PPO logs
      ppo_logs = {f"Policy/{k}": v for k, v in ppo_logs.items()}
      policy_learn_time = time.time() - start_policy_time

      # --- 4. Logging ---
      loss_dict = {**wm_logs, **ppo_logs}

      self.logger.log(
        it=iteration,
        total_it=total_it,
        collect_time=collect_time,
        wm_learn_time=wm_learn_time,
        policy_learn_time=policy_learn_time,
        loss_dict=loss_dict,
        num_collected_steps=num_collected_steps,
      )

      # Save checkpoints
      if (iteration + 1) % 10 == 0:
        torch.save(self.alg.save(), log_dir / "actor.pt")
        torch.save(
          self.world_model_learner.model.state_dict(), log_dir / "world_model.pt"
        )
        wandb.save(str(log_dir / "actor.pt"), base_path=str(log_dir), policy="now")
        wandb.save(
          str(log_dir / "world_model.pt"), base_path=str(log_dir), policy="now"
        )

        # ONNX Export
        # Pass the actor MLPModel (which has .mlp for the parent exporter)
        # and its obs_normalizer separately for the export pipeline
        actor = self.alg.actor
        export_motion_policy_as_onnx(
          self.env.unwrapped,
          actor,
          path=str(log_dir),
          normalizer=actor.obs_normalizer,
          filename="actor.onnx",
        )
        attach_onnx_metadata(
          self.env.unwrapped,
          wandb.run.name,
          path=str(log_dir),
          filename="actor.onnx",
        )
        wandb.save(str(log_dir / "actor.onnx"), base_path=str(log_dir), policy="now")

    wandb.finish()
