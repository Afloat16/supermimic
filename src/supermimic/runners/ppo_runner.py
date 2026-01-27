import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import wandb
from mjlab.tasks.tracking.rl.exporter import attach_onnx_metadata
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic

from supermimic.models import WorldModel
from supermimic.replay_buffer import ReplayBuffer
from supermimic.runners.exporter import export_motion_policy_as_onnx
from supermimic.runners.learner import (
  WorldModelLearner,
  WorldModelLearnerCfg,
)
from supermimic.runners.world_model_env import WorldModelEnv
from supermimic.utils.helpers import (
  BodyFrameStateLayout,
  construct_policy_input,
)
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
    state_obs = obs["full_state"] if hasattr(obs, "keys") else obs
    num_full_state = state_obs.shape[-1]
    goal_obs = obs["goal"] if hasattr(obs, "keys") else obs
    num_goals = goal_obs.shape[-1]
    num_actions = self.env.num_actions

    print(f"[INFO] State Dim: {num_full_state}, Goal Dim: {num_goals}")

    self.dt = self.env.unwrapped.step_dt
    num_joints = (num_full_state - 13) // 2
    self.layout = BodyFrameStateLayout(num_joints)
    body_state_dim = self.layout.dim

    # Initialize World Model & Learner
    history = self.cfg.wm.history

    # Check for reward terms
    self.reward_manager = self.env.unwrapped.reward_manager
    self.reward_names = getattr(self.reward_manager, "active_terms", ["reward"])
    num_rewards = len(self.reward_names)

    wm = WorldModel(body_state_dim, num_actions, num_goals, num_rewards, history).to(
      self.device
    )
    self.world_model_learner = WorldModelLearner(
      wm,
      self.cfg.wm,
      dt=self.dt,
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

    # Initialize Motion Sampler (Match PolicyLearner)
    motion_file = self.env.unwrapped.cfg.commands["motion"].motion_file

    # Calculate policy input dimension
    policy_input_dim = body_state_dim + num_goals - 3
    critic_input_dim = body_state_dim + num_goals

    print(
      f"[INFO] Policy Input Dim: {policy_input_dim}, Critic Input Dim: {critic_input_dim}"
    )

    # Create ActorCritic
    obs_tensors = {
      "policy": torch.zeros(self.env.num_envs, policy_input_dim, device=self.device),
      "critic": torch.zeros(self.env.num_envs, critic_input_dim, device=self.device),
    }
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    actor_critic = ActorCritic(
      obs=obs_tensors,
      obs_groups=obs_groups,
      num_actions=num_actions,
      actor_hidden_dims=self.cfg.actor_hidden_dims,
      critic_hidden_dims=self.cfg.critic_hidden_dims,
      activation=self.cfg.activation,
      init_noise_std=self.cfg.init_noise_std,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
    ).to(self.device)

    self.obs_shape_spec = {
      "policy": (policy_input_dim,),
      "critic": (critic_input_dim,),
    }

    # Initialize PPO
    self.alg = PPO(
      policy=actor_critic,
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

    # Initialize PPO Storage (Imagination)
    # This is the ONLY storage used for training (on imagined data).
    # Real data is only used for the World Model via ReplayBuffer.
    obs_tensors_imagination = {
      "policy": torch.zeros(self.cfg.batch_size, policy_input_dim, device=self.device),
      "critic": torch.zeros(self.cfg.batch_size, critic_input_dim, device=self.device),
    }

    self.alg.init_storage(
      "rl",
      self.cfg.batch_size,
      self.cfg.num_steps_per_env,
      obs_tensors_imagination,
      [num_actions],
    )

    # Buffer for World Model: [state, action, goal, reward, done]
    self.replay_buffer = ReplayBuffer(
      dim=[num_full_state, num_actions, num_goals, num_rewards, 1],
      buffer_size=self.cfg.buffer_size,
      device=self.device,
    )

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
      layout=self.layout,
      device=self.device,
    )

  def imagine(self):
    """Unroll World Model to generate imagined trajectories for PPO training."""
    self.alg.policy.eval()
    self.alg.storage.clear()

    # Unroll
    with torch.inference_mode():
      obs = self.wm_env.get_observations()
      # obs = self.wm_env.reset()
      for _ in range(self.cfg.num_steps_per_env):
        actions = self.alg.act(obs)
        self.alg.transition.observations = obs

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

    with torch.inference_mode():
      for _ in range(num_steps):
        steps_collected += 1
        curr_states, curr_goals = obs["full_state"], obs["goal"]
        policy_goal = torch.cat([curr_goals[..., :-9], curr_goals[..., -6:]], dim=-1)
        inputs = construct_policy_input(curr_states, policy_goal, self.layout)

        actions = self.alg.policy.act({"policy": inputs})

        next_obs, rewards, dones, extras = self.env.step(actions)
        proc_actions = self.action_term._processed_actions

        reward_vec = self.reward_manager._step_reward.to(self.device).clone()
        if reward_vec.dim() == 1:
          reward_vec = reward_vec.unsqueeze(1)

        dones_float = dones.float()
        if "time_outs" in extras:
          dones_float[extras["time_outs"]] = 2.0

        self.replay_buffer.insert(
          [
            curr_states.unsqueeze(1),
            proc_actions.unsqueeze(1),
            curr_goals.unsqueeze(1),
            reward_vec.unsqueeze(1).float(),
            dones_float.unsqueeze(1).unsqueeze(2),
          ]
        )
        self.world_model_learner.update_normalization(curr_states, actions, curr_goals)
        self.logger.process_env_step(rewards, dones, extras)
        obs = next_obs

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
    self.wm_env.reset()

    self.replay_buffer.step = 0
    self.replay_buffer.num_transitions = 0

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
        torch.save(self.alg.policy.state_dict(), log_dir / "policy.pt")
        torch.save(
          self.world_model_learner.model.state_dict(), log_dir / "world_model.pt"
        )
        wandb.save(str(log_dir / "policy.pt"), base_path=str(log_dir), policy="now")
        wandb.save(
          str(log_dir / "world_model.pt"), base_path=str(log_dir), policy="now"
        )

        # ONNX Export
        policy = self.alg.policy
        normalizer = policy.actor_obs_normalizer
        export_motion_policy_as_onnx(
          self.env.unwrapped,
          policy,
          path=str(log_dir),
          normalizer=normalizer,
          filename="policy.onnx",
        )
        attach_onnx_metadata(
          self.env.unwrapped,
          wandb.run.name,
          path=str(log_dir),
          filename="policy.onnx",
        )
        wandb.save(str(log_dir / "policy.onnx"), base_path=str(log_dir), policy="now")

    wandb.finish()
