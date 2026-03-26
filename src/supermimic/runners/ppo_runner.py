from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import wandb
from mjlab.tasks.tracking.rl.exporter import attach_onnx_metadata
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from supermimic.runners.exporter import export_motion_policy_as_onnx
from supermimic.runners.supermimic_runner import BaseRunner, BaseRunnerCfg


@dataclass
class PpoRunnerCfg(BaseRunnerCfg):
  max_iterations: int = 25000
  buffer_size: int = 1000
  log_interval: int = 1
  save_interval: int = 10
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
  buffer_ratio: float = 0.01

  # Policy Architecture
  actor_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
  critic_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
  activation: str = "elu"
  init_noise_std: float = 1.0


class PpoRunner(BaseRunner):
  def __init__(
    self,
    env,
    cfg: PpoRunnerCfg,
    task_id: str,
    registry_name: str | None = None,
    device: str = "cuda:0",
  ):
    super().__init__(env, cfg, task_id, registry_name, device)

  def _init_policy(self) -> None:
    num_actions = self._num_actions

    motion_file = self.env.unwrapped.cfg.commands["motion"].motion_file

    actor_input_dim, critic_input_dim = self._policy_input_dims()

    print(
      f"[INFO] Actor Input Dim: {actor_input_dim}, Critic Input Dim: {critic_input_dim}"
    )

    actor_model = self._build_mlp(
      "actor",
      actor_input_dim,
      num_actions,
      hidden_dims=self.cfg.actor_hidden_dims,
      activation=self.cfg.activation,
      distribution_cfg={
        "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
        "init_std": self.cfg.init_noise_std,
      },
    )

    critic_model = self._build_mlp(
      "critic",
      critic_input_dim,
      1,
      hidden_dims=self.cfg.critic_hidden_dims,
      activation=self.cfg.activation,
    )

    # Initialize PPO Storage (Imagination)
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

    self.wm_env = self._create_wm_env(
      motion_file, self.cfg.batch_size, self.cfg.num_steps_per_env
    )

  def imagine(self) -> dict[str, float]:
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

  def collect_data(self, num_steps: int) -> int:
    """Collects data from the environment."""
    steps_collected = 0
    obs = self.env.get_observations()
    norm_updates = []

    with torch.inference_mode():
      for _ in range(num_steps):
        steps_collected += 1
        curr_states, curr_goals = obs["full_state"], obs["goal"]
        actions = self.alg.actor(self._actor_obs(curr_states, curr_goals))

        next_obs, rewards, dones, extras = self._step_and_store(
          actions, curr_states, curr_goals
        )
        norm_updates.append((curr_states.clone(), actions.clone(), curr_goals.clone()))
        self.logger.process_env_step(rewards, dones, extras)
        obs = next_obs

    # Update normalization outside inference_mode for torch.compile compatibility.
    for states, acts, goals in norm_updates:
      self.world_model_learner.update_normalization(states, acts, goals)

    return steps_collected * self.env.num_envs

  def run(self) -> None:
    log_dir = self._create_log_dir("ppo")
    self._init_wandb(log_dir, name_suffix="-ppo")

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
      if (iteration + 1) % self.cfg.save_interval == 0:
        torch.save(self.alg.save(), log_dir / "actor.pt")
        torch.save(
          self.world_model_learner.model.state_dict(), log_dir / "world_model.pt"
        )
        wandb.save(str(log_dir / "actor.pt"), base_path=str(log_dir), policy="now")
        wandb.save(
          str(log_dir / "world_model.pt"), base_path=str(log_dir), policy="now"
        )

        # ONNX Export
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

    self._cleanup()
