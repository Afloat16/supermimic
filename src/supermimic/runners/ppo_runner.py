import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import wandb
from mjlab.tasks.tracking.rl.exporter import attach_onnx_metadata
from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic

from supermimic.models import WorldModel
from supermimic.replay_buffer import ReplayBuffer
from supermimic.runners.exporter import export_motion_policy_as_onnx
from supermimic.runners.learner import (
  PolicyLearnerCfg,
  SimpleMotionSampler,
  WorldModelLearner,
  WorldModelLearnerCfg,
)
from supermimic.utils.helpers import (
  YawAlignedStateLayout,
  apply_yaw_correction,
  compute_yaw_aligned_state,
  construct_policy_input,
  integrate_global_state,
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
    self.layout = YawAlignedStateLayout(num_joints)
    yaw_state_dim = self.layout.dim

    # Initialize World Model & Learner
    history = self.cfg.wm.history

    # Check for reward terms
    self.reward_manager = self.env.unwrapped.reward_manager
    self.reward_names = getattr(self.reward_manager, "active_terms", ["reward"])
    num_rewards = len(self.reward_names)

    wm = WorldModel(yaw_state_dim, num_actions, num_goals, num_rewards, history).to(
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
    sampler_cfg = PolicyLearnerCfg(motion_file=motion_file)
    self.sampler = SimpleMotionSampler(sampler_cfg, self.device)

    # Calculate policy input dimension
    dummy_state = torch.zeros(1, num_full_state, device=self.device)
    dummy_goal = torch.zeros(1, num_goals, device=self.device)
    policy_goal = torch.cat([dummy_goal[..., :-9], dummy_goal[..., -6:]], dim=-1)
    dummy_policy_input = construct_policy_input(dummy_state, policy_goal, self.layout)
    policy_input_dim = dummy_policy_input.shape[-1]
    dummy_yaw_state = compute_yaw_aligned_state(dummy_state, self.layout)
    critic_input_dim = dummy_yaw_state.shape[-1] + num_goals

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

  def get_obs_vec(self, o):
    return o["full_state"], o["goal"]

  def _get_inputs(self, obs):
    # Assume obs is from Env (dict or similar structure where we can extract vec)
    curr_state, curr_goal = self.get_obs_vec(obs)
    policy_goal = torch.cat([curr_goal[..., :-9], curr_goal[..., -6:]], dim=-1)

    # Policy Input
    policy_input = construct_policy_input(curr_state, policy_goal, self.layout)

    # Critic Input (Full Goal)
    yaw_state = compute_yaw_aligned_state(curr_state, self.layout)
    critic_input = torch.cat([yaw_state, curr_goal], dim=-1)

    return {"policy": policy_input, "critic": critic_input}

  def _sample_initial_states(self, batch_size):
    """Samples initial states, goals, and history from SimpleMotionSampler (MoCap)."""
    # Match PolicyLearner sampling logic
    full_state, goal_seq = self.sampler.sample(batch_size, self.cfg.num_steps_per_env)

    # 1. History: Repeat current state
    curr_yaw_state = compute_yaw_aligned_state(full_state, self.layout)
    yaw_state_hist = curr_yaw_state.unsqueeze(1).repeat(1, self.cfg.wm.history, 1)

    # 2. Last Action: Initialize to Zeros
    last_action = torch.zeros((batch_size, self.env.num_actions), device=self.device)

    return yaw_state_hist, last_action, goal_seq, full_state

  def imagine(self):
    """Unroll World Model to generate imagined trajectories for PPO training."""
    self.alg.policy.eval()

    # Sample Initial States
    (
      curr_yaw_state_hist,
      curr_last_action,
      goal_seq,
      curr_full_state,
    ) = self._sample_initial_states(self.cfg.batch_size)

    # Prepare PPO Storage for Imagination
    self.alg.storage.clear()

    # Unroll
    with torch.inference_mode():
      curr_yaw_state = curr_yaw_state_hist[:, -1, :]

      # Track global state
      curr_root_pos_w = curr_full_state[..., 0:3]
      curr_root_quat_w = curr_full_state[..., 3:7]

      # Initial PPO Obs (t=0)
      curr_goal_data = goal_seq[:, 0, :]
      ref_pos_w = curr_goal_data[..., -7:-4]
      ref_quat_w = curr_goal_data[..., -4:]
      motion_cmd = curr_goal_data[..., :-7]

      rel_pos_b, rel_ori_w = subtract_frame_transforms(
        curr_root_pos_w, curr_root_quat_w, ref_pos_w, ref_quat_w
      )
      rel_ori_mat = matrix_from_quat(rel_ori_w)
      rel_pos_b_flat = rel_pos_b.view(rel_pos_b.shape[0], -1)
      rel_ori_b_flat = rel_ori_mat[..., :2].reshape(rel_ori_mat.shape[0], -1)
      curr_goal_wm = torch.cat([motion_cmd, rel_pos_b_flat, rel_ori_b_flat], dim=-1)
      curr_goal_policy = torch.cat([motion_cmd, rel_ori_b_flat], dim=-1)
      curr_policy_input = torch.cat([curr_yaw_state, curr_goal_policy], dim=-1)
      curr_critic_input = torch.cat([curr_yaw_state, curr_goal_wm], dim=-1)
      curr_ppo_obs = {"policy": curr_policy_input, "critic": curr_critic_input}

      for t in range(self.cfg.num_steps_per_env):
        # 1. Act
        actions = self.alg.act(curr_ppo_obs)
        self.alg.transition.observations = curr_ppo_obs

        # 2. Step World Model
        rel_pos_b_flat = rel_pos_b.view(rel_pos_b.shape[0], -1)
        curr_goal_wm = torch.cat([motion_cmd, rel_pos_b_flat, rel_ori_b_flat], dim=-1)

        proc_action = (
          actions * self.action_scale + motion_cmd[..., : self.env.num_actions]
        )

        out = self.world_model_learner.model(
          curr_yaw_state_hist, proc_action, curr_last_action, curr_goal_wm
        )

        # Decode output
        num_rewards = self.world_model_learner.model.num_rewards
        delta = out[..., : -num_rewards - 1]
        pred_r = out[..., -num_rewards - 1 : -1]
        pred_d = out[..., -1:]
        reward = pred_r.sum(dim=-1) * self.dt
        dones = (pred_d > 0).float().squeeze(-1)

        # 3. Update State (Global Integration)
        # ------------------------------------
        next_yaw_state = curr_yaw_state + delta
        next_yaw_state = apply_yaw_correction(next_yaw_state, self.dt, self.layout)

        # Global State Integration
        curr_full_state = integrate_global_state(
          curr_full_state, next_yaw_state, self.dt, self.layout
        )
        curr_root_pos_w = curr_full_state[..., 0:3]
        curr_root_quat_w = curr_full_state[..., 3:7]

        # Handle Resets
        # -------------
        reset_indices = torch.where(dones > 0.5)[0]
        num_resets = len(reset_indices)

        if num_resets > 0:
          (
            reset_hist,
            reset_action,
            reset_goal_seq,
            reset_full_state,
          ) = self._sample_initial_states(num_resets)

          # Update Global State
          curr_full_state[reset_indices] = reset_full_state
          curr_root_pos_w[reset_indices] = reset_full_state[..., 0:3]
          curr_root_quat_w[reset_indices] = reset_full_state[..., 3:7]

          # Update Yaw State
          reset_yaw_state = reset_hist[:, -1, :]
          next_yaw_state[reset_indices] = reset_yaw_state

          # Update Goals
          len_remaining = self.cfg.num_steps_per_env - 1 - t
          if len_remaining > 0:
            goal_seq[reset_indices, t + 1 : t + 1 + len_remaining] = reset_goal_seq[
              :, :len_remaining
            ]

        # 4. Compute Next PPO Obs
        if t < self.cfg.num_steps_per_env - 1:
          next_goal_data = goal_seq[:, t + 1, :]
        else:
          next_goal_data = goal_seq[:, -1, :]

        next_ref_pos_w = next_goal_data[..., -7:-4]
        next_ref_quat_w = next_goal_data[..., -4:]
        next_motion_cmd = next_goal_data[..., :-7]

        next_rel_pos_b, next_rel_ori_w = subtract_frame_transforms(
          curr_root_pos_w, curr_root_quat_w, next_ref_pos_w, next_ref_quat_w
        )
        next_rel_ori_mat = matrix_from_quat(next_rel_ori_w)
        next_rel_ori_b_flat = next_rel_ori_mat[..., :2].reshape(
          next_rel_ori_mat.shape[0], -1
        )

        next_policy_goal = torch.cat([next_motion_cmd, next_rel_ori_b_flat], dim=-1)
        next_policy_input = torch.cat([next_yaw_state, next_policy_goal], dim=-1)

        next_rel_pos_b_flat = next_rel_pos_b.view(next_rel_pos_b.shape[0], -1)
        next_goal_wm = torch.cat(
          [next_motion_cmd, next_rel_pos_b_flat, next_rel_ori_b_flat], dim=-1
        )
        next_critic_input = torch.cat([next_yaw_state, next_goal_wm], dim=-1)

        next_ppo_obs = {
          "policy": next_policy_input,
          "critic": next_critic_input,
        }

        # 5. Process PPO Step
        # -------------------
        extras = {}
        self.alg.process_env_step(next_ppo_obs, reward, dones, extras)

        # Advance
        curr_yaw_state = next_yaw_state
        curr_yaw_state_hist = torch.cat(
          [curr_yaw_state_hist[:, 1:, :], next_yaw_state.unsqueeze(1)], dim=1
        )

        if num_resets > 0:
          curr_yaw_state_hist[reset_indices] = reset_hist
          proc_action[reset_indices] = reset_action

        curr_last_action = proc_action

        # Update loop vars
        motion_cmd = next_motion_cmd
        rel_pos_b = next_rel_pos_b
        rel_ori_b_flat = next_rel_ori_b_flat
        curr_ppo_obs = next_ppo_obs

    # Post-Rollout Update
    self.alg.compute_returns(next_ppo_obs)
    ppo_logs = self.alg.update()

    return ppo_logs

  def collect_data(self, num_steps):
    """Collects data from the environment."""
    steps_collected = 0
    obs = self.env.get_observations()

    with torch.inference_mode():
      for _ in range(num_steps):
        # Construct Policy Input
        ppo_obs = self._get_inputs(obs)

        # PPO Act (No storage update)
        actions = self.alg.act(ppo_obs)

        # Step Env
        next_obs, rewards, dones, extras = self.env.step(actions)

        # Store in ReplayBuffer for World Model
        proc_action = self.action_term._processed_actions
        curr_state, curr_goal = self.get_obs_vec(obs)

        reward_vec = self.reward_manager._step_reward.to(self.device).clone()
        if reward_vec.dim() == 1:
          reward_vec = reward_vec.unsqueeze(1)

        dones_float = dones.float()
        if "time_outs" in extras:
          dones_float[extras["time_outs"]] = 2.0

        self.replay_buffer.insert(
          [
            curr_state.unsqueeze(1),
            proc_action.unsqueeze(1),
            curr_goal.unsqueeze(1),
            reward_vec.unsqueeze(1).float(),
            dones_float.unsqueeze(1).unsqueeze(2),
          ]
        )
        self.logger.process_env_step(rewards, dones, extras)
        obs = next_obs
        steps_collected += 1

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

    # Initial Data Collection
    print("[INFO] Initial Data Collection...")
    self.replay_buffer.step = 0
    self.replay_buffer.num_transitions = 0
    self.collect_data(self.cfg.buffer_size)

    start_it = 0
    total_it = self.cfg.max_iterations

    print(f"[INFO] Starting MBPO training loop for {total_it} iterations...")

    for iteration in range(start_it, total_it):
      start_time = time.time()

      # --- 1. Real World Data Collection (Sim) ---
      num_collected_steps = self.collect_data(
        max(1, int(self.cfg.buffer_size * self.cfg.buffer_ratio))
      )

      stop_collect_time = time.time()
      collect_time = stop_collect_time - start_time

      # --- 2. Train World Model ---
      start_wm_time = time.time()
      self.world_model_learner.compute_stats(self.replay_buffer)
      wm_logs = self.world_model_learner.update(self.replay_buffer)
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
