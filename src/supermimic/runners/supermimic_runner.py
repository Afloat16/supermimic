import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import wandb
from mjlab.tasks.tracking.rl.exporter import attach_onnx_metadata
from mjlab.viewer import NativeMujocoViewer

from supermimic.models import Policy, WorldModel
from supermimic.replay_buffer import ReplayBuffer
from supermimic.runners.exporter import export_motion_policy_as_onnx
from supermimic.runners.learner import (
  PolicyLearner,
  PolicyLearnerCfg,
  WorldModelLearner,
  WorldModelLearnerCfg,
)
from supermimic.runners.world_model_env import WorldModelEnv
from supermimic.utils.helpers import YawAlignedStateLayout, construct_policy_input
from supermimic.utils.logger import Logger


@dataclass
class SuperMimicRunnerCfg:
  wandb_project: str = "supermimic"
  run_name: str | None = None
  wandb_run_path: str | None = None
  headless: bool = True

  max_iterations: int = 100000
  buffer_size: int = 500
  collection_buffer_ratio: float = 10 / buffer_size

  demo_policy: bool = False

  wm: WorldModelLearnerCfg = field(default_factory=WorldModelLearnerCfg)
  policy: PolicyLearnerCfg = field(default_factory=PolicyLearnerCfg)


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
    self.layout = YawAlignedStateLayout(num_joints)
    yaw_state_dim = self.layout.dim

    # Initialize World Model & Learner
    history = self.cfg.wm.history

    # Check for reward terms
    self.reward_manager = self.env.unwrapped.reward_manager
    self.reward_names = getattr(self.reward_manager, "active_terms", ["reward"])
    num_rewards = len(self.reward_names)
    print(f"[INFO] Reward Dim: {num_rewards} ({self.reward_names})")

    wm = WorldModel(yaw_state_dim, num_actions, num_goals, num_rewards, history).to(
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
    policy_input_dim = yaw_state_dim + num_goals - 3
    if self.cfg.demo_policy:
      policy_input_dim = obs["policy"].shape[-1]

    policy = Policy(policy_input_dim, num_actions, normalization=True).to(self.device)
    policy_learner_cfg = self.cfg.policy
    policy_learner_cfg.motion_file = self.env.unwrapped.cfg.commands[
      "motion"
    ].motion_file

    # Extract Action Scale & Offset
    self.action_manager = self.env.unwrapped.action_manager
    term_name = self.action_manager.active_terms[0]
    self.action_term = self.action_manager.get_term(term_name)

    self.wm_env = WorldModelEnv(
      world_model=wm,
      motion_file=policy_learner_cfg.motion_file,
      env=self.env,
      batch_size=policy_learner_cfg.batch_size,
      horizon=policy_learner_cfg.horizon,
      layout=self.layout,
      device=self.device,
    )

    self.policy_learner = PolicyLearner(
      policy,
      self.wm_env,
      policy_learner_cfg,
      env=self.env,
      dt=dt,
      layout=self.layout,
      device=self.device,
    )

    if self.cfg.demo_policy:
      from mjlab.scripts.gcs import ensure_default_checkpoint

      checkpoint_path = ensure_default_checkpoint()
      print(f"[INFO] Loading demo policy from {checkpoint_path}")
      self.policy_learner.policy.load_state_dict_from_rsl_rl_ckpt(
        checkpoint_path, self.device
      )
      self.policy_learner.policy.eval()
      self.policy_learner.policy.requires_grad_(False)
      print("[INFO] Demo policy loaded and frozen (eval mode, no grad).")

    # Buffer: [state, action, goal, reward, done]
    self.replay_buffer = ReplayBuffer(
      dim=[num_full_state, num_actions, num_goals, num_rewards, 1],
      buffer_size=self.cfg.buffer_size,
      device=self.device,
    )

  def get_obs_vec(self, o):
    # Helper to extract parts
    return o["full_state"], o["goal"]

  def collect_data(self, num_steps):
    viewer = None
    if not self.cfg.headless:
      viewer = NativeMujocoViewer(self.env, self.policy_learner.policy)
      viewer.setup()

    with torch.inference_mode():
      obs = self.env.get_observations()

      # Initialize history buffer
      curr_state, _ = self.get_obs_vec(obs)
      history = self.cfg.wm.history
      curr_state_hist = torch.stack([curr_state] * history, dim=1)

      steps_collected = 0
      for step in range(num_steps):
        steps_collected += 1
        curr_state, curr_goal = self.get_obs_vec(obs)
        if self.cfg.demo_policy:
          action = self.policy_learner.policy(obs["policy"])
        else:
          policy_goal = torch.cat([curr_goal[..., :-9], curr_goal[..., -6:]], dim=-1)
          inputs = construct_policy_input(curr_state, policy_goal, self.layout)

          action = self.policy_learner.policy(inputs)

        noise = torch.randn_like(action) * self.policy_learner.cfg.noise
        action = action + noise
        next_obs, reward, done, extras = self.env.step(action)

        if viewer and not viewer.is_running():
          break
        if viewer:
          viewer.sync_env_to_viewer()

        # Handle 3-state done: 0=not done, 1=terminated, 2=timeout
        dones_float = done.float()
        if "time_outs" in extras:
          time_outs = extras["time_outs"]
          dones_float[time_outs] = 2.0

        self.logger.process_env_step(reward, done, extras)

        # Update history for next step
        next_state, _ = self.get_obs_vec(next_obs)
        curr_state_hist = torch.cat(
          [curr_state_hist[:, 1:], next_state.unsqueeze(1)], dim=1
        )

        # Handle resets: fill history with new start state
        reset_env_ids = torch.where(done)[0]
        if len(reset_env_ids) > 0:
          reset_states = next_state[reset_env_ids]
          curr_state_hist[reset_env_ids] = torch.stack([reset_states] * history, dim=1)

        # Get reward vector from reward manager
        # Note: env.step returns sum of rewards, we want the vector
        # We assume reward_manager._step_reward is populated after step()
        reward_vec = self.reward_manager._step_reward.to(self.device).clone()
        # If reward_vec is (N, D), we need (N, 1, D)
        if reward_vec.dim() == 1:
          reward_vec = reward_vec.unsqueeze(1)  # Handle single env case

        proc_action = self.action_term._processed_actions

        self.replay_buffer.insert(
          [
            curr_state.unsqueeze(1),
            proc_action.unsqueeze(1),
            curr_goal.unsqueeze(1),
            reward_vec.unsqueeze(1).float(),
            dones_float.unsqueeze(1).unsqueeze(2),
          ]
        )
        obs = next_obs

    if viewer:
      viewer.close()

    return steps_collected * self.env.num_envs

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
      print(f"[INFO] Linking registry artifact: {self.registry_name}")
      artifact = wandb.Api().artifact(self.registry_name)
      wandb.run.use_artifact(artifact)

    self.env.reset()
    self.wm_env.reset()

    print("[INFO] Initial Data Collection...")
    self.replay_buffer.step = 0
    self.replay_buffer.num_transitions = 0
    self.collect_data(self.cfg.buffer_size)

    print(f"[INFO] Starting training loop for {self.cfg.max_iterations} iterations...")

    start_it = 0
    total_it = self.cfg.max_iterations

    for iteration in range(start_it, total_it):
      start_time = time.time()

      # 1. Collect Data
      num_collected_steps = self.collect_data(
        max(1, int(self.cfg.buffer_size * self.cfg.collection_buffer_ratio))
      )

      stop_collect_time = time.time()
      collect_time = stop_collect_time - start_time

      # 2. Train World Model
      start_wm_time = time.time()
      self.world_model_learner.compute_stats(self.replay_buffer)
      wm_logs = self.world_model_learner.update(self.replay_buffer)
      wm_learn_time = time.time() - start_wm_time

      # 3. Train Policy
      start_policy_time = time.time()
      policy_logs = {}
      if not self.cfg.demo_policy:
        self.policy_learner.compute_stats(self.replay_buffer)
        policy_logs = self.policy_learner.update()
      policy_learn_time = time.time() - start_policy_time

      stop_learn_time = time.time()
      learn_time = stop_learn_time - stop_collect_time

      # 4. Logging & Saving
      # Combine logs for Logger
      loss_dict = {**wm_logs}
      loss_dict.update(policy_logs)

      self.logger.log(
        it=iteration,
        total_it=total_it,
        collect_time=collect_time,
        wm_learn_time=wm_learn_time,
        policy_learn_time=policy_learn_time,
        loss_dict=loss_dict,
        num_collected_steps=num_collected_steps,
      )

      if (iteration + 1) % 10 == 0:
        torch.save(
          self.world_model_learner.model.state_dict(), log_dir / "world_model.pt"
        )
        torch.save(self.policy_learner.policy.state_dict(), log_dir / "policy.pt")
        wandb.save(
          str(log_dir / "world_model.pt"), base_path=str(log_dir), policy="now"
        )
        wandb.save(str(log_dir / "policy.pt"), base_path=str(log_dir), policy="now")

        # ONNX Export
        policy = self.policy_learner.policy
        normalizer = policy.actor_obs_normalizer if policy.normalization else None
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

    env_close_fn = getattr(self.env, "close", None)
    if env_close_fn:
      env_close_fn()

    wandb.finish()
