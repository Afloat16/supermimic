from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import wandb
from mjlab.tasks.tracking.rl.exporter import attach_onnx_metadata
from mjlab.viewer import NativeMujocoViewer

from supermimic.models import (
  Policy,
  WorldModel,
  YawAlignedStateLayout,
  construct_policy_input,
)
from supermimic.replay_buffer import ReplayBuffer
from supermimic.runners.exporter import export_motion_policy_as_onnx
from supermimic.runners.learner import (
  PolicyLearner,
  PolicyLearnerCfg,
  WorldModelLearner,
  WorldModelLearnerCfg,
)


@dataclass
class SuperMimicRunnerCfg:
  wandb_project: str = "supermimic"
  run_name: str | None = None
  wandb_run_path: str | None = None
  headless: bool = True

  max_iterations: int = 100000
  buffer_size: int = 500
  collection_buffer_ratio: float = 0.05

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
    reward_manager = self.env.unwrapped.reward_manager
    self.reward_names = getattr(reward_manager, "active_terms", ["reward"])
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
    policy_input_dim = yaw_state_dim + 2 + num_goals
    if self.cfg.demo_policy:
      policy_input_dim = obs["policy"].shape[-1]

    policy = Policy(policy_input_dim, num_actions, normalization=True).to(self.device)
    policy_learner_cfg = self.cfg.policy
    policy_learner_cfg.motion_file = self.env.unwrapped.cfg.commands[
      "motion"
    ].motion_file
    self.policy_learner = PolicyLearner(
      policy,
      wm,
      policy_learner_cfg,
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
    print(f"[INFO] Collecting {num_steps} steps...")

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

      ep_infos = []
      for step in range(num_steps):
        curr_state, curr_goal = self.get_obs_vec(obs)
        if self.cfg.demo_policy:
          action = self.policy_learner.policy(obs["policy"])
        else:
          motion_term = self.env.unwrapped.command_manager.get_term("motion")
          target_quat = motion_term.body_quat_w[..., 0, :].to(curr_state.device)

          inputs = construct_policy_input(
            curr_state, curr_goal, target_quat, self.layout
          )
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

        # Collect logs
        raw_logs = {
          **extras.get("log", {}),
          **{k: v for k, v in extras.items() if k not in ["log", "time_outs"]},
        }
        step_log = {}
        for k, v in raw_logs.items():
          if isinstance(v, torch.Tensor):
            v = v.item()
          if isinstance(v, (int, float)):
            step_log[k] = v

        if step_log:
          ep_infos.append(step_log)

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
        reward_vec = self.env.unwrapped.reward_manager._step_reward.to(
          self.device
        ).clone()
        # If reward_vec is (N, D), we need (N, 1, D)
        if reward_vec.dim() == 1:
          reward_vec = reward_vec.unsqueeze(1)  # Handle single env case

        self.replay_buffer.insert(
          [
            curr_state.unsqueeze(1),
            action.unsqueeze(1),
            curr_goal.unsqueeze(1),
            reward_vec.unsqueeze(1).float(),
            dones_float.unsqueeze(1).unsqueeze(2),
          ]
        )
        obs = next_obs

        if step % 100 == 0:
          print(f"Step {step}/{num_steps}")

    if viewer:
      viewer.close()

    # Consolidate logs
    logs = {}
    if len(ep_infos) > 0:
      all_keys = set().union(*ep_infos)
      for k in all_keys:
        values = [x[k] for x in ep_infos if k in x]
        if len(values) > 0:
          logs[k] = sum(values) / len(values)
    return logs

  def run(self):
    log_dir = Path("logs") / self.task_id / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging to {log_dir}")

    wandb_config = {
      "env_cfg": asdict(self.env.cfg),
      "runner_cfg": asdict(self.cfg),
    }

    wandb.init(
      project=self.cfg.wandb_project,
      name=self.cfg.run_name or f"{self.task_id}-{log_dir.name}",
      config=wandb_config,
    )

    if self.registry_name:
      print(f"[INFO] Linking registry artifact: {self.registry_name}")
      artifact = wandb.Api().artifact(self.registry_name)
      wandb.run.use_artifact(artifact)

    print("[INFO] Initial Data Collection...")
    # Reset buffer once at start
    self.env.reset()
    self.replay_buffer.step = 0
    self.replay_buffer.num_transitions = 0
    self.collect_data(self.cfg.buffer_size)
    print(f"[INFO] Starting training loop for {self.cfg.max_iterations} iterations...")
    for iteration in range(self.cfg.max_iterations):
      # 1. Collect Data
      collect_logs = self.collect_data(
        max(1, int(self.cfg.buffer_size * self.cfg.collection_buffer_ratio))
      )

      # 2. Train World Model
      self.world_model_learner.compute_stats(self.replay_buffer)
      wm_logs = self.world_model_learner.update(self.replay_buffer)

      # 3. Train Policy
      policy_logs = {}
      if not self.cfg.demo_policy:
        self.policy_learner.compute_stats(self.replay_buffer)
        policy_logs = self.policy_learner.update()

      # 4. Logging & Saving
      full_log = {**wm_logs, **policy_logs, **collect_logs}
      if full_log:
        wandb.log(full_log, step=iteration)

      if (iteration + 1) % 10 == 0:
        print(f"Iteration {iteration + 1}/{self.cfg.max_iterations} Complete")
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
