from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import wandb
from mjlab.viewer import NativeMujocoViewer

from supermimic.models import Policy, WorldModel
from supermimic.replay_buffer import ReplayBuffer
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
    num_state = state_obs.shape[-1]
    goal_obs = obs["goal"] if hasattr(obs, "keys") else obs
    num_goals = goal_obs.shape[-1]
    num_actions = self.env.num_actions

    print(f"[INFO] State Dim: {num_state}, Goal Dim: {num_goals}")

    dt = self.env.unwrapped.step_dt

    # Initialize World Model & Learner
    wm = WorldModel(num_state, num_actions, num_goals).to(self.device)
    self.world_model_learner = WorldModelLearner(
      wm, self.cfg.wm, dt=dt, device=self.device
    )

    # Initialize Policy & Learner
    policy_input_dim = num_state + num_goals
    if self.cfg.demo_policy:
      policy_input_dim = obs["policy"].shape[-1]

    policy = Policy(policy_input_dim, num_actions, normalization=True).to(self.device)
    policy_learner_cfg = self.cfg.policy
    policy_learner_cfg.motion_file = self.env.unwrapped.cfg.commands[
      "motion"
    ].motion_file
    self.policy_learner = PolicyLearner(
      policy, wm, policy_learner_cfg, dt=dt, device=self.device
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
      dim=[num_state, num_actions, num_goals, 1, 1],
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

      for step in range(num_steps):
        curr_state, curr_goal = self.get_obs_vec(obs)
        if self.cfg.demo_policy:
          action = self.policy_learner.policy(obs["policy"])
        else:
          action = self.policy_learner.get_action(curr_state, curr_goal)

        noise = torch.randn_like(action) * self.policy_learner.cfg.noise
        action = action + noise
        next_obs, reward, done, info = self.env.step(action)

        if viewer and not viewer.is_running():
          break
        if viewer:
          viewer.sync_env_to_viewer()

        # Handle 3-state done: 0=not done, 1=terminated, 2=timeout
        dones_float = done.float()
        if "time_outs" in info:
          time_outs = info["time_outs"]
          dones_float[time_outs] = 2.0

        self.replay_buffer.insert(
          [
            curr_state.unsqueeze(1),
            action.unsqueeze(1),
            curr_goal.unsqueeze(1),
            reward.unsqueeze(1).unsqueeze(2).float(),
            dones_float.unsqueeze(1).unsqueeze(2),
          ]
        )
        obs = next_obs

        if step % 100 == 0:
          print(f"Step {step}/{num_steps}")

    if viewer:
      viewer.close()

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
      self.collect_data(
        max(1, int(self.cfg.buffer_size * self.cfg.collection_buffer_ratio))
      )

      # 2. Train World Model
      self.world_model_learner.compute_stats(self.replay_buffer)
      wm_logs = self.world_model_learner.update(self.replay_buffer)

      # 3. Train Policy
      # self.policy_learner.compute_stats(self.replay_buffer)
      policy_logs = {}
      if not self.cfg.demo_policy:
        policy_logs = self.policy_learner.update()

      # 4. Logging & Saving
      full_log = {**wm_logs, **policy_logs}
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

    env_close_fn = getattr(self.env, "close", None)
    if env_close_fn:
      env_close_fn()

    wandb.finish()
