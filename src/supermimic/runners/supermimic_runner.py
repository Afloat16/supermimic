from datetime import datetime
from pathlib import Path

import torch
import wandb

from mjlab.viewer import NativeMujocoViewer


class SuperMimicRunner:
    def __init__(self, env, replay_buffer, world_model_learner, policy_learner, cfg):
        self.env = env
        self.replay_buffer = replay_buffer
        self.world_model_learner = world_model_learner
        self.policy_learner = policy_learner
        self.cfg = cfg
        self.device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

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
                action = self.policy_learner.get_action(curr_state, curr_goal)
                next_obs, reward, done, _ = self.env.step(action)

                if viewer and not viewer.is_running():
                    break
                if viewer:
                    viewer.sync_env_to_viewer()

                self.replay_buffer.insert([
                    curr_state.unsqueeze(1),
                    action.unsqueeze(1),
                    curr_goal.unsqueeze(1),
                    reward.unsqueeze(1).unsqueeze(2).float(),
                    done.unsqueeze(1).unsqueeze(2).float()
                ])
                obs = next_obs

                if step % 100 == 0:
                    print(f"Step {step}/{num_steps}")

        if viewer:
            viewer.close()

    def run(self):
        log_dir = Path("logs") / self.cfg.task_id / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Logging to {log_dir}")

        wandb.init(project=self.cfg.wandb_project, group=self.cfg.wandb_group,
                   name=f"{self.cfg.task_id}-{log_dir.name}", config=self.cfg)

        print("[INFO] Initial Data Collection...")
        # Reset buffer once at start
        self.env.reset()
        self.replay_buffer.step = 0
        self.replay_buffer.num_transitions = 0

        print(f"[INFO] Starting training loop for {self.cfg.max_iterations} iterations...")
        for iteration in range(self.cfg.max_iterations):
            # 1. Collect Data
            self.collect_data(max(1, int(self.cfg.buffer_size * self.cfg.collection_buffer_ratio)))

            # 2. Train World Model
            self.world_model_learner.compute_stats(self.replay_buffer)
            wm_logs = self.world_model_learner.update(self.replay_buffer)

            # 3. Train Policy
            policy_logs = self.policy_learner.update(self.replay_buffer)

            # 4. Logging & Saving
            full_log = {**wm_logs, **policy_logs}
            if full_log:
                wandb.log(full_log, step=iteration)

            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.cfg.max_iterations} Complete")
                torch.save(self.world_model_learner.model.state_dict(), log_dir / "world_model.pt")
                torch.save(self.policy_learner.policy.state_dict(), log_dir / "policy.pt")
                wandb.save(str(log_dir / "world_model.pt"), base_path=str(log_dir), policy="now")
                wandb.save(str(log_dir / "policy.pt"), base_path=str(log_dir), policy="now")

        env_close_fn = getattr(self.env, "close", None)
        if env_close_fn:
            env_close_fn()

        wandb.finish()
