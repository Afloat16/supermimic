from datetime import datetime
from pathlib import Path

import torch

from mjlab.viewer import NativeMujocoViewer


class SuperMimicRunner:
    def __init__(self, env, policy_learner, world_model_learner, replay_buffer, cfg):
        self.env = env
        self.policy_learner = policy_learner
        self.world_model_learner = world_model_learner
        self.replay_buffer = replay_buffer
        self.cfg = cfg
        self.device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_obs_vec(self, o):
        # Helper to extract parts
        return o["policy"], o["full_state"], o["goal"]

    def collect_data(self):
        print(f"[INFO] Collecting {self.cfg.buffer_size} steps...")

        viewer = None
        if not self.cfg.headless:
            # We assume policy_learner.policy is the model viewer needs
            viewer = NativeMujocoViewer(self.env, self.policy_learner.policy)
            viewer.setup()

        # Reset
        obs, _ = self.env.reset()

        # Reset Buffer pointers
        self.replay_buffer.step = 0
        self.replay_buffer.num_transitions = 0

        with torch.inference_mode():
            for step in range(self.cfg.buffer_size):
                curr_policy, curr_state, curr_goal = self.get_obs_vec(obs)
                action = self.policy_learner.get_action(curr_policy)
                next_obs, reward, done, _ = self.env.step(action)
                _, next_state, _ = self.get_obs_vec(next_obs)

                if viewer and not viewer.is_running():
                    break
                if viewer:
                    viewer.sync_env_to_viewer()

                self.replay_buffer.insert([
                    curr_state.unsqueeze(1),
                    action.unsqueeze(1),
                    next_state.unsqueeze(1),
                    reward.unsqueeze(1).unsqueeze(2).float(),
                    curr_goal.unsqueeze(1),
                    done.unsqueeze(1).unsqueeze(2).float()
                ])
                obs = next_obs

                if step % 100 == 0:
                    print(f"Step {step}/{self.cfg.buffer_size}")
                if self.replay_buffer.num_transitions >= self.cfg.buffer_size:
                    break

        if viewer:
            viewer.close()

    def train(self):
        # Compute Stats
        self.world_model_learner.compute_stats(self.replay_buffer)

        # Log Dir
        log_dir = Path("logs") / self.cfg.task_id / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Logging to {log_dir}")
        print("[INFO] Training World Model...")

        self.world_model_learner.update(self.replay_buffer, log_dir)

        env_close_fn = getattr(self.env, "close", None)
        if env_close_fn:
            env_close_fn()

    def run(self):
        self.collect_data()
        self.train()
