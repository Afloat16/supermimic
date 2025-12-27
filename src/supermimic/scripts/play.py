"""Script to visualize World Model prediction vs Ground Truth."""

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

import supermimic.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.gcs import ensure_default_checkpoint
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_from_matrix
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer
from supermimic.models import Policy, WorldModel


@dataclass
class PlayConfig:
    wandb_path: str
    task_id: str = "SuperMimic-WorldModel-G1"
    device: str | None = None
    num_steps_rollout: int = 50


class GhostVisualizer:
    def __init__(self, base_model):
        self.model = copy.deepcopy(base_model)
        self.model.geom_rgba[:, 3] = 0.8
        self.model.geom_rgba[:, 0:3] = [0.5, 0.5, 0.8]
        self.qpos = None

    def update(self, qpos):
        self.qpos = qpos

    def draw(self, viz):
        if self.qpos is not None:
            viz.add_ghost_mesh(self.qpos, model=self.model)


class SuperMimicNativeViewer(NativeMujocoViewer):
    def __init__(self, env, policy, wm, ghost_viz, num_steps_rollout, **kwargs):
        super().__init__(env, policy, **kwargs)
        self.wm = wm
        self.ghost_viz = ghost_viz
        self.num_steps_rollout = num_steps_rollout

        mj_model = self.env.unwrapped.sim.mj_model
        joint_names = [mj_model.joint(i).name for i in range(mj_model.njnt)]
        self.joint_q_adr = np.array(
            [mj_model.joint(n).qposadr[0] for n in joint_names if mj_model.joint(n).type != 0]
        )
        # Find free joint
        free_joint_id = next((i for i in range(mj_model.njnt) if mj_model.joint(i).type == 0), -1)
        self.free_joint_q_adr = None
        if free_joint_id >= 0:
            adr = mj_model.joint(free_joint_id).qposadr[0]
            self.free_joint_q_adr = np.arange(adr, adr + 7)

        self.step_in_rollout = 0
        self.curr_pred_state = None

    def reset_environment(self) -> None:
        super().reset_environment()
        self.step_in_rollout = 0
        self.curr_pred_state = None

    def _get_obs_vec(self, o):
        return o["policy"], o["full_state"], o["goal"]

    def step_simulation(self) -> None:
        if self._is_paused:
            return

        with torch.no_grad():
            with self._sim_timer.measure_time():
                obs = self.env.get_observations()
                curr_policy_gt, curr_gt_state, curr_goal = self._get_obs_vec(obs)

                if self.curr_pred_state is None or self.step_in_rollout == 0:
                    self.curr_pred_state = curr_gt_state.clone()

                # Policy Step
                action = self.policy(curr_policy_gt)
                action = action + torch.randn_like(action) * 0.2

                # Env Step
                _, _, dones, _ = self.env.step(action)
                self._step_count += 1

                # Predict Next State (Ghost)
                out = self.wm(self.curr_pred_state, action, curr_goal)
                self.curr_pred_state += out[..., :-1]

                # Visualization: Add Ghost
                state = self.curr_pred_state[0]
                root_pos = state[0:3].cpu().numpy()
                root_rot6d = state[3:9]

                # Reconstruct Quaternion from 6D (Rows 0,1)
                r1 = root_rot6d[:3]
                r2 = root_rot6d[3:]
                r3 = torch.linalg.cross(r1, r2)
                mat = torch.stack([r1, r2, r3], dim=0)
                root_quat = quat_from_matrix(mat).cpu().numpy()

                # Construct qpos
                qpos = np.zeros(self.env.unwrapped.sim.mj_model.nq)
                if self.free_joint_q_adr is not None:
                    qpos[self.free_joint_q_adr[0:3]] = root_pos
                    qpos[self.free_joint_q_adr[3:7]] = root_quat
                qpos[self.joint_q_adr] = state[15:44].cpu().numpy()

                self.ghost_viz.update(qpos)

                self.step_in_rollout += 1
                if dones.any() or self.step_in_rollout >= self.num_steps_rollout:
                    self.step_in_rollout = 0


def play(cfg: PlayConfig):
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Env
    print(f"[INFO] Loading environment: {cfg.task_id}")
    env_cfg = load_env_cfg(cfg.task_id, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = cfg.num_steps_rollout * env_cfg.sim.mujoco.timestep * env_cfg.decimation

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(base_env)

    # Load Policy
    resume_path = Path(ensure_default_checkpoint())
    obs, _ = env.reset()
    policy_obs = obs["policy"] if hasattr(obs, "keys") else obs
    num_obs = policy_obs.shape[-1]
    num_actions = env.num_actions

    policy = Policy(num_obs, num_actions, normalization=True).to(device)
    policy.load_state_dict_from_rsl_rl_ckpt(str(resume_path), device)
    policy.eval()

    # Load World Model
    state_obs = obs["full_state"] if hasattr(obs, "keys") else obs
    num_state = state_obs.shape[-1]

    goal_obs = obs["goal"] if hasattr(obs, "keys") else obs
    num_goals = goal_obs.shape[-1]

    wm = WorldModel(num_state, num_actions, num_goals).to(device)

    # Download World Model from WandB
    import wandb
    print(f"[INFO] Downloading from WandB: {cfg.wandb_path}")
    api = wandb.Api()
    run = api.run(cfg.wandb_path)
    download_dir = Path("logs") / cfg.task_id / run.id / "wandb_download"
    download_dir.mkdir(parents=True, exist_ok=True)

    # Download file
    run.file("world_model.pt").download(root=str(download_dir), replace=True)
    wm_path = download_dir / "world_model.pt"

    print(f"[INFO] Loading World Model from {wm_path}")
    wm.load_state_dict(torch.load(wm_path, map_location=device), strict=False)

    wm.eval()

    # Setup Ghost Visualizer
    ghost_viz = GhostVisualizer(base_env.sim.mj_model)
    # Patch the environment to call our visualizer
    base_env.update_visualizers = ghost_viz.draw

    # Viewer
    print("[INFO] Starting viewer...")
    viewer = SuperMimicNativeViewer(
        env=env,
        policy=policy,
        wm=wm,
        ghost_viz=ghost_viz,
        num_steps_rollout=cfg.num_steps_rollout,
    )
    viewer.run()

    env.close()


def main():
    args = tyro.cli(PlayConfig)
    play(args)


if __name__ == "__main__":
    main()
