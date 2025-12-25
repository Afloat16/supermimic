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
    num_envs: int = 1  # Visualize one env
    device: str | None = None


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


def play(cfg: PlayConfig):
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Env
    print(f"[INFO] Loading environment: {cfg.task_id}")
    env_cfg = load_env_cfg(cfg.task_id, play=True)
    env_cfg.scene.num_envs = cfg.num_envs

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

    # Indices helper
    joint_names = [base_env.sim.mj_model.joint(i).name for i in range(base_env.sim.mj_model.njnt)]
    joint_q_adr = np.array(
        [base_env.sim.mj_model.joint(n).qposadr[0] for n in joint_names if base_env.sim.mj_model.joint(n).type != 0])
    # Find free joint
    free_joint_id = -1
    for i in range(base_env.sim.mj_model.njnt):
        if base_env.sim.mj_model.joint(i).type == 0:  # mjJNT_FREE
            free_joint_id = i
            break
    free_joint_q_adr = None
    if free_joint_id >= 0:
        free_joint_q_adr = np.arange(base_env.sim.mj_model.joint(free_joint_id).qposadr[0],
                                     base_env.sim.mj_model.joint(free_joint_id).qposadr[0] + 7)

    # Viewer
    print("[INFO] Starting viewer...")
    viewer = NativeMujocoViewer(base_env, policy)
    viewer.setup()

    # State helpers
    def get_obs_vec(o):
        return o["policy"], o["full_state"], o["goal"]

    # Visualization Loop
    rollout_len = 50  # 1 sec / 0.02 dt
    step_in_rollout = 0

    # Initial GT
    curr_policy, curr_gt_state, _ = get_obs_vec(obs)
    curr_pred_state = curr_gt_state.clone()

    with torch.inference_mode():
        # Loop for a fixed number of steps
        step = 0

        while viewer.is_running():
            step += 1

            # 1. Sync Ghost to GT if needed
            if step_in_rollout == 0:
                _, curr_gt_state, _ = get_obs_vec(obs)
                curr_pred_state = curr_gt_state.clone()

            # 2. GT Step
            curr_policy_gt, _, curr_goal = get_obs_vec(obs)
            action = policy(curr_policy_gt)
            next_obs, _, dones, _ = env.step(action)

            # 3. Predict Next State (Ghost)
            out = wm(curr_pred_state, action, curr_goal)
            delta = out[..., :-1]
            next_pred_state = curr_pred_state + delta

            # 4. Visualization: Add Ghost
            # Viz state is (1, D)
            state = next_pred_state[0]  # assuming batch 1

            # Extract parts
            root_pos = state[0:3].cpu().numpy()
            root_rot6d = state[3:9]  # Keep in torch

            # Reconstruct Quaternion from 6D (Rows 0,1)
            r1 = root_rot6d[:3]
            r2 = root_rot6d[3:]
            r3 = torch.linalg.cross(r1, r2)
            mat = torch.stack([r1, r2, r3], dim=0)  # (3, 3)
            root_quat = quat_from_matrix(mat).cpu().numpy()

            # Use Pred Joints
            # Indices: 0:3 Pos, 3:9 Ori, 9:12 LinVel, 12:15 AngVel, 15:44 QPos
            joint_pos = state[15:44].cpu().numpy()

            # Construct qpos
            qpos = np.zeros(base_env.sim.mj_model.nq)

            if free_joint_q_adr is not None:
                qpos[free_joint_q_adr[0:3]] = root_pos
                qpos[free_joint_q_adr[3:7]] = root_quat

            qpos[joint_q_adr] = joint_pos

            # Update Visualization State
            ghost_viz.update(qpos)

            # Sync Viewer (Calls ghost_viz.draw internally via callback)
            if viewer:
                viewer.sync_env_to_viewer()

            # 5. Update State for next loop
            obs = next_obs
            curr_pred_state = next_pred_state

            # 6. Update Counter & Check Reset
            step_in_rollout += 1
            if dones.any() or step_in_rollout >= rollout_len:
                step_in_rollout = 0

    if viewer: viewer.close()
    env.close()


def main():
    args = tyro.cli(PlayConfig)
    play(args)


if __name__ == "__main__":
    main()
