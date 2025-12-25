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
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer
from supermimic.policy import Mlp
from supermimic.world_model import WorldModel


@dataclass
class PlayConfig:
    wandb_path: str
    task_id: str = "SuperMimic-WorldModel-G1"
    num_envs: int = 1  # Visualize one env
    device: str | None = None
    headless: bool = False  # Should be false for play by default (unless testing)


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

    policy = Mlp(num_obs, num_actions, normalization=True).to(device)
    policy.load_state_dict_from_rsl_rl_ckpt(str(resume_path), device)
    policy.eval()

    # Load World Model
    state_obs = obs["full_state"] if hasattr(obs, "keys") else obs
    num_state = state_obs.shape[-1]
    wm = WorldModel(num_state, num_actions).to(device)

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

    # Setup Ghost Model for Visualization
    ghost_model = copy.deepcopy(base_env.sim.mj_model)
    ghost_color = np.array([0.5, 0.5, 0.7, 1.0], dtype=np.float32)  # Blue transparent
    ghost_model.geom_rgba[:] = ghost_color

    # Robot Indexing
    robot = base_env.scene["robot"]
    indexing = robot.indexing
    if hasattr(indexing, "free_joint_q_adr"):
        free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()  # [0:3]=pos, [3:7]=quat
    else:
        free_joint_q_adr = None

    joint_q_adr = indexing.joint_q_adr.cpu().numpy()

    # Viewer
    viewer = None
    if not cfg.headless:
        print("[INFO] Starting viewer...")
        viewer = NativeMujocoViewer(base_env, policy)
        viewer.setup()
    else:
        print("[INFO] Headless mode: Running without viewer for debugging.")

    # State helpers
    def get_obs_vec(o):
        return o["policy"], o["full_state"]

    # Visualization Loop
    rollout_len = 50  # 1 sec / 0.02 dt
    step_in_rollout = 0

    # Initial GT
    curr_policy, curr_gt_state = get_obs_vec(obs)
    curr_pred_state = curr_gt_state.clone()

    with torch.inference_mode():
        # Loop for a fixed number of steps if headless, else while viewer runs
        max_steps = 200 if cfg.headless else float('inf')
        step = 0

        while (viewer is None or viewer.is_running()) and step < max_steps:
            step += 1

            # GT Step
            curr_policy_gt, _ = get_obs_vec(obs)
            action = policy(curr_policy_gt)
            next_obs, _, dones, _ = env.step(action)

            # Model Step
            if step_in_rollout == 0:
                # Reset to GT
                _, curr_gt_state = get_obs_vec(obs)
                curr_pred_state = curr_gt_state.clone()

            # Predict Next State
            next_pred_state = curr_pred_state + wm(curr_pred_state, action)

            # Visualization: Add Ghost
            if viewer and viewer.viewer:
                viewer.sync_env_to_viewer()

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

                # Add ghost
                viz = MujocoNativeDebugVisualizer(viewer.viewer.user_scn, viewer.mjm, viewer.env_idx)
                viz.add_ghost_mesh(qpos, model=ghost_model)

                # Sync again to update scene with ghost
                viewer.viewer.sync()

            # Update loop
            obs = next_obs
            curr_pred_state = next_pred_state

            step_in_rollout += 1
            if step_in_rollout >= rollout_len or dones.any():
                step_in_rollout = 0

    if viewer: viewer.close()
    env.close()


def main():
    args = tyro.cli(PlayConfig)
    play(args)


if __name__ == "__main__":
    main()
