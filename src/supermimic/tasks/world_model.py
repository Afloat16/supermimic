"""World model environment configurations."""

from typing import TYPE_CHECKING

import torch

import mjlab.envs.mdp as mdp
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scripts.gcs import ensure_default_motion
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

if TYPE_CHECKING:
    from mjlab.entity import Entity

from mjlab.utils.lab_api.math import matrix_from_quat


def root_pos(
        env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: "Entity" = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w


def root_ori(
        env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: "Entity" = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    mat = matrix_from_quat(quat)
    # Use first two rows (6D representation)
    return mat[..., :2, :].reshape(*mat.shape[:-2], -1)


def root_lin_vel(
        env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: "Entity" = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_w


def root_ang_vel(
        env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: "Entity" = env.scene[asset_cfg.name]
    return asset.data.root_link_ang_vel_w


def qpos(
        env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: "Entity" = env.scene[asset_cfg.name]
    return asset.data.joint_pos


def qvel(
        env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: "Entity" = env.scene[asset_cfg.name]
    return asset.data.joint_vel


def world_model_env_cfg(
        has_state_estimation: bool = True,
        play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create world model environment configuration based on G1 tracking."""
    cfg = unitree_g1_flat_tracking_env_cfg(
        has_state_estimation=has_state_estimation, play=False  # play=False for world model
    )

    cfg.commands["motion"].debug_vis = False
    cfg.commands["motion"].motion_file = ensure_default_motion()
    cfg.commands["motion"].sampling_mode = "uniform"

    # Add full state observation group
    cfg.observations["full_state"] = ObservationGroupCfg(
        terms={
            "root_pos": ObservationTermCfg(func=root_pos),
            "root_ori": ObservationTermCfg(func=root_ori),
            "root_lin_vel": ObservationTermCfg(func=root_lin_vel),
            "root_ang_vel": ObservationTermCfg(func=root_ang_vel),
            "qpos": ObservationTermCfg(func=qpos),
            "qvel": ObservationTermCfg(func=qvel),
        },
        concatenate_terms=True,
        enable_corruption=False,
    )

    # Add goal observation group
    cfg.observations["goal"] = ObservationGroupCfg(
        terms={
            "motion": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "motion"}),
        },
        concatenate_terms=True,
        enable_corruption=False,
    )

    cfg.events.pop("push_robot", None)

    cfg.rewards.pop("motion_global_root_pos", None)
    cfg.rewards.pop("motion_global_root_ori", None)
    cfg.rewards.pop("self_collisions", None)

    cfg.terminations.pop("ee_body_pos", None)

    return cfg
