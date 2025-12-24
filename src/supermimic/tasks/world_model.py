"""World model environment configurations."""

from typing import TYPE_CHECKING

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

if TYPE_CHECKING:
    from mjlab.entity import Entity


def root_pos(
    env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: "Entity" = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w


def root_quat(
    env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: "Entity" = env.scene[asset_cfg.name]
    return asset.data.root_link_quat_w


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
        has_state_estimation=has_state_estimation, play=play
    )

    # Add full state observation group
    cfg.observations["full_state"] = ObservationGroupCfg(
        terms={
            "root_pos": ObservationTermCfg(func=root_pos),
            "root_quat": ObservationTermCfg(func=root_quat),
            "root_lin_vel": ObservationTermCfg(func=root_lin_vel),
            "root_ang_vel": ObservationTermCfg(func=root_ang_vel),
            "qpos": ObservationTermCfg(func=qpos),
            "qvel": ObservationTermCfg(func=qvel),
        },
        concatenate_terms=True,
        enable_corruption=False,
    )

    return cfg
