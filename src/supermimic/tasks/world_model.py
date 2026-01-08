"""World model environment configurations."""

from typing import TYPE_CHECKING, cast

import mjlab.envs.mdp as mdp
import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scripts.gcs import ensure_default_motion
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)

from .actions import MotionResidualActionCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity


def root_pos(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  asset: "Entity" = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w - env.scene.env_origins


def root_ori(
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


def joint_pos(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  asset: "Entity" = env.scene[asset_cfg.name]
  return asset.data.joint_pos


def joint_vel(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  asset: "Entity" = env.scene[asset_cfg.name]
  return asset.data.joint_vel


def motion_root_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_body_pos_w[:, 0],
    command.robot_body_quat_w[:, 0],
    command.body_pos_w[:, 0],
    command.body_quat_w[:, 0],
  )

  return pos.view(env.num_envs, -1)


def motion_root_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  _, ori = subtract_frame_transforms(
    command.robot_body_pos_w[:, 0],
    command.robot_body_quat_w[:, 0],
    command.body_pos_w[:, 0],
    command.body_quat_w[:, 0],
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def world_model_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create world model environment configuration based on G1 tracking."""
  cfg = unitree_g1_flat_tracking_env_cfg(
    has_state_estimation=has_state_estimation,
    play=False,  # play=False for world model
  )

  cfg.commands["motion"].motion_file = ensure_default_motion()
  cfg.commands["motion"].sampling_mode = "uniform"

  # Replace action with MotionResidualAction
  old_action = cfg.actions["joint_pos"]
  cfg.actions["joint_pos"] = MotionResidualActionCfg(
    entity_name=old_action.entity_name,
    actuator_names=old_action.actuator_names,
    scale=old_action.scale,
    command_name="motion",
  )

  # Add full state observation group
  cfg.observations["full_state"] = ObservationGroupCfg(
    terms={
      "root_pos": ObservationTermCfg(func=root_pos),
      "root_ori": ObservationTermCfg(func=root_ori),
      "joint_pos": ObservationTermCfg(func=joint_pos),
      "root_lin_vel": ObservationTermCfg(func=root_lin_vel),
      "root_ang_vel": ObservationTermCfg(func=root_ang_vel),
      "joint_vel": ObservationTermCfg(func=joint_vel),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  # Add goal observation group
  cfg.observations["goal"] = ObservationGroupCfg(
    terms={
      "motion": ObservationTermCfg(
        func=mdp.generated_commands, params={"command_name": "motion"}
      ),
      "motion_root_pos_b": ObservationTermCfg(
        func=motion_root_pos_b,
        params={"command_name": "motion"},
        noise=cfg.observations["policy"].terms["base_lin_vel"].noise,
      ),
      "motion_root_ori_b": ObservationTermCfg(
        func=motion_root_ori_b,
        params={"command_name": "motion"},
        noise=cfg.observations["policy"].terms["base_ang_vel"].noise,
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  cfg.events.pop("push_robot", None)
  cfg.events.clear()

  cfg.rewards.pop("self_collisions", None)
  # cfg.rewards.pop("motion_global_root_pos", None)
  # cfg.rewards.pop("motion_global_root_ori", None)

  cfg.terminations.pop("ee_body_pos", None)

  return cfg
