"""World model environment configurations."""

from typing import TYPE_CHECKING, cast

import mjlab.envs.mdp as mdp
import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)

from supermimic.utils.helpers import (
  build_maxcoord_full_state,
  compute_maxcoord_rep_state,
)

from .actions import MotionResidualActionCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def root_pos(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: "Entity" = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w - env.scene.env_origins


def root_ori(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: "Entity" = env.scene[asset_cfg.name]
  return asset.data.root_link_quat_w


def root_lin_vel(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: "Entity" = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_w


def root_ang_vel(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: "Entity" = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_w


def joint_pos(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: "Entity" = env.scene[asset_cfg.name]
  return asset.data.joint_pos


def joint_vel(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
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


# ---------------------------------------------------------------------------
# Maximal coordinate observation terms
# ---------------------------------------------------------------------------


def body_pos_w(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
  """Per-body positions in world frame, flattened. Shape: (N, B*3)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  pos = command.robot_body_pos_w  # (N, B, 3)
  # Subtract env origins from all bodies
  pos = pos - env.scene.env_origins.unsqueeze(1)
  return pos.reshape(env.num_envs, -1)


def body_quat_w(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
  """Per-body quaternions in world frame, flattened. Shape: (N, B*4)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  quat = command.robot_body_quat_w  # (N, B, 4)
  return quat.reshape(env.num_envs, -1)


def body_lin_vel_w(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  """Per-body linear velocities in world frame, flattened. Shape: (N, B*3)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  vel = command.robot_body_lin_vel_w  # (N, B, 3)
  return vel.reshape(env.num_envs, -1)


def body_ang_vel_w(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  """Per-body angular velocities in world frame, flattened. Shape: (N, B*3)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  vel = command.robot_body_ang_vel_w  # (N, B, 3)
  return vel.reshape(env.num_envs, -1)


def target_maxcoord_body_rep(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  """Maxcoord rep state of the target body state from motion. Shape: (N, B*16+3)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  t_pos = command.body_pos_w - env.scene.env_origins.unsqueeze(1)  # (N, B, 3)
  target_full = build_maxcoord_full_state(
    t_pos, command.body_quat_w, command.body_lin_vel_w, command.body_ang_vel_w
  )
  num_bodies = t_pos.shape[1]
  return compute_maxcoord_rep_state(target_full, num_bodies)


# ---------------------------------------------------------------------------
# Configuration factories
# ---------------------------------------------------------------------------


def world_model_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create world model environment configuration based on G1 tracking."""
  cfg = unitree_g1_flat_tracking_env_cfg(
    has_state_estimation=has_state_estimation,
    play=False,  # play=False for world model
  )

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
        noise=cfg.observations["actor"].terms["base_lin_vel"].noise,
      ),
      "motion_root_ori_b": ObservationTermCfg(
        func=motion_root_ori_b,
        params={"command_name": "motion"},
        noise=cfg.observations["actor"].terms["base_ang_vel"].noise,
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


def maxcoord_world_model_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create maximal-coordinate world model environment configuration.

  Extends the base world_model_env_cfg by replacing the full_state and goal
  observation groups with per-body maximal coordinate versions.
  """
  cfg = world_model_env_cfg(has_state_estimation=has_state_estimation, play=play)

  # Override full state: per-body maximal coordinates
  # Layout: [body_pos(B*3), body_quat(B*4), body_lin_vel(B*3), body_ang_vel(B*3)]
  cfg.observations["full_state"] = ObservationGroupCfg(
    terms={
      "body_pos_w": ObservationTermCfg(
        func=body_pos_w,
        params={"command_name": "motion"},
      ),
      "body_quat_w": ObservationTermCfg(
        func=body_quat_w,
        params={"command_name": "motion"},
      ),
      "body_lin_vel_w": ObservationTermCfg(
        func=body_lin_vel_w,
        params={"command_name": "motion"},
      ),
      "body_ang_vel_w": ObservationTermCfg(
        func=body_ang_vel_w,
        params={"command_name": "motion"},
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  # Override goal: [target_body_rep(B*16+3), rel_pos_b(3), rel_ori_b(6)]
  cfg.observations["goal"] = ObservationGroupCfg(
    terms={
      "target_body_rep": ObservationTermCfg(
        func=target_maxcoord_body_rep,
        params={"command_name": "motion"},
      ),
      "motion_root_pos_b": ObservationTermCfg(
        func=motion_root_pos_b,
        params={"command_name": "motion"},
        noise=cfg.observations["actor"].terms["base_lin_vel"].noise,
      ),
      "motion_root_ori_b": ObservationTermCfg(
        func=motion_root_ori_b,
        params={"command_name": "motion"},
        noise=cfg.observations["actor"].terms["base_ang_vel"].noise,
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  return cfg
