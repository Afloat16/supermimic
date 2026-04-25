"""Environment configurations for SuperMimic representations."""

from __future__ import annotations

from typing import cast

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

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _make_asset_obs_fn(attr_name: str, *, subtract_origins: bool = False):
  """Factory for observation terms that read a single asset data attribute."""

  def fn(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
  ) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    val = getattr(asset.data, attr_name)
    return val - env.scene.env_origins if subtract_origins else val

  return fn


root_pos = _make_asset_obs_fn("root_link_pos_w", subtract_origins=True)
root_ori = _make_asset_obs_fn("root_link_quat_w")
root_lin_vel = _make_asset_obs_fn("root_link_lin_vel_w")
root_ang_vel = _make_asset_obs_fn("root_link_ang_vel_w")
joint_pos = _make_asset_obs_fn("joint_pos")
joint_vel = _make_asset_obs_fn("joint_vel")


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


def _make_body_obs_fn(attr_name: str, *, subtract_origins: bool = False):
  """Factory for observation terms that read a MotionCommand body attribute."""

  def fn(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    val = getattr(command, attr_name)
    if subtract_origins:
      val = val - env.scene.env_origins.unsqueeze(1)
    return val.reshape(env.num_envs, -1)

  return fn


body_pos_w = _make_body_obs_fn("robot_body_pos_w", subtract_origins=True)
body_quat_w = _make_body_obs_fn("robot_body_quat_w")
body_lin_vel_w = _make_body_obs_fn("robot_body_lin_vel_w")
body_ang_vel_w = _make_body_obs_fn("robot_body_ang_vel_w")


# ---------------------------------------------------------------------------
# Configuration factories
# ---------------------------------------------------------------------------


def gencoord_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create generalized-coordinate world model environment configuration based on G1 tracking."""
  cfg = unitree_g1_flat_tracking_env_cfg(
    has_state_estimation=has_state_estimation,
    play=play,
  )

  cfg.commands["motion"].sampling_mode = "uniform"

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


def maxcoord_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create maximal-coordinate world model environment configuration.

  Extends the base gencoord_env_cfg by replacing the full_state and goal
  observation groups with per-body maximal coordinate versions.
  """
  cfg = gencoord_env_cfg(has_state_estimation=has_state_estimation, play=play)

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

  # Override goal: [joint_target, rel_pos_b(3), rel_ori_b(6)]
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

  return cfg
