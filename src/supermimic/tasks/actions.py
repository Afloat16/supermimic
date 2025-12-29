from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class MotionResidualAction(JointPositionAction):
  """Joint position action with residuals from motion command."""

  cfg: MotionResidualActionCfg

  def __init__(self, cfg: MotionResidualActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._command_term = self._env.command_manager.get_term(self.cfg.command_name)

  def process_actions(self, actions: torch.Tensor):
    self._offset = self._command_term.joint_pos.clone()
    super().process_actions(actions)


@dataclass
class MotionResidualActionCfg(JointPositionActionCfg):
  command_name: str = "motion"

  def build(self, env: ManagerBasedRlEnv) -> MotionResidualAction:
    return MotionResidualAction(self, env)
