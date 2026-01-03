import os
from typing import cast

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.tracking.mdp import MotionCommand
from mjlab.utils.lab_api.rl.exporter import _OnnxPolicyExporter


def export_motion_policy_as_onnx(
  env: ManagerBasedRlEnv,
  actor_critic: object,
  path: str,
  normalizer: object | None = None,
  filename="policy.onnx",
  verbose=False,
):
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
  policy_exporter = _OnnxSuperMimicPolicyExporter(
    env, actor_critic, normalizer, verbose
  )
  policy_exporter.export(path, filename)


class _OnnxSuperMimicPolicyExporter(_OnnxPolicyExporter):
  def __init__(
    self, env: ManagerBasedRlEnv, actor_critic, normalizer=None, verbose=False
  ):
    super().__init__(actor_critic, normalizer, verbose)
    cmd = cast(MotionCommand, env.command_manager.get_term("motion"))

    self.joint_pos = cmd.motion.joint_pos.to("cpu")
    self.joint_vel = cmd.motion.joint_vel.to("cpu")
    self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
    self.time_step_total: int = self.joint_pos.shape[0]

  def forward(self, x, time_step):  # type: ignore[invalid-method-override]
    time_step_clamped = torch.clamp(
      time_step.long().squeeze(-1), max=self.time_step_total - 1
    )
    return (
      self.actor(self.normalizer(x)),
      self.joint_pos[time_step_clamped],
      self.joint_vel[time_step_clamped],
      self.body_quat_w[time_step_clamped],
    )

  def export(self, path, filename):
    self.to("cpu")
    obs = torch.zeros(1, self.actor[0].in_features)
    time_step = torch.zeros(1, 1)
    torch.onnx.export(
      self,
      (obs, time_step),
      os.path.join(path, filename),
      export_params=True,
      opset_version=11,
      verbose=self.verbose,
      input_names=["obs", "time_step"],
      output_names=[
        "actions",
        "joint_pos",
        "joint_vel",
        "body_quat_w",
      ],
      dynamic_axes={},
      dynamo=False,
    )
