"""World model visualization viewers for SuperMimic.

Provides mixin and concrete viewer classes that overlay world model
predictions as ghost entities on top of the ground-truth simulation.
"""

import copy
import time

import numpy as np
import torch
from mjlab.utils.lab_api.math import subtract_frame_transforms
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from tensordict import TensorDict

from supermimic.utils.helpers import extract_action_scale_offset, quat_to_rot6d


def create_wm_vis_model(base_model):
  model = copy.deepcopy(base_model)
  model.geom_rgba[:, 3] = 0.8
  model.geom_rgba[:, 0:3] = [0.5, 0.5, 0.8]
  return model


class SuperMimicViewerMixin:
  def setup_supermimic(
    self,
    wm,
    num_steps_rollout,
    history,
  ):
    self.wm = wm
    self.num_steps_rollout = num_steps_rollout
    self.history = history
    self.action_scale, self.action_offset = extract_action_scale_offset(self.env)

    mj_model = self.env.unwrapped.sim.mj_model
    joint_names = [mj_model.joint(i).name for i in range(mj_model.njnt)]
    self.joint_q_adr = np.array(
      [mj_model.joint(n).qposadr[0] for n in joint_names if mj_model.joint(n).type != 0]
    )
    # Find free joint
    free_joint_id = next(
      (i for i in range(mj_model.njnt) if mj_model.joint(i).type == 0), -1
    )
    self.free_joint_q_adr = None
    if free_joint_id >= 0:
      adr = mj_model.joint(free_joint_id).qposadr[0]
      self.free_joint_q_adr = np.arange(adr, adr + 7)

    self.step_in_rollout = 0
    self.curr_pred_body_states_hist = None
    self.curr_wm_states = None
    self.last_actions = None

    # For per-body tracking: map entity-local body indexes to MuJoCo body IDs
    if self.wm.representation.uses_per_body_tracking:
      motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
      entity_body_indexes = motion_cmd.body_indexes
      mj_body_ids = motion_cmd.robot.indexing.body_ids[entity_body_indexes]
      self.mj_body_ids = mj_body_ids.cpu().numpy()

  def reset_env_supermimic(self):
    self.step_in_rollout = 0
    self.curr_pred_body_states_hist = None
    self.curr_wm_states = None
    self.last_actions = None

  def setup_wm_visualizer(
    self, wm_vis_model, *, update_every_n_frames: int | None = None
  ):
    self.wm_vis_model = wm_vis_model
    self.wm_vis_qpos = None
    self.wm_vis_mocap_pos = None
    self.wm_vis_mocap_quat = None
    self.wm_vis_xpos = None
    self.wm_vis_xquat = None

    # For per-body tracking: hide geoms belonging to non-tracked bodies (needed
    # for native viewer's mjv_addGeoms which renders all visible geoms).
    if self.wm.representation.uses_per_body_tracking:
      tracked_set = set(self.mj_body_ids.tolist())
      for geom_id in range(wm_vis_model.ngeom):
        if wm_vis_model.geom_bodyid[geom_id] not in tracked_set:
          wm_vis_model.geom_rgba[geom_id, 3] = 0

    base_env = self.env.unwrapped
    original_vis_callback = base_env.update_visualizers

    def combined_vis_callback(viz):
      original_vis_callback(viz)
      if (
        update_every_n_frames is not None and self._counter % update_every_n_frames != 0
      ):
        return
      if self.wm.representation.uses_per_body_tracking:
        if self.wm_vis_xpos is not None:
          viz.add_ghost_bodies(
            model=self.wm_vis_model,
            body_ids=self.mj_body_ids,
            body_xpos=self.wm_vis_xpos,
            body_xquat=self.wm_vis_xquat,
            label="wm_prediction",
          )
      else:
        if self.wm_vis_qpos is not None:
          viz.add_ghost_entity(
            self.wm_vis_qpos,
            model=self.wm_vis_model,
            mocap_pos=self.wm_vis_mocap_pos,
            mocap_quat=self.wm_vis_mocap_quat,
            label="wm_prediction",
          )

    base_env.update_visualizers = combined_vis_callback

  def reset_wm_visualizer(self) -> None:
    self.reset_env_supermimic()
    self.wm_vis_qpos = None
    self.wm_vis_mocap_pos = None
    self.wm_vis_mocap_quat = None
    self.wm_vis_xpos = None
    self.wm_vis_xquat = None

  def step_wm_visualizer(self, *, force: bool = False) -> bool:
    return self.step_and_predict(force=force)

  def _execute_step(self) -> bool:
    # BaseViewer uses this for both running and paused single-step.
    return self.step_wm_visualizer(force=self._is_paused)

  def _extract_root_pos_quat(self, full_state):
    """Extract root position and quaternion from full state."""
    return self.wm.representation.extract_root_pos_quat(full_state)

  def _compute_wm_goal(self, curr_goals, wm_states, target_pos, target_quat):
    goal_extra = self.wm.representation.goal_dim_split()
    base_goal = curr_goals[..., : -(goal_extra + 6)]
    root_pos, root_quat = self._extract_root_pos_quat(wm_states)
    rel_pos, rel_quat = subtract_frame_transforms(
      root_pos, root_quat, target_pos, target_quat
    )

    rel_ori = quat_to_rot6d(rel_quat)

    return torch.cat([base_goal, rel_pos, rel_ori], dim=-1)

  def _get_obs_vec(self, o):
    return o["actor"], o["full_state"], o["goal"]

  def step_and_predict(self, *, force: bool = False) -> bool:
    if self._is_paused and not force:
      return False

    try:
      t0 = time.perf_counter()
      with torch.no_grad():
        obs = self.env.get_observations()
        curr_policy_gt, curr_gt_states, curr_goals = self._get_obs_vec(obs)

        # Extract targets for wm goal computation
        motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
        target_pos = motion_cmd.body_pos_w[:, 0].clone()
        target_quat = motion_cmd.body_quat_w[:, 0].clone()

        # Initialize wm state at start of rollout (or if reset)
        if self.curr_wm_states is None or self.step_in_rollout == 0:
          self.curr_wm_states = curr_gt_states.clone()

          curr_body_states = self.wm.representation.represent(self.curr_wm_states)
          # [Batch, History, Dim]
          self.curr_pred_body_states_hist = curr_body_states.unsqueeze(1).repeat(
            1, self.history, 1
          )

        # Initialize last_action if None (start of rollout)
        if self.last_actions is None:
          self.last_actions = torch.zeros(
            (1, self.wm.num_actions), device=curr_gt_states.device
          )

        # Actor Step (using Ground Truth for policy input)
        actor_goal = self.wm.representation.extract_actor_goal(curr_goals)
        actor_input = self.wm.representation.construct_policy_input(
          curr_gt_states, actor_goal
        )

        actor_obs = TensorDict(
          {"actor": actor_input},
          batch_size=actor_input.shape[:-1],
          device=actor_input.device,
        )
        actions = self.policy(actor_obs)

        # Env Step (Ground Truth simulation)
        _, _, dones, _ = self.env.step(actions)
        self._step_count += 1
        self._stats_steps += 1

        # Predict Next State (WM)
        wm_goal = self._compute_wm_goal(
          curr_goals, self.curr_wm_states, target_pos, target_quat
        )

        # Closed loop: Compute action based on CURRENT WM STATE
        wm_body_states = self.curr_pred_body_states_hist[:, -1, :]
        wm_actor_goal = self.wm.representation.extract_actor_goal(wm_goal)
        wm_actor_input = torch.cat([wm_body_states, wm_actor_goal], dim=-1)
        wm_actor_obs = TensorDict(
          {"actor": wm_actor_input},
          batch_size=wm_actor_input.shape[:-1],
          device=wm_actor_input.device,
        )
        wm_actions = self.policy(wm_actor_obs)

        # Process action for World Model
        proc_wm_actions = wm_actions * self.action_scale + self.action_offset

        # Predict next state using history
        next_full_state, next_rep_state, _, term_logit = self.wm.transition(
          self.curr_pred_body_states_hist,
          proc_wm_actions,
          self.last_actions,
          wm_goal,
          self.curr_wm_states,
        )
        self.last_actions = proc_wm_actions.clone().detach()

        # Update history
        self.curr_pred_body_states_hist = torch.cat(
          [self.curr_pred_body_states_hist[:, 1:, :], next_rep_state.unsqueeze(1)],
          dim=1,
        )

        # Update full state (already integrated by transition)
        self.curr_wm_states = next_full_state

        pred_done = (torch.sigmoid(term_logit) > 0.5).any().item()

        # Visualization: update state for ghost rendering
        state = self.curr_wm_states[0]  # Take 0-th env

        if self.wm.representation.uses_per_body_tracking:
          B = self.wm.representation.num_bodies
          self.wm_vis_xpos = state[: B * 3].reshape(B, 3).cpu().numpy()
          self.wm_vis_xquat = state[B * 3 : B * 7].reshape(B, 4).cpu().numpy()
        else:
          nq = self.env.unwrapped.sim.mj_model.nq
          qpos = np.zeros(nq)
          root_pos, root_quat = self._extract_root_pos_quat(state)
          if self.free_joint_q_adr is not None:
            qpos[self.free_joint_q_adr[0:3]] = root_pos.cpu().numpy()
            qpos[self.free_joint_q_adr[3:7]] = root_quat.cpu().numpy()
            self.wm_vis_mocap_pos = None
            self.wm_vis_mocap_quat = None
            joint_state_start = 7
          else:
            # Fixed-base models may still need mocap pose for ghost visualization.
            self.wm_vis_mocap_pos = root_pos.cpu().numpy()
            self.wm_vis_mocap_quat = root_quat.cpu().numpy()
            joint_state_start = 0
          joint_state = state[
            joint_state_start : joint_state_start + len(self.joint_q_adr)
          ]
          qpos[self.joint_q_adr] = joint_state.cpu().numpy()
          self.wm_vis_qpos = qpos

        self.step_in_rollout += 1
        if dones.any() or self.step_in_rollout >= self.num_steps_rollout or pred_done:
          self.step_in_rollout = 0
          self.curr_wm_states = None
          self.last_actions = None
      _ = time.perf_counter() - t0
      return True
    except Exception as exc:
      self._last_error = f"{type(exc).__name__}: {exc}"
      self.pause()
      return False


class SuperMimicNativeViewer(SuperMimicViewerMixin, NativeMujocoViewer):
  def __init__(
    self,
    env,
    actor,
    wm,
    wm_vis_model,
    num_steps_rollout,
    history,
    **kwargs,
  ):
    super().__init__(env, actor, **kwargs)
    self.setup_supermimic(
      wm,
      num_steps_rollout,
      history,
    )
    self.setup_wm_visualizer(wm_vis_model)

  def reset_environment(self) -> None:
    super().reset_environment()
    self.reset_wm_visualizer()


class SuperMimicViserViewer(SuperMimicViewerMixin, ViserPlayViewer):
  def __init__(
    self,
    env,
    actor,
    wm,
    wm_vis_model,
    num_steps_rollout,
    history,
    **kwargs,
  ):
    super().__init__(env, actor, **kwargs)
    self.setup_supermimic(
      wm,
      num_steps_rollout,
      history,
    )
    self.setup_wm_visualizer(wm_vis_model, update_every_n_frames=2)

  def setup(self) -> None:
    super().setup()
    # Enable camera tracking by default
    self._scene.camera_tracking_enabled = True

  def reset_environment(self) -> None:
    super().reset_environment()
    self.reset_wm_visualizer()
