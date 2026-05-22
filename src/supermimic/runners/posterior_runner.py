"""Posterior runner: trains a high-level MLP over frozen WM + VAE.

Sim collection runs the MLP + VAE prior to step the sim; raw task goals
are stored in the replay buffer and converted to goal obs via
``build_goal_obs()`` at each WM reset for BPTT training.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Annotated

import torch
import tyro
import wandb
from tensordict import TensorDict

from supermimic.envs import TASK_ENV_REGISTRY
from supermimic.envs.getup_env import GetupEnvCfg
from supermimic.envs.vae_world_model_env import VaeWorldModelEnv
from supermimic.envs.velocity_env import VelocityEnvCfg
from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
  BpttPolicyLearner,
  BpttPolicyLearnerCfg,
)
from supermimic.runners.exporter import export_posterior_as_onnx
from supermimic.runners.supermimic_runner import BaseRunner, BaseRunnerCfg, Ratio
from supermimic.runners.vae_utils import load_vae_from_wandb

BpttCfg = Annotated[BpttPolicyLearnerCfg, tyro.conf.subcommand(name="bptt")]
ActorCriticCfg = Annotated[
  ActorCriticPolicyLearnerCfg, tyro.conf.subcommand(name="actor-critic")
]

VelocityTaskCfg = Annotated[VelocityEnvCfg, tyro.conf.subcommand(name="velocity")]
GetupTaskCfg = Annotated[GetupEnvCfg, tyro.conf.subcommand(name="getup")]


@dataclass
class PosteriorRunnerCfg(BaseRunnerCfg):
  wandb_project: str = "posterior"
  max_iterations: int = 100000
  save_interval: int = 500
  replay_ratio: int = 32
  policy: BpttCfg | ActorCriticCfg = field(default_factory=BpttPolicyLearnerCfg)
  resample_interval: int = 100
  task: VelocityTaskCfg | GetupTaskCfg = field(default_factory=VelocityEnvCfg)
  """Task env cfg; ``task.task_env`` discriminates."""
  vae_wandb_path: str = ""
  """W&B run path for the trained VAE (e.g. ``entity/project/run_id``)."""


class PosteriorRunner(BaseRunner[PosteriorRunnerCfg]):
  def _init_common(self) -> None:
    if not self.cfg.vae_wandb_path:
      raise ValueError("PosteriorRunnerCfg.vae_wandb_path is required")
    self._vae, self._vae_dims, representation = load_vae_from_wandb(
      self.cfg.vae_wandb_path, self.device
    )
    self.cfg.representation = representation
    if self.cfg.wandb_run_path is None:
      self.cfg.wandb_run_path = self.cfg.vae_wandb_path
    super()._init_common()

  @property
  def _buffer_goal_dim(self) -> int:
    return TASK_ENV_REGISTRY[self.cfg.task.task_env][0].raw_goal_dim

  def _init_policy(self) -> None:
    cfg = self.cfg
    dims = self._vae_dims

    print(
      f"[INFO] VAE frozen: obs={dims.obs_dim}, goal={dims.goal_dim}, "
      f"action={dims.action_dim}, latent={dims.latent_dim}"
    )

    self._vae.eval()
    for p in self._dynamics.parameters():
      p.requires_grad_(False)
    self._dynamics.eval()
    for p in self._representation.parameters():
      p.requires_grad_(False)
    self._representation.eval()

    motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
    mj_body_ids = motion_cmd.robot.indexing.body_ids[motion_cmd.body_indexes]
    body_masses = self.env.unwrapped.sim.model.body_mass[0, mj_body_ids].clone()
    print(
      f"[INFO] Body masses ({len(body_masses)} bodies): {body_masses.cpu().tolist()}"
    )

    env_cls, _ = TASK_ENV_REGISTRY[cfg.task.task_env]
    self.wm_env: VaeWorldModelEnv = env_cls(
      cfg=cfg.task,
      body_masses=body_masses,
      dynamics=self._dynamics,
      representation=self._representation,
      vae=self._vae,
      batch_size=cfg.policy.batch_size,
      horizon=cfg.policy.horizon,
      replay_buffer=self.replay_buffer,
      latent_dim=dims.latent_dim,
      sim_context=self._sim_context,
      device=self.device,
    )
    if torch.cuda.is_available():
      self.wm_env._wm_step = torch.compile(self.wm_env._wm_step, mode="reduce-overhead")

    self._sensor_dim = (
      self._representation.proprio_obs_dim(self._num_actions)
      if self.cfg.actor_obs == "proprio"
      else self._representation.rep_state_dim
    )
    actor_input_dim = self._sensor_dim + self.wm_env.goal_dim
    actor = self._build_mlp(
      "actor", actor_input_dim, dims.latent_dim, hidden_dims=(256, 256, 256)
    )

    if isinstance(cfg.policy, ActorCriticPolicyLearnerCfg):
      critic_input_dim = (
        self._representation.rep_state_dim
        + self.wm_env.goal_dim
        + self._sim_context.output_dim
      )
      critic = self._build_mlp("critic", critic_input_dim, 1)
      self.policy_learner = ActorCriticPolicyLearner(
        actor, critic, self.wm_env, cfg.policy, device=self.device
      )
    else:
      self.policy_learner = BpttPolicyLearner(
        actor, self.wm_env, cfg.policy, device=self.device
      )

    self._collection_step = 0
    self._collection_raw_goal = self.wm_env.sample_raw_goal(self.env.num_envs)

  def step(self) -> int:
    with torch.inference_mode():
      obs = self.env.get_observations()
      curr_states = obs["full_state"]

      self._collection_step += 1
      if self._collection_step % self.cfg.resample_interval == 0:
        self._collection_raw_goal = self.wm_env.sample_raw_goal(self.env.num_envs)

      rep_state = self._representation.represent(curr_states)
      aux_state = self._representation.extract_aux_state(curr_states)
      goal_obs = self.wm_env.build_goal_obs(self._collection_raw_goal, aux_state)
      proprio = self.cfg.actor_obs == "proprio"
      vae_obs = self._representation.compute_actor_sensor(
        rep_state,
        self._num_actions,
        self._sim_context.read(self.env.unwrapped),
        sim_actor_obs=obs["actor"].to(self.device) if proprio else None,
        noisy=proprio,
      )
      actor_input = torch.cat([vae_obs, goal_obs], dim=-1)
      actor_obs = TensorDict(
        {"actor": actor_input},
        batch_size=[curr_states.shape[0]],
        device=self.device,
      )

      latent_residual = self.policy_learner.actor(actor_obs)
      actions = self.wm_env.decode_residual(vae_obs, latent_residual)

      _, rewards, _, extras = self._step_and_store(
        actions,
        curr_states,
        self._collection_raw_goal,
        extra_fields=self._sim_context.read_raw(self.env.unwrapped),
      )
      terminated = self.env.unwrapped.termination_manager.terminated
      self.logger.process_env_step(rewards, terminated, extras)

    # Only the actor normalizer needs sim-data updates; WM heads are frozen.
    self.policy_learner.actor.train()
    self.policy_learner.actor.update_normalization(actor_obs)
    return self.env.num_envs

  def run(self) -> None:
    log_dir = self._create_log_dir("posterior")
    self._init_wandb(log_dir, name_suffix="-posterior")

    self.env.reset()

    should_train = Ratio(
      self.env.num_envs
      / self.cfg.policy.num_batches
      / self.cfg.policy.batch_size
      / self.cfg.policy.horizon
      * self.cfg.replay_ratio
    )

    num_collected = 0
    start_time = time.time()
    policy_time = 0
    loss_dict = {}
    log_count = 0

    for step_idx in range(self.cfg.max_iterations):
      num_collected += self.step()

      for _ in range(should_train(step_idx)):
        t0 = time.time()
        loss_dict.update(self.policy_learner.update())
        policy_time += time.time() - t0

      if step_idx % self.cfg.log_interval == 0:
        elapsed = time.time() - start_time
        self.logger.log(
          it=log_count,
          total_it=self.cfg.max_iterations // self.cfg.log_interval,
          collect_time=elapsed - policy_time,
          wm_learn_time=0,
          policy_learn_time=policy_time,
          loss_dict=loss_dict,
          num_collected_steps=num_collected,
        )
        log_count += 1
        start_time = time.time()
        policy_time = 0
        num_collected = 0

      if step_idx % self.cfg.save_interval == 0:
        torch.save(self.policy_learner.actor.state_dict(), log_dir / "actor.pt")
        wandb.save(str(log_dir / "actor.pt"), base_path=str(log_dir), policy="now")
        if self.cfg.actor_obs == "proprio":
          export_posterior_as_onnx(
            self.env.unwrapped,
            self.policy_learner.actor,
            self._vae,
            self.wm_env._latent_scale,
            self._sensor_dim,
            self.wm_env.goal_dim,
            str(log_dir),
            run_name=wandb.run.name if wandb.run else "local",
          )
          wandb.save(str(log_dir / "policy.onnx"), base_path=str(log_dir), policy="now")

    self._cleanup()
