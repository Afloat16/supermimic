"""Play a trained posterior agent."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mjlab.tasks  # noqa: F401
import torch
import tyro
import wandb as wandb_module
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from viser import Icon

import supermimic.tasks  # noqa: F401
from supermimic.envs import TASK_ENV_REGISTRY
from supermimic.envs.vae_world_model_env import (
  DiscreteGoalSpec,
  GoalSpec,
  VaeWorldModelEnv,
)
from supermimic.representations import (
  StateRepresentation,
  create_representation,
)
from supermimic.runners.vae_utils import (
  load_vae_from_wandb,
  modify_env_cfg_for_vae_prior,
)
from supermimic.scripts.train_utils import download_artifact_cached
from supermimic.sim_context import SimContext
from supermimic.utils.helpers import unpack_obs_dims


@dataclass(frozen=True)
class PlayConfig:
  wandb_path: str = ""
  """W&B run path for the trained posterior."""
  num_envs: int | None = None
  device: str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"


class PosteriorPolicy:
  """obs → rep_state → MLP → VAE decode → joint actions.

  Goal observations are computed via ``env_class.build_goal_obs()`` — the
  same staticmethod used during training.
  """

  def __init__(
    self,
    vae: torch.nn.Module,
    representation: StateRepresentation,
    env_class: type[VaeWorldModelEnv],
    goal_specs: list[GoalSpec],
    mlp: MLPModel,
    latent_scale: float,
    device: str,
  ) -> None:
    self._vae = vae
    self._representation = representation
    self._env_class = env_class
    self._mlp = mlp
    self._latent_scale = latent_scale
    self._device = device
    self._goal = torch.tensor(
      [[s.default for s in goal_specs]], device=device, dtype=torch.float32
    )

  def set_goal_dim(self, idx: int, value: float) -> None:
    self._goal[0, idx] = value

  @torch.no_grad()
  def __call__(self, obs: TensorDict) -> torch.Tensor:
    full_state = obs["full_state"]
    aux_state = self._representation.extract_aux_state(full_state)
    num_envs = full_state.shape[0]

    raw_goal = self._goal.expand(num_envs, -1)
    goal_obs = self._env_class.build_goal_obs(raw_goal, aux_state)

    if self._representation.actor_obs_mode == "proprio":
      vae_obs = obs["actor"].to(self._device)
    else:
      vae_obs = self._representation.represent(full_state)
    actor_input = torch.cat([vae_obs, goal_obs], dim=-1)
    actor_obs = TensorDict(
      {"actor": actor_input}, batch_size=[num_envs], device=self._device
    )
    residual = self._latent_scale * torch.tanh(self._mlp(actor_obs))
    z = self._vae.encode(vae_obs) + residual
    return self._vae.decode(z, vae_obs)


class PosteriorViserViewer(ViserPlayViewer):
  """Viser viewer with auto-generated controls from goal specs."""

  def __init__(
    self,
    env: Any,
    policy: PosteriorPolicy,
    goal_specs: list[GoalSpec],
  ) -> None:
    super().__init__(env, policy)
    self._goal_specs = goal_specs

  def setup(self) -> None:
    super().setup()
    self._scene.camera_tracking_enabled = True

    policy = self.policy
    specs = self._goal_specs

    with self._server.gui.add_folder("Task Goals"):
      controls: list[Any] = []
      for i, spec in enumerate(specs):
        if isinstance(spec, DiscreteGoalSpec):
          options = tuple(f"{c:g}" for c in spec.choices)
          dropdown = self._server.gui.add_dropdown(
            spec.name, options, initial_value=f"{spec.default:g}"
          )

          @dropdown.on_update
          def _(_ev, _idx=i, _ctrl=dropdown) -> None:
            policy.set_goal_dim(_idx, float(_ctrl.value))

          controls.append(dropdown)
        else:
          slider = self._server.gui.add_slider(
            spec.name,
            min=spec.min,
            max=spec.max,
            step=spec.step,
            initial_value=spec.default,
          )

          @slider.on_update
          def _(_ev, _idx=i, _ctrl=slider) -> None:
            policy.set_goal_dim(_idx, _ctrl.value)

          controls.append(slider)

      reset_btn = self._server.gui.add_button("Reset", icon=Icon.SQUARE_X)

      @reset_btn.on_click
      def _(_) -> None:
        for j, (ctrl, sp) in enumerate(zip(controls, specs, strict=True)):
          ctrl.value = (
            f"{sp.default:g}" if isinstance(sp, DiscreteGoalSpec) else sp.default
          )
          policy.set_goal_dim(j, sp.default)


def _load_trained_mlp(
  wandb_path: str,
  input_dim: int,
  output_dim: int,
  device: str,
) -> MLPModel:
  api = wandb_module.Api()
  run = api.run(wandb_path)

  name = "actor.pt"
  run_id = wandb_path.split("/")[-1]
  download_dir = Path("logs") / "posterior_checkpoints" / run_id
  download_dir.mkdir(parents=True, exist_ok=True)
  checkpoint_path = download_dir / name

  if not checkpoint_path.exists():
    print(f"[INFO] Downloading {name} from {wandb_path}...")
    run.file(name).download(root=str(download_dir), replace=True)

  state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

  obs_td = TensorDict(
    {"actor": torch.zeros(1, input_dim, device=device)},
    batch_size=[1],
    device=device,
  )
  mlp = MLPModel(
    obs=obs_td,
    obs_groups={"actor": ["actor"]},
    obs_set="actor",
    output_dim=output_dim,
    hidden_dims=(256, 256, 256),
    obs_normalization=True,
  ).to(device)
  mlp.load_state_dict(state_dict)
  mlp.eval()
  return mlp


def run_play(task_id: str, cfg: PlayConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  if not cfg.wandb_path:
    raise ValueError("--wandb-path is required.")

  api = wandb_module.Api()
  run_config = api.run(cfg.wandb_path).config
  runner_cfg = run_config.get("runner_cfg", {})
  vae_wandb_path = runner_cfg.get("wandb_run_path", "")
  if not vae_wandb_path:
    raise ValueError(
      f"Training run {cfg.wandb_path} has no wandb_run_path in runner_cfg."
    )
  print(f"[INFO] Resolved VAE/WM source: {vae_wandb_path}")
  task_cfg = runner_cfg.get("task", {})
  task_env = task_cfg.get("task_env")
  if task_env is None:
    raise ValueError(f"Training run {cfg.wandb_path} has no runner_cfg.task.task_env.")
  latent_scale_mult = task_cfg.get("latent_scale_mult")
  if latent_scale_mult is None:
    raise ValueError(
      f"Training run {cfg.wandb_path} has no runner_cfg.task.latent_scale_mult."
    )
  actor_obs_mode = runner_cfg.get("actor_obs", "rep_state")

  env_class, env_cfg_cls = TASK_ENV_REGISTRY[task_env]
  specs = env_cfg_cls().goal_specs()

  vae, dims, rep_name = load_vae_from_wandb(vae_wandb_path, device)
  latent_dim = dims.latent_dim
  latent_scale = latent_scale_mult * vae.fixed_std

  env_cfg = load_env_cfg(task_id, play=True)

  arts = [a for a in api.run(vae_wandb_path).used_artifacts() if a.type == "motions"]
  if not arts:
    raise RuntimeError(
      f"No motion artifact in wandb run {vae_wandb_path}; VAE training must link one."
    )
  motion_files = [str(Path(download_artifact_cached(a)) / "motion.npz") for a in arts]
  env_cfg.commands["motion"].motion_file = (
    motion_files[0] if len(motion_files) == 1 else motion_files
  )
  env_cfg.commands["motion"].debug_vis = False

  modify_env_cfg_for_vae_prior(env_cfg)
  env_cfg.observations["actor"].enable_corruption = True

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env_wrapped = RslRlVecEnvWrapper(env)

  obs, _ = env_wrapped.reset()
  num_state, num_goals = unpack_obs_dims(obs)
  num_actions = env_wrapped.num_actions
  reward_names = getattr(env.unwrapped.reward_manager, "active_terms", ["reward"])
  default_joint_pos = env.unwrapped.scene["robot"].data.default_joint_pos[0].to(device)
  has_state_estimation = (
    "base_lin_vel" in env.unwrapped.observation_manager.active_terms["actor"]
  )
  sim_context = SimContext.from_env(env.unwrapped, device)
  representation = create_representation(
    rep_name,
    num_state,
    num_actions,
    num_goals,
    len(reward_names),
    sim_context.output_dim,
    env.unwrapped.step_dt,
    default_joint_pos,
    has_state_estimation,
    actor_obs_mode,
  ).to(device)
  representation.eval()

  obs_dim = (
    representation.proprio_obs_dim(num_actions)
    if actor_obs_mode == "proprio"
    else representation.rep_state_dim
  )
  actor_input_dim = obs_dim + env_class.goal_dim
  mlp = _load_trained_mlp(cfg.wandb_path, actor_input_dim, latent_dim, device)
  policy = PosteriorPolicy(
    vae, representation, env_class, specs, mlp, latent_scale, device
  )

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env_wrapped, policy).run()
  else:
    PosteriorViserViewer(env_wrapped, policy, specs).run()

  env.close()


def main() -> None:
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
