"""Register SuperMimic tasks."""

from mjlab.tasks.registry import load_rl_cfg, register_mjlab_task

from .world_model import world_model_env_cfg

# Register World Model task
# We reuse the RL config from the base tracking task
BASE_TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1"

try:
  register_mjlab_task(
    task_id="SuperMimic-WorldModel-G1",
    env_cfg=world_model_env_cfg(),
    play_env_cfg=world_model_env_cfg(play=True),
    rl_cfg=load_rl_cfg(BASE_TASK_ID),
  )
except ValueError:
  # Already registered
  pass
