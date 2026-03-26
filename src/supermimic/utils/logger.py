from __future__ import annotations

import statistics
import time
from collections import deque

import torch


class Logger:
  """Logger to save the learning metrics to different logging services."""

  def __init__(
    self,
    cfg: dict,
    num_envs: int,
    device: str,
    disable_logs: bool = False,
  ) -> None:
    self.log_dir = None
    self.cfg = cfg
    self.num_envs = num_envs
    self.device = device
    self.disable_logs = disable_logs

    self.tot_timesteps = 0
    self.tot_time = 0

    # Create buffers
    self.ep_extras = []
    self.rewbuffer = deque(maxlen=100)
    self.lenbuffer = deque(maxlen=100)
    self.cur_reward_sum = torch.zeros(
      self.num_envs, dtype=torch.float, device=self.device
    )
    self.cur_episode_length = torch.zeros(
      self.num_envs, dtype=torch.float, device=self.device
    )

    self.writer = None

  def process_env_step(
    self,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict,
  ) -> None:
    """Add metrics from the environment step to the buffers."""
    if self.log_dir is not None:
      if "episode" in extras:
        self.ep_extras.append(extras["episode"])
      elif "log" in extras:
        self.ep_extras.append(extras["log"])

      # Update rewards and episode length
      self.cur_reward_sum += rewards
      self.cur_episode_length += 1

      # Clear data for completed episodes
      new_ids = (dones > 0).nonzero(as_tuple=False)
      if len(new_ids) > 0:
        self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
        self.lenbuffer.extend(
          self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
        )
        self.cur_reward_sum[new_ids] = 0
        self.cur_episode_length[new_ids] = 0

  def _add_scalar(self, tag: str, value: float, step: int) -> None:
    if self.writer:
      self.writer.log({tag: value}, step=step)

  def _log_extras(self, it: int, pad: int) -> str:
    """Aggregate episode extras, log to W&B, and return formatted string."""
    extras_string = ""
    if not self.ep_extras:
      return extras_string

    first_extra = self.ep_extras[0]
    for key in first_extra:
      infotensor = torch.tensor([], device=self.device)
      for ep_info in self.ep_extras:
        if key not in ep_info:
          continue
        val = ep_info[key]
        if not isinstance(val, torch.Tensor):
          val = torch.tensor([val], device=self.device)
        else:
          val = val.to(self.device).reshape(-1)

        if val.ndim == 0:
          val = val.unsqueeze(0)
        infotensor = torch.cat((infotensor, val))

      if len(infotensor) > 0:
        value = torch.mean(infotensor).item()
        if "/" in key:
          self._add_scalar(key, value, it)
          extras_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
        else:
          self._add_scalar("Episode/" + key, value, it)
          extras_string += f"""{f"Episode {key}:":>{pad}} {value:.4f}\n"""

    return extras_string

  def _log_scalars(
    self,
    it: int,
    fps: int,
    collect_time: float,
    wm_learn_time: float,
    policy_learn_time: float,
    loss_dict: dict,
  ) -> None:
    """Log loss, performance, and training scalars to W&B."""
    for key, value in loss_dict.items():
      self._add_scalar(key, value, it)

    self._add_scalar("Perf/total_fps", fps, it)
    self._add_scalar("Perf/collection_time", collect_time, it)
    self._add_scalar("Perf/wm_learning_time", wm_learn_time, it)
    self._add_scalar("Perf/policy_learning_time", policy_learn_time, it)

    if len(self.rewbuffer) > 0:
      self._add_scalar("Train/mean_reward", statistics.mean(self.rewbuffer), it)
      self._add_scalar("Train/mean_episode_length", statistics.mean(self.lenbuffer), it)

  def _build_console_log(
    self,
    it: int,
    total_it: int,
    fps: int,
    collect_time: float,
    wm_learn_time: float,
    policy_learn_time: float,
    loss_dict: dict,
    extras_string: str,
    iteration_time: float,
    width: int,
    pad: int,
  ) -> str:
    """Build the formatted console log string."""
    log_string = f"""{"#" * width}\n"""
    log_string += (
      f"""\033[1m{f" Learning iteration {it}/{total_it} ".center(width)}\033[0m \n\n"""
    )

    run_name = self.cfg.get("run_name", "SuperMimic")
    log_string += f"""{"Run name:":>{pad}} {run_name}\n"""

    log_string += (
      f"""{"Total steps:":>{pad}} {self.tot_timesteps} \n"""
      f"""{"Steps per second:":>{pad}} {fps:.0f} \n"""
      f"""{"Collection time:":>{pad}} {collect_time:.3f}s \n"""
      f"""{"WM Learning time:":>{pad}} {wm_learn_time:.3f}s \n"""
      f"""{"Policy Learning time:":>{pad}} {policy_learn_time:.3f}s \n"""
    )

    for key, value in loss_dict.items():
      log_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""

    if len(self.rewbuffer) > 0:
      log_string += (
        f"""{"Mean reward:":>{pad}} {statistics.mean(self.rewbuffer):.2f}\n"""
      )
      log_string += (
        f"""{"Mean episode length:":>{pad}} {statistics.mean(self.lenbuffer):.2f}\n"""
      )

    log_string += extras_string

    done_it = it + 1
    remaining_it = total_it - done_it
    eta = self.tot_time / done_it * remaining_it if done_it > 0 else 0.0
    log_string += (
      f"""{"-" * width}\n"""
      f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
      f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
      f"""{"ETA:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(eta))}\n"""
    )

    return log_string

  def log(
    self,
    it: int,
    total_it: int,
    collect_time: float,
    wm_learn_time: float,
    policy_learn_time: float,
    loss_dict: dict,
    num_collected_steps: int,
    width: int = 80,
    pad: int = 40,
  ) -> None:
    if self.log_dir is not None and not self.disable_logs:
      iteration_time = collect_time + wm_learn_time + policy_learn_time
      self.tot_timesteps += num_collected_steps
      self.tot_time += iteration_time
      fps = int(num_collected_steps / (iteration_time + 1e-6))

      extras_string = self._log_extras(it, pad)
      self._log_scalars(
        it, fps, collect_time, wm_learn_time, policy_learn_time, loss_dict
      )
      print(
        self._build_console_log(
          it,
          total_it,
          fps,
          collect_time,
          wm_learn_time,
          policy_learn_time,
          loss_dict,
          extras_string,
          iteration_time,
          width,
          pad,
        )
      )
      self.ep_extras.clear()
