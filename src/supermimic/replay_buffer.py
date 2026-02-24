import torch


class ReplayBuffer:
  """Fixed-size circular buffer storing a list of per-field tensors.

  Each field is a separate tensor of shape (num_envs, buffer_size, dim_i).
  Fields with dim=0 are stored as ``None``.

  The last field is assumed to hold done/reset flags and is used to
  generate valid (non-episode-crossing) sequence indices.
  """

  def __init__(self, dim: list[int], buffer_size: int, device: str):
    """Initialize a ReplayBuffer object.

    Args:
      dim: List of feature dimensions, one per field.
      buffer_size: Maximum number of timesteps stored per environment.
      device: Torch device string.
    """
    self.dim = dim
    self.buffer_size = buffer_size
    self.device = device

    self.replay_buf: list[torch.Tensor | None] | None = None
    self.num_envs: int | None = None
    self.step = 0
    self.num_transitions = 0

  def _initialize_buffer(self, num_envs: int):
    """Initialize the replay buffer dynamically."""
    self.num_envs = num_envs
    self.replay_buf = [
      torch.zeros(num_envs, self.buffer_size, d, device=self.device) if d > 0 else None
      for d in self.dim
    ]

  def initialize(self, num_envs: int):
    """Pre-initialize buffer outside inference mode for torch.compile compatibility."""
    if self.replay_buf is None:
      self._initialize_buffer(num_envs)

  def insert_transition(
    self, states, actions, goals, reward_vec, motion_time, dones_float
  ):
    """Insert a single-timestep transition with standard unsqueeze formatting."""
    self.insert(
      [
        states.unsqueeze(1),
        actions.unsqueeze(1),
        goals.unsqueeze(1),
        reward_vec.unsqueeze(1).float(),
        motion_time.unsqueeze(1).unsqueeze(2).float(),
        dones_float.unsqueeze(1).unsqueeze(2),
      ]
    )

  def insert(self, input_buf: list[torch.Tensor]):
    """Add new states to memory in a circular manner."""
    if self.replay_buf is None:
      self._initialize_buffer(input_buf[0].shape[0])

    def _insert_into_buffer(r_buf, i_buf):
      num_inputs = i_buf.shape[1]
      end_idx = self.step + num_inputs
      if end_idx > self.buffer_size:
        r_buf[:, self.step : self.buffer_size] = i_buf[
          :, : self.buffer_size - self.step
        ]
        r_buf[:, : end_idx - self.buffer_size] = i_buf[
          :, self.buffer_size - self.step :
        ]
      else:
        r_buf[:, self.step : end_idx] = i_buf
      return num_inputs

    num_inputs = 0
    for r_buf, i_buf in zip(self.replay_buf, input_buf, strict=True):
      if r_buf is not None:
        num_inputs = _insert_into_buffer(r_buf, i_buf)

    self.num_transitions = min(self.buffer_size, self.num_transitions + num_inputs)
    self.step = (self.step + num_inputs) % self.buffer_size

  def mini_batch_generator(self, sequence_length, num_mini_batch, mini_batch_size):
    """Yield mini-batches for training."""
    assert self.replay_buf is not None, "Replay buffer is not initialized."

    def _pad_and_get_replay_buf():
      padding_size = sequence_length - self.num_transitions
      return [
        torch.cat(
          [
            torch.zeros(buf.shape[0], padding_size, buf.shape[-1], device=self.device),
            buf[:, : self.num_transitions],
          ],
          dim=1,
        )
        if buf is not None
        else None
        for buf in self.replay_buf
      ]

    replay_buf = (
      _pad_and_get_replay_buf()
      if self.num_transitions < sequence_length
      else self.replay_buf
    )
    reset_data = replay_buf[-1]

    valid_indices = None
    if reset_data is not None:
      valid_indices = self._generate_valid_indices(reset_data, sequence_length)

    for _ in range(num_mini_batch):
      yield self._generate_batch(
        replay_buf, valid_indices, sequence_length, mini_batch_size
      )

  def _generate_valid_indices(self, reset_data, sequence_length):
    """Generate valid start indices for sequences."""
    reset_flags = reset_data[:, : max(self.num_transitions, sequence_length)].to(
      torch.bool
    )
    valid_mask = ~reset_flags.unfold(1, sequence_length, 1).squeeze(-2)[:, :, :-1].any(
      dim=2
    )
    env_indices, start_indices = torch.where(valid_mask)

    # Mask out indices that would cross the circular buffer write pointer
    if self.num_transitions >= self.buffer_size:
      # The boundary is between self.step-1 and self.step
      # Invalid starts are s such that s <= step-1 AND s+len > step
      # s in [step - len + 1, step - 1]
      invalid_start = self.step - sequence_length + 1
      invalid_end = self.step  # exclusive

      # Clamping to valid bounds
      r_start = max(0, invalid_start)
      r_end = min(valid_mask.shape[1], invalid_end)

      if r_start < r_end:
        # We need to re-compute indices after masking
        valid_mask[:, r_start:r_end] = False
        env_indices, start_indices = torch.where(valid_mask)

    return env_indices, start_indices

  def _generate_batch(
    self, replay_buf, valid_indices, sequence_length, mini_batch_size
  ):
    """Generate a mini-batch by sampling valid sequences."""
    if valid_indices is None:
      if self.num_transitions < self.buffer_size:
        max_start_idx = max(self.num_transitions - sequence_length, 0) + 1
        sampled_starts = torch.randint(
          0, max_start_idx, size=(mini_batch_size,), device=self.device
        )
      else:
        # Full buffer: Two valid segments [0, step-len] and [step, size-len]
        end1 = self.step - sequence_length + 1
        start2 = self.step
        end2 = self.buffer_size - sequence_length + 1

        valid_starts = []
        if end1 > 0:
          valid_starts.append(torch.arange(0, end1, device=self.device))
        if end2 > start2:
          valid_starts.append(torch.arange(start2, end2, device=self.device))

        if not valid_starts:
          raise RuntimeError(
            f"No valid sequences of length {sequence_length} in buffer of size {self.buffer_size} with step {self.step}"
          )

        all_valid = torch.cat(valid_starts)
        sampled_starts = all_valid[
          torch.randint(0, len(all_valid), size=(mini_batch_size,), device=self.device)
        ]

      sampled_envs = torch.randint(
        0, self.num_envs, size=(mini_batch_size,), device=self.device
      )
    else:
      env_indices, start_indices = valid_indices
      sampled_idxs = torch.randint(
        0, len(env_indices), (mini_batch_size,), device=self.device
      )
      sampled_envs = env_indices[sampled_idxs]
      sampled_starts = start_indices[sampled_idxs]

    offsets = torch.arange(sequence_length, device=self.device)
    return [
      buf[sampled_envs[:, None], sampled_starts[:, None] + offsets]
      if buf is not None
      else None
      for buf in replay_buf
    ]
