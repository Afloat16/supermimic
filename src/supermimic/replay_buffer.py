from dataclasses import dataclass

import torch
from tensordict import TensorDict


@dataclass(frozen=True)
class FieldSpec:
  dim: int = 0  # 0 = scalar (no trailing axis); >0 = vector with trailing dim
  dtype: torch.dtype = torch.float32

  def shape(self, num_envs: int, buffer_size: int) -> tuple[int, ...]:
    return (num_envs, buffer_size, self.dim) if self.dim else (num_envs, buffer_size)


class ReplayBuffer:
  """Per-env circular buffer over a TensorDict, with env-stratified window sampling."""

  def __init__(
    self,
    schema: dict[str, FieldSpec],
    buffer_size: int,
    num_envs: int,
    device: str = "cpu",
  ):
    if not {"terminated", "truncated"} <= schema.keys():
      raise ValueError("schema needs 'terminated' and 'truncated'")
    self.num_transitions = 0
    self.buffer_size = buffer_size
    self.num_envs = num_envs
    self.device = device
    self._step = 0
    self._storage = TensorDict(
      {
        name: torch.zeros(
          spec.shape(num_envs, buffer_size), dtype=spec.dtype, device=device
        )
        for name, spec in schema.items()
      },
      batch_size=[num_envs, buffer_size],
    )

  def insert(self, td: TensorDict) -> None:
    if td.batch_size != torch.Size([self.num_envs]):
      raise ValueError(
        f"insert expects batch_size=[{self.num_envs}], got {list(td.batch_size)}"
      )
    if set(td.keys()) != set(self._storage.keys()):
      raise ValueError("schema mismatch")
    self._storage[:, self._step] = td
    self._step = (self._step + 1) % self.buffer_size
    self.num_transitions = min(self.buffer_size, self.num_transitions + 1)

  def sample(self, sequence_length: int, batch_size: int) -> TensorDict:
    L, B, E, T = sequence_length, batch_size, self.num_envs, self.buffer_size
    dev, n = self.device, self.num_transitions
    if L > T:
      raise ValueError(f"sequence_length {L} exceeds buffer_size {T}")
    # Index by age (0 = oldest filled, n-1 = newest); slot = (base + age) % T.
    # In age-space windows are linear, so wrap and warm-up collapse into one predicate.
    base = self._step if n == T else 0
    ages = torch.arange(T, device=dev)
    ended = (self._storage["terminated"] | self._storage["truncated"])[
      :, (base + ages) % T
    ].long()
    # Interior [a, a+L-2] is done-free iff c[a+L-1] == c[a] (c is cumsum with leading 0).
    c = torch.cat(
      [ended.new_zeros(E, 1), ended.cumsum(1), ended.new_zeros(E, L - 1)], dim=1
    )
    valid = (c[:, L - 1 : L - 1 + T] == c[:, :T]) & (ages + L - 1 < n)
    csum = valid.cumsum(dim=1, dtype=torch.long)
    total = csum[:, -1]
    if not (total > 0).any():
      raise RuntimeError(f"no valid sequences L={L} n={n}")
    env_idx = torch.multinomial((total > 0).float(), B, replacement=True)
    u = (torch.rand(B, device=dev) * total[env_idx].float()).long()
    age_start = torch.searchsorted(csum[env_idx], u[:, None], right=True).squeeze(-1)
    storage_idx = (base + age_start[:, None] + torch.arange(L, device=dev)) % T
    return self._storage[env_idx[:, None], storage_idx]
