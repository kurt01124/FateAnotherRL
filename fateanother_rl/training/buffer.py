"""LSTM-aware rollout buffer with sequence chunk iteration.

Stores per-agent trajectories and provides time-ordered sequence
chunks for PPO training with BPTT.

Supports two data sources:
  1. Online: store() transitions one at a time (from TCP env)
  2. Offline: from_tensor_data() to load batched .pt rollout files (from C++)
     Uses vectorized tensor operations — no per-transition Python loop.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """Single timestep data for one agent (online mode only)."""
    obs: dict           # numpy arrays from state_parser
    masks: dict         # head_name -> bool array
    action: dict        # head_name -> value (int or float tensor)
    log_prob: float
    value: float
    reward: float
    done: bool
    hx: tuple           # (h, c) LSTM hidden state at this step

    # Filled after GAE computation
    advantage: float = 0.0
    ret: float = 0.0


@dataclass
class SequenceChunk:
    """Batched sequence chunk for PPO training."""
    obs: dict[str, torch.Tensor]          # (B, T, ...)
    masks: dict[str, torch.Tensor]        # (B, T, head_size)
    actions: dict[str, torch.Tensor]      # (B, T, ...)
    old_log_probs: torch.Tensor           # (B, T)
    values: torch.Tensor                  # (B, T)
    rewards: torch.Tensor                 # (B, T)
    advantages: torch.Tensor              # (B, T)
    returns: torch.Tensor                 # (B, T)
    hx_init: tuple[torch.Tensor, torch.Tensor]  # (h, c) each (1, B, H)

    def to(self, device: torch.device) -> SequenceChunk:
        """Move all tensors to device."""
        def _to(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            elif isinstance(x, dict):
                return {k: _to(v) for k, v in x.items()}
            elif isinstance(x, tuple):
                return tuple(_to(v) for v in x)
            return x

        return SequenceChunk(
            obs=_to(self.obs),
            masks=_to(self.masks),
            actions=_to(self.actions),
            old_log_probs=_to(self.old_log_probs),
            values=_to(self.values),
            rewards=_to(self.rewards),
            advantages=_to(self.advantages),
            returns=_to(self.returns),
            hx_init=_to(self.hx_init),
        )


class RolloutBuffer:
    """LSTM-aware rollout buffer.

    Two modes:
      - Online (Transition-based): store() individual transitions
      - Offline (Tensor-based): from_tensor_data() loads C++ rollout tensors
        and uses vectorized operations for GAE and sequence iteration.
    """

    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam
        # Online mode
        self.trajectories: dict[tuple[int, int], list[Transition]] = defaultdict(list)
        # Offline tensor mode
        self._td: dict | None = None
        self._gae_computed = False

    @classmethod
    def from_tensor_data(cls, data: dict, gamma: float = 0.99, lam: float = 0.95) -> RolloutBuffer:
        """Create buffer from C++ FATE binary rollout tensors.

        Stores raw tensors directly — no per-transition Python loop.
        Vectorized GAE and sequence iteration via tensor slicing.

        Expected keys from C++ RolloutWriter:
            self_vecs (T,12,77), ally_vecs (T,12,5,37), enemy_vecs (T,12,6,41),
            global_vecs (T,12,6), grids (T,12,3,25,48),
            log_probs (T,12), values (T,12), rewards (T,12), dones (T,12),
            hx_h (T,12,1,256), hx_c (T,12,1,256),
            mask_* (T,12,N), act_* (T,12,...) with possible extra (1,) dim
        """
        buf = cls(gamma=gamma, lam=lam)

        # Extract masks: mask_skill → "skill", etc.
        masks = {}
        for k, v in data.items():
            if k.startswith("mask_"):
                masks[k[5:]] = v.float()  # bool → float for model compatibility

        # Extract actions: act_skill → "skill", etc.
        # Squeeze extra (1,) dim from C++ stacking:
        #   (T,12,1) → (T,12), (T,12,1,2) → (T,12,2)
        actions = {}
        for k, v in data.items():
            if k.startswith("act_"):
                head_name = k[4:]
                while v.dim() > 2 and v.shape[2] == 1:
                    v = v.squeeze(2)
                actions[head_name] = v

        T = data["self_vecs"].shape[0]
        A = data["self_vecs"].shape[1]

        buf._td = {
            "self_vec": data["self_vecs"].float(),       # (T, A, 77)
            "ally_vec": data["ally_vecs"].float(),        # (T, A, 5, 37)
            "enemy_vec": data["enemy_vecs"].float(),      # (T, A, 6, 41)
            "global_vec": data["global_vecs"].float(),    # (T, A, 6)
            "grid": data["grids"].float(),                # (T, A, 3, 25, 48)
            "log_probs": data["log_probs"].float(),       # (T, A)
            "values": data["values"].float(),             # (T, A)
            "rewards": data["rewards"].float(),           # (T, A)
            "dones": data["dones"].float(),               # (T, A) 0.0 or 1.0
            "hx_h": data["hx_h"].float(),                 # (T, A, 1, 256)
            "hx_c": data["hx_c"].float(),                 # (T, A, 1, 256)
            "masks": masks,
            "actions": actions,
            # Filled by compute_gae:
            "advantages": None,
            "returns": None,
        }

        logger.info(
            "Buffer loaded: T=%d, agents=%d, transitions=%d, actions=%s",
            T, A, T * A,
            {k: list(v.shape) for k, v in actions.items()},
        )
        return buf

    def store(self, env_id: int, agent_id: int, obs: dict, masks: dict,
              action: dict, log_prob: float, value: float, reward: float,
              done: bool, hx: tuple):
        """Store a single transition (online mode)."""
        key = (env_id, agent_id)
        self.trajectories[key].append(Transition(
            obs=obs, masks=masks, action=action,
            log_prob=log_prob, value=value, reward=reward,
            done=done, hx=hx,
        ))
        self._gae_computed = False

    def store_terminal(self, env_id: int, terminal_rewards: np.ndarray):
        """Add terminal rewards to last transition of each agent and mark done."""
        for agent_id in range(12):
            key = (env_id, agent_id)
            traj = self.trajectories.get(key, [])
            if traj:
                traj[-1].reward += terminal_rewards[agent_id]
                traj[-1].done = True
        self._gae_computed = False

    def total_transitions(self) -> int:
        if self._td is not None:
            return self._td["values"].shape[0] * self._td["values"].shape[1]
        return sum(len(t) for t in self.trajectories.values())

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------
    def compute_gae(self, last_values=None):
        """Compute GAE advantages and returns."""
        if self._gae_computed:
            return

        if self._td is not None:
            self._compute_gae_tensor(last_values)
        else:
            self._compute_gae_online(last_values)

        self._gae_computed = True

    def _compute_gae_tensor(self, bootstrap_values=None):
        """Vectorized GAE for offline tensor data. Processes all agents at once."""
        d = self._td
        values = d["values"]     # (T, A)
        rewards = d["rewards"]   # (T, A)
        dones = d["dones"]       # (T, A) float, 1.0=done

        T, A = values.shape
        not_done = 1.0 - dones   # (T, A) mask: 1 if alive, 0 if done

        # Bootstrap value for last timestep (0 if done)
        if bootstrap_values is not None:
            if isinstance(bootstrap_values, torch.Tensor):
                boot = bootstrap_values.float()
            else:
                boot = torch.zeros(A)
        else:
            boot = torch.zeros(A)

        advantages = torch.zeros_like(values)
        last_gae = torch.zeros(A)

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = boot * not_done[t]
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * not_done[t] - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lam * not_done[t] * last_gae

        d["advantages"] = advantages
        d["returns"] = advantages + values
        logger.info("GAE computed: T=%d, agents=%d", T, A)

    def _compute_gae_online(self, last_values=None):
        """Per-trajectory GAE for online mode."""
        if last_values is None:
            last_values = {}

        for key, traj in self.trajectories.items():
            T = len(traj)
            if T == 0:
                continue

            advantages = np.zeros(T, dtype=np.float32)
            last_gae = 0.0
            last_val = last_values.get(key, 0.0)

            for t in reversed(range(T)):
                if t == T - 1:
                    next_val = last_val if not traj[t].done else 0.0
                else:
                    next_val = traj[t + 1].value

                mask = 0.0 if traj[t].done else 1.0
                delta = traj[t].reward + self.gamma * next_val * mask - traj[t].value
                advantages[t] = last_gae = delta + self.gamma * self.lam * mask * last_gae

            returns = advantages + np.array([t.value for t in traj])

            for t in range(T):
                traj[t].advantage = float(advantages[t])
                traj[t].ret = float(returns[t])

    # ------------------------------------------------------------------
    # Sequence iteration
    # ------------------------------------------------------------------
    def iterate_sequences(self, seq_len: int, batch_size: int) -> Iterator[SequenceChunk]:
        if not self._gae_computed:
            self.compute_gae()

        if self._td is not None:
            yield from self._iterate_tensor(seq_len, batch_size)
        else:
            yield from self._iterate_online(seq_len, batch_size)

    def _iterate_tensor(self, seq_len: int, batch_size: int) -> Iterator[SequenceChunk]:
        """Fast tensor-based sequence iteration for offline data.

        Splits each agent's trajectory into seq_len chunks,
        shuffles, and yields batched SequenceChunks via tensor slicing.
        """
        d = self._td
        T, A = d["values"].shape

        # Build chunk indices: (agent_idx, start_t, end_t)
        indices = []
        for a in range(A):
            for start in range(0, T, seq_len):
                end = start + seq_len
                if end <= T:
                    indices.append((a, start, end))

        if not indices:
            logger.warning("No valid sequence chunks (T=%d < seq_len=%d)", T, seq_len)
            return

        random.shuffle(indices)
        logger.info("Sequence chunks: %d (seq_len=%d, batch_size=%d)", len(indices), seq_len, batch_size)

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            yield self._build_tensor_batch(batch_idx, seq_len)

    def _build_tensor_batch(self, indices: list[tuple[int, int, int]], seq_len: int) -> SequenceChunk:
        """Build a SequenceChunk from list of (agent_idx, start, end).

        All tensor slicing, no Transition objects.
        Returns shapes: obs (B,T,...), masks (B,T,N), actions (B,T,...), etc.
        """
        d = self._td
        B = len(indices)

        # Observations: (B, T, ...)
        obs = {
            "self_vec":  torch.stack([d["self_vec"][s:e, a]  for a, s, e in indices]),
            "ally_vec":  torch.stack([d["ally_vec"][s:e, a]  for a, s, e in indices]),
            "enemy_vec": torch.stack([d["enemy_vec"][s:e, a] for a, s, e in indices]),
            "global_vec":torch.stack([d["global_vec"][s:e, a]for a, s, e in indices]),
            "grid":      torch.stack([d["grid"][s:e, a]      for a, s, e in indices]),
        }

        # Masks: (B, T, N)
        masks = {
            k: torch.stack([v[s:e, a] for a, s, e in indices])
            for k, v in d["masks"].items()
        }

        # Actions: (B, T) for discrete, (B, T, 2) for continuous
        actions = {
            k: torch.stack([v[s:e, a] for a, s, e in indices])
            for k, v in d["actions"].items()
        }

        # Scalars: (B, T)
        old_log_probs = torch.stack([d["log_probs"][s:e, a]   for a, s, e in indices])
        values        = torch.stack([d["values"][s:e, a]      for a, s, e in indices])
        rewards       = torch.stack([d["rewards"][s:e, a]     for a, s, e in indices])
        advantages    = torch.stack([d["advantages"][s:e, a]  for a, s, e in indices])
        returns       = torch.stack([d["returns"][s:e, a]     for a, s, e in indices])

        # LSTM hidden at chunk start: hx_h is (T, A, 1, 256)
        # d["hx_h"][s, a] → (1, 256). Stack B of them → (B, 1, 256).
        # Permute to (1, B, 256) for LSTM format.
        hx_h = torch.stack([d["hx_h"][s, a] for a, s, e in indices])  # (B, 1, 256)
        hx_c = torch.stack([d["hx_c"][s, a] for a, s, e in indices])  # (B, 1, 256)
        hx_init = (hx_h.permute(1, 0, 2).contiguous(),   # (1, B, 256)
                   hx_c.permute(1, 0, 2).contiguous())

        return SequenceChunk(
            obs=obs, masks=masks, actions=actions,
            old_log_probs=old_log_probs, values=values,
            rewards=rewards, advantages=advantages, returns=returns,
            hx_init=hx_init,
        )

    # ------------------------------------------------------------------
    # Online mode (Transition-based, unchanged)
    # ------------------------------------------------------------------
    def _iterate_online(self, seq_len: int, batch_size: int) -> Iterator[SequenceChunk]:
        chunks = []
        for key, traj in self.trajectories.items():
            T = len(traj)
            for start in range(0, T, seq_len):
                end = min(start + seq_len, T)
                if end - start < seq_len:
                    continue
                chunk_traj = traj[start:end]
                chunks.append(self._build_chunk(chunk_traj))

        if not chunks:
            return

        random.shuffle(chunks)

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            if len(batch_chunks) > 0:
                yield self._collate_chunks(batch_chunks)

    def _build_chunk(self, traj: list[Transition]) -> SequenceChunk:
        """Build a single SequenceChunk from a trajectory slice."""
        obs_keys = list(traj[0].obs.keys())
        obs = {}
        for k in obs_keys:
            obs[k] = torch.from_numpy(np.stack([t.obs[k] for t in traj]))

        mask_keys = list(traj[0].masks.keys())
        masks = {}
        for k in mask_keys:
            masks[k] = torch.from_numpy(np.stack([t.masks[k] for t in traj]))

        action_keys = list(traj[0].action.keys())
        actions = {}
        for k in action_keys:
            vals = [t.action[k] for t in traj]
            if isinstance(vals[0], (int, float)):
                actions[k] = torch.tensor(vals)
            elif isinstance(vals[0], np.ndarray):
                actions[k] = torch.from_numpy(np.stack(vals))
            elif isinstance(vals[0], torch.Tensor):
                actions[k] = torch.stack(vals)
            else:
                actions[k] = torch.tensor(vals)

        old_log_probs = torch.tensor([t.log_prob for t in traj], dtype=torch.float32)
        values = torch.tensor([t.value for t in traj], dtype=torch.float32)
        rewards = torch.tensor([t.reward for t in traj], dtype=torch.float32)
        advantages = torch.tensor([t.advantage for t in traj], dtype=torch.float32)
        returns = torch.tensor([t.ret for t in traj], dtype=torch.float32)

        hx_init = (traj[0].hx[0].detach(), traj[0].hx[1].detach())

        return SequenceChunk(
            obs=obs, masks=masks, actions=actions,
            old_log_probs=old_log_probs, values=values,
            rewards=rewards, advantages=advantages, returns=returns,
            hx_init=hx_init,
        )

    def _collate_chunks(self, chunks: list[SequenceChunk]) -> SequenceChunk:
        """Stack multiple chunks into a batch."""
        obs = {k: torch.stack([c.obs[k] for c in chunks]) for k in chunks[0].obs}
        masks = {k: torch.stack([c.masks[k] for c in chunks]) for k in chunks[0].masks}
        actions = {k: torch.stack([c.actions[k] for c in chunks]) for k in chunks[0].actions}

        return SequenceChunk(
            obs=obs, masks=masks, actions=actions,
            old_log_probs=torch.stack([c.old_log_probs for c in chunks]),
            values=torch.stack([c.values for c in chunks]),
            rewards=torch.stack([c.rewards for c in chunks]),
            advantages=torch.stack([c.advantages for c in chunks]),
            returns=torch.stack([c.returns for c in chunks]),
            hx_init=(
                torch.cat([c.hx_init[0] for c in chunks], dim=1),
                torch.cat([c.hx_init[1] for c in chunks], dim=1),
            ),
        )

    def clear(self):
        """Clear all stored data."""
        self.trajectories.clear()
        self._td = None
        self._gae_computed = False


class TensorRolloutBuffer:
    """Vectorized rollout buffer that operates on (T, num_agents, ...) tensors.

    Instead of creating individual Transition objects (O(T*12) Python objects),
    this keeps data as contiguous tensors and uses direct slicing for GAE and
    sequence iteration. ~100x faster than RolloutBuffer.from_tensor_data().
    """

    def __init__(self, data: dict, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam

        # Core observation tensors (T, 12, ...)
        self.obs = {
            "self_vec":   data["self_vecs"].float(),
            "ally_vec":   data["ally_vecs"].float(),
            "enemy_vec":  data["enemy_vecs"].float(),
            "global_vec": data["global_vecs"].float(),
            "grid":       data["grids"].float(),
        }

        # Scalar fields (T, 12)
        self.log_probs = data["log_probs"].float()
        self.values = data["values"].float()
        self.rewards = data["rewards"].float()
        self.dones = data["dones"].bool() if data["dones"].dtype != torch.bool else data["dones"]

        # LSTM hidden states (T, 12, 1, 256)
        self.hx_h = data["hx_h"].float()
        self.hx_c = data["hx_c"].float()

        # Masks: {name: (T, 12, N)} — strip "mask_" prefix
        self.masks = {}
        for k, v in data.items():
            if k.startswith("mask_"):
                self.masks[k[5:]] = v.bool() if v.dtype != torch.bool else v

        # Actions: {name: (T, 12) or (T, 12, 2)} — strip "act_" prefix
        self.actions = {}
        for k, v in data.items():
            if k.startswith("act_"):
                self.actions[k[4:]] = v

        self.T = self.log_probs.shape[0]
        self.num_agents = self.log_probs.shape[1]

        # GAE results (filled by compute_gae)
        self.advantages = None
        self.returns = None

        logger.info(
            "TensorRolloutBuffer: T=%d, agents=%d, total=%d",
            self.T, self.num_agents, self.T * self.num_agents,
        )

    @classmethod
    def merge(cls, buffers: list["TensorRolloutBuffer"]) -> "TensorRolloutBuffer":
        """Merge multiple TensorRolloutBuffers by concatenating along time axis.

        Each buffer has shape (T_i, 12, ...). Result has (sum(T_i), 12, ...).
        GAE must be recomputed after merging since episode boundaries differ.
        """
        merged = cls.__new__(cls)
        merged.gamma = buffers[0].gamma
        merged.lam = buffers[0].lam
        merged.num_agents = buffers[0].num_agents

        # Concat observations along time dimension
        merged.obs = {}
        for k in buffers[0].obs:
            merged.obs[k] = torch.cat([b.obs[k] for b in buffers], dim=0)

        # Concat scalar fields
        merged.log_probs = torch.cat([b.log_probs for b in buffers], dim=0)
        merged.values = torch.cat([b.values for b in buffers], dim=0)
        merged.rewards = torch.cat([b.rewards for b in buffers], dim=0)
        merged.dones = torch.cat([b.dones for b in buffers], dim=0)

        # Concat LSTM states
        merged.hx_h = torch.cat([b.hx_h for b in buffers], dim=0)
        merged.hx_c = torch.cat([b.hx_c for b in buffers], dim=0)

        # Concat masks and actions
        merged.masks = {}
        for k in buffers[0].masks:
            merged.masks[k] = torch.cat([b.masks[k] for b in buffers], dim=0)

        merged.actions = {}
        for k in buffers[0].actions:
            merged.actions[k] = torch.cat([b.actions[k] for b in buffers], dim=0)

        merged.T = merged.log_probs.shape[0]
        merged.advantages = None
        merged.returns = None

        logger.info("Merged %d buffers: total T=%d, agents=%d, transitions=%d",
                    len(buffers), merged.T, merged.num_agents, merged.T * merged.num_agents)
        return merged

    def slice_agent(self, agent_idx: int) -> "TensorRolloutBuffer":
        """Extract a single agent's data from (T, 12, ...) tensors.

        Returns a new TensorRolloutBuffer with num_agents=1 for per-hero PPO.
        """
        sliced = TensorRolloutBuffer.__new__(TensorRolloutBuffer)
        sliced.gamma = self.gamma
        sliced.lam = self.lam

        sliced.obs = {k: v[:, agent_idx:agent_idx+1] for k, v in self.obs.items()}
        sliced.log_probs = self.log_probs[:, agent_idx:agent_idx+1]
        sliced.values = self.values[:, agent_idx:agent_idx+1]
        sliced.rewards = self.rewards[:, agent_idx:agent_idx+1]
        sliced.dones = self.dones[:, agent_idx:agent_idx+1]
        sliced.hx_h = self.hx_h[:, agent_idx:agent_idx+1]
        sliced.hx_c = self.hx_c[:, agent_idx:agent_idx+1]
        sliced.masks = {k: v[:, agent_idx:agent_idx+1] for k, v in self.masks.items()}
        sliced.actions = {k: v[:, agent_idx:agent_idx+1] for k, v in self.actions.items()}

        sliced.T = self.T
        sliced.num_agents = 1

        # Copy GAE results if computed
        sliced.advantages = self.advantages[:, agent_idx:agent_idx+1] if self.advantages is not None else None
        sliced.returns = self.returns[:, agent_idx:agent_idx+1] if self.returns is not None else None

        return sliced

    def total_transitions(self) -> int:
        return self.T * self.num_agents

    def compute_gae(self, bootstrap_values=None):
        """Vectorized GAE: single reverse sweep, vectorized over 12 agents."""
        T = self.T
        rewards = self.rewards
        values = self.values
        not_dones = (~self.dones).float()

        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(self.num_agents)

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = torch.zeros(self.num_agents)
                if bootstrap_values is not None:
                    next_val = bootstrap_values
                next_val = next_val * not_dones[t]
            else:
                next_val = values[t + 1]

            mask = not_dones[t]
            delta = rewards[t] + self.gamma * next_val * mask - values[t]
            last_gae = delta + self.gamma * self.lam * mask * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = advantages + values
        logger.info("GAE computed: T=%d, mean_adv=%.4f, mean_ret=%.4f",
                     T, advantages.mean().item(), self.returns.mean().item())

    def iterate_sequences(self, seq_len: int, batch_size: int) -> Iterator[SequenceChunk]:
        """Yield SequenceChunks via direct tensor slicing."""
        if self.advantages is None:
            self.compute_gae()

        chunk_indices = []
        for a in range(self.num_agents):
            for start in range(0, self.T, seq_len):
                if start + seq_len <= self.T:
                    chunk_indices.append((a, start))

        if not chunk_indices:
            return

        random.shuffle(chunk_indices)

        for i in range(0, len(chunk_indices), batch_size):
            batch_idx = chunk_indices[i:i + batch_size]
            if batch_idx:
                yield self._build_batch(batch_idx, seq_len)

    def _build_batch(self, indices: list[tuple[int, int]], seq_len: int) -> SequenceChunk:
        """Build SequenceChunk from (agent_idx, start_t) pairs by tensor slicing."""
        obs = {}
        for k, v in self.obs.items():
            obs[k] = torch.stack([v[s:s+seq_len, a] for a, s in indices])

        masks = {}
        for k, v in self.masks.items():
            masks[k] = torch.stack([v[s:s+seq_len, a] for a, s in indices])

        actions = {}
        for k, v in self.actions.items():
            actions[k] = torch.stack([v[s:s+seq_len, a] for a, s in indices])

        old_log_probs = torch.stack([self.log_probs[s:s+seq_len, a] for a, s in indices])
        values = torch.stack([self.values[s:s+seq_len, a] for a, s in indices])
        rewards = torch.stack([self.rewards[s:s+seq_len, a] for a, s in indices])
        advantages = torch.stack([self.advantages[s:s+seq_len, a] for a, s in indices])
        returns = torch.stack([self.returns[s:s+seq_len, a] for a, s in indices])

        # LSTM hidden at chunk start: hx[s, a] is (1, 256)
        # Stack → (B, 1, 256) → permute → (1, B, 256)
        hx_h = torch.stack([self.hx_h[s, a] for a, s in indices])
        hx_c = torch.stack([self.hx_c[s, a] for a, s in indices])
        hx_init = (hx_h.permute(1, 0, 2).contiguous(),
                    hx_c.permute(1, 0, 2).contiguous())

        return SequenceChunk(
            obs=obs, masks=masks, actions=actions,
            old_log_probs=old_log_probs, values=values,
            rewards=rewards, advantages=advantages, returns=returns,
            hx_init=hx_init,
        )
