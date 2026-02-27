"""Offline Rollout Trainer: reads .pt rollout files from C++ inference server.

Architecture:
  C++ inference server: collects rollouts in game, writes .pt files
  Python trainer: loads .pt files, runs PPO update, exports TorchScript model
  C++ hot-reloads updated model for next rollout

Synchronous batch training: waits for sync_rollouts files (default 15),
merges into one big buffer, then runs PPO. Includes adaptive entropy
coefficient, gamma annealing, and per-hero auto-rollback.

Supports two rollout formats:
  - FATE: Original format from training server
  - FSTR: Streaming format from distributed clients (FateAnotherRL-together)
"""

import logging
import struct
import time
import yaml
import json
from collections import deque
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from fateanother_rl.data.constants import NUM_HEROES, HERO_IDS, DISCRETE_HEADS
from fateanother_rl.model.policy import FateModel
from fateanother_rl.model.export import export_model
from fateanother_rl.training.buffer import TensorRolloutBuffer
from fateanother_rl.training.ppo import ppo_loss
from fateanother_rl.utils.logger import Logger

logger = logging.getLogger(__name__)

# Number of agents (heroes) per game
NUM_AGENTS = 12

# ============================================================
# Reward Configuration
# ============================================================

@dataclass
class RewardConfig:
    """Reward weights for relabeling. Load from YAML/JSON for easy tuning.

    Event types from protocol.h:
      EVT_KILL(1), EVT_CREEP_KILL(2), EVT_LEVEL_UP(3), EVT_PORTAL(4)
    """

    # Event-based rewards (from EventType enum)
    kill_hero: float = 1.0      # EVT_KILL: 영웅 처치
    kill_creep: float = 0.1     # EVT_CREEP_KILL: 크립 처치
    level_up: float = 0.2       # EVT_LEVEL_UP: 레벨업
    portal_use: float = 0.05    # EVT_PORTAL: 포탈/건물 진입

    # HP/damage rewards (per-tick normalized)
    damage_dealt: float = 0.001
    damage_taken: float = -0.0005
    healing: float = 0.0005

    # Score-based rewards
    score_diff: float = 0.01

    # Win/lose terminal rewards
    win: float = 10.0
    lose: float = -10.0

    # Shaping coefficients (experimental)
    hp_shaping: float = 0.0
    position_shaping: float = 0.0

    @classmethod
    def from_file(cls, path: str) -> "RewardConfig":
        """Load reward config from YAML or JSON file."""
        path = Path(path)
        if not path.exists():
            logger.warning("Reward config not found: %s, using defaults", path)
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            if path.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(f) or {}
            elif path.suffix == ".json":
                data = json.load(f)
            else:
                logger.warning("Unknown config format: %s, using defaults", path.suffix)
                return cls()

        # Only use keys that exist in the dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}

        logger.info("Loaded reward config from %s: %d params", path, len(filtered))
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Export config as dict for logging."""
        return {
            "kill_hero": self.kill_hero,
            "kill_creep": self.kill_creep,
            "level_up": self.level_up,
            "portal_use": self.portal_use,
            "damage_dealt": self.damage_dealt,
            "damage_taken": self.damage_taken,
            "healing": self.healing,
            "score_diff": self.score_diff,
            "win": self.win,
            "lose": self.lose,
            "hp_shaping": self.hp_shaping,
            "position_shaping": self.position_shaping,
        }


# ============================================================
# Rollout Loaders
# ============================================================

# C++ ScalarType enum -> torch dtype mapping
_DTYPE_MAP = {
    0: torch.uint8,
    1: torch.int8,
    2: torch.int16,
    3: torch.int32,
    4: torch.int64,
    5: torch.float16,
    6: torch.float32,
    7: torch.float64,
    11: torch.bool,
    15: torch.bfloat16,
}


def load_fate_rollout(path: str) -> dict:
    """Load a FATE binary rollout file as a dict of tensors.

    Custom binary format written by C++ RolloutWriter:
        Header: "FATE"(4) + num_entries(4)
        Per entry: name_len(4) + name + dtype(1) + ndim(4)
                   + shape(8*ndim) + nbytes(8) + raw_data
    """
    result = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"FATE":
            raise ValueError(f"Not a FATE rollout file (magic={magic!r}): {path}")

        num_entries = struct.unpack("<I", f.read(4))[0]

        for _ in range(num_entries):
            name_len = struct.unpack("<I", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")

            dtype_code = struct.unpack("<B", f.read(1))[0]
            ndim = struct.unpack("<I", f.read(4))[0]
            shape = tuple(struct.unpack("<q", f.read(8))[0] for _ in range(ndim))

            nbytes = struct.unpack("<q", f.read(8))[0]
            raw_data = f.read(nbytes)

            dtype = _DTYPE_MAP.get(dtype_code)
            if dtype is None:
                raise ValueError(f"Unknown dtype code {dtype_code} for tensor '{name}'")

            tensor = torch.frombuffer(bytearray(raw_data), dtype=dtype).reshape(shape)
            result[name] = tensor

    logger.info("Loaded FATE rollout: %d tensors from %s", len(result), path)
    return result


def load_fstr_rollout(path: str) -> dict:
    """Load a FSTR (streaming) binary rollout file as a dict of tensors.

    FSTR format from FateAnotherRL-together client:
        Header: "FSTR"(4) + version(4) + episode_id(8) + chunk_seq(4)
                + num_timesteps(4) + num_agents(4)
        Per transition: timestep(4) + agent_idx(4) + obs_tensors + scalars
                        + masks + actions + v2_fields
        End marker: "CONT"(4) or "TERM"(4) + terminal_rewards(48)
    """
    result = {}

    with open(path, "rb") as f:
        # Read header
        magic = f.read(4)
        if magic != b"FSTR":
            raise ValueError(f"Not a FSTR rollout file (magic={magic!r}): {path}")

        version = struct.unpack("<I", f.read(4))[0]
        episode_id = struct.unpack("<Q", f.read(8))[0]
        chunk_seq = struct.unpack("<I", f.read(4))[0]
        num_timesteps = struct.unpack("<I", f.read(4))[0]
        num_agents = struct.unpack("<I", f.read(4))[0]

        if num_agents != NUM_AGENTS:
            logger.warning("FSTR num_agents=%d, expected %d", num_agents, NUM_AGENTS)

        # Pre-allocate tensors
        T = num_timesteps
        A = num_agents

        self_vecs = torch.zeros(T, A, 77, dtype=torch.float32)
        ally_vecs = torch.zeros(T, A, 5, 37, dtype=torch.float32)
        enemy_vecs = torch.zeros(T, A, 6, 43, dtype=torch.float32)
        global_vecs = torch.zeros(T, A, 6, dtype=torch.float32)
        grids = torch.zeros(T, A, 6, 25, 48, dtype=torch.float32)
        hx_h = torch.zeros(T, A, 1, 256, dtype=torch.float32)
        hx_c = torch.zeros(T, A, 1, 256, dtype=torch.float32)
        log_probs = torch.zeros(T, A, dtype=torch.float32)
        values = torch.zeros(T, A, dtype=torch.float32)
        rewards = torch.zeros(T, A, dtype=torch.float32)
        dones = torch.zeros(T, A, dtype=torch.bool)

        # v2 fields
        model_versions = torch.zeros(T, A, dtype=torch.int32)
        game_times = torch.zeros(T, dtype=torch.float32)
        prev_hp = torch.zeros(T, A, dtype=torch.float32)
        prev_max_hp = torch.zeros(T, A, dtype=torch.float32)
        unit_alive = torch.zeros(T, A, dtype=torch.int32)
        unit_level = torch.zeros(T, A, dtype=torch.int32)
        unit_x = torch.zeros(T, A, dtype=torch.float32)
        unit_y = torch.zeros(T, A, dtype=torch.float32)
        skill_points = torch.zeros(T, A, dtype=torch.int32)
        prev_score_t0 = torch.zeros(T, dtype=torch.int32)
        prev_score_t1 = torch.zeros(T, dtype=torch.int32)

        # Events: collect all then stack
        events_list = [[[] for _ in range(A)] for _ in range(T)]
        event_counts = torch.zeros(T, A, dtype=torch.int32)

        # Masks and actions: accumulate names
        masks_data = {}
        actions_data = {}

        # Read transitions
        for _ in range(T * A):
            ts = struct.unpack("<i", f.read(4))[0]
            aidx = struct.unpack("<i", f.read(4))[0]

            if ts < 0 or ts >= T or aidx < 0 or aidx >= A:
                logger.error("Invalid ts=%d or aidx=%d in %s", ts, aidx, path)
                continue

            # Observation tensors (raw float32 binary)
            self_vecs[ts, aidx] = torch.frombuffer(
                bytearray(f.read(77 * 4)), dtype=torch.float32
            )
            ally_vecs[ts, aidx] = torch.frombuffer(
                bytearray(f.read(5 * 37 * 4)), dtype=torch.float32
            ).reshape(5, 37)
            enemy_vecs[ts, aidx] = torch.frombuffer(
                bytearray(f.read(6 * 43 * 4)), dtype=torch.float32
            ).reshape(6, 43)
            global_vecs[ts, aidx] = torch.frombuffer(
                bytearray(f.read(6 * 4)), dtype=torch.float32
            )
            grids[ts, aidx] = torch.frombuffer(
                bytearray(f.read(6 * 25 * 48 * 4)), dtype=torch.float32
            ).reshape(6, 25, 48)

            # LSTM hidden states
            hx_h[ts, aidx, 0] = torch.frombuffer(
                bytearray(f.read(256 * 4)), dtype=torch.float32
            )
            hx_c[ts, aidx, 0] = torch.frombuffer(
                bytearray(f.read(256 * 4)), dtype=torch.float32
            )

            # Scalars
            log_probs[ts, aidx] = struct.unpack("<f", f.read(4))[0]
            values[ts, aidx] = struct.unpack("<f", f.read(4))[0]
            rewards[ts, aidx] = struct.unpack("<f", f.read(4))[0]
            done_byte = struct.unpack("<B", f.read(1))[0]
            dones[ts, aidx] = done_byte != 0

            # Masks
            num_masks = struct.unpack("<I", f.read(4))[0]
            for _ in range(num_masks):
                name_len = struct.unpack("<I", f.read(4))[0]
                name = f.read(name_len).decode("utf-8")
                nbytes = struct.unpack("<q", f.read(8))[0]
                data = f.read(nbytes)

                key = f"mask_{name}"
                if key not in masks_data:
                    # Determine mask shape from first occurrence
                    n_elements = nbytes  # bool tensors: 1 byte per element
                    masks_data[key] = torch.zeros(T, A, n_elements, dtype=torch.bool)
                masks_data[key][ts, aidx] = torch.frombuffer(
                    bytearray(data), dtype=torch.bool
                )

            # Actions
            num_actions = struct.unpack("<I", f.read(4))[0]
            for _ in range(num_actions):
                name_len = struct.unpack("<I", f.read(4))[0]
                name = f.read(name_len).decode("utf-8")
                nbytes = struct.unpack("<q", f.read(8))[0]
                data = f.read(nbytes)

                key = f"act_{name}"
                if key not in actions_data:
                    # Actions are typically int64 (8 bytes) or float32
                    n_elements = nbytes // 8 if nbytes >= 8 else nbytes // 4
                    if n_elements == 0:
                        n_elements = 1
                    dtype = torch.int64 if nbytes % 8 == 0 else torch.float32
                    if n_elements == 1:
                        actions_data[key] = torch.zeros(T, A, dtype=dtype)
                    else:
                        actions_data[key] = torch.zeros(T, A, n_elements, dtype=dtype)
                    actions_data[key + "_dtype"] = dtype

                dtype = actions_data.get(key + "_dtype", torch.int64)
                tensor = torch.frombuffer(bytearray(data), dtype=dtype)
                if actions_data[key].dim() == 2:
                    actions_data[key][ts, aidx] = tensor.item() if tensor.numel() == 1 else tensor[0]
                else:
                    actions_data[key][ts, aidx] = tensor

            # v2 fields
            model_versions[ts, aidx] = struct.unpack("<i", f.read(4))[0]
            game_time = struct.unpack("<f", f.read(4))[0]
            if aidx == 0:
                game_times[ts] = game_time

            # Events
            num_events = struct.unpack("<I", f.read(4))[0]
            event_counts[ts, aidx] = num_events
            for _ in range(num_events):
                ev_data = struct.unpack("<4i", f.read(16))
                events_list[ts][aidx].append(ev_data)

            # Unit state
            prev_hp[ts, aidx] = struct.unpack("<f", f.read(4))[0]
            prev_max_hp[ts, aidx] = struct.unpack("<f", f.read(4))[0]
            alive_byte = struct.unpack("<B", f.read(1))[0]
            unit_alive[ts, aidx] = alive_byte
            unit_level[ts, aidx] = struct.unpack("<i", f.read(4))[0]
            unit_x[ts, aidx] = struct.unpack("<f", f.read(4))[0]
            unit_y[ts, aidx] = struct.unpack("<f", f.read(4))[0]
            skill_points[ts, aidx] = struct.unpack("<i", f.read(4))[0]

            score_t0 = struct.unpack("<i", f.read(4))[0]
            score_t1 = struct.unpack("<i", f.read(4))[0]
            if aidx == 0:
                prev_score_t0[ts] = score_t0
                prev_score_t1[ts] = score_t1

        # Read end marker
        end_marker = f.read(4)
        if end_marker == b"TERM":
            # Read terminal rewards
            terminal_rewards = torch.frombuffer(
                bytearray(f.read(A * 4)), dtype=torch.float32
            )
            # Add terminal rewards to last timestep
            if T > 0:
                rewards[T - 1] += terminal_rewards
                dones[T - 1] = True
            logger.info("FSTR terminal chunk: added terminal rewards")
        elif end_marker == b"CONT":
            logger.info("FSTR continuation chunk (more data expected)")
        else:
            logger.warning("Unknown end marker: %s", end_marker)

    # Build result dict matching FATE format
    result["self_vecs"] = self_vecs
    result["ally_vecs"] = ally_vecs
    result["enemy_vecs"] = enemy_vecs
    result["global_vecs"] = global_vecs
    result["grids"] = grids
    result["hx_h"] = hx_h
    result["hx_c"] = hx_c
    result["log_probs"] = log_probs
    result["values"] = values
    result["rewards"] = rewards
    result["dones"] = dones

    # Add masks and actions (remove dtype keys)
    for k, v in masks_data.items():
        if not k.endswith("_dtype"):
            result[k] = v
    for k, v in actions_data.items():
        if not k.endswith("_dtype"):
            result[k] = v

    # v2 fields
    result["__version__"] = torch.tensor([2], dtype=torch.int32)
    result["model_version"] = model_versions[:, 0]  # Same across agents
    result["game_time"] = game_times
    result["prev_hp"] = prev_hp
    result["prev_max_hp"] = prev_max_hp
    result["unit_alive"] = unit_alive
    result["unit_level"] = unit_level
    result["unit_x"] = unit_x
    result["unit_y"] = unit_y
    result["skill_points"] = skill_points
    result["prev_score_t0"] = prev_score_t0
    result["prev_score_t1"] = prev_score_t1
    result["event_counts"] = event_counts

    # Events tensor: (T, A, max_events, 4)
    max_events = max(
        max(len(events_list[t][a]) for a in range(A))
        for t in range(T)
    ) if T > 0 else 0
    max_events = max(max_events, 4)  # At least 4 slots
    events = torch.zeros(T, A, max_events, 4, dtype=torch.int32)
    for t in range(T):
        for a in range(A):
            for i, ev in enumerate(events_list[t][a]):
                if i < max_events:
                    events[t, a, i] = torch.tensor(ev, dtype=torch.int32)
    result["events"] = events

    # Metadata
    result["_fstr_episode_id"] = episode_id
    result["_fstr_chunk_seq"] = chunk_seq
    result["_fstr_version"] = version

    logger.info(
        "Loaded FSTR rollout: T=%d, A=%d, ep=%d, chunk=%d from %s",
        T, A, episode_id, chunk_seq, path
    )
    return result


def load_rollout(path: str) -> dict:
    """Auto-detect and load rollout file (FATE or FSTR format)."""
    with open(path, "rb") as f:
        magic = f.read(4)

    if magic == b"FATE":
        return load_fate_rollout(path)
    elif magic == b"FSTR":
        return load_fstr_rollout(path)
    else:
        raise ValueError(f"Unknown rollout format (magic={magic!r}): {path}")


def relabel_rewards(data: dict, config: RewardConfig) -> dict:
    """Relabel rewards using v2 fields and RewardConfig.

    Overwrites data["rewards"] with newly computed rewards based on:
    - events (kills, deaths, assists)
    - HP changes (damage dealt/taken)
    - score changes

    Args:
        data: Rollout data dict from load_rollout()
        config: RewardConfig with reward weights

    Returns:
        Same dict with rewards tensor replaced
    """
    # Check if we have v2 fields for relabeling
    if data.get("events") is None:
        logger.warning("No v2 fields for relabeling, keeping original rewards")
        return data

    T, A = data["rewards"].shape
    new_rewards = torch.zeros(T, A, dtype=torch.float32)

    # Event-based rewards (kills, deaths)
    # events shape: (T, A, max_events, 4) where [type, killer, victim, tick]
    events = data["events"]
    event_counts = data["event_counts"]

    # Event types (from C++ EventType enum in protocol.h)
    EVT_KILL = 1        # Hero kill
    EVT_CREEP_KILL = 2  # Creep kill
    EVT_LEVEL_UP = 3    # Level up
    EVT_PORTAL = 4      # Portal entry (building enter/exit)

    for t in range(T):
        for a in range(A):
            n_events = event_counts[t, a].item() if event_counts is not None else 0
            for e in range(min(n_events, events.shape[2])):
                ev_type = events[t, a, e, 0].item()
                if ev_type == EVT_KILL:
                    new_rewards[t, a] += config.kill_hero
                elif ev_type == EVT_CREEP_KILL:
                    new_rewards[t, a] += config.kill_creep
                elif ev_type == EVT_LEVEL_UP:
                    new_rewards[t, a] += config.level_up
                elif ev_type == EVT_PORTAL:
                    new_rewards[t, a] += config.portal_use

    # HP-based rewards (damage dealt/taken)
    if data.get("prev_hp") is not None and data.get("prev_max_hp") is not None:
        prev_hp = data["prev_hp"].float()  # (T, A)
        prev_max_hp = data["prev_max_hp"].float()

        # HP change from previous timestep
        # Negative delta = took damage, positive delta = healed
        for t in range(1, T):
            hp_delta = prev_hp[t] - prev_hp[t - 1]
            # Normalize by max HP for fair comparison across heroes
            hp_ratio = hp_delta / (prev_max_hp[t].clamp(min=1.0))

            # Damage taken (negative hp_delta)
            damage_taken = (-hp_ratio).clamp(min=0)
            new_rewards[t] += damage_taken * config.damage_taken * 100  # Scale up

            # Healing (positive hp_delta)
            healing = hp_ratio.clamp(min=0)
            new_rewards[t] += healing * config.healing * 100

    # Score-based rewards
    if data.get("prev_score_t0") is not None and data.get("prev_score_t1") is not None:
        score_t0 = data["prev_score_t0"].float()  # (T,)
        score_t1 = data["prev_score_t1"].float()

        for t in range(1, T):
            # Team 0 = agents 0-5, Team 1 = agents 6-11
            score_delta_t0 = score_t0[t] - score_t0[t - 1]
            score_delta_t1 = score_t1[t] - score_t1[t - 1]

            # Team 0 rewards
            new_rewards[t, :6] += score_delta_t0 * config.score_diff
            new_rewards[t, :6] -= score_delta_t1 * config.score_diff

            # Team 1 rewards (opposite)
            new_rewards[t, 6:] += score_delta_t1 * config.score_diff
            new_rewards[t, 6:] -= score_delta_t0 * config.score_diff

    # Replace rewards
    data["rewards"] = new_rewards

    logger.debug("Relabeled rewards: mean=%.4f, std=%.4f",
                 new_rewards.mean().item(), new_rewards.std().item())
    return data


class RolloutTrainer:
    """Offline trainer that reads .pt rollout files from C++ inference server.

    Synchronous batch workflow per iteration:
      1. Wait for sync_rollouts .pt files (default 15)
      2. Load all rollouts, build TensorRolloutBuffers
      3. Merge buffers into one large (sum_T, 12, ...) buffer
      4. Compute GAE with annealed gamma
      5. Per-hero PPO update with adaptive entropy
      6. Export TorchScript models for C++ hot-reload
      7. Auto-rollback heroes whose EMA reward drops >threshold below best
      8. Delete processed rollout files
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Training device: %s", self.device)

        # --- Per-hero Models ---
        model_cfg = config.get("model", {})
        self.model_kwargs = dict(
            self_dim=model_cfg.get("self_dim", 77),
            ally_dim=model_cfg.get("ally_dim", 37),
            enemy_dim=model_cfg.get("enemy_dim", 43),
            global_dim=model_cfg.get("global_dim", 6),
            grid_channels=model_cfg.get("grid_channels", 3),
            hidden_dim=model_cfg.get("hidden_dim", 256),
        )

        ppo_cfg = config.get("ppo", {})
        lr = float(ppo_cfg.get("lr", 3e-4))

        self.models = {}
        self.optimizers = {}
        for hid in HERO_IDS:
            m = FateModel(**self.model_kwargs).to(self.device)
            self.models[hid] = m
            self.optimizers[hid] = optim.Adam(m.parameters(), lr=lr)

        single_params = sum(p.numel() for p in self.models[HERO_IDS[0]].parameters())
        logger.info("Per-hero model params: %d, total: %d", single_params, single_params * NUM_HEROES)

        # --- Directories ---
        train_cfg = config.get("training", {})
        self.rollout_dir = Path(train_cfg.get("rollout_dir", "/data/rollouts"))
        self.model_dir = Path(train_cfg.get("model_dir", "/data/models"))
        self.save_dir = Path(train_cfg.get("save_dir", "checkpoints"))
        self.rollout_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # --- PPO config ---
        self.ppo_epochs = int(ppo_cfg.get("ppo_epochs", 4))
        self.batch_size = int(ppo_cfg.get("batch_size", 4096))
        self.seq_len = int(ppo_cfg.get("seq_len", 16))
        self.clip_eps = float(ppo_cfg.get("clip_eps", 0.2))
        self.vf_coef = float(ppo_cfg.get("vf_coef", 0.5))
        self.max_grad_norm = float(ppo_cfg.get("max_grad_norm", 0.5))

        # --- Training config ---
        self.max_iterations = int(train_cfg.get("max_iterations", 100000))
        self.save_interval = int(train_cfg.get("save_interval", 50))
        self.log_interval = int(train_cfg.get("log_interval", 1))
        self.poll_interval = float(train_cfg.get("poll_interval", 1.0))
        self.sync_rollouts = int(train_cfg.get("sync_rollouts", 15))

        # --- GAE config (initial, may be annealed) ---
        self.gamma = float(ppo_cfg.get("gamma", 0.998))
        self.gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))

        # --- Reward config (for relabeling with hot-reload) ---
        reward_cfg_path = train_cfg.get("reward_config", None)
        self._reward_config_path = reward_cfg_path
        self._reward_config_mtime = 0.0
        if reward_cfg_path and Path(reward_cfg_path).exists():
            self.reward_config = RewardConfig.from_file(reward_cfg_path)
            self._reward_config_mtime = Path(reward_cfg_path).stat().st_mtime
            logger.info("Reward relabeling enabled: %s", reward_cfg_path)
        else:
            self.reward_config = None
            logger.info("Reward relabeling disabled (no reward_config in config)")

        # --- Streaming PPO config (memory-safe) ---
        # Instead of loading all rollouts at once (OOM risk),
        # load rollout_batch_size files at a time and run PPO
        self.rollout_batch_size = int(train_cfg.get("rollout_batch_size", 10))
        logger.info("Streaming PPO: batch_size=%d files (memory-safe mode)",
                    self.rollout_batch_size)

        # --- Adaptive entropy + gamma ---
        adaptive_cfg = config.get("adaptive", {})
        self.adaptive_cfg = adaptive_cfg
        self.ent_coef = float(adaptive_cfg.get("ent_coef_init", 0.01))
        self.reward_history: deque = deque(maxlen=1000)

        # --- Logger ---
        self.tb_logger = Logger(log_dir=str(train_cfg.get("log_dir", "runs")))

        # --- Counters ---
        self.iteration = 0
        self.total_transitions = 0

        # --- Best model tracking (per-hero EMA reward) ---
        self.ema_alpha = 0.05  # smoothing factor: ~20 iteration window
        self.ema_reward = {hid: 0.0 for hid in HERO_IDS}
        self.best_reward = {hid: float("-inf") for hid in HERO_IDS}
        self.best_dir = self.save_dir / "best"
        self.best_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def train(self):
        """Main training loop. Waits for sync_rollouts files per iteration."""
        logger.info("=== RolloutTrainer Start (sync=%d) ===", self.sync_rollouts)
        logger.info("Rollout dir: %s", self.rollout_dir)
        logger.info("Model dir: %s", self.model_dir)

        # Export initial model so C++ can start inference immediately
        self._export_model()
        logger.info("Initial model exported to %s", self.model_dir)

        try:
            for _ in count():
                if self.iteration >= self.max_iterations:
                    logger.info("Max iterations reached (%d). Stopping.", self.max_iterations)
                    break

                # 1. Wait for sync_rollouts files
                rollout_paths = self._wait_for_rollouts()
                t_start = time.time()

                # 1.5. Hot-reload reward config if changed
                self._maybe_reload_reward_config()

                # 2. Streaming PPO: process files in batches to save memory
                #    Instead of loading all 100 files (80GB), load 10 at a time (8GB)
                loaded_paths = []
                failed_paths = []
                n_transitions = 0
                all_losses = []
                all_rewards_for_tracking = []

                # Shuffle rollout paths for better data mixing across epochs
                import random
                shuffled_paths = list(rollout_paths)

                # Each PPO epoch processes all files in batches
                for epoch in range(self.ppo_epochs):
                    random.shuffle(shuffled_paths)

                    for batch_start in range(0, len(shuffled_paths), self.rollout_batch_size):
                        batch_paths = shuffled_paths[batch_start:batch_start + self.rollout_batch_size]

                        # Load batch
                        buffers = []
                        for rp in batch_paths:
                            if epoch == 0:
                                logger.info("Loading rollout: %s", rp.name)
                            try:
                                data = load_rollout(str(rp))

                                # Relabel rewards if config is provided
                                if self.reward_config is not None:
                                    data = relabel_rewards(data, self.reward_config)

                                buf = TensorRolloutBuffer(data, gamma=self.gamma, lam=self.gae_lambda)
                                if buf.total_transitions() > 0:
                                    buffers.append(buf)
                                    if epoch == 0 and rp not in loaded_paths:
                                        loaded_paths.append(rp)
                            except Exception as e:
                                logger.error("Failed to load %s: %s", rp.name, e)
                                if rp not in failed_paths:
                                    failed_paths.append(rp)

                        if not buffers:
                            continue

                        # Merge batch
                        if len(buffers) == 1:
                            buffer = buffers[0]
                        else:
                            buffer = TensorRolloutBuffer.merge(buffers)

                        if epoch == 0:
                            n_transitions += buffer.total_transitions()

                        # Compute GAE
                        buffer.gamma = self.gamma
                        buffer.compute_gae()

                        # PPO update (single epoch per batch)
                        losses = self._ppo_update_single_epoch(buffer)
                        all_losses.append(losses)

                        # Track rewards for adaptive entropy (first epoch only)
                        if epoch == 0:
                            all_rewards_for_tracking.append(buffer.rewards.float().mean().item())

                        # Free memory immediately
                        del buffers, buffer

                    logger.info("Epoch %d/%d complete", epoch + 1, self.ppo_epochs)

                if not loaded_paths:
                    # All loads failed -- clean up and retry
                    for rp in rollout_paths:
                        rp.unlink(missing_ok=True)
                    continue

                self.total_transitions += n_transitions
                self.iteration += 1
                t_elapsed = time.time() - t_start

                # Average losses across all batches
                if all_losses:
                    losses = {k: float(np.mean([l[k] for l in all_losses if k in l]))
                              for k in all_losses[0]}
                else:
                    losses = {}

                # Track reward for adaptive entropy
                if all_rewards_for_tracking:
                    self.reward_history.append(np.mean(all_rewards_for_tracking))

                # 5. Adaptive entropy update
                self._adapt_entropy()

                # 6. Gamma annealing
                self._anneal_gamma()

                # 7. Export model for C++ hot-reload
                self._export_model()

                # 8. Cleanup processed rollouts
                for rp in loaded_paths:
                    rp.unlink(missing_ok=True)
                for rp in failed_paths:
                    rp.unlink(missing_ok=True)

                # 9. Logging
                if self.iteration % self.log_interval == 0:
                    self._log(self.iteration, losses, n_transitions, t_elapsed,
                              merged_count=len(loaded_paths))

                if self.iteration % self.save_interval == 0:
                    self._save_checkpoint(self.iteration)

        except KeyboardInterrupt:
            logger.info("Interrupted. Saving checkpoint...")
            self._save_checkpoint(self.iteration)
        finally:
            self.tb_logger.close()
            logger.info("=== RolloutTrainer stopped (iter=%d, total_trans=%d) ===",
                        self.iteration, self.total_transitions)

    # ------------------------------------------------------------------
    # Rollout file loading
    # ------------------------------------------------------------------
    def _wait_for_rollouts(self) -> list[Path]:
        """Wait for sync_rollouts files. Proceeds with 80%+ after 10min patience.

        Supports both FATE (.pt) and FSTR (.fatestream) formats.
        """
        first_seen = None
        while True:
            # Find both FATE (.pt) and FSTR (.fatestream) files
            pt_files = list(self.rollout_dir.glob("rollout_*.pt"))
            fstr_files = list(self.rollout_dir.glob("rollout_*.fatestream"))
            files = sorted(pt_files + fstr_files, key=lambda p: p.stat().st_mtime)
            if len(files) >= self.sync_rollouts:
                return files[:self.sync_rollouts]
            # Safety: don't hang forever if some WC3 instances crash
            if files and first_seen is None:
                first_seen = time.time()
            if first_seen and len(files) >= max(1, int(self.sync_rollouts * 0.8)):
                if time.time() - first_seen > 600:
                    logger.warning("Patience expired: got %d/%d rollouts, proceeding",
                                   len(files), self.sync_rollouts)
                    first_seen = None
                    return files
            if not files:
                first_seen = None
            time.sleep(self.poll_interval)

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------
    def _ppo_update(self, buffer: TensorRolloutBuffer) -> dict:
        """Run per-hero PPO on the given buffer with adaptive entropy."""
        ent_coef = self.ent_coef

        all_stats = []

        for hero_idx, hero_id in enumerate(HERO_IDS):
            model = self.models[hero_id]
            optimizer = self.optimizers[hero_id]
            model.train()

            hero_buffer = buffer.slice_agent(hero_idx)

            for epoch in range(self.ppo_epochs):
                for batch in hero_buffer.iterate_sequences(self.seq_len, self.batch_size):
                    batch = batch.to(self.device)

                    new_lp, new_values, new_entropy = model.forward_sequence(
                        batch.obs, batch.hx_init, batch.masks, batch.actions,
                    )

                    loss, stats = ppo_loss(
                        new_lp, batch.old_log_probs,
                        new_values, batch.returns,
                        new_entropy, batch.advantages,
                        clip_eps=self.clip_eps,
                        vf_coef=self.vf_coef,
                        ent_coef=ent_coef,
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    optimizer.step()
                    all_stats.append(stats)

        if all_stats:
            return {k: float(np.mean([s[k] for s in all_stats])) for k in all_stats[0]}
        return {}

    def _ppo_update_single_epoch(self, buffer: TensorRolloutBuffer) -> dict:
        """Run single PPO epoch on the buffer (for streaming mode).

        Unlike _ppo_update which runs multiple epochs, this runs just one pass.
        Used by streaming PPO where epochs are handled in the outer loop.
        """
        ent_coef = self.ent_coef
        all_stats = []

        for hero_idx, hero_id in enumerate(HERO_IDS):
            model = self.models[hero_id]
            optimizer = self.optimizers[hero_id]
            model.train()

            hero_buffer = buffer.slice_agent(hero_idx)

            # Single epoch (no inner epoch loop)
            for batch in hero_buffer.iterate_sequences(self.seq_len, self.batch_size):
                batch = batch.to(self.device)

                new_lp, new_values, new_entropy = model.forward_sequence(
                    batch.obs, batch.hx_init, batch.masks, batch.actions,
                )

                loss, stats = ppo_loss(
                    new_lp, batch.old_log_probs,
                    new_values, batch.returns,
                    new_entropy, batch.advantages,
                    clip_eps=self.clip_eps,
                    vf_coef=self.vf_coef,
                    ent_coef=ent_coef,
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()
                all_stats.append(stats)

        if all_stats:
            return {k: float(np.mean([s[k] for s in all_stats])) for k in all_stats[0]}
        return {}

    # ------------------------------------------------------------------
    # Adaptive entropy
    # ------------------------------------------------------------------
    def _adapt_entropy(self):
        """Adapt entropy coefficient based on reward plateau detection.

        If reward is improving, reduce exploration (decay ent_coef).
        If reward is plateauing/declining, increase exploration (raise ent_coef).
        """
        cfg = self.adaptive_cfg
        window = int(cfg.get("plateau_window", 20))

        if len(self.reward_history) < window:
            return  # Not enough data yet

        recent = list(self.reward_history)[-window:]
        if len(self.reward_history) >= window * 2:
            old = list(self.reward_history)[-window * 2:-window]
        else:
            old = list(self.reward_history)[:window]

        recent_mean = sum(recent) / len(recent)
        old_mean = sum(old) / len(old)

        improving = recent_mean > old_mean + 1e-4

        if improving:
            # Reward improving -> reduce exploration
            self.ent_coef = max(
                float(cfg.get("ent_coef_min", 0.003)),
                self.ent_coef * float(cfg.get("ent_down", 0.995)),
            )
        else:
            # Plateau/declining -> increase exploration
            self.ent_coef = min(
                float(cfg.get("ent_coef_max", 0.02)),
                self.ent_coef * float(cfg.get("ent_up", 1.01)),
            )

        logger.info("Adaptive entropy: %.5f (improving=%s, recent=%.4f, old=%.4f)",
                     self.ent_coef, improving, recent_mean, old_mean)

    # ------------------------------------------------------------------
    # Reward config hot-reload
    # ------------------------------------------------------------------
    def _maybe_reload_reward_config(self):
        """Check if reward config file changed and reload if needed.

        Enables hot-reload: edit yaml file → next iteration uses new weights.
        No container restart required!
        """
        if not self._reward_config_path:
            return

        config_path = Path(self._reward_config_path)
        if not config_path.exists():
            return

        current_mtime = config_path.stat().st_mtime
        if current_mtime > self._reward_config_mtime:
            try:
                self.reward_config = RewardConfig.from_file(str(config_path))
                self._reward_config_mtime = current_mtime
                logger.info("Reward config hot-reloaded! New weights: %s",
                            self.reward_config.to_dict())
            except Exception as e:
                logger.error("Failed to reload reward config: %s", e)

    # ------------------------------------------------------------------
    # Gamma annealing
    # ------------------------------------------------------------------
    def _anneal_gamma(self):
        """Linear gamma annealing from gamma_init to gamma_final."""
        cfg = self.adaptive_cfg
        gamma_init = float(cfg.get("gamma_init", 0.998))
        gamma_final = float(cfg.get("gamma_final", 0.9995))
        anneal_iters = int(cfg.get("gamma_anneal_iters", 50000))

        progress = min(1.0, self.iteration / max(anneal_iters, 1))
        self.gamma = gamma_init + (gamma_final - gamma_init) * progress

    # ------------------------------------------------------------------
    # Best model tracking + auto-rollback
    # ------------------------------------------------------------------
    def _update_best_and_rollback(self, buffers: list[TensorRolloutBuffer]):
        """Update per-hero EMA reward, save best models, auto-rollback on collapse.

        Rollback restores a hero's weights from the best saved state_dict if its
        EMA reward drops more than rollback_threshold below the best EMA.
        """
        # Compute mean reward across all loaded buffers
        all_rewards = torch.cat([b.rewards for b in buffers], dim=0)  # (sum_T, 12)
        per_agent = all_rewards.float().mean(dim=0)  # (12,)

        rollback_threshold = float(self.adaptive_cfg.get("rollback_threshold", 0.30))
        any_rollback = False

        for hero_idx, hero_id in enumerate(HERO_IDS):
            r = per_agent[hero_idx].item()
            # EMA update
            self.ema_reward[hero_id] = (
                self.ema_alpha * r + (1 - self.ema_alpha) * self.ema_reward[hero_id]
            )

            # Check best (skip first 10 iterations for EMA warmup)
            if self.iteration > 10 and self.ema_reward[hero_id] > self.best_reward[hero_id]:
                self.best_reward[hero_id] = self.ema_reward[hero_id]
                # Save TorchScript model (for C++ inference)
                best_path = str(self.best_dir / f"best_{hero_id}.pt")
                export_model(self.models[hero_id], best_path, device="cpu")
                # Save raw state_dict (for Python rollback -- can't easily load
                # TorchScript back into nn.Module)
                state_path = str(self.best_dir / f"best_{hero_id}_state.pt")
                torch.save(self.models[hero_id].state_dict(), state_path)
                logger.info("New best %s: EMA=%.4f (iter %d)",
                            hero_id, self.best_reward[hero_id], self.iteration)

            # Auto-rollback: if EMA drops >threshold below best
            if (self.iteration > 30 and
                    self.best_reward[hero_id] > 0 and
                    self.ema_reward[hero_id] < self.best_reward[hero_id] * (1 - rollback_threshold)):
                state_path = self.best_dir / f"best_{hero_id}_state.pt"
                if state_path.exists():
                    logger.warning(
                        "ROLLBACK %s: EMA=%.4f < best=%.4f * %.0f%%. Restoring best model.",
                        hero_id, self.ema_reward[hero_id], self.best_reward[hero_id],
                        (1 - rollback_threshold) * 100,
                    )
                    state = torch.load(str(state_path), map_location=self.device, weights_only=True)
                    self.models[hero_id].load_state_dict(state)
                    # Reset EMA to near-best to prevent repeated rollbacks
                    self.ema_reward[hero_id] = self.best_reward[hero_id] * 0.95
                    any_rollback = True

        if any_rollback:
            # Bump entropy to help re-explore after rollback
            old_ent = self.ent_coef
            self.ent_coef = min(
                float(self.adaptive_cfg.get("ent_coef_max", 0.02)),
                self.ent_coef * 1.5,
            )
            logger.info("Post-rollback entropy bump: %.5f -> %.5f", old_ent, self.ent_coef)
            # Re-export the rolled-back models
            self._export_model()

        # Track mean reward for plateau detection (adaptive entropy)
        mean_r = per_agent.mean().item()
        self.reward_history.append(mean_r)

    # ------------------------------------------------------------------
    # Model export
    # ------------------------------------------------------------------
    def _export_model(self):
        """Export per-hero TorchScript models for C++ hot-reload.

        Exports one .pt file per hero (e.g. H000.pt, H001.pt, ...).
        The C++ server loads each hero's model independently.
        """
        for hero_id, model in self.models.items():
            export_path = str(self.model_dir / f"{hero_id}.pt")
            try:
                export_model(model, export_path, device="cpu")
            except Exception as e:
                logger.error("Model export failed for %s: %s", hero_id, e)
        logger.debug("All %d hero models exported", len(self.models))

    # ------------------------------------------------------------------
    # Logging and checkpoints
    # ------------------------------------------------------------------
    def _log(self, iteration: int, losses: dict, n_transitions: int, elapsed: float,
             merged_count: int = 1):
        """Log training metrics to TensorBoard and console."""
        self.tb_logger.log_iteration(iteration, losses)
        self.tb_logger.log_scalar("train/total_transitions", self.total_transitions, iteration)
        self.tb_logger.log_scalar("train/rollout_transitions", n_transitions, iteration)
        self.tb_logger.log_scalar("train/update_time", elapsed, iteration)
        self.tb_logger.log_scalar("train/merged_rollouts", merged_count, iteration)

        # Adaptive metrics
        self.tb_logger.log_scalar("adaptive/ent_coef", self.ent_coef, iteration)
        self.tb_logger.log_scalar("adaptive/gamma", self.gamma, iteration)

        # Reward stats from reward_history
        if self.reward_history:
            self.tb_logger.log_scalar("rollout/reward_mean", self.reward_history[-1], iteration)
        if len(self.reward_history) >= 2:
            recent = list(self.reward_history)[-20:]
            self.tb_logger.log_scalar("rollout/reward_mean_20", sum(recent) / len(recent), iteration)

        # Per-hero EMA reward
        for hero_idx, hero_id in enumerate(HERO_IDS):
            self.tb_logger.log_scalar(f"best/{hero_id}_ema", self.ema_reward[hero_id], iteration)
            self.tb_logger.log_scalar(f"best/{hero_id}_best", self.best_reward[hero_id], iteration)

        # Flush to disk immediately (protect against OOM kills)
        self.tb_logger.writer.flush()

        logger.info(
            "Iter %d | %d trans (%d rollouts) | %.1fs | policy=%.4f value=%.4f "
            "ent=%.4f kl=%.4f | ent_coef=%.5f gamma=%.5f",
            iteration, n_transitions, merged_count, elapsed,
            losses.get("policy_loss", 0), losses.get("value_loss", 0),
            losses.get("entropy", 0), losses.get("approx_kl", 0),
            self.ent_coef, self.gamma,
        )

    def _save_checkpoint(self, iteration: int):
        """Save training checkpoint (per-hero models + optimizers + adaptive state)."""
        path = self.save_dir / f"checkpoint_{iteration:06d}.pt"
        torch.save({
            "iteration": iteration,
            "total_transitions": self.total_transitions,
            "model_states": {hid: m.state_dict() for hid, m in self.models.items()},
            "optimizer_states": {hid: o.state_dict() for hid, o in self.optimizers.items()},
            "ema_reward": self.ema_reward,
            "best_reward": self.best_reward,
            "ent_coef": self.ent_coef,
            "gamma": self.gamma,
            "reward_history": list(self.reward_history),
        }, str(path))
        logger.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint and resume state (including adaptive params)."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Support both old single-model and new per-hero checkpoint formats
        if "model_states" in ckpt:
            for hid, state in ckpt["model_states"].items():
                if hid in self.models:
                    self.models[hid].load_state_dict(state)
            for hid, state in ckpt["optimizer_states"].items():
                if hid in self.optimizers:
                    self.optimizers[hid].load_state_dict(state)
        elif "model_state" in ckpt:
            # Legacy: load shared model into all heroes
            for model in self.models.values():
                model.load_state_dict(ckpt["model_state"])
            logger.warning("Loaded legacy shared-model checkpoint into all heroes")

        self.iteration = ckpt.get("iteration", 0)
        self.total_transitions = ckpt.get("total_transitions", 0)
        if "ema_reward" in ckpt:
            self.ema_reward.update(ckpt["ema_reward"])
        if "best_reward" in ckpt:
            self.best_reward.update(ckpt["best_reward"])

        # Restore adaptive state
        self.ent_coef = ckpt.get("ent_coef", self.ent_coef)
        self.gamma = ckpt.get("gamma", self.gamma)
        if "reward_history" in ckpt:
            self.reward_history = deque(ckpt["reward_history"], maxlen=1000)

        logger.info("Loaded checkpoint: %s (iter=%d, ent_coef=%.5f, gamma=%.5f)",
                     path, self.iteration, self.ent_coef, self.gamma)
        # Re-export models so C++ picks up resumed weights
        self._export_model()
