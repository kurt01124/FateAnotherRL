"""Offline Rollout Trainer: reads .pt rollout files from C++ inference server.

Architecture:
  C++ inference server: collects rollouts in game, writes .pt files
  Python trainer: loads .pt files, runs PPO update, exports TorchScript model
  C++ hot-reloads updated model for next rollout

No TCP, no env module, no Docker/Wine -- pure offline training loop.
"""

import logging
import struct
import time
from itertools import count
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from fateanother_rl.data.constants import NUM_HEROES, HERO_IDS, DISCRETE_HEADS
from fateanother_rl.model.policy import FateModel
from fateanother_rl.model.export import export_model
from fateanother_rl.training.buffer import TensorRolloutBuffer
from fateanother_rl.training.ppo import ppo_loss, get_entropy_coef
from fateanother_rl.utils.logger import Logger

logger = logging.getLogger(__name__)

# C++ ScalarType enum → torch dtype mapping
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


class RolloutTrainer:
    """Offline trainer that reads .pt rollout files from C++ inference server.

    Workflow per iteration:
      1. Wait for rollout_*.pt in rollout_dir
      2. Load tensor data
      3. Build RolloutBuffer from loaded data
      4. Compute GAE advantages
      5. PPO update (same loss, same sequence chunking)
      6. Export TorchScript model for C++ hot-reload
      7. Delete processed rollout file
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
        self.batch_size = int(ppo_cfg.get("batch_size", 64))
        self.seq_len = int(ppo_cfg.get("seq_len", 16))
        self.clip_eps = float(ppo_cfg.get("clip_eps", 0.2))
        self.vf_coef = float(ppo_cfg.get("vf_coef", 0.5))
        self.max_grad_norm = float(ppo_cfg.get("max_grad_norm", 0.5))

        # --- Training config ---
        self.max_iterations = int(train_cfg.get("max_iterations", 100000))
        self.save_interval = int(train_cfg.get("save_interval", 50))
        self.log_interval = int(train_cfg.get("log_interval", 10))
        self.poll_interval = float(train_cfg.get("poll_interval", 1.0))

        # --- GAE config ---
        self.gamma = float(ppo_cfg.get("gamma", 0.99))
        self.gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))

        # --- Logger ---
        self.tb_logger = Logger(log_dir=str(train_cfg.get("log_dir", "runs")))

        # --- Counters ---
        self.iteration = 0
        self.total_transitions = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def train(self):
        """Main training loop. Runs until max_iterations or KeyboardInterrupt."""
        logger.info("=== RolloutTrainer Start ===")
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

                # 1. Wait for rollout file
                rollout_path = self._wait_for_rollout()
                t_start = time.time()

                # 2. Load rollout data (with retry for partial writes)
                logger.info("Loading rollout: %s", rollout_path.name)
                data = None
                for attempt in range(3):
                    try:
                        data = load_fate_rollout(str(rollout_path))
                        break
                    except Exception as e:
                        if attempt < 2:
                            logger.warning("Load attempt %d failed for %s: %s. Retrying in 2s...",
                                           attempt + 1, rollout_path.name, e)
                            time.sleep(2)
                        else:
                            logger.error("Failed to load rollout %s after 3 attempts: %s",
                                         rollout_path.name, e)
                            rollout_path.unlink(missing_ok=True)
                if data is None:
                    continue

                # 3. Build buffer from loaded data
                try:
                    buffer = self._build_buffer(data)
                except Exception as e:
                    logger.error("Failed to build buffer from %s: %s", rollout_path.name, e)
                    rollout_path.unlink(missing_ok=True)
                    continue

                n_transitions = buffer.total_transitions()
                self.total_transitions += n_transitions

                if n_transitions == 0:
                    logger.warning("Empty rollout file: %s. Skipping.", rollout_path.name)
                    rollout_path.unlink(missing_ok=True)
                    continue

                # 4. Compute GAE
                last_values = data.get("bootstrap_values", None)
                buffer.compute_gae(last_values)

                # 5. PPO update
                losses = self._ppo_update(buffer)
                self.iteration += 1
                t_elapsed = time.time() - t_start

                # 6. Export model for C++ hot-reload
                self._export_model()

                # 7. Cleanup processed rollout
                rollout_path.unlink(missing_ok=True)

                # 8. Logging
                if self.iteration % self.log_interval == 0:
                    self._log(self.iteration, losses, n_transitions, t_elapsed, data)

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
    def _wait_for_rollout(self) -> Path:
        """Poll rollout directory for new .pt files.

        Files are sorted by name (expected format: rollout_{timestamp}.pt)
        so oldest files are processed first.

        Returns:
            Path to the first available rollout file.
        """
        while True:
            files = sorted(self.rollout_dir.glob("rollout_*.pt"))
            if files:
                return files[0]
            time.sleep(self.poll_interval)

    def _build_buffer(self, data: dict) -> TensorRolloutBuffer:
        """Convert loaded tensor data into TensorRolloutBuffer (vectorized).

        Uses direct tensor slicing instead of creating per-transition objects.
        ~100x faster than old RolloutBuffer.from_tensor_data().
        """
        return TensorRolloutBuffer(
            data,
            gamma=self.gamma,
            lam=self.gae_lambda,
        )

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------
    def _ppo_update(self, buffer) -> dict:
        """Run per-hero PPO on the given buffer."""
        ent_coef = get_entropy_coef(self.iteration, self.max_iterations)

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
             data: dict = None):
        """Log training metrics."""
        self.tb_logger.log_iteration(iteration, losses)
        self.tb_logger.log_scalar("train/total_transitions", self.total_transitions, iteration)
        self.tb_logger.log_scalar("train/rollout_transitions", n_transitions, iteration)
        self.tb_logger.log_scalar("train/update_time", elapsed, iteration)

        # Log reward and episode stats from rollout data
        if data is not None:
            rewards = data.get("rewards")  # (T, 12)
            dones = data.get("dones")      # (T, 12)
            if rewards is not None:
                rewards_f = rewards.float()
                self.tb_logger.log_scalar("rollout/reward_mean", rewards_f.mean().item(), iteration)
                self.tb_logger.log_scalar("rollout/reward_std", rewards_f.std().item(), iteration)
                self.tb_logger.log_scalar("rollout/reward_min", rewards_f.min().item(), iteration)
                self.tb_logger.log_scalar("rollout/reward_max", rewards_f.max().item(), iteration)
                # Per-agent mean reward
                per_agent = rewards_f.mean(dim=0)  # (12,)
                self.tb_logger.log_scalar("rollout/reward_team0", per_agent[:6].mean().item(), iteration)
                self.tb_logger.log_scalar("rollout/reward_team1", per_agent[6:].mean().item(), iteration)
            if dones is not None:
                n_done = dones.sum().item()
                self.tb_logger.log_scalar("rollout/num_dones", n_done, iteration)
                if rewards is not None and n_done > 0:
                    # Compute episode return: sum of rewards per agent in this rollout
                    ep_return = rewards.float().sum(dim=0).mean().item()  # mean over 12 agents
                    self.tb_logger.log_scalar("rollout/episode_return", ep_return, iteration)
                    T = rewards.shape[0]
                    self.tb_logger.log_scalar("rollout/episode_length", T, iteration)

        logger.info(
            "Iter %d | %d trans | %.1fs | policy=%.4f value=%.4f ent=%.4f kl=%.4f",
            iteration, n_transitions, elapsed,
            losses.get("policy_loss", 0), losses.get("value_loss", 0),
            losses.get("entropy", 0), losses.get("approx_kl", 0),
        )

    def _save_checkpoint(self, iteration: int):
        """Save training checkpoint (per-hero models + optimizers + counters)."""
        path = self.save_dir / f"checkpoint_{iteration:06d}.pt"
        torch.save({
            "iteration": iteration,
            "total_transitions": self.total_transitions,
            "model_states": {hid: m.state_dict() for hid, m in self.models.items()},
            "optimizer_states": {hid: o.state_dict() for hid, o in self.optimizers.items()},
        }, str(path))
        logger.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint and resume state."""
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
        logger.info("Loaded checkpoint: %s (iter=%d)", path, self.iteration)
        # Re-export models so C++ picks up resumed weights
        self._export_model()
