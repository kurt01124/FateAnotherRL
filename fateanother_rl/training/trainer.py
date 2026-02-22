"""Offline Rollout Trainer: reads .pt rollout files from C++ inference server.

Architecture:
  C++ inference server: collects rollouts in game, writes .pt files
  Python trainer: loads .pt files, runs PPO update, exports TorchScript model
  C++ hot-reloads updated model for next rollout

Synchronous batch training: waits for sync_rollouts files (default 15),
merges into one big buffer, then runs PPO. Includes adaptive entropy
coefficient, gamma annealing, and per-hero auto-rollback.
"""

import logging
import struct
import time
from collections import deque
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
from fateanother_rl.training.ppo import ppo_loss
from fateanother_rl.utils.logger import Logger

logger = logging.getLogger(__name__)

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

                # 2. Load all rollouts and merge
                buffers = []
                loaded_paths = []
                for rp in rollout_paths:
                    logger.info("Loading rollout: %s", rp.name)
                    try:
                        data = load_fate_rollout(str(rp))
                        buf = TensorRolloutBuffer(data, gamma=self.gamma, lam=self.gae_lambda)
                        if buf.total_transitions() > 0:
                            buffers.append(buf)
                            loaded_paths.append(rp)
                    except Exception as e:
                        logger.error("Failed to load %s: %s", rp.name, e)

                if not buffers:
                    # All loads failed -- clean up and retry
                    for rp in rollout_paths:
                        rp.unlink(missing_ok=True)
                    continue

                # Merge all buffers
                if len(buffers) == 1:
                    buffer = buffers[0]
                else:
                    buffer = TensorRolloutBuffer.merge(buffers)

                n_transitions = buffer.total_transitions()
                self.total_transitions += n_transitions

                # 3. Compute GAE with current (possibly annealed) gamma
                buffer.gamma = self.gamma
                buffer.compute_gae()

                # 4. PPO update with adaptive entropy
                losses = self._ppo_update(buffer)
                self.iteration += 1
                t_elapsed = time.time() - t_start

                # 5. Adaptive entropy update
                self._adapt_entropy()

                # 6. Gamma annealing
                self._anneal_gamma()

                # 7. Export model for C++ hot-reload
                self._export_model()

                # 8. Update best models + auto rollback check
                self._update_best_and_rollback(buffers)

                # 9. Cleanup processed rollouts
                for rp in loaded_paths:
                    rp.unlink(missing_ok=True)
                # Also clean up any rollouts that failed to load
                for rp in rollout_paths:
                    if rp not in loaded_paths:
                        rp.unlink(missing_ok=True)

                # 10. Logging
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
        """Wait for sync_rollouts files. Proceeds with 80%+ after 10min patience."""
        first_seen = None
        while True:
            files = sorted(self.rollout_dir.glob("rollout_*.pt"))
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
