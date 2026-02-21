"""Self-play manager: checkpoint pool and opponent loading."""

import logging
import os
import random

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SelfPlayManager:
    """Manages self-play checkpoints and opponent selection.

    Modes:
        pure: Both teams use the same current model. Both teams' experience
              is collected (2x data). This is the default mode.
        pool: One team uses current model, opponent is randomly selected
              from a pool of past checkpoints.
    """

    def __init__(self, mode: str = "pure", pool_size: int = 20,
                 checkpoint_dir: str = "checkpoints"):
        self.mode = mode
        self.pool_size = pool_size
        self.checkpoint_dir = checkpoint_dir
        self.pool: list[tuple[int, str]] = []  # (iteration, path)
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, model: nn.Module, iteration: int):
        """Save model checkpoint and add to opponent pool."""
        path = os.path.join(self.checkpoint_dir, f"model_{iteration:06d}.pth")
        torch.save(model.state_dict(), path)
        self.pool.append((iteration, path))
        if len(self.pool) > self.pool_size:
            self.pool.pop(0)
        logger.info("Self-play checkpoint saved: %s (pool size=%d)",
                     path, len(self.pool))

    def load_opponent(self, model_class: type) -> nn.Module | None:
        """Load a random opponent from pool. Returns None in pure mode."""
        if self.mode == "pure" or len(self.pool) == 0:
            return None

        _, path = random.choice(self.pool)
        opp = model_class()
        opp.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        opp.eval()
        logger.info("Loaded opponent from: %s", path)
        return opp

    def switch_to_pool(self):
        """Switch from pure to pool self-play mode."""
        self.mode = "pool"
        logger.info("Self-play mode switched to: pool")
