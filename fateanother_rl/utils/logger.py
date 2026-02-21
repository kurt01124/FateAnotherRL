"""TensorBoard logging wrapper for FateAnother RL."""

import logging
from collections import deque
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class Logger:
    """TensorBoard logging wrapper with episode tracking."""

    def __init__(self, log_dir: str = "runs"):
        from torch.utils.tensorboard import SummaryWriter

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_path))
        self._episode_results = deque(maxlen=200)
        self._episode_count = 0

    def log_iteration(self, iteration: int, losses: dict):
        """Log PPO update losses."""
        for key, val in losses.items():
            self.writer.add_scalar(f"loss/{key}", val, iteration)

    def log_episode(self, env_id: int, done_info: dict):
        """Log episode completion."""
        self._episode_count += 1
        ep = self._episode_count

        winner = done_info.get("winner", "unknown")
        score_ally = done_info.get("score_ally", 0)
        score_enemy = done_info.get("score_enemy", 0)

        self.writer.add_scalar("episode/score_ally", score_ally, ep)
        self.writer.add_scalar("episode/score_enemy", score_enemy, ep)

        # Track win/loss for winrate calculation
        if winner == "ally":
            self._episode_results.append(1.0)
        elif winner == "enemy":
            self._episode_results.append(0.0)
        else:
            self._episode_results.append(0.5)

        if len(self._episode_results) >= 10:
            wr = self.recent_winrate()
            self.writer.add_scalar("episode/winrate", wr, ep)

        logger.info(
            "Episode %d | env=%d | winner=%s | score=%d-%d",
            ep, env_id, winner, score_ally, score_enemy,
        )

    def log_rewards(self, iteration: int, mean_reward: float):
        """Log mean episode reward."""
        self.writer.add_scalar("reward/mean", mean_reward, iteration)

    def log_scalar(self, tag: str, value: float, step: int):
        """Log arbitrary scalar."""
        self.writer.add_scalar(tag, value, step)

    def recent_winrate(self, window: int = 100) -> float:
        """Recent winrate over last `window` episodes."""
        if not self._episode_results:
            return 0.5
        recent = list(self._episode_results)[-window:]
        return float(np.mean(recent))

    def close(self):
        """Flush and close writer."""
        self.writer.close()
