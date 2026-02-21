"""Event-based reward calculator for FateAnother RL v2.

Processes C# events (KILL, CREEP, LVUP) and per-tick penalties
to compute per-agent rewards.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_REWARD_CONFIG = {
    "kill_personal": 3.0,
    "kill_team": 1.0,
    "death_personal": -2.0,
    "death_team": -0.5,
    "creep": 0.2,
    "levelup": 0.5,
    "idle_penalty": -0.01,
    "skill_points_held": -0.02,
    "stat_points_held": -0.01,
}


def get_reward_weights(iteration: int, max_iterations: int = 100000) -> dict:
    """Soft reward weight schedule based on training progress.

    Early: higher personal, lower team
    Late: higher team bonuses
    """
    progress = min(iteration / max(max_iterations, 1), 1.0)
    return {
        "kill_personal": 3.0,
        "kill_team": 0.2 + 1.3 * progress,        # 0.2 -> 1.5
        "death_personal": -2.0,
        "death_team": -0.1 - 0.4 * progress,       # -0.1 -> -0.5
        "creep": 0.3 - 0.1 * progress,              # 0.3 -> 0.2
        "levelup": 0.5,
        "idle_penalty": -0.01,
        "skill_points_held": -0.02,
        "stat_points_held": -0.01,
    }


class RewardCalculator:
    """Compute per-agent rewards from events and state.

    Unlike v1 which estimated kills from HP transitions, v2 uses
    explicit events from C# (KILL, CREEP_KILL, LEVEL_UP).
    """

    def __init__(self, config: dict | None = None):
        self.cfg = DEFAULT_REWARD_CONFIG.copy()
        if config:
            self.cfg.update(config)
        self.prev_positions = {}  # agent_idx -> (x, y) for idle detection

    def reset(self):
        """Reset state for new episode."""
        self.prev_positions = {}

    def compute(self, state: dict, iteration: int = 0) -> np.ndarray:
        """Compute rewards for all 12 agents in one tick.

        Args:
            state: Full STATE JSON dict (with units, events, global)
            iteration: Training iteration (for reward weight scheduling)

        Returns:
            np.ndarray of shape (12,) with per-agent rewards.
        """
        rewards = np.zeros(12, dtype=np.float32)
        units = state.get("units", [])
        events = state.get("events", [])

        if len(units) != 12:
            return rewards

        # Get reward weights for current iteration
        weights = get_reward_weights(iteration)

        # --- Process events ---
        for ev in events:
            ev_type = ev.get("type", "")

            if ev_type == "KILL":
                killer = ev.get("killer", -1)
                victim = ev.get("victim", -1)
                if 0 <= killer < 12 and 0 <= victim < 12:
                    killer_team = 0 if killer < 6 else 1

                    # Personal kill reward to killer
                    rewards[killer] += weights["kill_personal"]

                    # Team kill reward
                    for i in range(6):
                        rewards[killer_team * 6 + i] += weights["kill_team"]

                    # Death penalty to victim
                    victim_team = 0 if victim < 6 else 1
                    rewards[victim] += weights.get("death_personal", -2.0)

                    # Team death penalty
                    for i in range(6):
                        rewards[victim_team * 6 + i] += weights["death_team"]

            elif ev_type == "CREEP_KILL":
                killer = ev.get("killer", -1)
                if 0 <= killer < 12:
                    rewards[killer] += weights["creep"]

            elif ev_type == "LEVEL_UP":
                unit_idx = ev.get("unit_idx", -1)
                if 0 <= unit_idx < 12:
                    rewards[unit_idx] += weights["levelup"]

        # --- Per-tick penalties ---
        for i, unit in enumerate(units):
            if not unit.get("alive", False):
                continue

            # Idle penalty (moved less than threshold)
            x, y = unit.get("x", 0), unit.get("y", 0)
            prev = self.prev_positions.get(i)
            if prev is not None:
                dx = x - prev[0]
                dy = y - prev[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < 10.0:  # Threshold for "idle"
                    rewards[i] += self.cfg["idle_penalty"]
            self.prev_positions[i] = (x, y)

            # Skill points held penalty
            sp = unit.get("skill_points", 0)
            if sp > 0:
                rewards[i] += self.cfg["skill_points_held"] * sp

            # Stat points held penalty
            stat_p = unit.get("stat_points", 0)
            if stat_p > 0:
                rewards[i] += self.cfg["stat_points_held"] * min(stat_p, 10)

        return rewards

    def compute_terminal(self, done_info: dict) -> np.ndarray:
        """Compute terminal rewards from DONE info.

        In v2, terminal rewards are minimal (kills already rewarded).
        Only timeout-draw gets a small penalty.
        """
        rewards = np.zeros(12, dtype=np.float32)
        reason = done_info.get("reason", "")

        if reason == "timeout_draw" or done_info.get("winner") == "draw":
            # Small penalty to discourage passive play
            rewards[:] = -2.0

        return rewards
