"""v2 Environment test: connect to Docker container, validate STATE/ACTION cycle.

Usage:
    python -m fateanother_rl.scripts.test_env [host] [port] [max_ticks]
    Default: localhost 9999 500
"""

import sys
import os
import time
import random
import logging

# Add project root
_THIS_FILE = os.path.abspath(__file__)
_SCRIPTS_DIR = os.path.dirname(_THIS_FILE)
_RL_PKG_DIR = os.path.dirname(_SCRIPTS_DIR)
_PROJECT_ROOT = os.path.dirname(_RL_PKG_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 9999
    max_ticks = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    from fateanother_rl.env.wc3_env import WC3Env
    from fateanother_rl.env.state_parser import split_perspectives, batch_observations
    from fateanother_rl.model.action_utils import actions_to_json
    from fateanother_rl.training.reward import RewardCalculator

    print(f"=== v2 Env Test: {host}:{port} (max {max_ticks} ticks) ===")

    env = WC3Env(host, port)
    reward_calc = RewardCalculator()

    try:
        env.connect(max_retries=60, interval=2.0)
        print("Connected!")
    except RuntimeError as e:
        print(f"Connection failed: {e}")
        return

    tick = 0
    total_rewards = [0.0] * 12
    t_start = time.time()

    try:
        while tick < max_ticks:
            state = env.recv_state()

            if state is None:
                elapsed = time.time() - t_start
                print(f"\nDONE received at tick {tick} ({elapsed:.1f}s)")
                print(f"Episode info: {env.episode_info}")
                break

            tick += 1

            # Test state parsing
            agent_obs = split_perspectives(state)
            if not agent_obs:
                print(f"WARNING: split_perspectives returned empty at tick {tick}")
                continue

            # Test reward computation
            rewards = reward_calc.compute(state, iteration=0)
            for i in range(12):
                total_rewards[i] += rewards[i]

            # Print summary periodically
            if tick % 50 == 1 or tick == 1:
                units = state["units"]
                alive_t0 = sum(1 for u in units[:6] if u.get("alive"))
                alive_t1 = sum(1 for u in units[6:] if u.get("alive"))
                g = state.get("global", {})
                events = state.get("events", [])
                print(f"  tick={tick:5d} gt={g.get('game_time',0):.1f} "
                      f"score={g.get('score_ally',0)}-{g.get('score_enemy',0)} "
                      f"alive={alive_t0}v{alive_t1} events={len(events)} "
                      f"self_dim={agent_obs[0].self_vec.shape} "
                      f"ally_dim={agent_obs[0].ally_vecs.shape}")

                for ev in events:
                    print(f"    EVENT: {ev}")

            # Send random move actions
            actions = {}
            for i in range(12):
                actions[i] = {
                    "move": [random.uniform(-1, 1), random.uniform(-1, 1)],
                    "point": [0.0, 0.0],
                    "skill": 0,
                    "unit_target": 0,
                    "skill_levelup": 0,
                    "stat_upgrade": 0,
                    "attribute": 0,
                    "item_buy": 0,
                    "item_use": 0,
                    "seal_use": 0,
                    "faire_send": 0,
                    "faire_request": 0,
                    "faire_respond": 0,
                }

            action_json = actions_to_json(actions, list(range(12)))
            env.send_action(action_json)

    except KeyboardInterrupt:
        print(f"\nInterrupted at tick {tick}")
    except Exception as e:
        print(f"\nError at tick {tick}: {e}")

    elapsed = time.time() - t_start
    print(f"\n=== Summary ===")
    print(f"Ticks: {tick}")
    print(f"Time: {elapsed:.1f}s ({tick / max(elapsed, 0.01):.1f} ticks/s)")
    print(f"Total rewards: {[f'{r:.2f}' for r in total_rewards]}")
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
