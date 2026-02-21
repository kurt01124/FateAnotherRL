"""FateAnother RL Offline Training Entry Point.

Reads .pt rollout files from C++ inference server and runs PPO updates.
Exports TorchScript models for C++ hot-reload after each iteration.

Usage:
    python -m fateanother_rl.scripts.train
    python -m fateanother_rl.scripts.train --config fateanother_rl/config/default.yaml
    python -m fateanother_rl.scripts.train --training.rollout_dir /data/rollouts
    python -m fateanother_rl.scripts.train --training.model_dir /data/models

Config priority: CLI overrides > YAML file > hardcoded defaults
"""

import argparse
import ast
import logging
import os
import sys

import yaml

# ---------------------------------------------------------------------------
# Add project root to PYTHONPATH
# ---------------------------------------------------------------------------
_THIS_FILE = os.path.abspath(__file__)
_SCRIPTS_DIR = os.path.dirname(_THIS_FILE)
_RL_PKG_DIR = os.path.dirname(_SCRIPTS_DIR)       # fateanother_rl/
_PROJECT_ROOT = os.path.dirname(_RL_PKG_DIR)       # FateAnother/

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = os.path.join(_RL_PKG_DIR, "config", "default.yaml")


# ---------------------------------------------------------------------------
# CLI override parser
# ---------------------------------------------------------------------------

def _apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply dotted-key CLI overrides to config dict."""
    i = 0
    while i < len(overrides):
        raw_key = overrides[i]
        if not raw_key.startswith("--"):
            i += 1
            continue

        key_path = raw_key[2:]
        if "=" in key_path:
            key_path, raw_val = key_path.split("=", 1)
        else:
            i += 1
            if i >= len(overrides):
                logger.warning("Override key '%s' has no value, skipping.", raw_key)
                continue
            raw_val = overrides[i]

        value = _parse_value(raw_val)

        parts = key_path.split(".")
        d = config
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
        logger.info("Override: %s = %r", key_path, value)

        i += 1

    return config


def _parse_value(s: str):
    """Auto-convert string to Python value."""
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.startswith("[") or s.startswith("{"):
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            pass
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FateAnother RL v2 Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        help=f"YAML config file path (default: {_DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="CKPT_PATH",
        help="Checkpoint file path to resume from",
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args, extra = parser.parse_known_args()
    logging.getLogger().setLevel(getattr(logging, args.loglevel))

    # Load YAML config
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded config: %s", config_path)

    # Apply CLI overrides
    if extra:
        config = _apply_overrides(config, extra)

    # Import trainer (after PYTHONPATH setup)
    from fateanother_rl.training.trainer import RolloutTrainer

    trainer = RolloutTrainer(config)

    # Resume from checkpoint
    if args.resume:
        logger.info("Resuming from: %s", args.resume)
        trainer.load_checkpoint(args.resume)

    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.exception("Unhandled exception: %s", e)
        try:
            trainer._save_checkpoint(trainer.iteration)
        except Exception as save_err:
            logger.error("Emergency checkpoint save failed: %s", save_err)
        sys.exit(1)


if __name__ == "__main__":
    main()
