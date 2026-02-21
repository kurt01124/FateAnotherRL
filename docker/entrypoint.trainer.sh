#!/bin/bash
# FateAnother RL - Python Trainer Only
# Initializes models, then runs PPO training loop
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/data/models}"
ROLLOUT_DIR="${ROLLOUT_DIR:-/data/rollouts}"

export PYTHONPATH="/app:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== FateAnother Python Trainer ==="
echo "  Model: ${MODEL_DIR}, Rollout: ${ROLLOUT_DIR}"

# Initialize models (creates model_latest.pt if not exists)
echo "Initializing models..."
python3 -m fateanother_rl.scripts.init_models --model-dir "${MODEL_DIR}" --device cpu
echo "Models ready."

echo "Starting training..."
exec python3 -m fateanother_rl.scripts.train \
    --training.rollout_dir "${ROLLOUT_DIR}" \
    --training.model_dir "${MODEL_DIR}" \
    --training.save_dir /data/checkpoints \
    --training.log_dir /data/runs
