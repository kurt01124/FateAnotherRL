#!/bin/bash
# FateAnother RL - Python Trainer Only
# Initializes models, then runs PPO training loop
#
# Environment variables:
#   FRESH_START=1  → wipe all data and start from scratch
#   RESUME=auto    → find latest checkpoint and resume (default)
#   RESUME=<path>  → resume from specific checkpoint
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/data/models}"
ROLLOUT_DIR="${ROLLOUT_DIR:-/data/rollouts}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/checkpoints}"
LOG_DIR="${LOG_DIR:-/data/runs}"
FRESH_START="${FRESH_START:-0}"
RESUME="${RESUME:-auto}"

export PYTHONPATH="/app:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== FateAnother Python Trainer ==="
echo "  Model: ${MODEL_DIR}, Rollout: ${ROLLOUT_DIR}"
echo "  Checkpoint: ${CHECKPOINT_DIR}, Logs: ${LOG_DIR}"
echo "  FRESH_START=${FRESH_START}, RESUME=${RESUME}"

# Fresh start: wipe everything
if [ "${FRESH_START}" = "1" ]; then
    echo "[FRESH] Wiping all data for fresh start..."
    rm -rf "${MODEL_DIR:?}"/* "${CHECKPOINT_DIR:?}"/* "${ROLLOUT_DIR:?}"/* "${LOG_DIR:?}"/*
    echo "[FRESH] All data cleared."
fi

# Initialize models (creates per-hero .pt if not exists)
echo "Initializing models..."
python3 -m fateanother_rl.scripts.init_models --model-dir "${MODEL_DIR}" --device cpu
echo "Models ready."

# Find resume checkpoint
RESUME_ARG=""
if [ "${FRESH_START}" != "1" ] && [ "${RESUME}" = "auto" ]; then
    # Find the latest checkpoint by name (lexicographic = numeric order)
    LATEST=$(ls -1 "${CHECKPOINT_DIR}"/checkpoint_*.pt 2>/dev/null | sort | tail -n1 || true)
    if [ -n "${LATEST}" ]; then
        echo "[RESUME] Auto-detected latest checkpoint: ${LATEST}"
        RESUME_ARG="--resume ${LATEST}"
    else
        echo "[RESUME] No checkpoint found, starting fresh training."
    fi
elif [ "${FRESH_START}" != "1" ] && [ "${RESUME}" != "auto" ] && [ "${RESUME}" != "0" ]; then
    echo "[RESUME] Using specified checkpoint: ${RESUME}"
    RESUME_ARG="--resume ${RESUME}"
fi

echo "Starting training..."
exec python3 -m fateanother_rl.scripts.train \
    ${RESUME_ARG} \
    --training.rollout_dir "${ROLLOUT_DIR}" \
    --training.model_dir "${MODEL_DIR}" \
    --training.save_dir "${CHECKPOINT_DIR}" \
    --training.log_dir "${LOG_DIR}"
