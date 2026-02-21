#!/bin/bash
# FateAnother RL - Inference Server Only (C++ libtorch)
# Waits for model file from trainer, then runs inference server
set -euo pipefail

PORT="${INFERENCE_PORT:-7777}"
ACTION_PORT="${ACTION_PORT:-7778}"
MODEL_DIR="${MODEL_DIR:-/data/models}"
ROLLOUT_DIR="${ROLLOUT_DIR:-/data/rollouts}"
ROLLOUT_SIZE="${ROLLOUT_SIZE:-2048}"
DEVICE="${DEVICE:-cuda}"

echo "=== FateAnother Inference Server ==="
echo "  Port: ${PORT}, Action: ${ACTION_PORT}, Device: ${DEVICE}"
echo "  Model: ${MODEL_DIR}, Rollout: ${ROLLOUT_DIR}, Size: ${ROLLOUT_SIZE}"

# Wait for trainer to create initial model
echo "Waiting for model_latest.pt..."
while [ ! -f "${MODEL_DIR}/model_latest.pt" ]; do
    sleep 2
done
echo "Model found! Starting inference server..."

exec fate_inference_server \
    --port "${PORT}" \
    --action-port "${ACTION_PORT}" \
    --device "${DEVICE}" \
    --model-dir "${MODEL_DIR}" \
    --rollout-dir "${ROLLOUT_DIR}" \
    --rollout-size "${ROLLOUT_SIZE}"
