#!/bin/bash
#=============================================================
# FateAnother RL - Linux Native Training
#
# 1 Inference Server + 1 Trainer (WC3 runs separately via Wine)
# Requires: Python 3.10+, PyTorch, cmake, g++
#=============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# -- Colors --
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

echo -e "${CYAN}"
echo "+===================================================+"
echo "   FateAnother RL - Linux Native Training"
echo "   1 Inference + 1 Trainer"
echo "+===================================================+"
echo -e "${NC}"

# -- Cleanup on exit --
PIDS=()
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo -e "${GREEN}Done.${NC}"
}
trap cleanup EXIT INT TERM

# -- 1. Prerequisites --
echo -e "${YELLOW}[1/5] Checking prerequisites...${NC}"

PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "${RED}ERROR: Python not found. Install Python 3.10+${NC}"
    exit 1
fi

$PYTHON -c "import torch" 2>/dev/null || {
    echo -e "${RED}ERROR: PyTorch not found. Run: pip install torch${NC}"
    exit 1
}

if ! command -v cmake &>/dev/null; then
    echo -e "${RED}ERROR: cmake not found. Install cmake.${NC}"
    exit 1
fi

DEVICE=$($PYTHON -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null)
if [ "$DEVICE" = "cuda" ]; then
    echo -e "${GREEN}  [OK] CUDA available${NC}"
else
    echo -e "${YELLOW}  [!] CUDA not available, using CPU (slower)${NC}"
fi

echo -e "${GREEN}  [OK] Prerequisites OK${NC}"

# -- 2. Build Inference Server --
echo -e "${YELLOW}[2/5] Building C++ inference server...${NC}"

if [ ! -f "inference_server/build/fate_inference_server" ]; then
    cd inference_server
    bash build_linux.sh
    cd "$SCRIPT_DIR"
fi

if [ ! -f "inference_server/build/fate_inference_server" ]; then
    echo -e "${RED}ERROR: Build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}  [OK] fate_inference_server built${NC}"

# -- 3. Data Directories --
echo -e "${YELLOW}[3/5] Setting up data directories...${NC}"

mkdir -p data/{models,rollouts,checkpoints,runs}
echo -e "${GREEN}  [OK] Data directories ready${NC}"

# -- 4. Initialize Models --
echo -e "${YELLOW}[4/5] Initializing models...${NC}"

if [ ! -f "data/models/model_latest.pt" ]; then
    $PYTHON -m fateanother_rl.scripts.init_models --model-dir data/models
    echo -e "${GREEN}  [OK] Models initialized${NC}"
else
    echo -e "${GREEN}  [OK] Models already exist${NC}"
fi

# -- 5. Start Services --
echo -e "${YELLOW}[5/5] Starting services...${NC}"

# Inference Server
./inference_server/build/fate_inference_server \
    --port 7777 --action-port 7778 \
    --device "$DEVICE" \
    --model-dir data/models \
    --rollout-dir data/rollouts \
    --rollout-size 2048 \
    > data/inference.log 2>&1 &
PIDS+=($!)
echo -e "${GREEN}  [OK] Inference server PID=$! (port 7777/7778)${NC}"

sleep 2

# Python Trainer
$PYTHON -m fateanother_rl.scripts.train \
    --rollout-dir data/rollouts \
    --model-dir data/models \
    --save-dir data/checkpoints \
    --log-dir data/runs \
    > data/trainer.log 2>&1 &
PIDS+=($!)
echo -e "${GREEN}  [OK] Trainer PID=$!${NC}"

# TensorBoard (optional)
if $PYTHON -c "import tensorboard" 2>/dev/null; then
    $PYTHON -m tensorboard.main --logdir data/runs --port 6006 --bind_all \
        > /dev/null 2>&1 &
    PIDS+=($!)
    echo -e "${GREEN}  [OK] TensorBoard (http://localhost:6006)${NC}"
fi

echo ""
echo -e "${GREEN}=========================================================${NC}"
echo -e "${GREEN}  All services running!${NC}"
echo -e "${GREEN}  TensorBoard: http://localhost:6006${NC}"
echo -e "${GREEN}=========================================================${NC}"
echo ""
echo "  Logs:"
echo "    tail -f data/inference.log"
echo "    tail -f data/trainer.log"
echo ""
echo "  WC3 must be started separately (Wine):"
echo "    cd War3Client && wine JNLoader.exe -loadfile 'Maps\\rl\\fateanother_rl.w3x' -window"
echo ""
echo "  Press Ctrl+C to stop all services."
echo ""

# Wait for processes
wait
