#!/bin/bash
#=============================================================
# FateAnother RL - Docker Distributed Training
#
# 1 Trainer + 5 Inference + 15 WC3 + TensorBoard
# Requires: Docker, nvidia-container-toolkit, War3Client.zip
#=============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# -- Colors --
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

echo -e "${CYAN}"
echo "+===================================================+"
echo "   FateAnother RL - Docker Distributed Training"
echo "   15 WC3 + 5 Inference + 1 Trainer + TB"
echo "+===================================================+"
echo -e "${NC}"

# -- 1. Prerequisites Check --
echo -e "${YELLOW}[1/5] Checking prerequisites...${NC}"

if ! command -v docker &>/dev/null; then
    echo -e "${RED}ERROR: Docker not found. Install Docker first.${NC}"
    exit 1
fi

if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    if ! command -v nvidia-smi &>/dev/null; then
        echo -e "${YELLOW}WARNING: nvidia-smi not found. GPU may not be available.${NC}"
    fi
fi

# Check War3Client.zip OR assembled War3Client/
if [ ! -d "War3Client" ] && [ ! -f "War3Client.zip" ]; then
    echo -e "${RED}ERROR: War3Client.zip not found.${NC}"
    echo "  Place your Warcraft III 1.26a client as War3Client.zip"
    exit 1
fi

echo -e "${GREEN}  [OK] Prerequisites OK${NC}"

# -- 2. Assemble War3Client --
echo -e "${YELLOW}[2/5] Assembling War3Client...${NC}"

if [ ! -f "War3Client/JNLoader.exe" ]; then
    echo "  Running assemble.py..."
    python3 assemble.py --skip-map --skip-rlcomm 2>&1 | tail -5
    echo -e "${GREEN}  [OK] War3Client assembled${NC}"
else
    echo -e "${GREEN}  [OK] War3Client already assembled${NC}"
fi

# Ensure patched map exists
if [ ! -f "War3Client/Maps/rl/fateanother_rl.w3x" ]; then
    echo -e "${YELLOW}  Patching RL map...${NC}"
    if [ -f "MapPatch/rl_patch.py" ] && command -v python3 &>/dev/null; then
        cd MapPatch
        python3 rl_patch.py 2>&1 | tail -5
        mkdir -p ../War3Client/Maps/rl
        cp fateanother_now.w3x ../War3Client/Maps/rl/fateanother_rl.w3x
        cd "$SCRIPT_DIR"
        echo -e "${GREEN}  [OK] Map patched${NC}"
    else
        echo -e "${RED}  ERROR: No patched map and cannot run rl_patch.py${NC}"
        exit 1
    fi
fi

# -- 3. Build Docker Images --
echo -e "${YELLOW}[3/5] Building Docker images...${NC}"

# Base image (Wine + .NET + libGL) - slow first time, cached after
if ! docker image inspect wc3-base:latest &>/dev/null; then
    echo "  Building wc3-base (first time, ~10min)..."
    docker build -f docker/Dockerfile.gpu.base -t wc3-base:latest .
fi

# GPU image (C++ inference server + Python trainer)
echo "  Building fate-gpu..."
docker build -f docker/Dockerfile.gpu -t fate-gpu:latest .

# WC3 image
if [ -f "docker/Dockerfile.wc3" ]; then
    echo "  Building docker-wc3..."
    docker build -f docker/Dockerfile.wc3 -t docker-wc3:latest .
fi

echo -e "${GREEN}  [OK] Docker images built${NC}"

# -- 4. Start Services --
echo -e "${YELLOW}[4/5] Starting services...${NC}"

docker compose -f docker/docker-compose.v4.yml up -d 2>&1 | grep -v "orphan"

echo -e "${GREEN}  [OK] All services started${NC}"

# -- 5. Status --
echo -e "${YELLOW}[5/5] Service status:${NC}"
sleep 5

echo ""
echo -e "${CYAN}  Trainer:${NC}"
docker logs fate-trainer 2>&1 | grep -E "RolloutTrainer|device" | tail -2

echo ""
echo -e "${CYAN}  Inference servers:${NC}"
for i in 1 2 3 4 5; do
    STATUS=$(docker inspect -f '{{.State.Status}}' fate-inf-$i 2>/dev/null || echo "not found")
    echo "    fate-inf-$i: $STATUS"
done

echo ""
echo -e "${CYAN}  WC3 instances:${NC}"
WC3_COUNT=$(docker ps --filter "name=wc3" --format "{{.Names}}" | wc -l)
echo "    $WC3_COUNT containers running"

echo ""
echo -e "${GREEN}=========================================================${NC}"
echo -e "${GREEN}  TensorBoard: http://localhost:6006${NC}"
echo -e "${GREEN}=========================================================${NC}"
echo ""
echo "Commands:"
echo "  docker compose -f docker/docker-compose.v4.yml logs -f trainer"
echo "  docker compose -f docker/docker-compose.v4.yml logs -f inference-1"
echo "  docker compose -f docker/docker-compose.v4.yml down"
