#!/bin/bash
# Build fate_inference_server on Linux
# Requires: PyTorch (pip install torch) or standalone libtorch

set -e

# Detect libtorch path
TORCH_CMAKE=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null || \
              python -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null)

if [ -z "$TORCH_CMAKE" ]; then
    echo "ERROR: PyTorch not found. Install with: pip install torch"
    exit 1
fi
echo "Using libtorch from: $TORCH_CMAKE"

# Create build directory
mkdir -p build && cd build

# Configure
cmake -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" \
      -DCMAKE_BUILD_TYPE=Release \
      ..

# Build
cmake --build . -j$(nproc)

echo ""
echo "Build successful! Binary: build/fate_inference_server"
