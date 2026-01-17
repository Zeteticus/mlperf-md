#!/bin/bash
# MLPerf GPU Benchmark - Direct Device Mapping Version
# Use this if CDI (nvidia.com/gpu=all) doesn't work

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}MLPerf GPU Benchmark (Direct Device Mode)${NC}"
echo "======================================"

# Check for Podman
if ! command -v podman &> /dev/null; then
    echo -e "${RED}Error: Podman not found${NC}"
    exit 1
fi

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}Warning: nvidia-smi not found${NC}"
else
    echo -e "${GREEN}GPU detected:${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi
echo ""

# Verify device files exist
echo "Checking GPU device files..."
for dev in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm; do
    if [ -e "$dev" ]; then
        echo -e "${GREEN}✓ $dev${NC}"
    else
        echo -e "${RED}✗ $dev not found${NC}"
    fi
done
echo ""

# Create results directory
mkdir -p results

# Build if needed
if ! podman image exists mlperf-benchmark:latest; then
    echo "Building container..."
    podman build -f Containerfile -t mlperf-benchmark:latest .
fi

# Stop any existing container
podman stop mlperf-gpu-benchmark 2>/dev/null || true
podman rm mlperf-gpu-benchmark 2>/dev/null || true

# Get duration
DURATION=${1:-4}
echo "Benchmark duration: ${DURATION} hours"

# Update config
sed -i "s/duration_hours:.*/duration_hours: $DURATION/" config.yaml

# Run with direct device mapping
echo "Starting benchmark..."
podman run -d \
    --name mlperf-gpu-benchmark \
    --device /dev/nvidia0:/dev/nvidia0 \
    --device /dev/nvidiactl:/dev/nvidiactl \
    --device /dev/nvidia-uvm:/dev/nvidia-uvm \
    --device /dev/nvidia-modeset:/dev/nvidia-modeset \
    --security-opt=label=disable \
    --group-add=keep-groups \
    -v "$SCRIPT_DIR/results:/workspace/results:Z" \
    -v "$SCRIPT_DIR/config.yaml:/workspace/config.yaml:ro,Z" \
    -v "$SCRIPT_DIR/run_benchmark.py:/workspace/run_benchmark.py:ro,Z" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    mlperf-benchmark:latest

echo -e "${GREEN}Benchmark started!${NC}"
echo ""
echo "Commands:"
echo "  View logs:  podman logs -f mlperf-gpu-benchmark"
echo "  Stop:       podman stop mlperf-gpu-benchmark"
echo "  Shell:      podman exec -it mlperf-gpu-benchmark /bin/bash"
echo ""
echo "Results will be in: $SCRIPT_DIR/results/"
