#!/bin/bash
# MLPerf GPU Benchmark - Build and Run Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}MLPerf GPU Benchmark Setup${NC}"
echo "======================================"

# Check for Podman
if ! command -v podman &> /dev/null; then
    echo -e "${RED}Error: Podman not found${NC}"
    echo "Install with: sudo dnf install -y podman"
    exit 1
fi

CONTAINER_CMD="podman"
COMPOSE_CMD="podman-compose"
echo -e "${GREEN}Using Podman${NC}"

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}Warning: nvidia-smi not found. GPU may not be available.${NC}"
else
    echo -e "${GREEN}GPU detected:${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi

# Create results directory
mkdir -p results
echo "Results directory: $SCRIPT_DIR/results"

# Function to display usage
usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build       - Build the container image"
    echo "  run         - Run the benchmark (default 4 hours)"
    echo "  run-short   - Run quick test (1 hour)"
    echo "  run-long    - Run extended test (8 hours)"
    echo "  stop        - Stop running benchmark"
    echo "  logs        - Show benchmark logs"
    echo "  clean       - Remove container and results"
    echo "  shell       - Open shell in container"
    echo ""
}

# Build image
build() {
    echo -e "${GREEN}Building MLPerf benchmark image...${NC}"
    $CONTAINER_CMD build -f Containerfile -t mlperf-benchmark:latest .
    echo -e "${GREEN}Build complete!${NC}"
}

# Run benchmark
run_benchmark() {
    local duration=${1:-4}
    
    echo -e "${GREEN}Starting MLPerf benchmark (${duration}h duration)...${NC}"
    
    # Update config
    sed -i "s/duration_hours:.*/duration_hours: $duration/" config.yaml
    
    # Run container with Podman
    $CONTAINER_CMD run -d \
        --name mlperf-gpu-benchmark \
        --device nvidia.com/gpu=all \
        --security-opt=label=disable \
	--replace \
        -v "$SCRIPT_DIR/results:/workspace/results:Z" \
        -v "$SCRIPT_DIR/config.yaml:/workspace/config.yaml:ro,Z" \
        -v "$SCRIPT_DIR/run_benchmark.py:/workspace/run_benchmark.py:ro,Z" \
        -e CUDA_VISIBLE_DEVICES=0 \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        --annotation=run.oci.keep_original_groups=1 \
        mlperf-benchmark:latest
    
    echo -e "${GREEN}Benchmark started!${NC}"
    echo "Container: mlperf-gpu-benchmark"
    echo "To view logs: $0 logs"
    echo "To stop: $0 stop"
}

# Stop benchmark
stop() {
    echo -e "${YELLOW}Stopping benchmark...${NC}"
    $CONTAINER_CMD stop mlperf-gpu-benchmark 2>/dev/null || true
    $CONTAINER_CMD rm mlperf-gpu-benchmark 2>/dev/null || true
    echo -e "${GREEN}Stopped${NC}"
}

# Show logs
logs() {
    if $CONTAINER_CMD ps -a | grep -q mlperf-gpu-benchmark; then
        $CONTAINER_CMD logs -f mlperf-gpu-benchmark
    else
        echo -e "${RED}Container not running${NC}"
        if [ -f results/benchmark.log ]; then
            echo -e "${YELLOW}Showing latest log file:${NC}"
            tail -f results/benchmark.log
        fi
    fi
}

# Clean up
clean() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    stop
    read -p "Remove results directory? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf results/*
        echo -e "${GREEN}Results cleaned${NC}"
    fi
}

# Shell access
shell() {
    if ! $CONTAINER_CMD ps | grep -q mlperf-gpu-benchmark; then
        echo -e "${YELLOW}Starting temporary container...${NC}"
        $CONTAINER_CMD run -it --rm \
            --device nvidia.com/gpu=all \
            --security-opt=label=disable \
            --annotation=run.oci.keep_original_groups=1 \
            -v "$SCRIPT_DIR/results:/workspace/results:Z" \
            -v "$SCRIPT_DIR/run_benchmark.py:/workspace/run_benchmark.py:ro,Z" \
            mlperf-benchmark:latest /bin/bash
    else
        $CONTAINER_CMD exec -it mlperf-gpu-benchmark /bin/bash
    fi
}

# Main command handling
case "${1:-}" in
    build)
        build
        ;;
    run)
        build
        run_benchmark 4
        ;;
    run-short)
        build
        run_benchmark 1
        ;;
    run-long)
        build
        run_benchmark 8
        ;;
    stop)
        stop
        ;;
    logs)
        logs
        ;;
    clean)
        clean
        ;;
    shell)
        shell
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage
        exit 1
        ;;
esac
