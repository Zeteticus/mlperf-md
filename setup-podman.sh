#!/bin/bash
# Quick Setup for Podman on RHEL 9/10

set -e

echo "========================================"
echo "MLPerf GPU Benchmark - Podman Setup"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if running on RHEL
if [ -f /etc/redhat-release ]; then
    echo -e "${GREEN}Detected RHEL:${NC}"
    cat /etc/redhat-release
else
    echo -e "${YELLOW}Warning: Not running on RHEL${NC}"
fi
echo ""

# Step 1: Check NVIDIA drivers
echo "Step 1: Checking NVIDIA drivers..."
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA drivers installed${NC}"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo -e "${RED}✗ NVIDIA drivers not found${NC}"
    echo "Install with: sudo dnf install -y nvidia-driver nvidia-driver-cuda"
    exit 1
fi
echo ""

# Step 2: Check Podman
echo "Step 2: Checking Podman..."
if command -v podman &> /dev/null; then
    echo -e "${GREEN}✓ Podman installed${NC}"
    podman --version
else
    echo -e "${YELLOW}! Podman not found. Installing...${NC}"
    sudo dnf install -y podman
fi
echo ""

# Step 3: Check NVIDIA Container Toolkit
echo "Step 3: Checking NVIDIA Container Toolkit..."
if [ -f /etc/cdi/nvidia.yaml ]; then
    echo -e "${GREEN}✓ NVIDIA Container Toolkit configured${NC}"
else
    echo -e "${YELLOW}! NVIDIA Container Toolkit not configured${NC}"
    read -p "Install NVIDIA Container Toolkit? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Adding NVIDIA repository..."
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
            sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
        
        echo "Installing toolkit..."
        sudo dnf install -y nvidia-container-toolkit
        
        echo "Generating CDI configuration..."
        sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
        
        echo -e "${GREEN}✓ Installation complete${NC}"
    fi
fi
echo ""

# Step 4: Test GPU access in container
echo "Step 4: Testing GPU access in container..."

# First, check if CDI devices are listed
echo "Checking for CDI devices..."
if [ -f /etc/cdi/nvidia.yaml ]; then
    DEVICE_COUNT=$(grep -c "name: nvidia.com/gpu" /etc/cdi/nvidia.yaml 2>/dev/null || echo "0")
    echo "Found $DEVICE_COUNT GPU device(s) in CDI configuration"
fi

# Try multiple methods to access GPU
TEST_PASSED=false

# Method 1: CDI with nvidia.com/gpu=all
echo "Testing Method 1: CDI (nvidia.com/gpu=all)..."
if podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.3.1-base-ubi9 nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ GPU accessible via CDI${NC}"
    TEST_PASSED=true
# Method 2: Direct device mapping (fallback)
elif podman run --rm \
    --device /dev/nvidia0 \
    --device /dev/nvidiactl \
    --device /dev/nvidia-uvm \
    --security-opt=label=disable \
    nvidia/cuda:12.3.1-base-ubi9 nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ GPU accessible via direct device mapping${NC}"
    echo -e "${YELLOW}Note: Using direct device mapping instead of CDI${NC}"
    TEST_PASSED=true
else
    echo -e "${RED}✗ GPU not accessible in containers${NC}"
    echo ""
    echo "Troubleshooting steps:"
    echo "  1. Verify nvidia-smi works on host:"
    echo "     nvidia-smi"
    echo ""
    echo "  2. Check CDI configuration exists:"
    echo "     ls -l /etc/cdi/nvidia.yaml"
    echo ""
    echo "  3. Try regenerating CDI:"
    echo "     sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
    echo ""
    echo "  4. Check device files exist:"
    echo "     ls -l /dev/nvidia*"
    echo ""
    echo "  5. Try direct device mapping:"
    echo "     podman run --rm --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm nvidia/cuda:12.3.1-base-ubi9 nvidia-smi"
fi

if [ "$TEST_PASSED" = false ]; then
    exit 1
fi
echo ""

# Step 5: Create results directory
echo "Step 5: Setting up workspace..."
mkdir -p results
chmod 755 results
echo -e "${GREEN}✓ Results directory created${NC}"
echo ""

# Step 6: Build container
echo "Step 6: Building benchmark container..."
read -p "Build container now? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Building (this may take 10-15 minutes)..."
    podman build -f Containerfile -t mlperf-benchmark:latest .
    echo -e "${GREEN}✓ Container built successfully${NC}"
fi
echo ""

# Summary
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Review config: vim config.yaml"
echo "  2. Run benchmark: ./benchmark.sh run"
echo "  3. Monitor logs:  ./benchmark.sh logs"
echo ""
echo "Quick test (1 hour): ./benchmark.sh run-short"
echo ""
echo "For detailed documentation, see:"
echo "  - README-PODMAN.md (Podman-specific guide)"
echo "  - README.md (Full documentation)"
echo ""
