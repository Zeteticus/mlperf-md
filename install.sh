#!/bin/bash
# Install MLPerf benchmark files to current directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "MLPerf Benchmark - File Installation"
echo "====================================="
echo ""
echo "This will copy the benchmark files to: $(pwd)"
echo ""

# List of required files
FILES=(
    "Containerfile"
    "Dockerfile"
    "config.yaml"
    "run_benchmark.py"
    "benchmark.sh"
    "run-direct-device.sh"
    "setup-podman.sh"
    "README.md"
    "README-PODMAN.md"
    "TROUBLESHOOTING.md"
)

# Check if any files would be overwritten
OVERWRITE=false
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        OVERWRITE=true
        break
    fi
done

if [ "$OVERWRITE" = true ]; then
    echo -e "${YELLOW}Warning: Some files already exist in this directory.${NC}"
    read -p "Overwrite existing files? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
fi

# Copy files
echo "Copying files..."
for file in "${FILES[@]}"; do
    if [ -f "$SCRIPT_DIR/$file" ]; then
        cp "$SCRIPT_DIR/$file" .
        echo "  ✓ $file"
    else
        echo "  ✗ $file (not found in source)"
    fi
done

# Make scripts executable
chmod +x benchmark.sh run-direct-device.sh setup-podman.sh 2>/dev/null || true

# Create results directory
mkdir -p results
echo "  ✓ results/ directory"

echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "Files installed to: $(pwd)"
echo ""
echo "Next steps:"
echo "  1. Build container:  ./benchmark.sh build"
echo "  2. Run benchmark:    ./benchmark.sh run"
echo "  or"
echo "  2. Run direct:       ./run-direct-device.sh 4"
echo ""
