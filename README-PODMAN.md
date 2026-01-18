# MLPerf GPU Benchmark with Podman (RHEL 9/10)

Containerized MLPerf inference benchmark for extended GPU stress testing on RHEL systems using Podman.

## Quick Start

```bash
# Build and run 4-hour benchmark
./benchmark.sh run

# View logs
./benchmark.sh logs

# Stop
./benchmark.sh stop
```

## Prerequisites

### 1. Install Podman

```bash
# RHEL 9/10
sudo dnf install -y podman

# Verify installation
podman --version
```

### 2. Install NVIDIA Drivers

```bash
# Install NVIDIA drivers (if not already installed)
sudo dnf install -y kernel-devel kernel-headers gcc make
sudo dnf install -y nvidia-driver nvidia-driver-cuda

# Verify
nvidia-smi
```

### 3. Install NVIDIA Container Toolkit

```bash
# Add NVIDIA repository
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# Install toolkit
sudo dnf install -y nvidia-container-toolkit

# Configure for Podman
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# Verify
podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.3.1-base-ubi9 nvidia-smi
```

## Usage

### Basic Commands

```bash
# Build and run benchmarks
./benchmark.sh build          # Build container image
./benchmark.sh run            # Run 4-hour benchmark
./benchmark.sh run-short      # Run 1-hour test
./benchmark.sh run-long       # Run 8-hour test

# Monitoring
./benchmark.sh logs           # Follow logs
./benchmark.sh shell          # Open shell in container

# Cleanup
./benchmark.sh stop           # Stop benchmark
./benchmark.sh clean          # Remove containers and results
```

### Manual Podman Commands

```bash
# Build
podman build -t mlperf-benchmark:latest .

# Run
podman run -d \
    --name mlperf-gpu-benchmark \
    --device nvidia.com/gpu=all \
    --security-opt=label=disable \
    -v ./results:/workspace/results:Z \
    -v ./config.yaml:/workspace/config.yaml:ro,Z \
    -e NVIDIA_VISIBLE_DEVICES=all \
    mlperf-benchmark:latest

# Monitor
podman logs -f mlperf-gpu-benchmark

# Stop
podman stop mlperf-gpu-benchmark
podman rm mlperf-gpu-benchmark
```

## Configuration

Edit `config.yaml`:

```yaml
# Test duration (hours)
duration_hours: 4

# Inference scenarios
scenarios:
  - SingleStream  # Latency-focused
  - Offline       # Throughput-focused

# Max iterations
iterations: 10
```

## Benchmarks

**ResNet50** (Image Classification)
- SingleStream: Batch size 1, 1000 iterations
- Offline: Batch size 8, 1000 iterations
- Metrics: FPS, latency percentiles

**BERT-base** (NLP)
- SingleStream: Batch size 1, 500 iterations  
- Offline: Batch size 4, 500 iterations
- Metrics: QPS, latency percentiles

## Results

All results saved to `results/` directory:

```
results/
├── benchmark.log              # Execution log
├── benchmark_summary.json     # Aggregated results
├── resnet50_*.json           # ResNet50 results
└── bert_*.json               # BERT results
```

### Example Result

```json
{
  "model": "ResNet50",
  "scenario": "Offline",
  "batch_size": 8,
  "mean_latency_ms": 12.34,
  "p99_latency_ms": 15.80,
  "throughput_fps": 648.5
}
```

## Monitoring

### GPU Utilization

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Detailed metrics
nvidia-smi dmon -s pucvmet
```

### Container Status

```bash
# Check container
podman ps -a

# Resource usage
podman stats mlperf-gpu-benchmark

# Logs
tail -f results/benchmark.log
```

## Multi-GPU Setup

```bash
# GPU 0 only
podman run --device nvidia.com/gpu=0 ...

# GPU 1 only  
podman run --device nvidia.com/gpu=1 ...

# All GPUs
podman run --device nvidia.com/gpu=all ...
```

## Troubleshooting

### GPU Not Visible

```bash
# Verify drivers
nvidia-smi

# Check CDI configuration
ls -l /etc/cdi/nvidia.yaml

# Test GPU access
podman run --rm --device nvidia.com/gpu=all \
    nvidia/cuda:12.3.1-base-ubi9 nvidia-smi
```

### SELinux Issues

```bash
# Volume mounts use :Z flag (already in scripts)
-v ./results:/workspace/results:Z

# Or temporarily disable
sudo setenforce 0
```

### Permission Errors

```bash
# Ensure results directory exists and is writable
mkdir -p results
chmod 755 results

# Run as rootless (default for Podman)
podman run --user $(id -u):$(id -g) ...
```

### Container Build Fails

```bash
# Clear cache
podman system prune -a

# Rebuild without cache
podman build --no-cache -t mlperf-benchmark:latest .
```

## Performance Expectations

Approximate throughput on common GPUs:

| GPU        | ResNet50 (Offline) | BERT (Offline) |
|------------|-------------------|----------------|
| RTX 3090   | ~2000 FPS         | ~400 QPS       |
| RTX 4090   | ~2800 FPS         | ~600 QPS       |
| A100 40GB  | ~3500 FPS         | ~800 QPS       |
| A100 80GB  | ~3800 FPS         | ~850 QPS       |

## RHEL-Specific Notes

### RHEL 10 Beta

If using RHEL 10 beta packages, you may encounter GPG key warnings. The benchmark container uses RHEL 9 base image (UBI9) which is stable and well-supported.

### Firewall

Podman networking typically doesn't require firewall changes for local containers.

### Updates

```bash
# Update Podman
sudo dnf update -y podman

# Update NVIDIA drivers
sudo dnf update -y nvidia-driver*

# Rebuild container after updates
./benchmark.sh build
```

## Advanced Usage

### Custom Models

Edit `run_benchmark.py` to add new models. Follow the pattern:

```python
def run_custom_benchmark(self, scenario='Offline', iteration=1):
    # Your model code here
    pass
```

### Long-Running Tests

For multi-day tests:

```bash
# Edit config
sed -i 's/duration_hours:.*/duration_hours: 48/' config.yaml

# Run in detached mode (already done by benchmark.sh)
./benchmark.sh run

# Monitor periodically
watch -n 300 'tail -20 results/benchmark.log'
```

### Systemd Service

Create a systemd service for automatic startup:

```bash
# Create service file
sudo tee /etc/systemd/system/mlperf-benchmark.service << EOF
[Unit]
Description=MLPerf GPU Benchmark
After=network.target

[Service]
Type=simple
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/benchmark.sh run
Restart=on-failure
User=$(whoami)

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable mlperf-benchmark
sudo systemctl start mlperf-benchmark
```

## Support

Common issues:
- GPU not detected: Verify `nvidia-smi` works and CDI is configured
- Permission denied: Use `:Z` flag on volumes for SELinux
- Build failures: Check network access to registry.access.redhat.com

For Podman-specific issues:
```bash
# Check Podman configuration
podman info

# View system logs
journalctl -u podman
```

## References

- [Podman Documentation](https://docs.podman.io/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [MLPerf Inference](https://github.com/mlcommons/inference)
- [RHEL Container Tools](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/9/html/building_running_and_managing_containers/)
