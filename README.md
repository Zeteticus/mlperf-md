# MLPerf GPU Benchmark Container

A containerized MLPerf inference benchmark suite designed for extended GPU stress testing on RHEL 9 and RHEL 10 systems. This benchmark runs ResNet50 (image classification) and BERT (NLP) workloads continuously for specified durations.

## Features

- **Multi-hour benchmarking**: Configurable duration (default 4 hours, supports up to days)
- **Multiple scenarios**: SingleStream and Offline inference patterns
- **Comprehensive logging**: Detailed performance metrics, system information, and GPU monitoring
- **RHEL compatibility**: Tested on RHEL 9 and RHEL 10
- **Containerized**: Uses NVIDIA CUDA base images with all dependencies included
- **Results aggregation**: JSON output with statistical analysis

## Prerequisites

### System Requirements

- RHEL 9 or RHEL 10
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed (version 525+ recommended)
- Container runtime: Podman or Docker
- At least 16GB RAM
- 20GB free disk space

### Software Dependencies

1. **NVIDIA Container Toolkit** (for GPU access in containers)

   For RHEL 9/10:
   ```bash
   # Add NVIDIA repository
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
     sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
   
   # Install toolkit
   sudo dnf install -y nvidia-container-toolkit
   
   # Configure for Podman (required)
   sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
   ```

2. **Podman** (primary container runtime)
   ```bash
   sudo dnf install -y podman
   
   # Optional: podman-compose for compose file support
   sudo dnf install -y podman-compose
   ```

   *Note: Docker is also supported but Podman is recommended for RHEL*

### Verification

Check GPU visibility:
```bash
nvidia-smi
```

Verify Podman can access GPU:
```bash
podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.3.1-base-ubi9 nvidia-smi
```

Check CDI configuration:
```bash
ls -l /etc/cdi/nvidia.yaml
```

## Quick Start

### Basic Usage

```bash
# Build and run 4-hour benchmark
./benchmark.sh run

# Run 1-hour quick test
./benchmark.sh run-short

# Run 8-hour extended test
./benchmark.sh run-long

# View real-time logs
./benchmark.sh logs

# Stop benchmark
./benchmark.sh stop
```

### Manual Build and Run

```bash
# Build image
podman build -t mlperf-benchmark:latest .

# Run benchmark
podman run -d \
    --name mlperf-gpu-benchmark \
    --device nvidia.com/gpu=all \
    --security-opt=label=disable \
    -v ./results:/workspace/results:Z \
    -v ./config.yaml:/workspace/config.yaml:ro,Z \
    mlperf-benchmark:latest
```

## Configuration

Edit `config.yaml` to customize benchmark behavior:

```yaml
# Duration in hours
duration_hours: 4

# Scenarios to test
scenarios:
  - SingleStream  # Simulates real-time inference (latency-focused)
  - Offline       # Simulates batch processing (throughput-focused)

# Maximum iterations before stopping
iterations: 10

# GPU device to use
gpu_device: 0
```

### Benchmark Models

The benchmark tests two workloads:

1. **ResNet50**: Image classification (computer vision)
   - SingleStream: Batch size 1 (latency test)
   - Offline: Batch size 8 (throughput test)
   - 1,000 iterations per run

2. **BERT-base**: Natural language processing
   - SingleStream: Batch size 1
   - Offline: Batch size 4
   - 500 iterations per run

## Results

Results are saved to the `results/` directory:

```
results/
├── benchmark.log              # Detailed execution log
├── benchmark_summary.json     # Aggregated results
├── resnet50_Offline_iter1.json
├── resnet50_SingleStream_iter1.json
├── bert_Offline_iter1.json
└── bert_SingleStream_iter1.json
```

### Metrics Captured

Each benchmark result includes:
- **Latency statistics**: mean, median, p90, p95, p99, min, max (in milliseconds)
- **Throughput**: FPS (frames per second) for ResNet50, QPS (queries per second) for BERT
- **System information**: GPU name, memory, driver version
- **Timestamps**: Start and end times for each iteration

### Example Result

```json
{
  "model": "ResNet50",
  "scenario": "Offline",
  "iteration": 1,
  "batch_size": 8,
  "num_iterations": 1000,
  "mean_latency_ms": 12.34,
  "median_latency_ms": 12.10,
  "p90_latency_ms": 13.50,
  "p95_latency_ms": 14.20,
  "p99_latency_ms": 15.80,
  "throughput_fps": 648.5,
  "timestamp": "2025-01-15T10:30:45.123456"
}
```

## Advanced Usage

### Custom Duration

Modify the config file before running:

```bash
# Edit config.yaml
sed -i 's/duration_hours:.*/duration_hours: 24/' config.yaml

# Run benchmark
./benchmark.sh run
```

### Shell Access

```bash
# Access running container
./benchmark.sh shell

# Or manually
podman exec -it mlperf-gpu-benchmark /bin/bash
```

### Using Docker Compose

```bash
# Start with compose
podman-compose up -d

# View logs
podman-compose logs -f

# Stop
podman-compose down
```

### Multi-GPU Systems

To benchmark specific GPUs, modify the run command:

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 podman run ...

# GPU 1
CUDA_VISIBLE_DEVICES=1 podman run ...

# Multiple GPUs
CUDA_VISIBLE_DEVICES=0,1 podman run ...
```

## Monitoring

### Real-time GPU Monitoring

While benchmark is running:

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Detailed monitoring
nvidia-smi dmon -s pucvmet
```

### Log Analysis

```bash
# Follow benchmark log
tail -f results/benchmark.log

# Search for errors
grep -i error results/benchmark.log

# View summary
cat results/benchmark_summary.json | jq
```

## Troubleshooting

### GPU Not Detected

```bash
# Verify drivers
nvidia-smi

# Check container toolkit
nvidia-ctk --version

# Test GPU access
podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.3.1-base-ubi9 nvidia-smi
```

### SELinux Issues (RHEL)

If you encounter permission errors:

```bash
# Temporarily set permissive mode
sudo setenforce 0

# Or add :Z to volume mounts (already done in scripts)
-v ./results:/workspace/results:Z
```

### Out of Memory

Reduce batch sizes in the benchmark code or limit GPU memory:

```bash
# Run with memory fraction
podman run -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 ...
```

### Container Build Fails

```bash
# Clean build cache
podman system prune -a

# Build with no cache
podman build --no-cache -t mlperf-benchmark:latest .
```

## Performance Expectations

Typical results on various GPUs (approximate):

| GPU Model | ResNet50 (Offline) | BERT (Offline) |
|-----------|-------------------|----------------|
| RTX 3090  | ~2000 FPS         | ~400 QPS       |
| A100      | ~3500 FPS         | ~800 QPS       |
| RTX 4090  | ~2800 FPS         | ~600 QPS       |
| V100      | ~1500 FPS         | ~300 QPS       |

*Note: Actual performance varies based on CPU, memory, drivers, and thermal conditions*

## Technical Details

### Architecture

```
┌─────────────────────────────────────────┐
│   RHEL 9/10 Host System                 │
├─────────────────────────────────────────┤
│   NVIDIA Drivers + CUDA Runtime         │
├─────────────────────────────────────────┤
│   Container Runtime (Podman/Docker)     │
├─────────────────────────────────────────┤
│   ┌───────────────────────────────────┐ │
│   │  CUDA Container (UBI9)            │ │
│   │  ├─ Python 3.11                   │ │
│   │  ├─ PyTorch 2.1 (CUDA 12.1)       │ │
│   │  ├─ Transformers                  │ │
│   │  └─ MLPerf Loadgen                │ │
│   │                                   │ │
│   │  ┌─────────────────────────────┐ │ │
│   │  │  Benchmark Runner            │ │ │
│   │  │  ├─ ResNet50 Inference       │ │ │
│   │  │  ├─ BERT Inference           │ │ │
│   │  │  └─ Result Aggregation       │ │ │
│   │  └─────────────────────────────┘ │ │
│   └───────────────────────────────────┘ │
│              │                           │
│              ▼                           │
│   ┌─────────────────┐                   │
│   │  results/       │                   │
│   │  - JSON files   │                   │
│   │  - Logs         │                   │
│   └─────────────────┘                   │
└─────────────────────────────────────────┘
```

### Container Size

- Base image: ~4GB
- With dependencies: ~8GB
- Peak memory usage: ~10GB GPU VRAM (depending on GPU)

## Maintenance

### Update Dependencies

To update PyTorch or other packages, modify the Dockerfile:

```dockerfile
RUN pip install --no-cache-dir \
    torch==2.2.0 \  # Update version
    torchvision==0.17.0 \
    ...
```

### Clean Up

```bash
# Remove all results and containers
./benchmark.sh clean

# Remove images
podman rmi mlperf-benchmark:latest
```

## Contributing

To add new benchmarks:

1. Add model code to `run_benchmark.py`
2. Create a new benchmark method following the pattern of `run_resnet50_benchmark`
3. Update the main loop in the `run()` method
4. Add configuration options to `config.yaml`

## License

This benchmark suite uses MLPerf (Apache 2.0), PyTorch (BSD), and Transformers (Apache 2.0).

## Support

For issues:
- Check logs: `./benchmark.sh logs`
- Review GPU status: `nvidia-smi`
- Verify container runtime: `podman version` or `docker version`

## References

- [MLPerf Inference](https://github.com/mlcommons/inference)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
