# MLPerf Inference Benchmark Container
# Compatible with RHEL 9 and RHEL 10
FROM nvidia/cuda:12.3.1-devel-ubi9

# Set working directory
WORKDIR /workspace

# Install Python and dependencies
RUN dnf install -y \
    python3.11 \
    python3.11-pip \
    python3.11-devel \
    git \
    wget \
    cmake \
    gcc \
    gcc-c++ \
    make \
    && dnf clean all

# Create Python virtual environment
RUN python3.11 -m venv /opt/mlperf-venv
ENV PATH="/opt/mlperf-venv/bin:$PATH"

# Upgrade pip and install core Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (newer version for compatibility)
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install MLPerf Inference dependencies
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    opencv-python-headless \
    pillow \
    pycocotools \
    onnx \
    onnxruntime-gpu \
    transformers==4.37.2 \
    datasets \
    scikit-learn \
    tqdm \
    pyyaml

# Clone MLPerf Inference repository
RUN git clone --depth 1 --branch v4.0 https://github.com/mlcommons/inference.git /workspace/mlperf-inference

# Set up MLPerf loadgen
WORKDIR /workspace/mlperf-inference/loadgen
RUN CFLAGS="-std=c++14" pip install --no-cache-dir .

# Create directories for results and data
RUN mkdir -p /workspace/results /workspace/data /workspace/models

# Set environment variables
ENV PYTHONPATH=/workspace/mlperf-inference:/workspace/mlperf-inference/vision/classification_and_detection:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

WORKDIR /workspace

# Copy benchmark files at runtime via volume mounts
# Default command runs the benchmark
CMD ["python3", "/workspace/run_benchmark.py"]
