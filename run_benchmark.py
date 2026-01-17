#!/usr/bin/env python3
"""
MLPerf GPU Benchmark Runner
Runs multiple MLPerf inference benchmarks in sequence for extended duration
Includes comprehensive logging and GPU monitoring
"""

import os
import sys
import time
import json
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/results/benchmark.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MLPerfBenchmark:
    """MLPerf Inference Benchmark Runner"""
    
    def __init__(self, config_path='config.yaml'):
        self.start_time = datetime.now()
        self.results_dir = Path('/workspace/results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.duration_hours = self.config.get('duration_hours', 4)
        self.scenarios = self.config.get('scenarios', ['SingleStream', 'Offline'])
        self.iterations = self.config.get('iterations', 10)
        
        logger.info(f"Initialized benchmark - Duration: {self.duration_hours}h, Scenarios: {self.scenarios}")
    
    def check_gpu(self):
        """Verify GPU availability and log details"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                capture_output=True, text=True, check=True
            )
            gpu_info = result.stdout.strip()
            logger.info(f"GPU detected: {gpu_info}")
            
            # Log GPU utilization
            result = subprocess.run(
                ['nvidia-smi', 'dmon', '-c', '1'],
                capture_output=True, text=True
            )
            logger.info(f"GPU status:\n{result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"GPU check failed: {e}")
            return False
        except FileNotFoundError:
            logger.error("nvidia-smi not found - NVIDIA drivers may not be installed")
            return False
    
    def log_system_info(self):
        """Log system information"""
        logger.info("="*80)
        logger.info("SYSTEM INFORMATION")
        logger.info("="*80)
        
        # OS info
        try:
            with open('/etc/os-release', 'r') as f:
                os_info = f.read()
                for line in os_info.split('\n'):
                    if line.startswith('PRETTY_NAME'):
                        os_name = line.split('=')[1].strip('"')
                        logger.info(f"OS: {os_name}")
        except:
            pass
        
        # CPU info
        try:
            result = subprocess.run(['nproc'], capture_output=True, text=True)
            logger.info(f"CPU cores: {result.stdout.strip()}")
        except:
            pass
        
        # Memory info
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        mem_kb = int(line.split()[1])
                        mem_gb = mem_kb / (1024**2)
                        logger.info(f"RAM: {mem_gb:.1f} GB")
                        break
        except:
            pass
        
        logger.info("="*80)
    
    def run_resnet50_benchmark(self, scenario='Offline', iteration=1):
        """Run ResNet50 image classification benchmark"""
        logger.info(f"Starting ResNet50 benchmark - Scenario: {scenario}, Iteration: {iteration}")
        
        benchmark_start = time.time()
        
        # ResNet50 benchmark using TorchVision
        code = f"""
import torch
import torchvision.models as models
import time
import json
from datetime import datetime

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {{device}}")

if device.type == 'cuda':
    print(f"GPU: {{torch.cuda.get_device_name(0)}}")
    print(f"Memory: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}} GB")

# Load ResNet50 model
print("Loading ResNet50 model...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model = model.to(device)
model.eval()

# Create dummy input
batch_size = {'8' if scenario == 'Offline' else '1'}
dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)

# Warmup
print("Warming up...")
with torch.no_grad():
    for _ in range(10):
        _ = model(dummy_input)

# Benchmark
num_iterations = 1000
print(f"Running {{num_iterations}} iterations...")
times = []

with torch.no_grad():
    for i in range(num_iterations):
        start = time.perf_counter()
        output = model(dummy_input)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
        
        if (i + 1) % 100 == 0:
            print(f"Progress: {{i + 1}}/{{num_iterations}}")

# Calculate statistics
import numpy as np
times = np.array(times) * 1000  # Convert to ms
results = {{
    'model': 'ResNet50',
    'scenario': '{scenario}',
    'iteration': {iteration},
    'batch_size': batch_size,
    'num_iterations': num_iterations,
    'mean_latency_ms': float(np.mean(times)),
    'median_latency_ms': float(np.median(times)),
    'p90_latency_ms': float(np.percentile(times, 90)),
    'p95_latency_ms': float(np.percentile(times, 95)),
    'p99_latency_ms': float(np.percentile(times, 99)),
    'min_latency_ms': float(np.min(times)),
    'max_latency_ms': float(np.max(times)),
    'throughput_fps': float(batch_size * 1000 / np.mean(times)),
    'timestamp': datetime.now().isoformat()
}}

print("\\nResults:")
print(json.dumps(results, indent=2))

# Save results
with open('/workspace/results/resnet50_{scenario}_iter{iteration}.json', 'w') as f:
    json.dump(results, f, indent=2)
"""
        
        result_file = self.results_dir / f"resnet50_{scenario}_iter{iteration}.py"
        with open(result_file, 'w') as f:
            f.write(code)
        
        try:
            result = subprocess.run(
                ['python3', str(result_file)],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout per benchmark
            )
            
            logger.info(result.stdout)
            if result.stderr:
                logger.warning(f"Stderr: {result.stderr}")
            
            duration = time.time() - benchmark_start
            logger.info(f"ResNet50 benchmark completed in {duration:.2f}s")
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            logger.error("Benchmark timeout!")
            return False
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return False
    
    def run_bert_benchmark(self, scenario='Offline', iteration=1):
        """Run BERT benchmark for NLP"""
        logger.info(f"Starting BERT benchmark - Scenario: {scenario}, Iteration: {iteration}")
        
        benchmark_start = time.time()
        
        code = f"""
import torch
from transformers import BertModel, BertTokenizer
import time
import json
from datetime import datetime
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {{device}}")

# Load BERT model
print("Loading BERT model...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model = model.to(device)
model.eval()

# Prepare input
text = "This is a sample sentence for BERT inference benchmark testing. " * 10
batch_size = {'4' if scenario == 'Offline' else '1'}
inputs = tokenizer([text] * batch_size, return_tensors='pt', padding=True, truncation=True, max_length=512)
inputs = {{k: v.to(device) for k, v in inputs.items()}}

# Warmup
print("Warming up...")
with torch.no_grad():
    for _ in range(5):
        _ = model(**inputs)

# Benchmark
num_iterations = 500
print(f"Running {{num_iterations}} iterations...")
times = []

with torch.no_grad():
    for i in range(num_iterations):
        start = time.perf_counter()
        outputs = model(**inputs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
        
        if (i + 1) % 50 == 0:
            print(f"Progress: {{i + 1}}/{{num_iterations}}")

times = np.array(times) * 1000
results = {{
    'model': 'BERT-base',
    'scenario': '{scenario}',
    'iteration': {iteration},
    'batch_size': batch_size,
    'num_iterations': num_iterations,
    'mean_latency_ms': float(np.mean(times)),
    'median_latency_ms': float(np.median(times)),
    'p90_latency_ms': float(np.percentile(times, 90)),
    'p95_latency_ms': float(np.percentile(times, 95)),
    'p99_latency_ms': float(np.percentile(times, 99)),
    'throughput_qps': float(batch_size * 1000 / np.mean(times)),
    'timestamp': datetime.now().isoformat()
}}

print("\\nResults:")
print(json.dumps(results, indent=2))

with open('/workspace/results/bert_{scenario}_iter{iteration}.json', 'w') as f:
    json.dump(results, f, indent=2)
"""
        
        result_file = self.results_dir / f"bert_{scenario}_iter{iteration}.py"
        with open(result_file, 'w') as f:
            f.write(code)
        
        try:
            result = subprocess.run(
                ['python3', str(result_file)],
                capture_output=True,
                text=True,
                timeout=1800
            )
            
            logger.info(result.stdout)
            if result.stderr:
                logger.warning(f"Stderr: {result.stderr}")
            
            duration = time.time() - benchmark_start
            logger.info(f"BERT benchmark completed in {duration:.2f}s")
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return False
    
    def aggregate_results(self):
        """Aggregate all benchmark results"""
        logger.info("Aggregating results...")
        
        all_results = []
        for result_file in self.results_dir.glob('*.json'):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    all_results.append(data)
            except Exception as e:
                logger.error(f"Failed to load {result_file}: {e}")
        
        summary = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'total_benchmarks': len(all_results),
            'results': all_results
        }
        
        summary_file = self.results_dir / 'benchmark_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to {summary_file}")
        
        # Print summary
        logger.info("="*80)
        logger.info("BENCHMARK SUMMARY")
        logger.info("="*80)
        logger.info(f"Total benchmarks run: {len(all_results)}")
        logger.info(f"Total duration: {summary['total_duration_hours']:.2f} hours")
        
        if all_results:
            for model_type in ['ResNet50', 'BERT-base']:
                model_results = [r for r in all_results if r['model'] == model_type]
                if model_results:
                    avg_throughput = sum(r.get('throughput_fps', r.get('throughput_qps', 0)) 
                                       for r in model_results) / len(model_results)
                    logger.info(f"{model_type} average throughput: {avg_throughput:.2f}")
        
        logger.info("="*80)
    
    def run(self):
        """Main benchmark loop"""
        logger.info("="*80)
        logger.info("MLPERF GPU BENCHMARK STARTING")
        logger.info("="*80)
        
        self.log_system_info()
        
        if not self.check_gpu():
            logger.error("GPU not available - cannot run benchmark")
            return
        
        end_time = self.start_time + timedelta(hours=self.duration_hours)
        logger.info(f"Benchmark will run until {end_time}")
        
        iteration = 1
        benchmarks_completed = 0
        
        while datetime.now() < end_time:
            remaining = (end_time - datetime.now()).total_seconds() / 3600
            logger.info(f"\n{'='*80}")
            logger.info(f"Iteration {iteration} - {remaining:.2f}h remaining")
            logger.info(f"{'='*80}\n")
            
            for scenario in self.scenarios:
                if datetime.now() >= end_time:
                    break
                
                # Run ResNet50
                if self.run_resnet50_benchmark(scenario, iteration):
                    benchmarks_completed += 1
                
                # Check time again
                if datetime.now() >= end_time:
                    break
                
                # Run BERT
                if self.run_bert_benchmark(scenario, iteration):
                    benchmarks_completed += 1
                
                # Log GPU status between benchmarks
                self.check_gpu()
            
            iteration += 1
            
            # Break if we've exceeded target iterations
            if iteration > self.iterations:
                logger.info(f"Reached maximum iterations ({self.iterations})")
                break
        
        logger.info(f"\nBenchmark loop completed - {benchmarks_completed} benchmarks run")
        self.aggregate_results()
        
        logger.info("="*80)
        logger.info("BENCHMARK COMPLETED")
        logger.info("="*80)


if __name__ == '__main__':
    benchmark = MLPerfBenchmark()
    benchmark.run()
