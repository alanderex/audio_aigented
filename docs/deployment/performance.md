# Performance Tuning

This guide covers optimization strategies to maximize the performance of the Audio Transcription Pipeline.

## Performance Benchmarks

### GPU Performance (RTX 3090)

| Model | Audio Duration | Processing Time | Speed Factor | GPU Memory |
|-------|----------------|-----------------|--------------|------------|
| Small | 5 min | 20s | 15x | 2GB |
| Medium | 5 min | 38s | 8x | 3GB |
| Large | 5 min | 75s | 4x | 4GB |

### CPU Performance (16-core)

| Model | Audio Duration | Processing Time | Speed Factor | RAM Usage |
|-------|----------------|-----------------|--------------|-----------|
| Small | 5 min | 200s | 1.5x | 4GB |
| Medium | 5 min | 375s | 0.8x | 6GB |
| Large | 5 min | 750s | 0.4x | 8GB |

## GPU Optimization

### CUDA Configuration

```python
import torch

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.get_device_name()}")

# Set GPU device
torch.cuda.set_device(0)  # Use first GPU
```

### Memory Management

```yaml
# Optimize GPU memory usage
transcription:
  batch_size: 1                    # Reduce for less memory
  model:
    compute_dtype: "float16"       # Use mixed precision
    
  gpu:
    memory_fraction: 0.8           # Reserve 20% for system
    allow_growth: true             # Dynamic allocation
```

### Multi-GPU Support

```python
# Use multiple GPUs
class MultiGPUPipeline:
    def __init__(self, gpu_ids=[0, 1]):
        self.models = []
        for gpu_id in gpu_ids:
            model = load_model(device=f"cuda:{gpu_id}")
            self.models.append(model)
    
    def process_batch(self, files):
        # Distribute files across GPUs
        results = []
        for i, file in enumerate(files):
            gpu_idx = i % len(self.models)
            result = self.models[gpu_idx].transcribe(file)
            results.append(result)
        return results
```

## CPU Optimization

### Thread Configuration

```yaml
# CPU optimization settings
processing:
  num_threads: 16                  # Match CPU cores
  use_mkl: true                    # Intel MKL acceleration
  
torch:
  num_threads: 8                   # PyTorch threads
  num_interop_threads: 4           # Inter-op parallelism
```

### NUMA Optimization

```bash
# Pin process to NUMA node
numactl --cpunodebind=0 --membind=0 python main.py

# Check NUMA configuration
numactl --hardware
```

## Batch Processing Optimization

### Optimal Batch Sizes

| GPU Memory | Batch Size (Small) | Batch Size (Medium) | Batch Size (Large) |
|------------|-------------------|---------------------|-------------------|
| 4GB | 16 | 8 | 4 |
| 8GB | 32 | 16 | 8 |
| 16GB | 64 | 32 | 16 |
| 24GB | 96 | 48 | 24 |

### Dynamic Batching

```python
def get_optimal_batch_size(gpu_memory_gb, model_size):
    """Calculate optimal batch size based on GPU memory"""
    base_sizes = {
        "small": 16,
        "medium": 8,
        "large": 4
    }
    
    memory_multiplier = gpu_memory_gb / 4.0
    optimal_size = int(base_sizes[model_size] * memory_multiplier)
    
    return max(1, optimal_size)
```

## Pipeline Optimization

### Preprocessing Cache

```python
class CachedAudioLoader:
    def __init__(self, cache_dir=".cache/audio"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def load(self, audio_path):
        cache_path = self.cache_dir / f"{audio_path.stem}_16k.npy"
        
        if cache_path.exists():
            return np.load(cache_path)
        
        # Load and preprocess
        audio, sr = sf.read(audio_path)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # Cache preprocessed audio
        np.save(cache_path, audio)
        return audio
```

### Parallel Pipeline

```python
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

class ParallelPipeline:
    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        
    def process_directory(self, input_dir):
        files = list(Path(input_dir).glob("*.wav"))
        
        # Parallel audio loading
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            audio_data = list(executor.map(self.load_audio, files))
        
        # Parallel diarization
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            diarization_results = list(executor.map(self.diarize, audio_data))
        
        # Batch transcription on GPU
        transcription_results = self.transcribe_batch(audio_data, diarization_results)
        
        return transcription_results
```

## Model Optimization

### Model Quantization

```python
# Quantize model for faster inference
def quantize_model(model):
    # Dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear, torch.nn.Conv1d}, 
        dtype=torch.qint8
    )
    return quantized_model
```

### ONNX Export

```python
# Export to ONNX for optimized inference
def export_to_onnx(model, output_path):
    dummy_input = torch.randn(1, 16000)  # 1 second audio
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['audio'],
        output_names=['transcription'],
        dynamic_axes={'audio': {0: 'batch_size', 1: 'sequence'}}
    )
```

## Storage Optimization

### Output Compression

```yaml
output:
  compression:
    enable: true
    format: "gzip"              # or "lz4" for faster
    level: 6                    # 1-9, higher = smaller
    
  # Only save required formats
  formats: ["json"]             # Skip txt if not needed
  
  # Reduce JSON size
  json:
    pretty: false               # Minified JSON
    exclude_words: true         # Skip word-level data
```

### Streaming Output

```python
class StreamingWriter:
    def __init__(self, output_file):
        self.output_file = output_file
        self.file = open(output_file, 'w')
        self.file.write('{"segments": [')
        self.first = True
    
    def write_segment(self, segment):
        if not self.first:
            self.file.write(',')
        self.file.write(json.dumps(segment))
        self.file.flush()
        self.first = False
    
    def close(self):
        self.file.write(']}')
        self.file.close()
```

## Network Optimization

### Model Download Cache

```bash
# Pre-download models
export TORCH_HOME=/shared/models
export HF_HOME=/shared/models/huggingface

# Share across machines
mkdir -p /shared/models
chmod 755 /shared/models
```

### Distributed Processing

```python
# Process on multiple machines
from multiprocessing.managers import BaseManager

class DistributedPipeline:
    def __init__(self, worker_addresses):
        self.workers = []
        for addr, port in worker_addresses:
            manager = BaseManager(address=(addr, port))
            manager.connect()
            self.workers.append(manager)
    
    def process_files(self, files):
        # Distribute files to workers
        chunk_size = len(files) // len(self.workers)
        futures = []
        
        for i, worker in enumerate(self.workers):
            start = i * chunk_size
            end = start + chunk_size if i < len(self.workers)-1 else len(files)
            chunk = files[start:end]
            
            future = worker.process_async(chunk)
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            results.extend(future.get())
        
        return results
```

## Monitoring and Profiling

### Performance Metrics

```python
import psutil
import GPUtil

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.start_cpu = psutil.cpu_percent()
        self.start_memory = psutil.virtual_memory().percent
        
    def log_metrics(self):
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # GPU metrics
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            gpu_util = gpu.load * 100
            gpu_memory = gpu.memoryUtil * 100
        else:
            gpu_util = gpu_memory = 0
        
        metrics = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_gb": memory.used / (1024**3),
            "gpu_util": gpu_util,
            "gpu_memory": gpu_memory,
            "elapsed_time": time.time() - self.start_time
        }
        
        return metrics
```

### Profiling Tools

```bash
# Profile with cProfile
python -m cProfile -o profile.stats main.py

# Analyze profile
python -m pstats profile.stats

# GPU profiling with nvprof
nvprof python main.py

# PyTorch profiler
python -m torch.profiler main.py
```

## Best Practices

### Configuration Templates

#### High Throughput

```yaml
# config/high_throughput.yaml
model:
  name: "stt_en_conformer_ctc_small"
  
transcription:
  batch_size: 32
  device: "cuda"
  compute_dtype: "float16"
  
diarization:
  enable: false                    # Skip for speed
  
processing:
  parallel_workers: 8
  
output:
  formats: ["json"]                # Minimal output
  compression:
    enable: true
    format: "lz4"
```

#### High Accuracy

```yaml
# config/high_accuracy.yaml
model:
  name: "stt_en_conformer_ctc_large"
  
transcription:
  batch_size: 4
  device: "cuda"
  compute_dtype: "float32"         # Full precision
  
diarization:
  enable: true
  embedding:
    window_length: 2.0             # Longer windows
    
output:
  formats: ["json", "txt", "attributed_txt"]
  include_word_timestamps: true
```

### Optimization Checklist

1. **Hardware**
   - [ ] GPU drivers updated
   - [ ] CUDA toolkit installed
   - [ ] Sufficient RAM/VRAM

2. **Configuration**
   - [ ] Optimal batch size
   - [ ] Appropriate model size
   - [ ] Mixed precision enabled

3. **Pipeline**
   - [ ] Preprocessing cached
   - [ ] Parallel processing enabled
   - [ ] Output compression configured

4. **Monitoring**
   - [ ] Resource usage tracked
   - [ ] Bottlenecks identified
   - [ ] Performance logged

## Troubleshooting Performance

### Slow Processing

1. **Check GPU utilization**
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Increase batch size**
   ```yaml
   transcription:
     batch_size: 16  # From 4
   ```

3. **Use smaller model**
   ```bash
   --model stt_en_conformer_ctc_small
   ```

### Out of Memory

1. **Reduce batch size**
   ```yaml
   transcription:
     batch_size: 1
   ```

2. **Enable gradient checkpointing**
   ```python
   model.gradient_checkpointing_enable()
   ```

3. **Clear cache**
   ```python
   torch.cuda.empty_cache()
   ```

### High Latency

1. **Enable model caching**
   ```python
   @lru_cache(maxsize=1)
   def get_model():
       return load_model()
   ```

2. **Warm up GPU**
   ```python
   # Process dummy input
   model.transcribe(torch.zeros(1, 16000))
   ```

3. **Use streaming**
   ```python
   for chunk in stream_process(audio):
       yield chunk
   ```