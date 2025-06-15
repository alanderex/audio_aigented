# Docker Setup for Audio Transcription Pipeline

This guide explains how to run the audio transcription pipeline using Docker.

## Prerequisites

1. **Docker** and **Docker Compose** installed
2. **NVIDIA Docker runtime** for GPU support:
   ```bash
   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

## Directory Structure

The Docker setup uses the following volume mappings:

```
audio_aigented/
├── inputs/       # Place your .wav files here
├── outputs/      # Transcription results will be saved here
├── models/       # NeMo models will be cached here
├── cache/        # General cache directory
└── config/       # Configuration files
```

## Quick Start

### 1. Build the Docker Image

```bash
docker-compose build
```

### 2. Test Setup

Run the test script to verify everything is configured correctly:

```bash
./scripts/docker-test.sh
```

### 3. Process Audio Files

Place your `.wav` files in the `inputs/` directory, then run:

```bash
# Using the helper script (recommended)
./scripts/run-transcription.sh

# Or manually with docker-compose
echo "Y" | docker-compose run --rm audio-transcription

# Process specific files with custom output
docker-compose run --rm \
  -v /path/to/audio:/data/inputs:ro \
  -v /path/to/results:/data/outputs \
  audio-transcription
```

### 4. Development Mode

For interactive development with the container:

```bash
docker-compose run --rm --entrypoint bash audio-transcription
```

## Docker Components

### Dockerfile

Multi-stage build with:
- NVIDIA CUDA 11.8 base image for GPU support
- Python 3.10 and all required system dependencies
- Non-root user for security
- Proper cache directory configuration for NeMo models
- Optimized layer caching for faster rebuilds

### docker-compose.yml

Main orchestration file with:
- GPU runtime configuration
- Volume mappings for inputs, outputs, models, and cache
- Resource limits (16GB RAM)
- Development profile for interactive debugging

### docker-entrypoint.sh

Smart entrypoint script that:
- Checks GPU availability
- Validates input/output directories
- Provides helpful error messages
- Supports both batch processing and interactive mode

## Processing Options

### Default Processing (with diarization)
```bash
./scripts/run-transcription.sh
```

### Fast Processing (without diarization)
```bash
docker-compose run --rm audio-transcription python main.py --input-dir /data/inputs --output-dir /data/outputs --disable-diarization
```

### Custom Configuration
```bash
docker-compose run --rm \
  -v ./custom_config.yaml:/app/config/default.yaml:ro \
  audio-transcription
```

## Troubleshooting

### GPU Not Detected

If you see "No GPU detected" warning:
1. Check NVIDIA drivers: `nvidia-smi`
2. Verify Docker GPU access: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`
3. Ensure NVIDIA Docker runtime is installed

### Permission Issues

If you encounter permission errors:
- The container runs as user `appuser` (UID 1000)
- Ensure input/output directories have appropriate permissions
- Use `--user $(id -u):$(id -g)` flag if needed

### Model Download Issues

Models are automatically downloaded on first use:
- Downloads are cached in `models/` directory
- Ensure you have ~3GB free space
- Check internet connectivity

## Advanced Usage

### Running with Specific GPU
```bash
docker-compose run --rm -e CUDA_VISIBLE_DEVICES=0 audio-transcription
```

### Using Different Models
Edit `config/default.yaml` or mount a custom config:
```yaml
model:
  name: "stt_en_conformer_ctc_large"  # or another NeMo model
```

### Batch Processing Multiple Directories
```bash
for dir in /path/to/audio/*/; do
  docker-compose run --rm \
    -v "$dir":/data/inputs:ro \
    -v "./outputs/$(basename "$dir")":/data/outputs \
    audio-transcription
done
```

## Resource Requirements

- **GPU**: NVIDIA GPU with 4GB+ VRAM (8GB+ recommended)
- **RAM**: 16GB minimum
- **Storage**: ~3GB for models, plus space for audio files
- **CUDA**: Version 11.8 (handled by Docker)