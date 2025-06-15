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

Create the following directories in your project root:

```
audio_aigented/
├── inputs/       # Place your .wav files here
├── outputs/      # Transcription results will be saved here
├── models/       # NeMo models will be cached here
├── cache/        # General cache directory
└── config/       # Configuration files (already exists)
```

```bash
mkdir -p inputs outputs models cache
```

## Basic Usage

### 1. Build the Docker Image

```bash
docker compose build
```

### 2. Process Audio Files

**Important:** The pipeline prompts for confirmation. Use `echo "Y"` to auto-confirm:

```bash
# Using the helper script (recommended)
./run-transcription.sh

# Or manually with auto-confirm
echo "Y" | docker compose run --rm audio-transcription

# Process with specific options
echo "Y" | docker compose run --rm audio-transcription --disable-diarization

# Process files from a specific directory
echo "Y" | docker compose run --rm \
  -v /path/to/your/audio/files:/data/inputs:ro \
  -v /path/to/output:/data/outputs:rw \
  audio-transcription \
  --input-dir /data/inputs \
  --output-dir /data/outputs
```

### 3. Process Test Files

To process the test audio files with two speakers:

```bash
# Using the helper script
./run-transcription.sh /home/hendorf/code/audio_ai/app/data/input /home/hendorf/code/audio_ai/app/data/output

# Or manually
echo "Y" | docker compose run --rm \
  -v /home/hendorf/code/audio_ai/app/data/input:/data/inputs:ro \
  -v /home/hendorf/code/audio_ai/app/data/output:/data/outputs:rw \
  audio-transcription \
  --input-dir /data/inputs \
  --output-dir /data/outputs \
  --device cpu
```

## Advanced Usage

### Custom Input/Output Directories

You can mount any local directory as input/output:

```bash
docker-compose run --rm \
  -v /path/to/input:/data/inputs:ro \
  -v /path/to/output:/data/outputs:rw \
  audio-transcription
```

### Development Mode

For development with live code updates:

```bash
# Start interactive container with code mounted
docker-compose run --rm --service-ports audio-transcription-dev

# Inside the container, run the pipeline
python main.py --input-dir /data/inputs --output-dir /data/outputs
```

### GPU Selection

If you have multiple GPUs:

```bash
# Use specific GPU (0, 1, 2, etc.)
docker-compose run --rm \
  -e CUDA_VISIBLE_DEVICES=1 \
  audio-transcription
```

### Resource Limits

The default docker-compose.yml limits memory to 16GB. Adjust if needed:

```yaml
deploy:
  resources:
    limits:
      memory: 32G  # Increase to 32GB
```

## Persistent Model Cache

Models are downloaded once and cached in the `models/` directory. This cache persists across container runs, saving download time.

## Troubleshooting

### 1. GPU Not Available

If the container doesn't detect GPU:

```bash
# Check if NVIDIA runtime is installed
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check Docker daemon configuration
cat /etc/docker/daemon.json
# Should contain: {"default-runtime": "nvidia", ...}
```

### 2. Permission Issues

If you get permission errors:

```bash
# Fix ownership of output directories
sudo chown -R $USER:$USER outputs/ models/ cache/
```

### 3. Out of Memory

For large audio files, you might need to:
- Increase Docker memory limits
- Use `--disable-diarization` to reduce memory usage
- Process files one at a time

### 4. View Logs

```bash
# View container logs
docker-compose logs -f audio-transcription

# Run with debug logging
docker-compose run --rm \
  -e LOG_LEVEL=DEBUG \
  audio-transcription
```

## Docker Commands Reference

```bash
# Build image
docker compose build

# Run pipeline
echo "Y" | docker compose run --rm audio-transcription [OPTIONS]

# Run interactive shell
docker compose run --rm audio-transcription bash

# Clean up containers
docker compose down

# Remove all data (careful!)
docker compose down -v

# Update image after code changes
docker compose build --no-cache
```

## Performance Notes

- First run will download NeMo models (~1-2GB), which are then cached
- GPU processing is significantly faster than CPU
- Speaker diarization requires more memory and processing time
- Typical processing speed: 10-50x real-time on GPU

## Example Workflow

```bash
# 1. Copy audio files to input directory
cp /path/to/audio/*.wav inputs/

# 2. Run transcription with speaker diarization
docker-compose run --rm audio-transcription

# 3. Check results
ls -la outputs/

# 4. View transcription
cat outputs/your_audio_file/transcript.txt
```