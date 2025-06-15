# Installation

This guide will help you set up the Audio Transcription Pipeline on your system.

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with 4GB+ VRAM (8GB+ recommended)
  - Tested on: RTX 3090, RTX 4090, RTX Titan, A100
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: ~3GB for models + space for audio files

### Software
- **Operating System**: Linux (Ubuntu 20.04+), Windows 10/11, macOS
- **Python**: 3.10 or higher
- **CUDA**: 11.8 or higher (for GPU acceleration)
- **NVIDIA Drivers**: 520.61.05 or newer

## Installation Methods

### Method 1: Using uv (Recommended)

First, install the `uv` package manager:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pipx
pipx install uv
```

Then install the package:

```bash
# Clone the repository
git clone https://github.com/yourusername/audio_aigented.git
cd audio_aigented

# Install the package
uv pip install -e .

# For development (includes testing and docs tools)
uv pip install -e ".[dev,docs]"
```

### Method 2: Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/audio_aigented.git
cd audio_aigented

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .

# For development
pip install -e ".[dev,docs]"
```

### Method 3: Using Docker

```bash
# Clone the repository
git clone https://github.com/yourusername/audio_aigented.git
cd audio_aigented

# Build the Docker image
docker-compose build

# Run the container
docker-compose run --rm audio-transcription
```

## NVIDIA Setup

### Installing CUDA Toolkit

1. Check your GPU compatibility:
   ```bash
   nvidia-smi
   ```

2. Install CUDA Toolkit 11.8+:
   ```bash
   # Ubuntu/Debian
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
   sudo dpkg -i cuda-keyring_1.0-1_all.deb
   sudo apt-get update
   sudo apt-get install cuda-11-8
   ```

3. Add to PATH:
   ```bash
   echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

### Installing PyTorch with CUDA

The package dependencies will automatically install the correct PyTorch version, but you can verify:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Model Downloads

Models are automatically downloaded on first use. To pre-download:

```python
from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.models.classification_models import EncDecClassificationModel

# Download ASR model
asr_model = EncDecCTCModel.from_pretrained("stt_en_conformer_ctc_large")

# Download VAD model
vad_model = EncDecClassificationModel.from_pretrained("vad_multilingual_marblenet")
```

Models are cached in:
- Linux/macOS: `~/.cache/torch/NeMo/`
- Windows: `%USERPROFILE%\.cache\torch\NeMo\`

## Verification

After installation, verify everything is working:

```bash
# Check installation
python -c "import audio_aigented; print('Package installed successfully')"

# Run tests
uv run pytest tests/test_models.py -v

# Process a test file
uv run python main.py --help
```

## Troubleshooting

### Common Issues

**ImportError: No module named 'nemo'**
- Solution: Install NeMo toolkit
  ```bash
  uv pip install "nemo-toolkit[asr]"
  ```

**CUDA out of memory**
- Reduce batch size in configuration
- Use a smaller model
- Switch to CPU processing

**No GPU detected**
- Check NVIDIA drivers: `nvidia-smi`
- Verify CUDA installation: `nvcc --version`
- Ensure PyTorch CUDA version matches system CUDA

**Permission denied errors**
- Ensure write permissions to output directory
- Check model cache directory permissions

## Next Steps

- Continue to [Quick Start](quickstart.md) to process your first audio file
- See [Configuration](configuration.md) for customization options
- Check [Docker Setup](../deployment/docker.md) for containerized deployment