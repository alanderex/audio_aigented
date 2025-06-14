# 🎙️ Audio Transcription Pipeline

A modular, GPU-accelerated audio processing pipeline that automates the transcription of spoken audio using NVIDIA NeMo's state-of-the-art ASR models.

## ✨ Features

- **🚀 GPU-Accelerated ASR** using NVIDIA NeMo conformer models
- **🎤 Speaker Diarization** for identifying who spoke when
- **📁 Batch Processing** of multiple audio files
- **🎯 High Accuracy** with confidence scoring
- **📊 Structured Output** in JSON and human-readable text formats
- **⚙️ Configurable Pipeline** with YAML configuration
- **🔄 Modular Architecture** for easy extension and maintenance
- **🧪 Comprehensive Testing** with pytest suite
- **💾 Smart Caching** to avoid re-processing files

## 🏗️ Architecture

The pipeline consists of 5 processing stages:

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ load_audio  │ -> │ transcribe   │ -> │ diarize     │ -> │ format      │ -> │ write_output │ -> │ Results     │
│             │    │              │    │             │    │             │    │              │    │             │
│ • Load .wav │    │ • NVIDIA     │    │ • Speaker   │    │ • Structure │    │ • JSON files │    │ • Per-file  │
│ • Validate  │    │   NeMo ASR   │    │   detection │    │ • Timestamps│    │ • TXT files  │    │   outputs   │
│ • Resample  │    │ • GPU accel  │    │ • Segments  │    │ • Confidence│    │ • Attributed │    │ • Summaries │
│             │    │ • Segments   │    │ • Clustering│    │ • Speakers  │    │   TXT files  │    │             │
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘    └──────────────┘    └─────────────┘
```
## 📄 Output Formats

The pipeline generates multiple output formats for each processed audio file:

### 1. JSON Format (`transcript.json`)
Structured data with complete metadata, timestamps, and confidence scores:
```json
{
  "audio_file": {
    "path": "audio.wav",
    "duration": 45.2,
    "sample_rate": 16000
  },
  "transcription": {
    "full_text": "Hello there! How are you today?",
    "segments": [
      {
        "text": "Hello there!",
        "start_time": 0.0,
        "end_time": 1.5,
        "confidence": 0.95,
        "speaker_id": "SPEAKER_00"
      }
    ]
  }
}
```

### 2. Human-Readable Text (`transcript.txt`)
Formatted report with statistics and detailed segment breakdown.

### 3. Theater-Style Attribution (`transcript_attributed.txt`)
**New Feature!** Dialog format for conversations with speaker labels:
```
SPEAKER_00: Hello there! How are you doing today?
SPEAKER_01: I'm doing great, thanks for asking!
SPEAKER_00: That's wonderful to hear.
```

This format maintains natural conversation flow and is perfect for:
- Meeting transcripts
- Interview recordings  
- Podcast dialog
- Conference call notes

## 📦 Installation

### Prerequisites

- Python 3.12+
- CUDA 12.8 (for GPU acceleration)
- NVIDIA RTX Titan or compatible GPU (recommended)
- [uv](https://docs.astral.sh/uv/) package manager

### Install uv

First, install uv if you haven't already:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pipx (recommended if you have it)
pipx install uv

# Or as a fallback with pip
pip install uv
```

### Install Dependencies

```bash
# Clone the repository
git clone <your-repo-url>
cd audio_aigented

# Install the package and dependencies
uv pip install -e .

# For development (includes testing tools)
uv pip install -e ".[dev]"
```

### NVIDIA NeMo Installation

The pipeline requires NVIDIA NeMo for ASR functionality:

```bash
uv pip install "nemo-toolkit[asr]"
```

## 🚀 Quick Start

### 1. Basic Usage with CLI

```bash
# Process all .wav files in ./inputs/ directory
uv run python main.py --input-dir ./inputs

# Use custom output directory
uv run python main.py --input-dir ./my_audio --output-dir ./my_results

# Use CPU instead of GPU
uv run python main.py --input-dir ./inputs --device cpu

# Disable speaker diarization (faster processing)
uv run python main.py --input-dir ./inputs --disable-diarization

# Show help for all options
uv run python main.py --help
```

### 2. Using the Python API

```python
from pathlib import Path
from src.audio_aigented.pipeline import TranscriptionPipeline

# Initialize pipeline with default settings
pipeline = TranscriptionPipeline()

# Process all files in input directory
results = pipeline.process_directory(Path("./inputs"))

# Process a single file
result = pipeline.process_single_file(Path("./inputs/meeting.wav"))
print(f"Transcription: {result.full_text}")
```

### 3. Custom Configuration

Create a custom configuration file:

```yaml
# config/my_config.yaml
input_dir: "./my_audio_files"
output_dir: "./my_results"

audio:
  sample_rate: 16000
  batch_size: 4

transcription:
  model_name: "stt_en_conformer_ctc_large"
  device: "cuda"
  enable_confidence_scores: true

processing:
  enable_diarization: true
  enable_caching: true
  log_level: "INFO"

output:
  formats: ["json", "txt", "attributed_txt"]
  include_timestamps: true
  pretty_json: true
```

Use with CLI:
```bash
uv run python main.py --config ./config/my_config.yaml
```

## 📁 Input and Output

### Input Structure
```
inputs/
├── meeting_recording.wav
├── interview_audio.wav
└── conference_call.wav
```

### Output Structure
```
outputs/
├── meeting_recording/
│   ├── transcript.json              # Structured data with timestamps & speakers
│   ├── transcript.txt               # Human-readable format with statistics
│   └── transcript_attributed.txt    # Theater-style speaker dialog
├── interview_audio/
│   ├── transcript.json
│   ├── transcript.txt
│   └── transcript_attributed.txt
└── processing_summary.txt           # Overall processing report
```

### Sample JSON Output
```json
{
  "audio_file": {
    "path": "./inputs/meeting.wav",
    "duration": 125.3,
    "sample_rate": 16000
  },
  "transcription": {
    "full_text": "Good morning everyone, let's begin the meeting...",
    "segments": [
      {
        "text": "Good morning everyone",
        "start_time": 0.0,
        "end_time": 2.1,
        "confidence": 0.95
      }
    ]
  },
  "processing": {
    "processing_time": 12.4,
    "model_info": {"name": "stt_en_conformer_ctc_large"}
  }
}
```

## ⚙️ Configuration Options

### Audio Processing
- `sample_rate`: Target sample rate (default: 16000 Hz)
- `batch_size`: Number of files to process in parallel
- `max_duration`: Maximum segment duration for processing

### ASR Settings
- `model_name`: NVIDIA NeMo model to use
  - `stt_en_conformer_ctc_small` (fast, lower accuracy)
  - `stt_en_conformer_ctc_medium` (balanced)
  - `stt_en_conformer_ctc_large` (slow, higher accuracy)
- `device`: Processing device (`cuda` or `cpu`)
- `enable_confidence_scores`: Include confidence scores in output

### Speaker Diarization
- `enable_diarization`: Enable/disable speaker identification (default: `true`)
- Command line: `--enable-diarization` / `--disable-diarization`
- When enabled, segments are automatically labeled with speaker IDs (SPEAKER_00, SPEAKER_01, etc.)
- Uses NVIDIA NeMo's clustering diarization for accurate speaker separation

### Processing Options
- `enable_caching`: Cache models and intermediate results
- `parallel_workers`: Number of parallel processing workers
- `log_level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

### Output Options
- `formats`: Output formats (`["json", "txt", "attributed_txt"]`)
- `include_timestamps`: Include timing information
- `include_confidence`: Include confidence scores in output
- `pretty_json`: Format JSON with indentation

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_models.py

# Run with verbose output
uv run pytest -v
```

## 🏃‍♂️ Performance

### GPU Performance (RTX Titan 24GB)
- Small model: ~15x real-time speed
- Medium model: ~8x real-time speed  
- Large model: ~4x real-time speed

### CPU Performance (16-core)
- Small model: ~1.5x real-time speed
- Medium model: ~0.8x real-time speed
- Large model: ~0.4x real-time speed

## 🛠️ Development

### Project Structure
```
audio_aigented/
├── src/audio_aigented/           # Main package
│   ├── audio/                    # Audio loading and preprocessing
│   ├── transcription/            # ASR processing with NeMo
│   ├── formatting/               # Output formatting
│   ├── output/                   # File writing
│   ├── config/                   # Configuration management
│   ├── models/                   # Pydantic data models
│   └── pipeline.py               # Main orchestration
├── tests/                        # Test suite
├── config/                       # Configuration files
├── examples/                     # Usage examples
└── main.py                       # CLI entry point
```

### Code Style
- Uses `ruff` for linting and formatting
- Follows PEP8 with type hints
- Google-style docstrings
- Maximum 500 lines per file

### Adding New Features
1. Create feature branch
2. Implement with comprehensive tests
3. Update documentation
4. Submit pull request

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Use smaller batch size
uv run python main.py --input-dir ./inputs --config config/cpu_config.yaml

# Or switch to CPU
uv run python main.py --input-dir ./inputs --device cpu
```

**No Audio Files Found**
- Ensure `.wav` files are in the input directory
- Check file permissions
- Use `--dry-run` to see what files would be processed

**Model Download Issues**
- NeMo models are downloaded automatically on first use
- Ensure internet connection for initial model download
- Models are cached in `~/.cache/torch/NeMo/`

## 📄 License

[Your License Here]

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Update documentation
5. Submit a pull request

## 📞 Support

- Create an issue for bugs or feature requests
- Check existing issues before creating new ones
- Provide detailed information including error logs

---

**Built with ❤️ using NVIDIA NeMo and modern Python practices**