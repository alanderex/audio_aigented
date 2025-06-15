# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Commands

```bash
# Install dependencies (using uv)
uv pip install -e .              # Install package
uv pip install -e ".[dev]"       # Install with dev dependencies

# Run the main pipeline
uv run python main.py --input-dir ./inputs
uv run python main.py --input-dir ./inputs --output-dir ./outputs --device cuda
uv run python main.py --input-dir ./inputs --disable-diarization  # Faster processing without speaker diarization

# Testing
uv run pytest                                        # Run all tests
uv run pytest --cov=src --cov-report=term-missing  # With coverage report
uv run pytest tests/test_models.py                  # Run specific test file
uv run pytest -v                                    # Verbose output

# Code quality
uv run ruff check .              # Lint code
uv run ruff format .             # Format code
uv run mypy src/                 # Type checking
```

## Architecture Overview

This is a GPU-accelerated audio transcription pipeline using NVIDIA NeMo for ASR (Automatic Speech Recognition) with speaker diarization capabilities.

### Pipeline Flow

The system follows a 5-stage pipeline architecture:

1. **Audio Loading** (`src/audio_aigented/audio/loader.py`):
   - Loads `.wav` files from input directory
   - Validates audio format and properties
   - Resamples to target sample rate (16kHz)

2. **ASR Transcription** (`src/audio_aigented/transcription/asr.py`):
   - Uses NVIDIA NeMo conformer models
   - GPU-accelerated processing
   - Generates timestamped segments with confidence scores

3. **Speaker Diarization** (`src/audio_aigented/diarization/diarizer.py`):
   - Identifies different speakers in audio
   - Uses NVIDIA NeMo's clustering diarization
   - Assigns speaker labels (SPEAKER_00, SPEAKER_01, etc.)

4. **Output Formatting** (`src/audio_aigented/formatting/formatter.py`):
   - Structures transcription results
   - Combines ASR output with speaker information
   - Prepares multiple output formats

5. **File Writing** (`src/audio_aigented/output/writer.py`):
   - Creates per-audio-file output directories
   - Writes JSON (structured data), TXT (human-readable), and attributed TXT (theater-style dialog)

### Key Design Patterns

- **Configuration-Driven**: Uses YAML configs via OmegaConf (`config/default.yaml`)
- **Pydantic Models**: Type-safe data validation (`src/audio_aigented/models/schemas.py`)
- **Pipeline Orchestration**: Main pipeline class coordinates all stages (`src/audio_aigented/pipeline.py`)
- **Error Handling**: Comprehensive error handling with fallbacks
- **Caching**: Smart caching to avoid re-processing files

### Main Entry Points

- **CLI**: `main.py` - Click-based command line interface
- **API**: `TranscriptionPipeline` class in `src/audio_aigented/pipeline.py`
- **Processing Methods**:
  - `process_directory(Path)` - Process all WAV files in a directory
  - `process_single_file(Path)` - Process a single audio file

### Output Structure

For each audio file, creates a directory with three output files:
- `transcript.json` - Structured data with timestamps, confidence scores, speaker IDs
- `transcript.txt` - Human-readable format with statistics
- `transcript_attributed.txt` - Theater-style dialog format with speaker labels

## Important Notes

- The project uses `uv` as the package manager (not pip directly)
- NVIDIA NeMo models are downloaded automatically on first use
- GPU (CUDA) is strongly recommended for performance
- Default model is `stt_en_conformer_ctc_large` (configurable)
- Speaker diarization can be disabled for faster processing
- All paths in the codebase should be absolute, not relative

## Previous Instructions

- Review the code thoroughly
- Read root directory
- Make Speaker Diarization work