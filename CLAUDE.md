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
uv run python main.py --input-dir ./inputs --num-speakers 4  # Hint for 4 speakers (improves diarization)
uv run python main.py --input-dir ./inputs --model-name nvidia/parakeet-tdt-0.6b-v2  # Use faster Parakeet model
uv run python main.py --input-dir ./inputs --clear-cache  # Clear cache before processing
uv run python main.py --clear-cache  # Clear cache only (no processing)
uv run python main.py --input-dir ./inputs --create-context-templates  # Create context files
uv run python main.py --input-dir ./inputs --vocabulary-file vocab.txt  # Use global vocabulary
uv run python main.py --input-dir ./inputs --content-file agenda.html  # Extract context from HTML
uv run python main.py --input-dir ./inputs --content-dir ./docs  # Extract from directory

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

The system follows a diarization-first pipeline architecture:

1. **Audio Loading** (`src/audio_aigented/audio/loader.py`):
   - Loads `.wav` files from input directory
   - Validates audio format and properties
   - Resamples to target sample rate (16kHz)

2. **Speaker Diarization** (`src/audio_aigented/diarization/diarizer.py`) - *When enabled*:
   - Processes entire audio file to identify speaker segments
   - Uses NVIDIA NeMo's clustering diarization with TitanNet embeddings
   - Outputs speaker segments with timestamps (start, end, speaker_id)
   - Parses RTTM files with support for filenames containing spaces

3. **ASR Transcription** (`src/audio_aigented/transcription/asr.py`):
   - If diarization enabled: Transcribes each speaker segment individually
   - If diarization disabled: Assumes single speaker (SPEAKER_00) and transcribes full audio
   - Uses NVIDIA NeMo conformer models with GPU acceleration
   - Generates timestamped segments with confidence scores

4. **Output Formatting** (`src/audio_aigented/formatting/formatter.py`):
   - Structures transcription results with speaker attribution
   - Aligns ASR output with diarization segments
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
- Alternative models available:
  - `nvidia/parakeet-tdt-0.6b-v2` - Faster transducer model with 600M parameters
  - `stt_en_conformer_ctc_small/medium` - Smaller CTC models for faster processing
- Speaker diarization can be disabled for faster processing
- Optional speaker count hint (--num-speakers) improves diarization accuracy for known speaker counts
- All paths in the codebase should be absolute, not relative
- Diarization is performed BEFORE transcription for better speaker attribution
- When testing, use audio files with multiple speakers from `/home/hendorf/code/audio_ai/app/data/input`

## Previous Instructions

- Review the code thoroughly
- Read root directory
- Make Speaker Diarization work