# Quick Start

Get transcribing in under 5 minutes.

## Install

```bash
# Clone and install
git clone https://github.com/yourusername/audio_aigented
cd audio_aigented
uv pip install -e .
```

## Basic Usage

```bash
# 1. Add audio files
mkdir inputs
cp your_audio.wav inputs/

# 2. Run transcription
python main.py -i ./inputs

# 3. Check results
cat outputs/your_audio/transcript.txt
```

## Common Commands

```bash
# Fast mode (no speakers)
python main.py -i ./inputs --disable-diarization

# With vocabulary
python main.py -i ./inputs --vocabulary-file terms.txt

# Fastest model
python main.py -i ./inputs --model-name nvidia/parakeet-tdt-0.6b-v2

# CPU only
python main.py -i ./inputs --device cpu
```

## Command Line Options

### Required Options
- `--input-dir`, `-i` - Directory containing .wav audio files to process

### Configuration Options
- `--config`, `-c` - Path to configuration YAML file
- `--output-dir`, `-o` - Output directory for transcription results
- `--log-level`, `-l` - Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO

### Model & Processing Options
- `--device` - Device for ASR processing (cuda, cpu)
- `--model-name` - NVIDIA NeMo model name to use for transcription
- `--enable-diarization` / `--disable-diarization` - Enable/disable speaker diarization. Default: enabled
- `--beam-size` - Beam search width for decoding (default: 4)

### Enhancement Options
- `--vocabulary-file` - Path to custom vocabulary file for improved accuracy
- `--content-dir` - Directory containing companion content files (can be used multiple times)

### Output Options
- `--formats` - Output formats (comma-separated: json,txt)

### Utility Options
- `--dry-run` - Show what would be processed without actually processing
- `--clear-cache` - Clear the cache before processing
- `--create-context-templates` - Create context template files for each audio file
- `--version` - Show version information

### Examples

```bash
# Basic usage
python main.py --input-dir ./audio

# Full configuration
python main.py -i ./audio -o ./results -c ./config.yaml --log-level DEBUG

# Fast processing with specific model
python main.py -i ./audio --model-name nvidia/parakeet-tdt-0.6b-v2 --disable-diarization

# With vocabulary and content directories
python main.py -i ./audio --vocabulary-file terms.txt --content-dir ./docs --content-dir ./notes

# Dry run to preview files
python main.py -i ./audio --dry-run

# Clear cache only
python main.py --clear-cache

# Create context templates
python main.py -i ./audio --create-context-templates
```

## Output Files

```
outputs/your_audio/
├── transcript.json           # Full data with timestamps
├── transcript.txt            # Human-readable summary  
└── transcript_attributed.txt # Speaker dialog
```

## Python API

```python
from pathlib import Path
from audio_aigented.pipeline import TranscriptionPipeline

# Process file
pipeline = TranscriptionPipeline()
result = pipeline.process_single_file(Path("audio.wav"))

# Get text
print(result.transcription.full_text)

# Get speakers
for seg in result.transcription.segments:
    print(f"{seg.speaker_id}: {seg.text}")
```

## Next Steps

- [Configuration Guide](configuration.md) - Customize settings
- [Improving Accuracy](../guide/improving-accuracy.md) - Add vocabulary
- [FAQ](../faq.md) - Troubleshooting