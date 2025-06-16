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
- [Improving Accuracy](../guide/improving-accuracy-condensed.md) - Add vocabulary
- [FAQ](../faq.md) - Troubleshooting