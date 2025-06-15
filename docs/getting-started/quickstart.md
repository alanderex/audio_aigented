# Quick Start

Get up and running with the Audio Transcription Pipeline in minutes.

## Prerequisites

Before starting, ensure you have:
- ✅ Installed the package ([Installation Guide](installation.md))
- ✅ NVIDIA GPU with CUDA support (or CPU for slower processing)
- ✅ Some `.wav` audio files to transcribe

## Basic Usage

### 1. Process Your First Audio File

Place your audio files in the `inputs/` directory:

```bash
mkdir -p inputs
cp your_audio.wav inputs/
```

Run the transcription:

```bash
uv run python main.py --input-dir ./inputs
```

The pipeline will:
1. Load and validate your audio files
2. Perform speaker diarization (identify who speaks when)
3. Transcribe each speaker segment
4. Generate output files in multiple formats

### 2. Check the Results

Your transcriptions will be in the `outputs/` directory:

```
outputs/
└── your_audio/
    ├── transcript.json              # Structured data with timestamps
    ├── transcript.txt               # Human-readable format
    └── transcript_attributed.txt    # Theater-style with speakers
```

## Common Use Cases

### Meeting Transcription

For meeting recordings with multiple speakers:

```bash
# Process with speaker identification
uv run python main.py --input-dir ./meetings --output-dir ./meeting_transcripts

# Review speaker-attributed transcript
cat outputs/team_meeting/transcript_attributed.txt
```

Output example:
```
SPEAKER_00: Good morning everyone, let's start with the status updates.
SPEAKER_01: I've completed the frontend implementation...
SPEAKER_02: The backend API is ready for testing...
```

### Single Speaker Audio

For podcasts or single-speaker content:

```bash
# Disable diarization for faster processing
uv run python main.py --input-dir ./podcasts --disable-diarization
```

### Batch Processing

Process multiple files at once:

```bash
# Create input structure
inputs/
├── interview_001.wav
├── interview_002.wav
└── interview_003.wav

# Process all files
uv run python main.py --input-dir ./inputs

# Check summary
cat outputs/processing_summary.txt
```

## Using Different Models

### Fast Processing (Lower Accuracy)

```bash
# Create custom config
cat > config/fast.yaml << EOF
model:
  name: "stt_en_conformer_ctc_small"
processing:
  batch_size: 8
EOF

# Use fast config
uv run python main.py --input-dir ./inputs --config config/fast.yaml
```

### High Accuracy (Slower)

```bash
# Use large model (default)
uv run python main.py --input-dir ./inputs --device cuda
```

### CPU Processing

```bash
# When no GPU available
uv run python main.py --input-dir ./inputs --device cpu
```

## Python API Quick Start

### Basic Example

```python
from pathlib import Path
from audio_aigented.pipeline import TranscriptionPipeline

# Initialize pipeline
pipeline = TranscriptionPipeline()

# Process a single file
result = pipeline.process_single_file(Path("meeting.wav"))

# Print the transcription
print(result.transcription.full_text)

# Access speaker segments
for segment in result.transcription.segments:
    print(f"{segment.speaker_id}: {segment.text}")
```

### Custom Configuration

```python
from audio_aigented.pipeline import TranscriptionPipeline
from audio_aigented.config import PipelineConfig

# Custom configuration
config = PipelineConfig(
    device="cuda",
    enable_diarization=True,
    output_formats=["json", "attributed_txt"]
)

# Initialize with custom config
pipeline = TranscriptionPipeline(config=config)

# Process files
results = pipeline.process_directory(Path("./inputs"))
```

### Accessing Results

```python
# Process file
result = pipeline.process_single_file(Path("audio.wav"))

# Access structured data
print(f"Duration: {result.audio_info.duration}s")
print(f"Processing time: {result.processing_info.processing_time}s")

# Iterate through segments
for segment in result.transcription.segments:
    print(f"[{segment.start_time:.1f}s - {segment.end_time:.1f}s] "
          f"{segment.speaker_id}: {segment.text} "
          f"(confidence: {segment.confidence:.2f})")
```

## Docker Quick Start

### Using Docker Compose

```bash
# Place audio files in inputs/
cp *.wav inputs/

# Run with Docker
docker-compose run --rm audio-transcription

# Results will be in outputs/
```

### Custom Docker Run

```bash
# Process specific directory
docker run --rm --gpus all \
  -v /path/to/audio:/data/inputs \
  -v /path/to/results:/data/outputs \
  audio-aigented
```

## Tips for Best Results

### Audio Quality
- Use high-quality recordings (16kHz or higher)
- Minimize background noise
- Ensure clear speech

### File Preparation
- Convert to WAV format if needed
- Split very long files (>1 hour) for better performance
- Name files descriptively for easier organization

### Performance Optimization
- Use GPU for 5-10x faster processing
- Disable diarization for single-speaker content
- Process files in batches for efficiency

## Troubleshooting Quick Fixes

### No Output Files
```bash
# Check for errors
uv run python main.py --input-dir ./inputs --log-level DEBUG

# Verify input files
ls -la inputs/*.wav
```

### Slow Processing
```bash
# Use smaller model
uv run python main.py --input-dir ./inputs \
  --config config/fast.yaml

# Check GPU usage
nvidia-smi
```

### Memory Issues
```bash
# Reduce batch size
echo "processing:
  batch_size: 1" > config/low_memory.yaml

uv run python main.py --config config/low_memory.yaml
```

## Next Steps

- Learn about [Configuration Options](configuration.md)
- Explore [Speaker Diarization](../guide/diarization.md)
- Understand [Output Formats](../guide/output-formats.md)
- Set up [Docker Deployment](../deployment/docker.md)