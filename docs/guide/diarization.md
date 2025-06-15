# Speaker Diarization

Speaker diarization is the process of identifying "who spoke when" in an audio recording. This guide covers how the pipeline performs speaker diarization using NVIDIA NeMo's neural diarization models.

## Overview

The diarization process consists of:
1. **Voice Activity Detection (VAD)** - Identify speech regions
2. **Speaker Embedding Extraction** - Create voice fingerprints
3. **Clustering** - Group similar voice segments
4. **Segment Refinement** - Clean up speaker boundaries

## How It Works

### 1. Voice Activity Detection

First, the pipeline detects which parts of the audio contain speech:

```python
# VAD identifies speech regions
vad_segments = vad_model.get_speech_segments(audio)
# Output: [(0.5, 2.3), (3.1, 5.7), ...]  # (start, end) times
```

### 2. Speaker Embeddings

For each speech segment, the pipeline extracts a speaker embedding (voice fingerprint):

```python
# Extract embeddings using TitanNet
embeddings = embedding_model.get_embeddings(speech_segments)
# Output: Array of 192-dimensional vectors
```

### 3. Spectral Clustering

Embeddings are clustered to identify unique speakers:

```python
# Cluster embeddings to find speakers
speaker_labels = clustering_backend.cluster(embeddings)
# Output: [0, 0, 1, 0, 1, 2, ...]  # Speaker IDs for each segment
```

### 4. Output Format

Diarization results are saved in RTTM format:

```
SPEAKER audio_file 1 0.500 1.800 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER audio_file 1 3.100 2.600 <NA> <NA> SPEAKER_01 <NA> <NA>
```

## Configuration

### Basic Settings

```yaml
diarization:
  enable: true                    # Enable/disable diarization
  device: "cuda"                  # GPU acceleration
  
  vad:
    model_path: "vad_multilingual_marblenet"
    onset: 0.8                    # Speech detection sensitivity
    offset: 0.6                   # Speech end sensitivity
    
  embedding:
    model_path: "titanet-l"       # Speaker embedding model
    window_length: 1.5            # Embedding window size
    shift_length: 0.75            # Window shift
    
  clustering:
    backend: "SC"                 # Spectral Clustering
    parameters:
      oracle_num_speakers: null   # Auto-detect speakers
      max_num_speakers: 8         # Maximum speakers
      enhanced_count_thresholding: true
```

### Advanced Options

```yaml
diarization:
  clustering:
    parameters:
      maj_vote_spk_count: true    # Majority voting
      min_samples_for_nmesc: 6    # Minimum samples for NME-SC
      
  postprocessing:
    onset: 0.5                    # Refine segment starts
    offset: 0.5                   # Refine segment ends
    min_duration_on: 0.1          # Minimum speech duration
    min_duration_off: 0.1         # Minimum silence duration
```

## Usage Examples

### Enable/Disable Diarization

```bash
# With diarization (default)
uv run python main.py --input-dir ./inputs

# Without diarization (faster)
uv run python main.py --input-dir ./inputs --disable-diarization
```

### Specify Number of Speakers

When you know the number of speakers:

```yaml
# config/two_speakers.yaml
diarization:
  clustering:
    parameters:
      oracle_num_speakers: 2
```

```bash
uv run python main.py --config config/two_speakers.yaml
```

### Python API

```python
from audio_aigented.diarization import Diarizer
from audio_aigented.config import DiarizationConfig

# Configure diarizer
config = DiarizationConfig(
    enable=True,
    oracle_num_speakers=3  # Known: 3 speakers
)

# Initialize and run
diarizer = Diarizer(config)
speaker_segments = diarizer.diarize(audio_path)

# Access results
for segment in speaker_segments:
    print(f"{segment.speaker}: {segment.start:.1f}s - {segment.end:.1f}s")
```

## Output Formats

### RTTM File Format

The standard Rich Transcription Time Marked (RTTM) format:

```
SPEAKER file_id 1 start_time duration <NA> <NA> speaker_id <NA> <NA>
```

Example:
```
SPEAKER meeting_audio 1 0.000 5.230 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER meeting_audio 1 5.230 3.120 <NA> <NA> SPEAKER_01 <NA> <NA>
SPEAKER meeting_audio 1 8.350 2.450 <NA> <NA> SPEAKER_00 <NA> <NA>
```

### JSON Format

Structured diarization data:

```json
{
  "speakers": {
    "SPEAKER_00": {
      "segments": [
        {"start": 0.0, "end": 5.23},
        {"start": 8.35, "end": 10.8}
      ],
      "total_duration": 7.68
    },
    "SPEAKER_01": {
      "segments": [
        {"start": 5.23, "end": 8.35}
      ],
      "total_duration": 3.12
    }
  },
  "total_speakers": 2
}
```

## Performance Optimization

### GPU Acceleration

Diarization is significantly faster on GPU:

| Component | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| VAD | 2.5s | 0.3s | 8.3x |
| Embeddings | 45s | 4.5s | 10x |
| Clustering | 1.5s | 1.5s | 1x |
| **Total** | **49s** | **6.3s** | **7.8x** |

*Times for 5-minute audio file*

### Batch Processing

Process multiple files efficiently:

```python
# Batch configuration
config = DiarizationConfig(
    batch_size=8,  # Process 8 files simultaneously
    num_workers=4  # Parallel data loading
)
```

### Memory Management

For long audio files:

```yaml
diarization:
  max_audio_length: 1800  # Process in 30-minute chunks
  overlap_duration: 5     # 5-second overlap between chunks
```

## Quality Tuning

### For Meetings/Conferences

```yaml
# Many speakers, overlapping speech
diarization:
  clustering:
    parameters:
      max_num_speakers: 15
      enhanced_count_thresholding: true
      
  vad:
    onset: 0.7   # More sensitive
    offset: 0.5
```

### For Interviews/Podcasts

```yaml
# Few speakers, clear turn-taking
diarization:
  clustering:
    parameters:
      oracle_num_speakers: 2  # Host + Guest
      
  vad:
    onset: 0.9   # Less sensitive
    offset: 0.7
```

### For Noisy Environments

```yaml
# Background noise, multiple speakers
diarization:
  vad:
    model_path: "vad_multilingual_marblenet"
    onset: 0.85
    offset: 0.65
    
  postprocessing:
    min_duration_on: 0.2   # Longer minimum speech
    min_duration_off: 0.3  # Longer silence gaps
```

## Troubleshooting

### Common Issues

#### Too Many Speakers Detected

**Problem**: System detects more speakers than actually present

**Solutions**:
1. Set `oracle_num_speakers` if known
2. Adjust clustering parameters:
   ```yaml
   clustering:
     parameters:
       max_num_speakers: 4
       enhanced_count_thresholding: true
   ```

#### Speaker Confusion

**Problem**: Same speaker identified as multiple speakers

**Solutions**:
1. Increase embedding window size:
   ```yaml
   embedding:
     window_length: 2.0  # Longer windows
     shift_length: 1.0
   ```

2. Use majority voting:
   ```yaml
   clustering:
     parameters:
       maj_vote_spk_count: true
   ```

#### Missing Short Utterances

**Problem**: Short speech segments not detected

**Solutions**:
```yaml
vad:
  onset: 0.7    # Lower threshold
  offset: 0.5
  
postprocessing:
  min_duration_on: 0.05  # Shorter minimum
```

### Performance Issues

#### Slow Processing

1. **Use GPU**: Ensure CUDA is available
   ```bash
   nvidia-smi  # Check GPU availability
   ```

2. **Reduce audio length**: Process in chunks
   ```yaml
   diarization:
     max_audio_length: 900  # 15-minute chunks
   ```

3. **Disable if not needed**:
   ```bash
   uv run python main.py --disable-diarization
   ```

#### Out of Memory

1. **Reduce batch size**:
   ```yaml
   diarization:
     batch_size: 1
   ```

2. **Use CPU clustering**:
   ```yaml
   clustering:
     backend: "SC_cpu"
   ```

## Best Practices

### Audio Quality
1. **Clear audio**: Minimize background noise
2. **Good separation**: Speakers should not talk over each other
3. **Consistent volume**: Similar audio levels for all speakers

### Configuration
1. **Start with defaults**: Work well for most cases
2. **Set oracle_num_speakers**: When known, always specify
3. **Test on samples**: Tune parameters on representative clips

### Integration
```python
# Example: Post-process diarization results
def merge_short_segments(segments, min_gap=0.5):
    """Merge segments from same speaker with small gaps"""
    merged = []
    for segment in segments:
        if merged and merged[-1].speaker == segment.speaker:
            if segment.start - merged[-1].end < min_gap:
                merged[-1].end = segment.end
                continue
        merged.append(segment)
    return merged
```

## Advanced Topics

### Custom Embedding Models

```python
# Use custom speaker embedding model
from nemo.collections.asr.models import EncDecSpeakerLabelModel

custom_model = EncDecSpeakerLabelModel.from_pretrained("your_model")
diarizer.embedding_model = custom_model
```

### Real-time Diarization

```python
# Streaming diarization (experimental)
async def stream_diarization(audio_stream):
    buffer = []
    for chunk in audio_stream:
        buffer.append(chunk)
        if len(buffer) > window_size:
            segments = diarizer.process_window(buffer)
            yield segments
            buffer = buffer[overlap:]
```

### Multi-stage Diarization

```python
# Hierarchical clustering for many speakers
def hierarchical_diarization(audio_path, max_speakers=20):
    # Stage 1: Initial clustering
    initial_segments = diarizer.diarize(
        audio_path, 
        max_speakers=max_speakers
    )
    
    # Stage 2: Refine with sub-clustering
    refined_segments = []
    for speaker_group in group_by_speaker(initial_segments):
        if len(speaker_group) > threshold:
            sub_segments = diarizer.diarize(
                speaker_group,
                max_speakers=5
            )
            refined_segments.extend(sub_segments)
    
    return refined_segments
```