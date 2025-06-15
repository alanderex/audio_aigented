# Transcription

This guide covers the automatic speech recognition (ASR) component of the pipeline, which converts audio to text using NVIDIA NeMo models.

## Overview

The transcription module:
- Uses state-of-the-art Conformer CTC models
- Processes audio segments from diarization
- Generates timestamped text with confidence scores
- Supports GPU acceleration for fast processing

## ASR Models

### Available Models

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| `stt_en_conformer_ctc_small` | Fast (15x real-time) | Good | Quick drafts, real-time |
| `stt_en_conformer_ctc_medium` | Medium (8x real-time) | Better | Balanced performance |
| `stt_en_conformer_ctc_large` | Slow (4x real-time) | Best | High accuracy needs |

### Model Architecture

The Conformer models combine:
- **Convolutional layers**: Capture local patterns
- **Self-attention**: Model long-range dependencies
- **Feed-forward networks**: Process representations
- **CTC decoder**: Convert to text

## How Transcription Works

### 1. Segment Processing

With diarization enabled:
```python
# Process each speaker segment
for segment in diarization_segments:
    audio_segment = extract_audio(segment.start, segment.end)
    text = asr_model.transcribe(audio_segment)
    results.append({
        "speaker": segment.speaker,
        "text": text,
        "start": segment.start,
        "end": segment.end
    })
```

### 2. Direct Transcription

Without diarization:
```python
# Process entire audio as one segment
full_text = asr_model.transcribe(audio_data)
results = [{
    "speaker": "SPEAKER_00",
    "text": full_text,
    "start": 0.0,
    "end": audio_duration
}]
```

### 3. Confidence Scoring

Each word includes confidence scores:
```python
# Word-level confidence
transcription = asr_model.transcribe_with_confidence(audio)
for word in transcription.words:
    print(f"{word.text}: {word.confidence:.2f}")
```

## Configuration

### Basic Settings

```yaml
transcription:
  model:
    name: "stt_en_conformer_ctc_large"
    
  device: "cuda"              # GPU acceleration
  batch_size: 4               # Parallel processing
  
  decoding:
    beam_size: 10             # Beam search width
    lm_weight: 0.0            # Language model weight
    word_insertion_penalty: 0.0
```

### Advanced Options

```yaml
transcription:
  preprocessing:
    normalize_audio: true     # Normalize input levels
    remove_noise: false       # Noise reduction
    
  model:
    compute_dtype: "float16"  # Mixed precision for speed
    
  postprocessing:
    punctuation: false        # Add punctuation (separate model)
    capitalize: false         # Capitalize sentences
    word_timestamps: true     # Include word timings
```

## Usage Examples

### Command Line

```bash
# Default large model
uv run python main.py --input-dir ./inputs

# Fast processing with small model
uv run python main.py --input-dir ./inputs \
  --model stt_en_conformer_ctc_small

# CPU processing
uv run python main.py --input-dir ./inputs --device cpu
```

### Python API

```python
from audio_aigented.transcription import Transcriber
from audio_aigented.config import TranscriptionConfig

# Configure transcriber
config = TranscriptionConfig(
    model_name="stt_en_conformer_ctc_medium",
    device="cuda",
    batch_size=8
)

# Initialize
transcriber = Transcriber(config)

# Transcribe audio
result = transcriber.transcribe(audio_path)
print(result.text)

# With timestamps
for segment in result.segments:
    print(f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}")
```

### Batch Processing

```python
# Process multiple files efficiently
audio_files = ["file1.wav", "file2.wav", "file3.wav"]
results = transcriber.transcribe_batch(audio_files)

for file, result in zip(audio_files, results):
    print(f"{file}: {result.text}")
```

## Output Format

### Segment Structure

```json
{
  "text": "Hello, how are you doing today?",
  "start_time": 0.5,
  "end_time": 2.3,
  "confidence": 0.95,
  "speaker_id": "SPEAKER_00",
  "words": [
    {"text": "Hello", "start": 0.5, "end": 0.8, "confidence": 0.98},
    {"text": "how", "start": 0.9, "end": 1.0, "confidence": 0.96},
    {"text": "are", "start": 1.0, "end": 1.1, "confidence": 0.97},
    {"text": "you", "start": 1.1, "end": 1.2, "confidence": 0.95},
    {"text": "doing", "start": 1.3, "end": 1.5, "confidence": 0.94},
    {"text": "today", "start": 1.6, "end": 1.9, "confidence": 0.93}
  ]
}
```

## Performance Optimization

### GPU Utilization

Monitor GPU usage:
```bash
# During transcription
watch -n 1 nvidia-smi

# Check memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Speed vs Accuracy Trade-offs

| Setting | Impact on Speed | Impact on Accuracy |
|---------|-----------------|-------------------|
| Smaller model | +++ | -- |
| Larger batch size | ++ | 0 |
| FP16 precision | ++ | - |
| Beam size reduction | + | - |
| Disable word timestamps | + | 0 |

### Memory Management

```yaml
# For limited GPU memory
transcription:
  batch_size: 1              # Process one at a time
  model:
    compute_dtype: "float16" # Use mixed precision
  
  memory:
    max_audio_length: 300    # Process in 5-minute chunks
```

## Quality Improvements

### Pre-processing

```python
# Enhance audio before transcription
def preprocess_for_asr(audio, sample_rate):
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    # High-pass filter for speech clarity
    from scipy import signal
    b, a = signal.butter(4, 80/(sample_rate/2), 'high')
    audio = signal.filtfilt(b, a, audio)
    
    return audio
```

### Post-processing

```python
# Clean up transcription output
def postprocess_text(text):
    # Remove filler words
    fillers = ["um", "uh", "er", "ah"]
    words = text.split()
    words = [w for w in words if w.lower() not in fillers]
    
    # Fix common errors
    replacements = {
        "gonna": "going to",
        "wanna": "want to",
        "kinda": "kind of"
    }
    
    text = " ".join(words)
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text
```

## Troubleshooting

### Common Issues

#### Low Accuracy

**Causes & Solutions**:

1. **Poor audio quality**
   ```python
   # Check audio statistics
   stats = analyze_audio(audio_path)
   if stats["snr"] < 10:  # Low signal-to-noise ratio
       print("Warning: Audio quality may affect accuracy")
   ```

2. **Wrong model for content**
   - Technical content → Use large model
   - Casual speech → Medium model often sufficient
   - Non-native speakers → Consider specialized models

3. **Preprocessing needed**
   ```yaml
   transcription:
     preprocessing:
       normalize_audio: true
       remove_noise: true
   ```

#### Slow Processing

**Solutions**:

1. **Use smaller model**
   ```bash
   uv run python main.py --model stt_en_conformer_ctc_small
   ```

2. **Increase batch size**
   ```yaml
   transcription:
     batch_size: 16  # If GPU memory allows
   ```

3. **Enable mixed precision**
   ```yaml
   transcription:
     model:
       compute_dtype: "float16"
   ```

#### Out of Memory

**Solutions**:

1. **Reduce batch size**
   ```yaml
   transcription:
     batch_size: 1
   ```

2. **Process in chunks**
   ```python
   # Split long audio
   def transcribe_long_audio(audio_path, chunk_duration=300):
       chunks = split_audio(audio_path, chunk_duration)
       results = []
       for chunk in chunks:
           result = transcriber.transcribe(chunk)
           results.append(result)
       return merge_results(results)
   ```

## Best Practices

### Model Selection

1. **Start with medium model** - Good balance
2. **Use large for**:
   - Legal/medical transcription
   - High-stakes content
   - Final production
3. **Use small for**:
   - Drafts and previews
   - Real-time processing
   - High volume, low stakes

### Audio Preparation

1. **Optimal recording**:
   - 16kHz sample rate minimum
   - Quiet environment
   - Good microphone placement

2. **Pre-screening**:
   ```python
   def screen_audio(audio_path):
       duration = get_duration(audio_path)
       if duration < 0.5:
           return False, "Too short"
       if duration > 3600:
           return False, "Too long"
       
       stats = analyze_audio(audio_path)
       if stats["silence_ratio"] > 0.9:
           return False, "Mostly silence"
       
       return True, "OK"
   ```

### Production Deployment

1. **Model caching**:
   ```python
   # Load model once, reuse
   class TranscriptionService:
       def __init__(self):
           self.model = load_model()
       
       def transcribe(self, audio):
           return self.model.transcribe(audio)
   ```

2. **Queue management**:
   ```python
   # Process queue of files
   from queue import Queue
   from threading import Thread
   
   def worker(queue, transcriber):
       while True:
           audio_path = queue.get()
           result = transcriber.transcribe(audio_path)
           save_result(result)
           queue.task_done()
   ```

## Advanced Topics

### Custom Vocabulary

```python
# Add domain-specific terms
custom_vocab = ["COVID-19", "mRNA", "blockchain"]
transcriber.add_vocabulary(custom_vocab)
```

### Streaming Transcription

```python
# Real-time transcription
async def stream_transcribe(audio_stream):
    buffer_size = 16000  # 1 second at 16kHz
    buffer = []
    
    async for chunk in audio_stream:
        buffer.extend(chunk)
        
        if len(buffer) >= buffer_size:
            # Process buffer
            text = transcriber.transcribe(buffer)
            yield text
            
            # Keep overlap for context
            buffer = buffer[-buffer_size//4:]
```

### Multi-language Support

```python
# Detect language and select model
def transcribe_multilingual(audio_path):
    language = detect_language(audio_path)
    
    model_map = {
        "en": "stt_en_conformer_ctc_large",
        "es": "stt_es_conformer_ctc_large",
        "fr": "stt_fr_conformer_ctc_large"
    }
    
    model_name = model_map.get(language, "stt_en_conformer_ctc_large")
    transcriber = Transcriber(model_name=model_name)
    
    return transcriber.transcribe(audio_path)
```