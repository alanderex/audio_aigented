# Audio Processing

This guide covers how the pipeline handles audio files, from loading to preprocessing.

## Supported Formats

### Primary Format
- **WAV** (`.wav`) - Uncompressed audio, best quality
  - Sample rates: 8kHz - 48kHz
  - Bit depths: 16-bit, 24-bit, 32-bit
  - Channels: Mono or stereo

### Format Conversion
For other formats (MP3, M4A, etc.), convert to WAV first:

```bash
# Using ffmpeg
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav

# Using sox
sox input.mp3 -r 16000 -c 1 output.wav
```

## Audio Loading Process

### 1. File Validation

The `AudioLoader` validates files before processing:

```python
def validate_audio_file(file_path: Path) -> bool:
    # Check file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Check file extension
    if file_path.suffix.lower() != '.wav':
        raise ValueError(f"Unsupported format: {file_path.suffix}")
    
    # Check file size
    if file_path.stat().st_size == 0:
        raise ValueError("Audio file is empty")
```

### 2. Audio Loading

Files are loaded using `soundfile`:

```python
import soundfile as sf

def load_audio(file_path: Path) -> Tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(file_path)
    return audio, sample_rate
```

### 3. Preprocessing

#### Resampling
All audio is resampled to 16kHz for optimal ASR performance:

```python
import librosa

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000):
    if orig_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return audio
```

#### Channel Conversion
Stereo audio is converted to mono:

```python
def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        # Average all channels
        audio = np.mean(audio, axis=1)
    return audio
```

#### Normalization
Audio is normalized to prevent clipping:

```python
def normalize_audio(audio: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.95
    return audio
```

## Audio Quality Considerations

### Optimal Input Characteristics
- **Sample Rate**: 16kHz or higher
- **Bit Depth**: 16-bit or higher
- **Dynamic Range**: -20dB to -3dB peaks
- **Background Noise**: < -40dB

### Common Issues and Solutions

#### Low Sample Rate
**Problem**: Audio recorded at 8kHz sounds muffled
**Solution**: Use upsampling with interpolation
```python
# Upsample from 8kHz to 16kHz
audio_16k = librosa.resample(audio_8k, orig_sr=8000, target_sr=16000)
```

#### Clipping/Distortion
**Problem**: Audio peaks exceed 0dB causing distortion
**Solution**: Apply dynamic range compression
```python
def reduce_clipping(audio: np.ndarray, threshold: float = 0.95):
    audio = np.tanh(audio / threshold) * threshold
    return audio
```

#### Background Noise
**Problem**: High background noise affects accuracy
**Solution**: Apply noise reduction (optional preprocessing)
```python
# Using noisereduce library
import noisereduce as nr
audio_clean = nr.reduce_noise(y=audio, sr=sample_rate)
```

## Batch Processing

### Efficient File Loading

Process multiple files efficiently:

```python
from concurrent.futures import ThreadPoolExecutor

def load_audio_batch(file_paths: List[Path], max_workers: int = 4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(load_and_preprocess, file_paths)
    return list(results)
```

### Memory Management

For large files, use chunked processing:

```python
def process_large_audio(file_path: Path, chunk_size: int = 30):
    """Process audio in 30-second chunks"""
    audio, sr = sf.read(file_path)
    chunk_samples = chunk_size * sr
    
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        yield chunk
```

## Audio Analysis

### Duration Calculation

```python
def get_audio_duration(file_path: Path) -> float:
    info = sf.info(file_path)
    return info.duration
```

### Audio Statistics

```python
def analyze_audio(audio: np.ndarray, sample_rate: int) -> dict:
    return {
        "duration": len(audio) / sample_rate,
        "rms_level": np.sqrt(np.mean(audio**2)),
        "peak_level": np.max(np.abs(audio)),
        "dynamic_range": 20 * np.log10(np.max(np.abs(audio)) / np.sqrt(np.mean(audio**2))),
        "silence_ratio": np.sum(np.abs(audio) < 0.01) / len(audio)
    }
```

## Configuration Options

### Audio Processing Settings

```yaml
audio:
  sample_rate: 16000          # Target sample rate
  mono: true                  # Convert to mono
  normalize: true             # Normalize audio levels
  max_duration: 3600          # Maximum duration in seconds
  min_duration: 0.5           # Minimum duration in seconds
  
preprocessing:
  remove_silence: false       # Remove leading/trailing silence
  noise_reduction: false      # Apply noise reduction
  compression: false          # Apply dynamic range compression
```

### Advanced Options

```yaml
audio:
  resampling_method: "kaiser_best"  # librosa resampling method
  normalization_target: -3.0        # Target peak level in dB
  silence_threshold: -40.0          # Silence detection threshold
  
quality_checks:
  min_sample_rate: 8000            # Reject files below this rate
  max_noise_level: -30.0           # Reject files with high noise
  check_clipping: true             # Check for clipped samples
```

## Troubleshooting

### Common Errors

#### "Audio file not found"
```python
# Check file path
file_path = Path("./inputs/audio.wav")
print(f"File exists: {file_path.exists()}")
print(f"Absolute path: {file_path.absolute()}")
```

#### "Unsupported audio format"
```python
# Check file format
import wave
with wave.open(str(file_path), 'rb') as wav_file:
    print(f"Channels: {wav_file.getnchannels()}")
    print(f"Sample rate: {wav_file.getframerate()}")
    print(f"Bit depth: {wav_file.getsampwidth() * 8}")
```

#### "Memory error with large files"
```python
# Use memory mapping for large files
import numpy as np
audio = np.memmap(file_path, dtype='float32', mode='r')
```

### Performance Tips

1. **Preprocessing Cache**: Cache preprocessed audio
   ```python
   cache_path = Path(f".cache/{file_path.stem}_16k.npy")
   if cache_path.exists():
       audio = np.load(cache_path)
   else:
       audio = preprocess_audio(file_path)
       np.save(cache_path, audio)
   ```

2. **Parallel Loading**: Load files in parallel
   ```python
   from multiprocessing import Pool
   
   with Pool(processes=4) as pool:
       audio_data = pool.map(load_audio, file_paths)
   ```

3. **Streaming Processing**: For real-time applications
   ```python
   def stream_audio(file_path: Path, block_size: int = 1024):
       with sf.SoundFile(file_path) as f:
           while True:
               audio_block = f.read(block_size)
               if audio_block.size == 0:
                   break
               yield audio_block
   ```

## Best Practices

### Input Preparation
1. Record at 16kHz or higher
2. Use good quality microphones
3. Minimize background noise
4. Avoid compression artifacts

### File Organization
```
inputs/
├── high_quality/      # 16kHz+, low noise
├── needs_processing/  # Requires preprocessing
└── archived/          # Processed files
```

### Quality Assurance
```python
def validate_audio_quality(file_path: Path) -> bool:
    audio, sr = load_audio(file_path)
    stats = analyze_audio(audio, sr)
    
    # Check quality criteria
    if stats["duration"] < 0.5:
        logger.warning(f"Audio too short: {stats['duration']}s")
        return False
    
    if stats["silence_ratio"] > 0.8:
        logger.warning(f"Too much silence: {stats['silence_ratio']*100}%")
        return False
    
    if stats["peak_level"] > 0.99:
        logger.warning("Audio may be clipped")
        return False
    
    return True
```