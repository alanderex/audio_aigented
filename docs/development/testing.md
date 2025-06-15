# Testing Guide

This guide covers the testing strategy, tools, and best practices for the Audio Transcription Pipeline.

## Testing Philosophy

We follow these principles:
- **Test-Driven Development (TDD)**: Write tests first when possible
- **Comprehensive Coverage**: Aim for >80% code coverage
- **Fast Feedback**: Tests should run quickly
- **Isolation**: Tests should not depend on external services
- **Clarity**: Tests should document behavior

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_audio_loader.py
│   ├── test_diarizer.py
│   └── test_transcriber.py
├── integration/             # Integration tests
│   ├── test_pipeline.py
│   └── test_output_formats.py
├── fixtures/               # Test data and fixtures
│   ├── audio/
│   └── config/
├── conftest.py            # Shared pytest configuration
└── test_e2e.py           # End-to-end tests
```

## Running Tests

### Basic Commands

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_audio_loader.py

# Run specific test
uv run pytest tests/test_audio_loader.py::test_load_wav_file

# Run tests matching pattern
uv run pytest -k "test_diarization"
```

### Coverage Reports

```bash
# Generate coverage report
uv run pytest --cov=src --cov-report=term-missing

# Generate HTML coverage report
uv run pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser

# Coverage with branch coverage
uv run pytest --cov=src --cov-branch --cov-report=term-missing
```

### Test Markers

```bash
# Run only fast tests
uv run pytest -m "not slow"

# Run only GPU tests
uv run pytest -m "gpu"

# Run integration tests
uv run pytest -m "integration"
```

## Writing Tests

### Unit Tests

Unit tests focus on individual functions or classes:

```python
# tests/unit/test_audio_loader.py
import pytest
import numpy as np
from pathlib import Path
from audio_aigented.audio.loader import AudioLoader, AudioLoadError

class TestAudioLoader:
    """Test AudioLoader functionality."""
    
    def test_load_valid_wav(self, sample_wav_path):
        """Test loading a valid WAV file."""
        loader = AudioLoader()
        audio, sample_rate = loader.load(sample_wav_path)
        
        assert isinstance(audio, np.ndarray)
        assert sample_rate == 16000
        assert len(audio) > 0
        
    def test_load_missing_file(self):
        """Test error handling for missing files."""
        loader = AudioLoader()
        
        with pytest.raises(AudioLoadError) as exc_info:
            loader.load(Path("nonexistent.wav"))
            
        assert "not found" in str(exc_info.value)
        
    def test_resample_audio(self):
        """Test audio resampling."""
        loader = AudioLoader(target_sample_rate=16000)
        
        # Create 8kHz audio
        audio_8k = np.random.randn(8000)
        
        # Resample to 16kHz
        audio_16k = loader.resample(audio_8k, 8000, 16000)
        
        assert len(audio_16k) == 16000
        
    @pytest.mark.parametrize("sample_rate,expected", [
        (8000, 16000),
        (22050, 16000),
        (44100, 16000),
        (48000, 16000),
    ])
    def test_various_sample_rates(self, sample_rate, expected):
        """Test resampling from various sample rates."""
        loader = AudioLoader(target_sample_rate=expected)
        audio = np.random.randn(sample_rate)
        
        resampled = loader.resample(audio, sample_rate, expected)
        
        assert len(resampled) == expected
```

### Integration Tests

Integration tests verify component interactions:

```python
# tests/integration/test_pipeline.py
import pytest
from pathlib import Path
from audio_aigented.pipeline import TranscriptionPipeline
from audio_aigented.config import PipelineConfig

class TestPipeline:
    """Test complete pipeline integration."""
    
    def test_process_single_file(self, sample_audio_file, tmp_path):
        """Test processing a single audio file."""
        config = PipelineConfig(
            output_dir=tmp_path,
            enable_diarization=True
        )
        
        pipeline = TranscriptionPipeline(config)
        result = pipeline.process_single_file(sample_audio_file)
        
        assert result.success
        assert result.transcription.full_text != ""
        assert len(result.transcription.segments) > 0
        
        # Check output files
        output_dir = tmp_path / sample_audio_file.stem
        assert (output_dir / "transcript.json").exists()
        assert (output_dir / "transcript.txt").exists()
        
    @pytest.mark.slow
    def test_process_directory(self, audio_directory, tmp_path):
        """Test batch processing of directory."""
        config = PipelineConfig(output_dir=tmp_path)
        pipeline = TranscriptionPipeline(config)
        
        results = pipeline.process_directory(audio_directory)
        
        assert len(results) > 0
        assert all(r.success for r in results)
        
    def test_pipeline_with_diarization_disabled(self, sample_audio_file):
        """Test pipeline without speaker diarization."""
        config = PipelineConfig(enable_diarization=False)
        pipeline = TranscriptionPipeline(config)
        
        result = pipeline.process_single_file(sample_audio_file)
        
        # All segments should have same speaker
        speakers = {s.speaker_id for s in result.transcription.segments}
        assert speakers == {"SPEAKER_00"}
```

### End-to-End Tests

E2E tests verify the complete system:

```python
# tests/test_e2e.py
import pytest
import subprocess
from pathlib import Path

@pytest.mark.e2e
class TestEndToEnd:
    """End-to-end tests using CLI."""
    
    def test_cli_basic_transcription(self, sample_wav_file, tmp_path):
        """Test basic CLI transcription."""
        cmd = [
            "python", "main.py",
            "--input-dir", str(sample_wav_file.parent),
            "--output-dir", str(tmp_path),
            "--device", "cpu"  # Use CPU for CI
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Processing complete" in result.stdout
        
        # Check outputs
        output_files = list(tmp_path.glob("**/*.json"))
        assert len(output_files) > 0
        
    def test_cli_with_config(self, config_file, audio_dir, tmp_path):
        """Test CLI with custom configuration."""
        cmd = [
            "python", "main.py",
            "--config", str(config_file),
            "--output-dir", str(tmp_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert result.returncode == 0
```

## Fixtures

### Common Fixtures

```python
# tests/conftest.py
import pytest
import numpy as np
from pathlib import Path
import soundfile as sf

@pytest.fixture
def sample_audio_data():
    """Generate sample audio data."""
    duration = 5  # seconds
    sample_rate = 16000
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, duration * sample_rate)
    audio = np.sin(2 * np.pi * frequency * t)
    
    return audio, sample_rate

@pytest.fixture
def sample_wav_file(tmp_path, sample_audio_data):
    """Create a temporary WAV file."""
    audio, sample_rate = sample_audio_data
    
    wav_path = tmp_path / "test_audio.wav"
    sf.write(wav_path, audio, sample_rate)
    
    return wav_path

@pytest.fixture
def mock_transcription_result():
    """Mock transcription result."""
    return {
        "segments": [
            {
                "text": "Hello world",
                "start_time": 0.0,
                "end_time": 1.5,
                "speaker_id": "SPEAKER_00",
                "confidence": 0.95
            },
            {
                "text": "How are you",
                "start_time": 2.0,
                "end_time": 3.5,
                "speaker_id": "SPEAKER_01",
                "confidence": 0.92
            }
        ]
    }

@pytest.fixture
def pipeline_config(tmp_path):
    """Create test pipeline configuration."""
    return {
        "input_dir": str(tmp_path / "inputs"),
        "output_dir": str(tmp_path / "outputs"),
        "model": {
            "name": "stt_en_conformer_ctc_small"
        },
        "device": "cpu",
        "enable_diarization": False
    }
```

### Audio Test Data

```python
# tests/fixtures/audio_generator.py
import numpy as np
import soundfile as sf

def create_test_audio(
    duration: float = 5.0,
    sample_rate: int = 16000,
    num_speakers: int = 1
) -> np.ndarray:
    """Create synthetic audio for testing."""
    samples = int(duration * sample_rate)
    
    if num_speakers == 1:
        # Simple sine wave
        frequency = 440
        t = np.linspace(0, duration, samples)
        audio = np.sin(2 * np.pi * frequency * t)
    else:
        # Multiple frequencies for different "speakers"
        audio = np.zeros(samples)
        segment_length = samples // num_speakers
        
        for i in range(num_speakers):
            start = i * segment_length
            end = start + segment_length
            frequency = 440 * (i + 1)
            t = np.linspace(0, segment_length / sample_rate, segment_length)
            audio[start:end] = np.sin(2 * np.pi * frequency * t)
    
    return audio

def create_speech_like_audio(
    duration: float = 5.0,
    sample_rate: int = 16000
) -> np.ndarray:
    """Create audio that resembles speech patterns."""
    samples = int(duration * sample_rate)
    
    # Combine multiple frequencies (formants)
    f1, f2, f3 = 700, 1220, 2600  # Typical formants
    t = np.linspace(0, duration, samples)
    
    audio = (
        0.5 * np.sin(2 * np.pi * f1 * t) +
        0.3 * np.sin(2 * np.pi * f2 * t) +
        0.2 * np.sin(2 * np.pi * f3 * t)
    )
    
    # Add envelope to simulate speech patterns
    envelope = np.concatenate([
        np.linspace(0, 1, samples // 4),
        np.ones(samples // 2),
        np.linspace(1, 0, samples // 4)
    ])
    
    return audio * envelope
```

## Mocking

### Mock External Dependencies

```python
# tests/unit/test_transcriber.py
from unittest.mock import Mock, patch
import pytest

class TestTranscriber:
    @patch('audio_aigented.transcription.asr.EncDecCTCModel')
    def test_transcribe_with_mock_model(self, mock_model_class):
        """Test transcription with mocked NeMo model."""
        # Setup mock
        mock_model = Mock()
        mock_model.transcribe.return_value = ["Hello world"]
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Test
        transcriber = Transcriber()
        result = transcriber.transcribe(np.zeros(16000))
        
        assert result.text == "Hello world"
        mock_model.transcribe.assert_called_once()
        
    @patch('torch.cuda.is_available')
    def test_device_fallback(self, mock_cuda):
        """Test fallback to CPU when GPU unavailable."""
        mock_cuda.return_value = False
        
        transcriber = Transcriber(device="cuda")
        
        assert transcriber.device == "cpu"
```

### Mock File System

```python
# tests/unit/test_output_writer.py
from unittest.mock import mock_open, patch

class TestOutputWriter:
    @patch('builtins.open', new_callable=mock_open)
    def test_write_json(self, mock_file):
        """Test JSON file writing."""
        writer = OutputWriter()
        data = {"test": "data"}
        
        writer.write_json(Path("test.json"), data)
        
        mock_file.assert_called_once_with(Path("test.json"), 'w')
        handle = mock_file()
        written_data = ''.join(call.args[0] for call in handle.write.call_args_list)
        
        assert json.loads(written_data) == data
```

## Performance Testing

### Benchmarking

```python
# tests/performance/test_benchmarks.py
import pytest
import time

class TestPerformance:
    @pytest.mark.benchmark
    def test_transcription_speed(self, benchmark, sample_audio):
        """Benchmark transcription speed."""
        transcriber = Transcriber()
        
        def transcribe():
            return transcriber.transcribe(sample_audio)
        
        result = benchmark(transcribe)
        
        # Assert performance requirements
        assert benchmark.stats['mean'] < 1.0  # Less than 1 second
        
    def test_memory_usage(self, sample_audio):
        """Test memory usage during transcription."""
        import tracemalloc
        
        tracemalloc.start()
        
        transcriber = Transcriber()
        result = transcriber.transcribe(sample_audio)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Assert memory usage is reasonable
        assert peak < 1024 * 1024 * 1024  # Less than 1GB
```

### Load Testing

```python
# tests/performance/test_load.py
import concurrent.futures
import pytest

class TestLoad:
    @pytest.mark.slow
    def test_concurrent_processing(self, audio_files):
        """Test concurrent file processing."""
        pipeline = TranscriptionPipeline()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(pipeline.process_single_file, f)
                for f in audio_files
            ]
            
            results = [f.result() for f in futures]
        
        assert all(r.success for r in results)
        assert len(results) == len(audio_files)
```

## Test Data Management

### Creating Test Data

```python
# scripts/create_test_data.py
"""Script to create test audio files."""

import numpy as np
import soundfile as sf
from pathlib import Path

def create_test_dataset():
    test_dir = Path("tests/fixtures/audio")
    test_dir.mkdir(exist_ok=True)
    
    # Short audio (1 second)
    short_audio = create_speech_like_audio(1.0)
    sf.write(test_dir / "short.wav", short_audio, 16000)
    
    # Medium audio (30 seconds)
    medium_audio = create_speech_like_audio(30.0)
    sf.write(test_dir / "medium.wav", medium_audio, 16000)
    
    # Multi-speaker simulation
    multi_speaker = create_test_audio(60.0, num_speakers=3)
    sf.write(test_dir / "multi_speaker.wav", multi_speaker, 16000)
    
    print(f"Created test data in {test_dir}")

if __name__ == "__main__":
    create_test_dataset()
```

## Continuous Integration

### GitHub Actions Configuration

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.10, 3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install uv
        uv pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        uv run pytest --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Testing Best Practices

### 1. Test Naming

```python
# Good test names
def test_load_wav_file_returns_audio_and_sample_rate():
    pass

def test_missing_file_raises_file_not_found_error():
    pass

def test_diarization_identifies_two_speakers():
    pass
```

### 2. Test Organization

```python
class TestAudioLoader:
    """Group related tests in classes."""
    
    class TestLoadingFiles:
        def test_wav_file(self):
            pass
            
        def test_missing_file(self):
            pass
    
    class TestPreprocessing:
        def test_resampling(self):
            pass
            
        def test_normalization(self):
            pass
```

### 3. Assertion Messages

```python
def test_audio_duration():
    audio, sr = load_audio("test.wav")
    expected_duration = 5.0
    actual_duration = len(audio) / sr
    
    assert actual_duration == pytest.approx(expected_duration, rel=0.01), \
        f"Expected {expected_duration}s, got {actual_duration}s"
```

### 4. Parametrized Tests

```python
@pytest.mark.parametrize("model_name,expected_speed", [
    ("stt_en_conformer_ctc_small", 15.0),
    ("stt_en_conformer_ctc_medium", 8.0),
    ("stt_en_conformer_ctc_large", 4.0),
])
def test_model_performance(model_name, expected_speed):
    """Test different model speeds."""
    config = {"model": {"name": model_name}}
    pipeline = TranscriptionPipeline(config)
    
    # Test performance
    actual_speed = measure_speed(pipeline)
    
    assert actual_speed >= expected_speed * 0.8  # Allow 20% variance
```

## Debugging Tests

### Using pytest debugger

```bash
# Drop into debugger on failure
uv run pytest --pdb

# Drop into debugger at start of test
uv run pytest --trace
```

### Verbose Output

```bash
# Show print statements
uv run pytest -s

# Show detailed test progress
uv run pytest -vv

# Show local variables on failure
uv run pytest -l
```

### Test Isolation

```python
# Ensure tests don't affect each other
@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment after each test."""
    yield
    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()
```