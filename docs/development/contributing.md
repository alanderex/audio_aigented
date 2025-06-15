# Contributing

Thank you for your interest in contributing to the Audio Transcription Pipeline! This guide will help you get started.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional, for testing GPU features)
- Git for version control
- `uv` package manager

### Development Setup

1. **Fork and Clone**
   ```bash
   # Fork the repository on GitHub, then:
   git clone https://github.com/YOUR_USERNAME/audio_aigented.git
   cd audio_aigented
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   uv pip install -e ".[dev,docs]"
   ```

4. **Set Up Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

## Development Workflow

### Branch Strategy

We use a simple branch strategy:
- `main` - Stable release branch
- `develop` - Integration branch for features
- `feature/*` - Feature branches
- `fix/*` - Bug fix branches
- `docs/*` - Documentation updates

### Making Changes

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Write clean, documented code
   - Follow the existing code style
   - Add tests for new functionality

3. **Run Tests**
   ```bash
   uv run pytest
   uv run pytest --cov=src --cov-report=term-missing
   ```

4. **Check Code Quality**
   ```bash
   uv run ruff check .
   uv run ruff format .
   uv run mypy src/
   ```

5. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add amazing new feature"
   ```

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc)
- `refactor:` Code refactoring
- `test:` Test additions or changes
- `chore:` Build process or auxiliary tool changes

Examples:
```
feat: add support for MP3 input files
fix: correct speaker assignment in diarization
docs: update installation guide for Windows
test: add integration tests for pipeline
```

## Code Standards

### Python Style Guide

We follow PEP 8 with these additions:
- Maximum line length: 88 characters (Black default)
- Use type hints for function signatures
- Use Google-style docstrings

Example:
```python
def process_audio(
    file_path: Path,
    sample_rate: int = 16000,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """Process audio file for transcription.
    
    Args:
        file_path: Path to the audio file
        sample_rate: Target sample rate in Hz
        normalize: Whether to normalize audio levels
        
    Returns:
        Tuple of (audio_array, actual_sample_rate)
        
    Raises:
        AudioLoadError: If file cannot be loaded
    """
    # Implementation
```

### Testing Guidelines

1. **Write Tests First** (TDD encouraged)
2. **Test File Naming**: `test_<module_name>.py`
3. **Test Structure**:
   ```python
   def test_feature_description():
       # Arrange
       input_data = create_test_data()
       
       # Act
       result = function_under_test(input_data)
       
       # Assert
       assert result.expected_field == expected_value
   ```

4. **Use Fixtures**:
   ```python
   @pytest.fixture
   def sample_audio():
       return np.random.randn(16000)  # 1 second at 16kHz
   ```

5. **Mock External Dependencies**:
   ```python
   @patch('audio_aigented.transcription.load_model')
   def test_transcription(mock_model):
       mock_model.return_value.transcribe.return_value = "test"
   ```

### Documentation Standards

1. **Docstrings**: All public functions, classes, and modules
2. **Type Hints**: All function parameters and returns
3. **Examples**: Include usage examples in docstrings
4. **User Guides**: Update relevant guides for new features

## Project Structure

```
audio_aigented/
├── src/audio_aigented/      # Source code
│   ├── audio/              # Audio processing
│   ├── diarization/        # Speaker diarization
│   ├── transcription/      # ASR transcription
│   ├── formatting/         # Output formatting
│   ├── output/            # File writing
│   └── pipeline.py        # Main pipeline
├── tests/                  # Test suite
├── docs/                   # Documentation
├── config/                 # Configuration files
└── examples/              # Usage examples
```

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_pipeline.py

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test
uv run pytest tests/test_pipeline.py::test_process_directory
```

### Test Categories

1. **Unit Tests**: Test individual functions/classes
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete pipeline
4. **Performance Tests**: Test speed and resource usage

### Writing Tests

```python
# tests/test_audio_loader.py
import pytest
from pathlib import Path
from audio_aigented.audio import AudioLoader

class TestAudioLoader:
    def test_load_valid_wav(self, sample_wav_file):
        loader = AudioLoader()
        audio, sr = loader.load(sample_wav_file)
        
        assert audio is not None
        assert sr == 16000
        assert len(audio) > 0
    
    def test_load_missing_file(self):
        loader = AudioLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load(Path("nonexistent.wav"))
```

## Adding New Features

### 1. Planning

Before implementing:
1. Open an issue to discuss the feature
2. Get feedback from maintainers
3. Consider backward compatibility

### 2. Implementation Checklist

- [ ] Create feature branch
- [ ] Write tests first
- [ ] Implement feature
- [ ] Update documentation
- [ ] Add usage examples
- [ ] Update changelog
- [ ] Run full test suite

### 3. Example: Adding New Output Format

```python
# src/audio_aigented/formatting/csv_formatter.py
from .base import BaseFormatter

class CSVFormatter(BaseFormatter):
    """Format transcription results as CSV."""
    
    def format(self, result: TranscriptionResult) -> str:
        output = "speaker,start,end,text,confidence\n"
        
        for segment in result.transcription.segments:
            output += f"{segment.speaker_id},"
            output += f"{segment.start_time},"
            output += f"{segment.end_time},"
            output += f'"{segment.text}",'
            output += f"{segment.confidence}\n"
        
        return output

# Register in pipeline
FORMATTERS['csv'] = CSVFormatter
```

## Pull Request Process

1. **Update Your Branch**
   ```bash
   git fetch upstream
   git rebase upstream/develop
   ```

2. **Push Changes**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request**
   - Use a clear, descriptive title
   - Reference any related issues
   - Include a description of changes
   - Add screenshots for UI changes

4. **PR Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
   
   ## Testing
   - [ ] Tests pass locally
   - [ ] Added new tests
   - [ ] Coverage maintained/improved
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-reviewed code
   - [ ] Updated documentation
   - [ ] No new warnings
   ```

## Review Process

### What We Look For

1. **Code Quality**
   - Clean, readable code
   - Appropriate abstractions
   - No code duplication

2. **Testing**
   - Adequate test coverage
   - Tests pass in CI
   - Edge cases covered

3. **Documentation**
   - Clear docstrings
   - Updated user guides
   - Changelog entry

4. **Performance**
   - No performance regressions
   - Efficient algorithms
   - Resource usage considered

### Responding to Feedback

- Be open to suggestions
- Ask for clarification if needed
- Update PR based on feedback
- Re-request review when ready

## Release Process

1. **Version Numbering**: Semantic Versioning (MAJOR.MINOR.PATCH)
2. **Changelog**: Update CHANGELOG.md
3. **Documentation**: Ensure docs are current
4. **Testing**: Full test suite passes
5. **Tag Release**: Create Git tag
6. **Deploy**: Update package repositories

## Getting Help

### Resources

- [Issue Tracker](https://github.com/yourusername/audio_aigented/issues)
- [Discussions](https://github.com/yourusername/audio_aigented/discussions)
- [Documentation](https://audio-aigented.readthedocs.io)

### Communication

- **Issues**: Bug reports and feature requests
- **Discussions**: General questions and ideas
- **Pull Requests**: Code contributions

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on what's best for the community

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Public or private harassment
- Publishing private information

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing! 🎉