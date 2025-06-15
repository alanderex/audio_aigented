# Configuration

The Audio Transcription Pipeline uses a flexible YAML-based configuration system with sensible defaults.

## Configuration Files

### Default Configuration

The default configuration is located at `config/default.yaml`:

```yaml
# Input/Output paths
input_dir: "./inputs"
output_dir: "./outputs"

# Audio processing settings
audio:
  sample_rate: 16000
  mono: true
  
# ASR model configuration  
model:
  name: "stt_en_conformer_ctc_large"
  
# Transcription settings
transcription:
  device: "cuda"  # or "cpu"
  batch_size: 4
  
# Diarization settings
diarization:
  enable: true
  embedding:
    model_path: "titanet-l"
  clustering:
    parameters:
      oracle_num_speakers: null  # Auto-detect
      max_num_speakers: 8
      
# Processing options
processing:
  enable_caching: true
  parallel_workers: 4
  log_level: "INFO"
  
# Output settings
output:
  formats: ["json", "txt", "attributed_txt"]
  include_timestamps: true
  include_confidence: true
  pretty_json: true
```

### Custom Configuration

Create custom configurations for different use cases:

```yaml
# config/fast_processing.yaml
model:
  name: "stt_en_conformer_ctc_small"
  
transcription:
  batch_size: 8
  
diarization:
  enable: false  # Disable for speed
  
output:
  formats: ["txt"]  # Minimal output
```

## Configuration Options

### Audio Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `audio.sample_rate` | int | 16000 | Target sample rate in Hz |
| `audio.mono` | bool | true | Convert to mono channel |
| `audio.max_duration` | float | null | Maximum audio duration in seconds |

### Model Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model.name` | string | "stt_en_conformer_ctc_large" | NeMo model name |
| `model.cache_dir` | string | "~/.cache/torch/NeMo" | Model cache directory |

Available models:
- `stt_en_conformer_ctc_small` - Fast, lower accuracy
- `stt_en_conformer_ctc_medium` - Balanced
- `stt_en_conformer_ctc_large` - Slow, high accuracy

### Transcription Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `transcription.device` | string | "cuda" | Device for processing ("cuda" or "cpu") |
| `transcription.batch_size` | int | 4 | Batch size for processing |
| `transcription.num_workers` | int | 0 | DataLoader workers |

### Diarization Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `diarization.enable` | bool | true | Enable speaker diarization |
| `diarization.embedding.model_path` | string | "titanet-l" | Speaker embedding model |
| `diarization.clustering.parameters.oracle_num_speakers` | int/null | null | Known number of speakers |
| `diarization.clustering.parameters.max_num_speakers` | int | 8 | Maximum speakers to detect |
| `diarization.clustering.parameters.enhanced_count_thresholding` | bool | true | Enhanced speaker detection |
| `diarization.clustering.parameters.maj_vote_spk_count` | bool | false | Majority vote for speaker count |

### Processing Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `processing.enable_caching` | bool | true | Cache intermediate results |
| `processing.parallel_workers` | int | 4 | Parallel processing workers |
| `processing.log_level` | string | "INFO" | Logging level |
| `processing.skip_existing` | bool | true | Skip already processed files |

### Output Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `output.formats` | list | ["json", "txt", "attributed_txt"] | Output file formats |
| `output.include_timestamps` | bool | true | Include timestamps in output |
| `output.include_confidence` | bool | true | Include confidence scores |
| `output.pretty_json` | bool | true | Format JSON with indentation |
| `output.segment_separator` | string | "\n\n" | Separator between segments |

## Using Configuration

### Command Line

```bash
# Use custom config file
uv run python main.py --config config/my_config.yaml

# Override specific options
uv run python main.py --input-dir /path/to/audio --device cpu

# Combine config file with overrides
uv run python main.py --config config/base.yaml --disable-diarization
```

### Python API

```python
from audio_aigented.config import PipelineConfig
from audio_aigented.pipeline import TranscriptionPipeline

# Load from file
config = PipelineConfig.from_yaml("config/custom.yaml")

# Create programmatically
config = PipelineConfig(
    device="cuda",
    model_name="stt_en_conformer_ctc_medium",
    enable_diarization=True,
    output_formats=["json", "attributed_txt"]
)

# Initialize pipeline
pipeline = TranscriptionPipeline(config=config)
```

### Environment Variables

Override configuration with environment variables:

```bash
# Override device
export AUDIO_TRANSCRIPTION_DEVICE=cpu

# Override model
export AUDIO_TRANSCRIPTION_MODEL=stt_en_conformer_ctc_small

# Run with overrides
uv run python main.py
```

## Configuration Examples

### High-Speed Processing

```yaml
# config/speed.yaml
model:
  name: "stt_en_conformer_ctc_small"
  
transcription:
  batch_size: 16
  
diarization:
  enable: false
  
processing:
  parallel_workers: 8
  
output:
  formats: ["txt"]
  include_confidence: false
```

### High-Accuracy Processing

```yaml
# config/accuracy.yaml
model:
  name: "stt_en_conformer_ctc_large"
  
transcription:
  batch_size: 1
  
diarization:
  enable: true
  clustering:
    parameters:
      enhanced_count_thresholding: true
      maj_vote_spk_count: true
      
output:
  formats: ["json", "txt", "attributed_txt"]
  include_confidence: true
  include_timestamps: true
```

### Meeting Transcription

```yaml
# config/meetings.yaml
diarization:
  enable: true
  clustering:
    parameters:
      max_num_speakers: 12
      enhanced_count_thresholding: true
      
output:
  formats: ["json", "attributed_txt"]
  segment_separator: "\n---\n"
```

### Podcast Processing

```yaml
# config/podcast.yaml
diarization:
  enable: true
  clustering:
    parameters:
      oracle_num_speakers: 2  # Host and guest
      
output:
  formats: ["txt", "attributed_txt"]
```

## Advanced Configuration

### Multi-Stage Processing

```yaml
# config/multi_stage.yaml
stages:
  - name: "diarization"
    config:
      enable: true
      
  - name: "transcription"
    config:
      model:
        name: "stt_en_conformer_ctc_large"
        
  - name: "post_processing"
    config:
      punctuation: true
      capitalization: true
```

### Resource Limits

```yaml
# config/resource_limited.yaml
transcription:
  device: "cuda"
  batch_size: 1
  
processing:
  max_memory_gb: 8
  max_gpu_memory_gb: 4
  
diarization:
  clustering:
    backend: "cpu"  # Use CPU for clustering
```

## Best Practices

1. **Start with Defaults**: The default configuration works well for most use cases
2. **Profile First**: Test with small batches to find optimal settings
3. **GPU Memory**: Reduce batch size if encountering OOM errors
4. **Model Selection**: Balance speed vs accuracy for your use case
5. **Output Formats**: Only generate formats you need to save processing time

## Troubleshooting

### Configuration Not Loading

```bash
# Validate configuration
uv run python -c "from audio_aigented.config import PipelineConfig; PipelineConfig.from_yaml('config/my_config.yaml')"
```

### Override Priority

Configuration priority (highest to lowest):
1. Command line arguments
2. Environment variables
3. Custom config file
4. Default config file

### Common Issues

**Invalid YAML syntax**
```yaml
# Bad - missing quotes
model:
  name: stt_en_conformer_ctc_large

# Good
model:
  name: "stt_en_conformer_ctc_large"
```

**Type mismatches**
```yaml
# Bad - string instead of int
transcription:
  batch_size: "4"

# Good
transcription:
  batch_size: 4
```