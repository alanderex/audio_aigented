# Quick Reference

## Common Commands

| Task | Command |
|------|---------|
| Basic transcription | `python main.py -i ./inputs` |
| Fast processing (no diarization) | `python main.py -i ./inputs --disable-diarization` |
| Use Parakeet (fastest model) | `python main.py -i ./inputs --model-name nvidia/parakeet-tdt-0.6b-v2` |
| With custom vocabulary | `python main.py -i ./inputs --vocabulary-file vocab.txt` |
| Extract context from docs | `python main.py -i ./inputs --content-file agenda.html` |
| Clear cache | `python main.py --clear-cache` |
| Create context templates | `python main.py -i ./inputs --create-context-templates` |
| Docker processing | `docker-compose run --rm audio-transcription` |

## CLI Options

### Basic Options
| Option | Description |
|--------|-------------|
| `--input-dir`, `-i` | **[REQUIRED]** Directory containing .wav audio files |
| `--output-dir`, `-o` | Output directory (default: `./outputs`) |
| `--config`, `-c` | Configuration YAML file (default: `config/default.yaml`) |
| `--log-level`, `-l` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Model & Processing Options
| Option | Description |
|--------|-------------|
| `--device` | Processing device: `cuda` (recommended) or `cpu` |
| `--model-name` | Model: `stt_en_conformer_ctc_large` (default), `nvidia/parakeet-tdt-0.6b-v2` (faster) |
| `--enable-diarization` | Enable speaker identification (default) |
| `--disable-diarization` | Disable speaker identification for faster processing |
| `--beam-size` | Beam search width (default: 4, higher = more accurate but slower) |

### Enhancement Options
| Option | Description |
|--------|-------------|
| `--vocabulary-file` | Custom vocabulary file (.txt) for domain-specific terms |
| `--content-dir` | Directory with companion content (can use multiple times) |

### Output Options
| Option | Description |
|--------|-------------|
| `--formats` | Output formats: `json,txt,attributed` (default: all three) |

### Utility Options
| Option | Description |
|--------|-------------|
| `--dry-run` | Preview files without processing |
| `--clear-cache` | Clear model cache before processing |
| `--create-context-templates` | Create editable context files |
| `--help`, `-h` | Show help message |
| `--version` | Show version information |

## Model Selection

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| `nvidia/parakeet-tdt-0.6b-v2` | ⚡⚡⚡ | ★★★ | Fast processing, good accuracy |
| `stt_en_conformer_ctc_small` | ⚡⚡ | ★★ | Quick drafts |
| `stt_en_conformer_ctc_large` | ⚡ | ★★★★ | Best accuracy (default) |

## Output Files

```
outputs/
└── audio_filename/
    ├── transcript.json           # Structured data with timestamps
    ├── transcript.txt            # Human-readable report
    └── transcript_attributed.txt # Speaker dialog format
```

## Context Files

| File Type | Purpose | Location |
|-----------|---------|----------|
| `.context.json` | Full context with speaker names | `audio.wav.context.json` |
| `.txt` | Simple vocabulary list | `audio.wav.txt` |
| `.content.*` | Raw content for extraction | `audio.wav.content.html` |

## Configuration Options

```yaml
# Key settings in config/default.yaml
transcription:
  model_name: "stt_en_conformer_ctc_large"
  device: "cuda"              # or "cpu"
  beam_size: 4                # 1-16 (higher = slower but more accurate)
  enable_file_context: true   # Enable per-file context

processing:
  enable_diarization: true    # Speaker identification
  enable_caching: true        # Avoid reprocessing

output:
  formats: ["json", "txt", "attributed_txt"]
```

## Performance Tips

| Issue | Solution |
|-------|----------|
| Out of memory | `--device cpu` or use smaller model |
| Slow processing | `--disable-diarization` or `--model-name nvidia/parakeet-tdt-0.6b-v2` |
| Poor accuracy | Add `--vocabulary-file` or `--content-file` |
| Reprocess files | `--clear-cache` before running |

## Vocabulary File Format

```text
# Corrections
neuro network -> neural network

# Acronyms  
AI:Artificial Intelligence

# Phrases
"machine learning pipeline"

# Technical terms
PyTorch
```