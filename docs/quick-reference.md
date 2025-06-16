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