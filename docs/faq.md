# FAQ

## Common Issues

### CUDA Out of Memory

**Error**: `torch.cuda.OutOfMemoryError`

**Solutions**:
1. Use CPU: `python main.py -i ./inputs --device cpu`
2. Use smaller model: `--model-name stt_en_conformer_ctc_small`
3. Reduce batch size in config
4. Close other GPU applications

### Model Download Fails

**Error**: `HTTPError downloading model`

**Solutions**:
1. Check internet connection
2. Retry with: `python main.py -i ./inputs` (auto-retry)
3. Clear cache: `rm -rf ~/.cache/torch/NeMo/`
4. Use different model

### No Audio Files Found

**Error**: `No .wav files found`

**Solutions**:
1. Check file extension (must be `.wav`)
2. Convert files: `ffmpeg -i audio.mp3 audio.wav`
3. Check input directory path
4. Ensure read permissions

### Poor Transcription Quality

**Solutions**:
1. Add vocabulary: `--vocabulary-file terms.txt`
2. Use context: `--content-file meeting_agenda.html`  
3. Try larger model: `--model-name stt_en_conformer_ctc_large`
4. Increase beam size: `--beam-size 16`

## Performance

### How to Speed Up Processing?

1. **Disable diarization**: `--disable-diarization` (2-3x faster)
2. **Use Parakeet model**: `--model-name nvidia/parakeet-tdt-0.6b-v2`
3. **Use GPU**: Ensure CUDA is available
4. **Process in parallel**: Increase batch size

### What Are Typical Processing Times?

| Setup | Speed |
|-------|-------|
| GPU + Large model | 4-6x realtime |
| GPU + Parakeet | 10-15x realtime |
| CPU + Large model | 0.5x realtime |
| No diarization | 2-3x faster |

## Features

### How to Add Speaker Names?

Create `audio.wav.context.json`:
```json
{
  "speakers": {
    "SPEAKER_00": "John Smith",
    "SPEAKER_01": "Jane Doe"
  }
}
```

### How to Handle Technical Terms?

1. **Vocabulary file**: List terms one per line
2. **Context file**: JSON with corrections
3. **Content files**: HTML/text documents
4. **All methods**: Can be combined

### Can I Process Non-English Audio?

Currently optimized for English. For other languages:
- Check NeMo's multilingual models
- Adjust VAD settings for language
- Disable vocabulary features

## Docker

### How to Use GPU in Docker?

1. Install nvidia-docker
2. Run: `docker-compose run --gpus all audio-transcription`
3. Verify: Check logs for "CUDA available"

### How to Mount Custom Configs?

```yaml
# docker-compose.override.yml
volumes:
  - ./my-config.yaml:/app/config/default.yaml
  - ./my-vocab.txt:/app/vocab.txt
```

## Advanced

### How to Process Large Files?

1. Increase max_duration in config
2. Use streaming mode (if available)
3. Split files: `ffmpeg -i long.wav -f segment -segment_time 300 out%03d.wav`

### Can I Customize the Pipeline?

Yes! See [Development Guide](development/structure.md):
- Extend base classes
- Add pre/post processors
- Implement custom models

### How to Debug Issues?

1. Enable debug logging: `--log-level DEBUG`
2. Check intermediate files in cache/
3. Process single file first
4. Verify model loading in logs