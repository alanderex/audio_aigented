# Speaker Diarization

Automatically identify "who spoke when" in audio recordings.

## Quick Start

```bash
# Enable diarization (default)
python main.py -i ./inputs

# Disable for faster processing  
python main.py -i ./inputs --disable-diarization
```

## How It Works

1. **VAD** → Detects speech regions
2. **Embeddings** → Creates voice fingerprints  
3. **Clustering** → Groups similar voices
4. **Output** → Speaker labels (SPEAKER_00, SPEAKER_01, etc.)

## Configuration

```yaml
# config/default.yaml
processing:
  enable_diarization: true

# Advanced settings (optional)
diarization:
  vad:
    onset: 0.8      # Speech detection sensitivity (0.5-0.95)
    offset: 0.6     # Speech end sensitivity (0.5-0.95)
  clustering:
    max_num_speakers: 8  # Limit speaker count
```

## Output Format

Diarization adds speaker IDs to each segment:

```json
{
  "segments": [
    {
      "text": "Hello everyone",
      "speaker_id": "SPEAKER_00",
      "start_time": 0.5,
      "end_time": 2.3
    }
  ]
}
```

## Speaker Names

Add real names via context files:

```json
// audio.wav.context.json
{
  "speakers": {
    "SPEAKER_00": "Alice Smith",
    "SPEAKER_01": "Bob Jones"
  }
}
```

## Performance

- **Processing time**: ~10% of audio duration  
- **GPU recommended**: 5-10x faster than CPU
- **Accuracy**: 85-95% depending on audio quality

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Too many speakers detected | Increase VAD thresholds |
| Speakers merged | Decrease clustering threshold |
| Slow processing | Use GPU or disable diarization |
| Poor accuracy | Ensure clear audio, minimal overlap |

## Technical Details

<details>
<summary>Models Used (click to expand)</summary>

- **VAD**: MarbleNet (multilingual)
- **Embeddings**: TitaNet-Large (192-dim)
- **Clustering**: Spectral Clustering with NME-SC

</details>

<details>
<summary>RTTM Output Format</summary>

```
SPEAKER file 1 start duration <NA> <NA> speaker_id <NA> <NA>
SPEAKER audio 1 0.5 1.8 <NA> <NA> SPEAKER_00 <NA> <NA>
```

</details>