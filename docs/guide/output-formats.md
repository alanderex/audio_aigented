# Output Formats

Three formats for different use cases.

## File Structure

```
outputs/
└── audio_filename/
    ├── transcript.json           # Structured data
    ├── transcript.txt            # Human-readable
    └── transcript_attributed.txt # Dialog format
```

## JSON Format

Complete structured data with metadata.

```json
{
  "audio_file": {
    "path": "meeting.wav",
    "duration": 125.34
  },
  "transcription": {
    "full_text": "Complete transcription...",
    "segments": [{
      "text": "Hello everyone",
      "speaker_id": "SPEAKER_00",
      "start_time": 0.5,
      "end_time": 2.3,
      "confidence": 0.95
    }]
  },
  "speakers": {
    "count": 2,
    "details": {
      "SPEAKER_00": {"duration": 45.2, "percentage": 36.1}
    }
  }
}
```

**Use for**: APIs, further processing, detailed analysis

## Text Format

Human-readable summary with statistics.

```
================================================================================
AUDIO TRANSCRIPTION REPORT
================================================================================

File: meeting.wav
Duration: 2:05 (125.34 seconds)
Speakers: 2
Model: stt_en_conformer_ctc_large

TRANSCRIPTION:
--------------------------------------------------------------------------------
Hello everyone. Let's start today's meeting. First, I'd like to discuss...

STATISTICS:
--------------------------------------------------------------------------------
- Total Words: 523
- Speaking Time: 120.5s (96.1%)
- Processing Time: 12.3s (10.2x realtime)

SPEAKER BREAKDOWN:
- SPEAKER_00: 45.2s (36.1%) - 15 segments
- SPEAKER_01: 75.3s (60.1%) - 22 segments
```

**Use for**: Reports, documentation, quick review

## Attributed Text Format

Theater-style dialog with speaker labels.

```
SPEAKER_00: Hello everyone. Let's start today's meeting.

SPEAKER_01: Thanks for joining. I have three items on the agenda.

SPEAKER_00: Sounds good. Should we start with the budget review?

SPEAKER_01: Yes, let's do that first.
```

**Use for**: Meeting minutes, transcripts, screenplays

## Choosing a Format

| Need | Format | Key Features |
|------|--------|--------------|
| API integration | JSON | Timestamps, confidence, metadata |
| Quick review | TXT | Summary stats, readable |
| Meeting minutes | Attributed | Speaker names, dialog flow |
| All details | JSON | Complete data structure |

## Custom Processing

```python
import json

# Load and process JSON output
with open('outputs/meeting/transcript.json') as f:
    data = json.load(f)
    
# Extract high-confidence segments
confident = [s for s in data['transcription']['segments'] 
             if s['confidence'] > 0.9]

# Get speaker statistics
for speaker, stats in data['speakers']['details'].items():
    print(f"{speaker}: {stats['percentage']:.1f}%")
```