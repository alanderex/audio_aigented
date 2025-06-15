# Output Formats

The Audio Transcription Pipeline generates multiple output formats to suit different use cases. This guide covers each format in detail.

## Output Structure

For each processed audio file, the pipeline creates:

```
outputs/
└── audio_filename/
    ├── transcript.json              # Structured data
    ├── transcript.txt               # Human-readable report
    └── transcript_attributed.txt    # Theater-style dialog
```

## JSON Format

### Overview

The JSON format provides complete structured data including:
- Audio metadata
- Full transcription with timestamps
- Speaker segments
- Confidence scores
- Processing information

### Structure

```json
{
  "audio_file": {
    "path": "./inputs/meeting.wav",
    "filename": "meeting.wav",
    "duration": 125.34,
    "sample_rate": 16000,
    "channels": 1,
    "size_bytes": 4013088
  },
  "transcription": {
    "full_text": "Good morning everyone. Let's begin today's meeting...",
    "segments": [
      {
        "text": "Good morning everyone.",
        "start_time": 0.5,
        "end_time": 2.3,
        "confidence": 0.95,
        "speaker_id": "SPEAKER_00",
        "words": [
          {
            "text": "Good",
            "start": 0.5,
            "end": 0.7,
            "confidence": 0.97
          },
          {
            "text": "morning",
            "start": 0.8,
            "end": 1.2,
            "confidence": 0.96
          },
          {
            "text": "everyone",
            "start": 1.3,
            "end": 2.1,
            "confidence": 0.93
          }
        ]
      }
    ]
  },
  "speakers": {
    "count": 3,
    "details": {
      "SPEAKER_00": {
        "segments": 15,
        "total_duration": 45.2,
        "percentage": 36.1
      },
      "SPEAKER_01": {
        "segments": 12,
        "total_duration": 38.7,
        "percentage": 30.9
      },
      "SPEAKER_02": {
        "segments": 8,
        "total_duration": 41.4,
        "percentage": 33.0
      }
    }
  },
  "processing": {
    "timestamp": "2024-01-15T10:30:45Z",
    "processing_time": 12.34,
    "real_time_factor": 10.16,
    "model": {
      "name": "stt_en_conformer_ctc_large",
      "version": "1.20.0"
    },
    "diarization": {
      "enabled": true,
      "model": "titanet-l"
    },
    "device": "cuda",
    "success": true
  }
}
```

### Usage Examples

```python
import json

# Load transcription results
with open("outputs/meeting/transcript.json") as f:
    data = json.load(f)

# Access full text
print(data["transcription"]["full_text"])

# Iterate through segments
for segment in data["transcription"]["segments"]:
    print(f"{segment['speaker_id']}: {segment['text']}")

# Get speaker statistics
for speaker, stats in data["speakers"]["details"].items():
    print(f"{speaker}: {stats['percentage']:.1f}% of conversation")
```

## Text Format

### Overview

The text format provides a human-readable report with:
- Summary statistics
- Formatted transcription
- Speaker transitions
- Processing metadata

### Example Output

```
================================================================================
                          TRANSCRIPTION REPORT
================================================================================

File: meeting.wav
Duration: 2:05 (125.34 seconds)
Processed: 2024-01-15 10:30:45
Processing Time: 12.34 seconds (10.2x faster than real-time)

--------------------------------------------------------------------------------
                           SPEAKER STATISTICS
--------------------------------------------------------------------------------

Total Speakers: 3

SPEAKER_00: 45.2 seconds (36.1%) - 15 segments
SPEAKER_01: 38.7 seconds (30.9%) - 12 segments  
SPEAKER_02: 41.4 seconds (33.0%) - 8 segments

--------------------------------------------------------------------------------
                              TRANSCRIPTION
--------------------------------------------------------------------------------

[00:00:00 - 00:00:02] SPEAKER_00 (confidence: 0.95)
Good morning everyone.

[00:00:02 - 00:00:05] SPEAKER_01 (confidence: 0.93)
Good morning. Thanks for joining today's meeting.

[00:00:06 - 00:00:10] SPEAKER_00 (confidence: 0.94)
Let's start with the agenda. We have three main topics to cover.

[00:00:11 - 00:00:15] SPEAKER_02 (confidence: 0.92)
Before we begin, I'd like to add one item to the agenda if that's okay.

================================================================================
                           END OF TRANSCRIPTION
================================================================================
```

### Configuration Options

```yaml
output:
  text:
    include_timestamps: true
    include_confidence: true
    include_speaker_stats: true
    timestamp_format: "[%H:%M:%S]"  # Or "[%M:%S]" for shorter
    confidence_threshold: 0.0        # Hide low confidence segments
```

## Attributed Text Format

### Overview

The attributed text format presents conversations in a theater-style dialog format:
- Clean, readable layout
- Speaker labels on each turn
- Natural conversation flow
- Perfect for meeting minutes

### Example Output

```
SPEAKER_00: Good morning everyone.

SPEAKER_01: Good morning. Thanks for joining today's meeting.

SPEAKER_00: Let's start with the agenda. We have three main topics to cover.

SPEAKER_02: Before we begin, I'd like to add one item to the agenda if that's okay.

SPEAKER_00: Of course, what would you like to discuss?

SPEAKER_02: I think we should review the Q4 results before moving to next year's planning.

SPEAKER_01: That's a great point. Let's add that as the first item.

SPEAKER_00: Agreed. So our updated agenda is: First, Q4 results review. Second, 2024 planning. Third, resource allocation. And finally, next steps.

SPEAKER_01: Perfect. Should we start with the Q4 review then?

SPEAKER_02: Yes, I have the slides ready to share.
```

### Use Cases

1. **Meeting Minutes**: Clean format for documentation
2. **Interview Transcripts**: Easy to follow Q&A
3. **Podcast Show Notes**: Reader-friendly dialog
4. **Legal Depositions**: Clear speaker attribution

### Customization

```yaml
output:
  attributed:
    speaker_format: "{speaker}: "    # Or "- {speaker}: " for bullets
    paragraph_break: true            # Add breaks between speakers
    merge_consecutive: true          # Merge same-speaker turns
    min_segment_words: 3            # Skip very short utterances
```

## Processing Summary

### Overview

A summary file is created for batch processing:

```
outputs/processing_summary.txt
```

### Example Content

```
================================================================================
                         BATCH PROCESSING SUMMARY
================================================================================

Processing Started: 2024-01-15 10:30:00
Processing Completed: 2024-01-15 10:35:45
Total Duration: 5 minutes 45 seconds

--------------------------------------------------------------------------------
                              FILES PROCESSED
--------------------------------------------------------------------------------

Successfully Processed: 8/10 files

✓ meeting_2024_01_15.wav
  - Duration: 2:05
  - Speakers: 3
  - Processing: 12.3s
  
✓ interview_john_doe.wav
  - Duration: 45:32
  - Speakers: 2
  - Processing: 3:12s

✓ podcast_episode_042.wav
  - Duration: 1:15:20
  - Speakers: 4
  - Processing: 5:45s

[... more files ...]

Failed: 2 files

✗ corrupted_audio.wav
  - Error: Invalid audio format
  
✗ empty_file.wav
  - Error: File is empty

--------------------------------------------------------------------------------
                              STATISTICS
--------------------------------------------------------------------------------

Total Audio Duration: 4:32:15
Total Processing Time: 0:25:32
Average Speed: 10.7x real-time

Total Speakers Detected: 23
Average Speakers per File: 2.9

Model Used: stt_en_conformer_ctc_large
Device: NVIDIA GeForce RTX 3090

================================================================================
```

## Custom Output Formats

### Adding New Formats

```python
from audio_aigented.formatting import BaseFormatter

class MarkdownFormatter(BaseFormatter):
    def format(self, result: TranscriptionResult) -> str:
        output = f"# Transcription: {result.audio_file.filename}\n\n"
        output += f"**Duration:** {result.audio_file.duration:.1f}s\n\n"
        
        for segment in result.transcription.segments:
            output += f"**{segment.speaker_id}** "
            output += f"*({segment.start_time:.1f}s)*: "
            output += f"{segment.text}\n\n"
        
        return output

# Register formatter
pipeline.add_formatter("markdown", MarkdownFormatter())
```

### Output Post-Processing

```python
# Add timestamps to existing format
def add_word_timestamps(json_data):
    output = []
    for segment in json_data["transcription"]["segments"]:
        for word in segment.get("words", []):
            output.append(f"[{word['start']:.2f}] {word['text']}")
    return " ".join(output)

# Generate SRT subtitles
def to_srt(json_data):
    srt_output = []
    for i, segment in enumerate(json_data["transcription"]["segments"], 1):
        start = format_srt_time(segment["start_time"])
        end = format_srt_time(segment["end_time"])
        text = f"{segment['speaker_id']}: {segment['text']}"
        
        srt_output.append(f"{i}\n{start} --> {end}\n{text}\n")
    
    return "\n".join(srt_output)
```

## Export Options

### CSV Export

```python
import csv

def export_to_csv(json_file, csv_file):
    with open(json_file) as f:
        data = json.load(f)
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Speaker', 'Start', 'End', 'Text', 'Confidence'])
        
        for segment in data['transcription']['segments']:
            writer.writerow([
                segment['speaker_id'],
                segment['start_time'],
                segment['end_time'],
                segment['text'],
                segment['confidence']
            ])
```

### Database Export

```python
import sqlite3

def export_to_db(json_file, db_file):
    with open(json_file) as f:
        data = json.load(f)
    
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    
    # Create table
    c.execute('''CREATE TABLE IF NOT EXISTS transcripts
                 (id INTEGER PRIMARY KEY,
                  filename TEXT,
                  speaker TEXT,
                  start_time REAL,
                  end_time REAL,
                  text TEXT,
                  confidence REAL)''')
    
    # Insert segments
    for segment in data['transcription']['segments']:
        c.execute('''INSERT INTO transcripts 
                     (filename, speaker, start_time, end_time, text, confidence)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (data['audio_file']['filename'],
                   segment['speaker_id'],
                   segment['start_time'],
                   segment['end_time'],
                   segment['text'],
                   segment['confidence']))
    
    conn.commit()
    conn.close()
```

## Best Practices

### Choosing Output Formats

1. **JSON**: For programmatic access and further processing
2. **TXT**: For human review and reports
3. **Attributed**: For meeting minutes and documentation

### File Organization

```python
# Organize outputs by date
outputs/
├── 2024-01-15/
│   ├── morning_meeting/
│   ├── client_call/
│   └── team_standup/
└── 2024-01-16/
    └── ...
```

### Compression

```python
# Compress large output files
import gzip
import shutil

def compress_outputs(output_dir):
    for json_file in output_dir.glob("**/*.json"):
        with open(json_file, 'rb') as f_in:
            with gzip.open(f"{json_file}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        json_file.unlink()  # Remove original
```

### Validation

```python
# Validate output completeness
def validate_output(output_dir):
    required_files = ['transcript.json', 'transcript.txt', 
                     'transcript_attributed.txt']
    
    for required in required_files:
        if not (output_dir / required).exists():
            raise FileNotFoundError(f"Missing {required}")
    
    # Validate JSON structure
    with open(output_dir / 'transcript.json') as f:
        data = json.load(f)
        assert 'transcription' in data
        assert 'segments' in data['transcription']
    
    return True
```