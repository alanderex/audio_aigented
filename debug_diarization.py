#!/usr/bin/env python3
"""Debug diarization with 4-speaker file."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Suppress NeMo output
import os
os.environ["NEMO_TESTING"] = "True"
import logging
logging.getLogger("nemo").setLevel(logging.ERROR)
logging.getLogger("nemo.collections.asr").setLevel(logging.ERROR)
logging.getLogger("nemo_logging").setLevel(logging.ERROR)

from src.audio_aigented.diarization.diarizer import NeMoDiarizer
from src.audio_aigented.models.schemas import AudioFile

# The test file
test_file = Path("/home/hendorf/Documents/youtube-div-in/rededfining-industrial-DLD25-[_6v9w2U2kto].wav")

# Create AudioFile object
audio_file = AudioFile(
    path=test_file,
    sample_rate=16000,
    duration=1319.125333,  # actual duration
    channels=1,
    format="wav"
)

print(f"Testing diarization on: {test_file.name}")
print("=" * 60)

# Test with speaker hint
print("\nTesting WITH 4-speaker hint:")
print("-" * 30)
diarizer = NeMoDiarizer(num_speakers=4)
segments = diarizer.diarize(audio_file)

# Analyze results
unique_speakers = set(s[2] for s in segments) if segments else set()
print(f"Number of segments: {len(segments)}")
print(f"Unique speakers found: {unique_speakers}")
print(f"Number of unique speakers: {len(unique_speakers)}")

if segments:
    print("\nFirst 10 segments:")
    for i, (start, end, speaker) in enumerate(segments[:10]):
        print(f"  {i+1}. {speaker}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
        
    # Show speaker distribution
    print("\nSpeaker distribution:")
    speaker_times = {}
    for start, end, speaker in segments:
        if speaker not in speaker_times:
            speaker_times[speaker] = 0
        speaker_times[speaker] += (end - start)
    
    for speaker, time in sorted(speaker_times.items()):
        print(f"  {speaker}: {time:.2f}s ({time/sum(speaker_times.values())*100:.1f}%)")