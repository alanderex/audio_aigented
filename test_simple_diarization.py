#!/usr/bin/env python3
"""
Simple test of diarization-first pipeline.
"""

import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.audio_aigented.pipeline import TranscriptionPipeline
from src.audio_aigented.models.schemas import ProcessingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress verbose numba logging
logging.getLogger('numba').setLevel(logging.WARNING)

def test():
    """Test the diarization-first pipeline."""
    
    # Configuration
    audio_file = Path("/home/hendorf/code/audio_ai/app/data/input/out_mono_How to build an AI-first organization.wav")
    
    # Create configuration
    config = ProcessingConfig(
        input_dir=audio_file.parent,
        output_dir=Path("test_simple_output"),
        processing={
            "enable_diarization": True,
            "log_level": "INFO"
        }
    )
    
    # Initialize and run pipeline
    print("Initializing pipeline...")
    pipeline = TranscriptionPipeline(config=config)
    
    print("Processing audio file...")
    result = pipeline.process_single_file(audio_file)
    
    # Show results
    print(f"\nResults:")
    print(f"  Total segments: {len(result.segments)}")
    print(f"  Text length: {len(result.full_text)}")
    print(f"  Speakers: {result.metadata.get('num_speakers', 0)}")
    
    # Count segments by speaker
    speaker_counts = {}
    for seg in result.segments:
        if seg.text.strip():
            speaker = seg.speaker_id or "UNKNOWN"
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
    
    print(f"\nSegments by speaker:")
    for speaker, count in sorted(speaker_counts.items()):
        print(f"  {speaker}: {count}")
    
    # Show first few segments
    print(f"\nFirst 3 segments:")
    for i, seg in enumerate(result.segments[:3]):
        if seg.text.strip():
            print(f"  {seg.speaker_id}: {seg.text[:100]}...")

if __name__ == "__main__":
    test()