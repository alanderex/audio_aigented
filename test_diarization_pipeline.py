#!/usr/bin/env python3
"""
Test the diarization-first pipeline with debugging.
"""

import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.audio_aigented.pipeline import TranscriptionPipeline
from src.audio_aigented.models.schemas import ProcessingConfig

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_diarization_pipeline():
    """Test the diarization-first pipeline."""
    
    # Configuration
    audio_file = Path("/home/hendorf/code/audio_ai/app/data/input/out_mono_How to build an AI-first organization.wav")
    
    if not audio_file.exists():
        print(f"Error: Audio file not found: {audio_file}")
        return
    
    # Create output directory
    output_dir = Path("test_diarization_debug")
    output_dir.mkdir(exist_ok=True)
    
    # Create configuration
    config = ProcessingConfig(
        input_dir=audio_file.parent,
        output_dir=output_dir,
        processing={
            "enable_diarization": True,
            "log_level": "DEBUG"
        }
    )
    
    # Initialize pipeline
    print("=== Initializing Pipeline ===")
    try:
        pipeline = TranscriptionPipeline(config=config)
        print("✓ Pipeline initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Process file
    print(f"\n=== Processing: {audio_file.name} ===")
    
    try:
        # Get audio file object
        audio_file_obj = pipeline.audio_loader.load_audio_file(audio_file)
        audio_data, sample_rate = pipeline.audio_loader.load_audio_data(audio_file_obj)
        
        print(f"Audio loaded: duration={audio_file_obj.duration:.2f}s, sample_rate={sample_rate}")
        
        # Test diarization only
        print("\n=== Testing Diarization ===")
        speaker_segments = pipeline.diarizer.diarize(audio_file_obj)
        
        if speaker_segments:
            unique_speakers = set(speaker for _, _, speaker in speaker_segments)
            print(f"Found {len(speaker_segments)} segments with {len(unique_speakers)} speakers")
            
            # Show first 10 segments
            print("\nFirst 10 speaker segments:")
            for i, (start, end, speaker) in enumerate(speaker_segments[:10]):
                duration = end - start
                print(f"  {i+1}. {speaker}: {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")
            
            # Check for very short segments
            short_segments = [(start, end, speaker) for start, end, speaker in speaker_segments if (end - start) < 0.5]
            if short_segments:
                print(f"\nWarning: Found {len(short_segments)} very short segments (<0.5s)")
                
        else:
            print("No speaker segments found!")
            
        # Now test full pipeline
        print("\n=== Testing Full Pipeline ===")
        result = pipeline.process_single_file(audio_file)
        
        print(f"\nResult summary:")
        print(f"  Total segments: {len(result.segments)}")
        print(f"  Full text length: {len(result.full_text)}")
        print(f"  Speakers detected: {result.metadata.get('num_speakers', 'Unknown')}")
        
        # Count segments by speaker
        speaker_counts = {}
        empty_segments = 0
        for seg in result.segments:
            if seg.text.strip():
                speaker = seg.speaker_id or "UNKNOWN"
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            else:
                empty_segments += 1
        
        print(f"\nSegments by speaker:")
        for speaker, count in sorted(speaker_counts.items()):
            print(f"  {speaker}: {count} segments")
        
        if empty_segments > 0:
            print(f"  Empty segments: {empty_segments}")
        
        # Show sample output
        print("\n=== First 5 attributed segments ===")
        current_speaker = None
        for i, seg in enumerate(result.segments[:5]):
            if seg.text.strip():
                if seg.speaker_id != current_speaker:
                    current_speaker = seg.speaker_id
                    print(f"\n{current_speaker}:")
                print(f"  {seg.text}")
        
        print("\n✓ Processing completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_diarization_pipeline()