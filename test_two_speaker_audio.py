#!/usr/bin/env python3
"""
Test script to verify diarization works with two-speaker audio files.
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
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_audio_files():
    """Test diarization on audio files in the specified directory."""
    
    # Audio files location
    audio_dir = Path("/home/hendorf/code/audio_ai/app/data/input")
    
    if not audio_dir.exists():
        print(f"Error: Audio directory not found: {audio_dir}")
        return
    
    # Find WAV files
    wav_files = list(audio_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} WAV files in {audio_dir}")
    
    if not wav_files:
        print("No WAV files found!")
        return
    
    # Show files
    print("\nAudio files found:")
    for i, f in enumerate(wav_files[:5], 1):  # Show first 5
        print(f"  {i}. {f.name}")
    if len(wav_files) > 5:
        print(f"  ... and {len(wav_files) - 5} more")
    
    # Create output directory
    output_dir = Path("test_diarization_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create configuration
    config = ProcessingConfig(
        input_dir=audio_dir,
        output_dir=output_dir,
        processing={
            "enable_diarization": True,
            "log_level": "INFO"
        }
    )
    
    # Initialize pipeline
    print("\n=== Initializing Pipeline ===")
    try:
        pipeline = TranscriptionPipeline(config=config)
        print("✓ Pipeline initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Process first file as a test
    test_file = wav_files[0]
    print(f"\n=== Processing: {test_file.name} ===")
    
    try:
        result = pipeline.process_single_file(test_file)
        
        # Analyze results
        print("\n=== Diarization Results ===")
        
        # Count speakers
        speaker_counts = {}
        for seg in result.segments:
            speaker = seg.speaker_id or "UNKNOWN"
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        print(f"Total segments: {len(result.segments)}")
        print(f"Unique speakers: {len(speaker_counts)}")
        
        for speaker, count in sorted(speaker_counts.items()):
            print(f"  {speaker}: {count} segments")
        
        # Check metadata
        if "num_speakers" in result.metadata:
            print(f"\nMetadata reports {result.metadata['num_speakers']} speakers")
        
        # Show sample attributed output
        print("\n=== Sample Attributed Transcript ===")
        current_speaker = None
        sample_lines = []
        
        for seg in result.segments[:20]:  # First 20 segments
            if seg.speaker_id != current_speaker:
                if current_speaker is not None:
                    sample_lines.append("")
                current_speaker = seg.speaker_id or "UNKNOWN"
                sample_lines.append(f"{current_speaker}: {seg.text}")
            else:
                # Append to current speaker's text
                if sample_lines:
                    sample_lines[-1] += f" {seg.text}"
        
        for line in sample_lines:
            print(line)
        
        if len(result.segments) > 20:
            print("...")
        
        # Check output files
        file_output_dir = output_dir / test_file.stem
        if file_output_dir.exists():
            print(f"\n=== Output Files ===")
            for f in file_output_dir.iterdir():
                print(f"  - {f.name}")
        
        print("\n✓ Processing completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_audio_files()