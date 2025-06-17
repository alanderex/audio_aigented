#!/usr/bin/env python3
"""Test the speaker hint feature with a 4-speaker audio file."""

import subprocess
import sys
from pathlib import Path

# The test file with 4 speakers
test_file = Path("/home/hendorf/Documents/youtube-div-in/rededfining-industrial-DLD25-[_6v9w2U2kto].wav")

if not test_file.exists():
    print(f"Error: Test file not found: {test_file}")
    sys.exit(1)

print(f"Testing speaker hint with: {test_file.name}")
print("=" * 60)

# Create a temporary directory for this test
import tempfile
with tempfile.TemporaryDirectory() as temp_dir:
    # Run the transcription pipeline with 4 speaker hint
    cmd = [
        "uv", "run", "python", "main.py",
        "--input-dir", str(test_file.parent),
        "--output-dir", temp_dir,
        "--num-speakers", "4",
        "--log-level", "INFO"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Check the output
    output_dir = Path(temp_dir) / test_file.stem
    if output_dir.exists():
        print("\n" + "=" * 60)
        print("Output files created:")
        for f in output_dir.iterdir():
            print(f"  - {f.name}")
            
        # Check the attributed transcript
        attributed_file = output_dir / "transcript_attributed.txt"
        if attributed_file.exists():
            print("\n" + "=" * 60)
            print("First 50 lines of speaker-attributed transcript:")
            print("-" * 60)
            with open(attributed_file) as f:
                lines = f.readlines()[:50]
                for line in lines:
                    print(line.rstrip())