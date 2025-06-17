#!/usr/bin/env python3
"""Test script to verify reduced verbosity in transcription output."""

import subprocess
import sys
from pathlib import Path

# Test with a small audio file
test_dir = Path("/home/hendorf/code/audio_ai/app/data/input")
test_files = list(test_dir.glob("*.mp3"))[:1]  # Just test with one file

if not test_files:
    print("No test files found")
    sys.exit(1)

print(f"Testing with: {test_files[0].name}")
print("=" * 60)

# Run the transcription pipeline
cmd = [
    "uv", "run", "python", "main.py",
    "--input-dir", str(test_files[0].parent),
    "--log-level", "INFO"
]

print(f"Running: {' '.join(cmd)}")
print("=" * 60)

result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)