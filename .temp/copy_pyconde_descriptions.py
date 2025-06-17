#!/usr/bin/env python3
"""
Script to copy PyCon DE 2025 talk descriptions to context directory.

Matches audio files with their corresponding description files by ID in square brackets.
"""

import re
from pathlib import Path
import shutil

# Define directories
audio_dir = Path("/home/hendorf/Documents/pyconde2025")
descriptions_dir = Path("/home/hendorf/Documents/pyconde2025-talkpage")
output_dir = Path("/home/hendorf/Documents/pyconde2025-context")

# Create output directory if it doesn't exist
output_dir.mkdir(exist_ok=True)

# Pattern to extract ID from filenames
id_pattern = re.compile(r'\[([A-Z0-9]+)\]')

# Process each audio file
for audio_file in audio_dir.iterdir():
    if audio_file.is_file():
        # Extract ID from audio filename
        match = id_pattern.search(audio_file.name)
        if match:
            file_id = match.group(1)
            
            # Find corresponding description file
            description_file = descriptions_dir / file_id / "contents.lr"
            
            if description_file.exists():
                # Create output filename (same as audio file but with .md extension)
                output_filename = audio_file.stem + ".md"
                output_path = output_dir / output_filename
                
                # Copy the description file
                shutil.copy2(description_file, output_path)
                print(f"✓ Copied: {file_id} -> {output_filename}")
            else:
                print(f"✗ Description not found for ID: {file_id} (file: {audio_file.name})")
        else:
            print(f"⚠ No ID found in filename: {audio_file.name}")

print(f"\nProcessing complete. Check {output_dir} for results.")