#!/usr/bin/env python3
"""Test that oracle_num_speakers is being set correctly."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.audio_aigented.diarization.config_builder import DiarizationConfigBuilder
from omegaconf import OmegaConf

# Test with base config
config_path = Path("config/diarization_config.yaml")
builder = DiarizationConfigBuilder(config_path)

# Build config with num_speakers
manifest_path = Path("/tmp/test_manifest.json")
output_dir = Path("/tmp/test_output")
audio_path = Path("/tmp/test.wav")

print("Testing oracle_num_speakers configuration...")
print("=" * 60)

# Test without num_speakers
cfg1 = builder.build_config(manifest_path, output_dir, audio_path, 100.0)
print(f"Without num_speakers: oracle_num_speakers = {cfg1.diarizer.clustering.parameters.oracle_num_speakers}")

# Test with num_speakers = 4
cfg2 = builder.build_config(manifest_path, output_dir, audio_path, 100.0, num_speakers=4)
print(f"With num_speakers=4: oracle_num_speakers = {cfg2.diarizer.clustering.parameters.oracle_num_speakers}")

# Print the full clustering config
print("\nFull clustering config with num_speakers=4:")
print(OmegaConf.to_yaml(cfg2.diarizer.clustering))

# Check if it's actually an integer
oracle_value = cfg2.diarizer.clustering.parameters.oracle_num_speakers
print(f"\nType of oracle_num_speakers: {type(oracle_value)}")
print(f"Value: {oracle_value}")
print(f"Is it 4? {oracle_value == 4}")