"""Speaker diarization using NVIDIA NeMo."""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import tempfile
import os

import torch
from nemo.collections.asr.models import ClusteringDiarizer
from omegaconf import OmegaConf

from ..models.schemas import AudioFile

logger = logging.getLogger(__name__)

class NeMoDiarizer:
    """Speaker diarization using NVIDIA NeMo."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the NeMo diarizer.
        
        Args:
            config_path: Optional path to diarization config file.
                        Defaults to config/diarization_config.yaml
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing NeMo Diarizer on device: {self.device}")

        # Set default config path
        if config_path is None:
            # Get absolute path relative to module location
            import os
            # Navigate from diarizer.py -> diarization -> audio_aigented -> src -> project_root
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "config" / "diarization_config.yaml"
        
        self.config_path = config_path
        logger.debug(f"Diarization config path: {self.config_path}")
        
        # Validate config file exists
        if not self.config_path.exists():
            logger.error(f"Diarization config file not found: {self.config_path}")
            logger.error(f"Current working directory: {Path.cwd()}")
            logger.error(f"Config path exists: {self.config_path.exists()}")
            logger.error(f"Config parent exists: {self.config_path.parent.exists()}")
            raise FileNotFoundError(f"Diarization config file not found: {self.config_path}")

        try:
            # Load configuration using OmegaConf
            self.base_cfg = OmegaConf.load(self.config_path)
            logger.info("NeMo Diarizer configuration loaded successfully")
            logger.debug(f"Base config keys: {list(self.base_cfg.keys()) if hasattr(self.base_cfg, 'keys') else 'N/A'}")
            
        except Exception as e:
            logger.error(f"Failed to load NeMo Diarizer config: {e}")
            raise

    def diarize(self, audio_file: AudioFile) -> List[Tuple[float, float, str]]:
        """
        Perform speaker diarization on the given audio file.

        Args:
            audio_file: The audio file to diarize.

        Returns:
            A list of tuples, where each tuple contains the start time, end time,
            and speaker ID for each segment, sorted by start time.
        """
        try:
            logger.info(f"Starting diarization for {audio_file.path.name}")
            
            # Validate audio file
            if not audio_file.path.exists():
                logger.error(f"Audio file not found: {audio_file.path}")
                return []
            
            # Validate audio file is actually a WAV file
            if audio_file.path.suffix.lower() not in ['.wav', '.wave']:
                logger.warning(f"Audio file may not be in WAV format: {audio_file.path}")
                # Continue anyway as NeMo might handle other formats
                
            # Create temporary output directory for diarization results
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create a temporary manifest file
                manifest_path = Path(temp_dir) / "input_manifest.json"
                
                # Create manifest entry for the audio file
                manifest_entry = {
                    "audio_filepath": str(audio_file.path),
                    "duration": audio_file.duration if audio_file.duration else 100.0,
                    "label": "infer",
                    "text": "-",
                    "offset": 0.0
                }
                
                with open(manifest_path, 'w') as f:
                    json.dump(manifest_entry, f)
                    f.write('\n')
                
                # Create a copy of the base configuration and update it
                cfg = OmegaConf.to_container(self.base_cfg, resolve=True)
                cfg['diarizer']['manifest_filepath'] = str(manifest_path)
                cfg['diarizer']['out_dir'] = temp_dir
                
                # Add required top-level attributes
                if 'verbose' not in cfg:
                    cfg['verbose'] = True  # Enable verbose output for debugging
                if 'device' not in cfg:
                    cfg['device'] = self.device  # Add device configuration
                
                # Also add device to sub-configurations that might need it
                for key in ['speaker_embeddings', 'vad', 'msdd_model']:
                    if key in cfg.get('diarizer', {}):
                        if 'device' not in cfg['diarizer'][key]:
                            cfg['diarizer'][key]['device'] = self.device
                
                # Convert back to OmegaConf
                cfg = OmegaConf.create(cfg)
                
                # Initialize diarizer with the updated configuration
                logger.debug(f"Initializing ClusteringDiarizer with config keys: {list(cfg.keys())}")
                logger.debug(f"Diarizer config keys: {list(cfg.diarizer.keys()) if hasattr(cfg, 'diarizer') else 'N/A'}")
                
                # Try multiple times with different fixes
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        if attempt == 0:
                            diarizer = ClusteringDiarizer(cfg=cfg)
                        else:
                            # Use the modified config from previous attempts
                            diarizer = ClusteringDiarizer(cfg=cfg)
                        break  # Success!
                    except Exception as init_error:
                        error_msg = str(init_error)
                        logger.error(f"Attempt {attempt + 1} failed: {error_msg}")
                        
                        if attempt < max_retries - 1:  # Don't retry on last attempt
                            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
                            
                            if "Key 'device' is not in struct" in error_msg:
                                logger.info("Adding/moving device parameter")
                                if 'device' in cfg_dict:
                                    cfg_dict['diarizer']['device'] = cfg_dict.pop('device')
                                else:
                                    cfg_dict['diarizer']['device'] = self.device
                                    
                            elif "Key 'collar' is not in struct" in error_msg:
                                logger.info("Adding missing collar parameter")
                                if 'collar' not in cfg_dict.get('diarizer', {}):
                                    cfg_dict['diarizer']['collar'] = 0.25
                                if 'ignore_overlap' not in cfg_dict.get('diarizer', {}):
                                    cfg_dict['diarizer']['ignore_overlap'] = True
                                    
                            elif "Key 'smoothing' is not in struct" in error_msg:
                                logger.info("Fixing smoothing parameter")
                                # Smoothing might need to be boolean
                                if 'vad' in cfg_dict.get('diarizer', {}):
                                    if 'parameters' in cfg_dict['diarizer']['vad']:
                                        cfg_dict['diarizer']['vad']['parameters']['smoothing'] = False
                            
                            cfg = OmegaConf.create(cfg_dict)
                            logger.debug(f"Retrying with modified config")
                        else:
                            raise  # Re-raise on last attempt
                
                diarizer.to(self.device)
                
                # Set verbose attribute if not present (some NeMo versions require this)
                if not hasattr(diarizer, 'verbose'):
                    diarizer.verbose = True
                
                # Run diarization
                logger.debug(f"Running diarization on {audio_file.path}")
                diarizer.diarize()

                # Read the output RTTM file
                speaker_segments = []
                
                # Check for RTTM files in the output directory
                pred_rttms_dir = Path(temp_dir) / "pred_rttms"
                logger.debug(f"Looking for RTTM files in: {pred_rttms_dir}")
                
                if pred_rttms_dir.exists():
                    rttm_files = list(pred_rttms_dir.glob("*.rttm"))
                    logger.debug(f"Found {len(rttm_files)} RTTM files: {[f.name for f in rttm_files]}")
                    
                    # Try to find RTTM file with different naming conventions
                    rttm_file = None
                    for possible_name in [f"{audio_file.path.stem}.rttm", 
                                        f"{Path(manifest_entry['audio_filepath']).stem}.rttm",
                                        "input_manifest.rttm"]:
                        possible_file = pred_rttms_dir / possible_name
                        if possible_file.exists():
                            rttm_file = possible_file
                            break
                    
                    # If no specific file found but there's exactly one RTTM file, use it
                    if not rttm_file and len(rttm_files) == 1:
                        rttm_file = rttm_files[0]
                        logger.debug(f"Using the only RTTM file found: {rttm_file.name}")
                else:
                    logger.warning(f"RTTM output directory not found: {pred_rttms_dir}")
                    rttm_file = None
                
                if rttm_file and rttm_file.exists():
                    logger.info(f"Reading RTTM file: {rttm_file}")
                    # Log first few lines of RTTM for debugging
                    with open(rttm_file, 'r') as f:
                        rttm_lines = f.readlines()
                        logger.debug(f"RTTM file has {len(rttm_lines)} lines")
                        for i, line in enumerate(rttm_lines[:3]):
                            logger.debug(f"RTTM line {i}: {line.strip()}")
                    
                    # Now parse the RTTM file
                    with open(rttm_file, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:  # Skip empty lines
                                continue
                            
                            # RTTM format: SPEAKER filename 1 start_time duration <NA> <NA> speaker_id <NA> <NA>
                            # The filename might contain spaces, so we need to parse more carefully
                            
                            if line.startswith("SPEAKER"):
                                try:
                                    # Split only on whitespace, but handle the filename with spaces
                                    parts = line.split()
                                    
                                    # Find the numeric fields - the "1" channel field helps us locate the right position
                                    # Look for pattern: ... 1 float float ...
                                    channel_idx = None
                                    for i, part in enumerate(parts):
                                        if part == "1" and i > 0 and i < len(parts) - 4:
                                            # Check if the next two fields are floats
                                            try:
                                                float(parts[i + 1])  # start_time
                                                float(parts[i + 2])  # duration
                                                channel_idx = i
                                                break
                                            except (ValueError, IndexError):
                                                continue
                                    
                                    if channel_idx is None:
                                        logger.warning(f"Could not find valid time fields in RTTM line {line_num}: {line}")
                                        continue
                                    
                                    # Extract fields based on the channel index
                                    start_time = float(parts[channel_idx + 1])
                                    duration = float(parts[channel_idx + 2])
                                    end_time = start_time + duration
                                    
                                    # Speaker ID is typically at position channel_idx + 5
                                    if channel_idx + 5 < len(parts):
                                        speaker_id = parts[channel_idx + 5]
                                    else:
                                        logger.warning(f"Could not find speaker ID in RTTM line {line_num}")
                                        continue
                                    
                                    # Validate values
                                    if start_time < 0 or duration < 0:
                                        logger.warning(f"Invalid time values at line {line_num}: start={start_time}, duration={duration}")
                                        continue
                                        
                                    speaker_segments.append((start_time, end_time, speaker_id))
                                    
                                except (ValueError, IndexError) as e:
                                    logger.warning(f"Failed to parse RTTM line {line_num}: {line} - Error: {e}")
                                    continue
                    
                    # Sort segments by start time
                    speaker_segments.sort(key=lambda x: x[0])
                    
                    # Log unique speakers found
                    unique_speakers = set(speaker for _, _, speaker in speaker_segments)
                    logger.info(f"Diarization completed for {audio_file.path.name}: "
                              f"found {len(speaker_segments)} segments with {len(unique_speakers)} unique speakers: {unique_speakers}")
                    
                    # If only one speaker detected, log a warning
                    if len(unique_speakers) == 1:
                        logger.warning(f"Only one speaker detected in {audio_file.path.name}. "
                                     "This might indicate: 1) Single speaker audio, 2) Similar voices, "
                                     "3) Need for parameter tuning")
                else:
                    logger.warning(f"No RTTM file found. Checking temp directory contents:")
                    # List all files in temp directory for debugging
                    for root, dirs, files in os.walk(temp_dir):
                        level = root.replace(temp_dir, '').count(os.sep)
                        indent = ' ' * 2 * level
                        logger.debug(f"{indent}{os.path.basename(root)}/")
                        subindent = ' ' * 2 * (level + 1)
                        for file in files:
                            logger.debug(f"{subindent}{file}")

                return speaker_segments

        except Exception as e:
            logger.error(f"Diarization failed for {audio_file.path.name}: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            
            # Provide specific guidance for common errors
            if "ConfigAttributeError" in str(type(e).__name__) or "Key" in str(e) and "is not in struct" in str(e):
                logger.error("This appears to be a configuration error. Please check that all required "
                           "parameters are present in the diarization config file.")
                if "smoothing" in str(e):
                    logger.error("The 'smoothing' parameter should be set to 'false' or a valid smoothing method.")
            
            import traceback
            logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            return []
            
    def get_speaker_mapping(self, speaker_segments: List[Tuple[float, float, str]],
                          speaker_mapping: Optional[dict] = None) -> List[Tuple[float, float, str]]:
        """
        Apply custom speaker name mapping to diarization results.
        
        Args:
            speaker_segments: List of (start, end, speaker_id) tuples
            speaker_mapping: Dict mapping speaker IDs to custom names
            
        Returns:
            List of segments with mapped speaker names
        """
        if not speaker_mapping:
            return speaker_segments
            
        mapped_segments = []
        for start, end, speaker_id in speaker_segments:
            mapped_name = speaker_mapping.get(speaker_id, speaker_id)
            mapped_segments.append((start, end, mapped_name))
            
        return mapped_segments