"""Speaker diarization using NVIDIA NeMo."""

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
            config_path = Path("config/diarization_config.yaml")
        
        self.config_path = config_path
        
        # Validate config file exists
        if not self.config_path.exists():
            raise FileNotFoundError(f"Diarization config file not found: {self.config_path}")

        try:
            # Load configuration using OmegaConf
            self.base_cfg = OmegaConf.load(self.config_path)
            logger.info("NeMo Diarizer configuration loaded successfully")
            
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
                
            # Create temporary output directory for diarization results
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create a copy of the base configuration and set the output directory
                cfg = OmegaConf.create(self.base_cfg)
                cfg.diarizer.out_dir = temp_dir
                
                # Initialize diarizer with the updated configuration
                diarizer = ClusteringDiarizer(cfg=cfg)
                diarizer.to(self.device)
                
                # Set up diarizer for this audio file and run
                logger.debug(f"Running diarization on {audio_file.path}")
                diarizer.diarize(paths2audio_files=[str(audio_file.path)])

                # Extract and sort speaker segments
                speaker_segments = []
                
                if hasattr(diarizer, 'speaker_segs') and diarizer.speaker_segs:
                    for speaker_id, segments in diarizer.speaker_segs.items():
                        for segment in segments:
                            start_time = float(segment.start)
                            end_time = float(segment.end)
                            speaker_segments.append((start_time, end_time, str(speaker_id)))
                    
                    # Sort segments by start time
                    speaker_segments.sort(key=lambda x: x[0])
                    
                    logger.info(f"Diarization completed for {audio_file.path.name}: "
                              f"found {len(speaker_segments)} speaker segments")
                else:
                    logger.warning(f"No speaker segments found for {audio_file.path.name}")

                return speaker_segments

        except Exception as e:
            logger.error(f"Diarization failed for {audio_file.path.name}: {e}")
            logger.debug(f"Diarization error details: {type(e).__name__}: {str(e)}")
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