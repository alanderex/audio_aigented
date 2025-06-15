"""Speaker diarization using NVIDIA NeMo."""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict

import torch
from nemo.collections.asr.models import ClusteringDiarizer
from omegaconf import DictConfig

from ..models.schemas import AudioFile
from ..utils import retry_on_error
from .config_builder import DiarizationConfigBuilder
from .rttm_parser import RTTMParser

logger = logging.getLogger(__name__)

class NeMoDiarizer:
    """Speaker diarization using NVIDIA NeMo."""

    def __init__(self, config_path: Path | None = None, 
                 collar: float = 0.25,
                 merge_segments: bool = True,
                 min_segment_duration: float = 0.1):
        """
        Initialize the NeMo diarizer.
        
        Args:
            config_path: Optional path to diarization config file.
                        Defaults to config/diarization_config.yaml
            collar: Time collar in seconds for segment boundaries
            merge_segments: Whether to merge adjacent segments from same speaker
            min_segment_duration: Minimum segment duration to keep
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing NeMo Diarizer on device: {self.device}")

        # Set default config path
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "config" / "diarization_config.yaml"

        self.config_path = config_path
        self.collar = collar
        self.merge_segments = merge_segments
        self.min_segment_duration = min_segment_duration
        
        # Initialize helper components
        self.config_builder = DiarizationConfigBuilder(config_path)
        self.rttm_parser = RTTMParser(collar=collar)
        
        logger.debug(f"Diarization config path: {self.config_path}")

    @retry_on_error(max_attempts=3, delay=1.0, exceptions=(RuntimeError, ValueError))
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
                # Prepare diarization
                manifest_path, cfg = self._prepare_diarization(
                    audio_file, Path(temp_dir)
                )
                
                # Initialize and run diarizer
                diarizer = self._initialize_diarizer(cfg)
                
                # Run diarization
                logger.debug(f"Running diarization on {audio_file.path}")
                diarizer.diarize()

                # Parse results
                speaker_segments = self._parse_results(
                    Path(temp_dir), audio_file
                )
                
                # Post-process segments if needed
                if speaker_segments:
                    if self.merge_segments:
                        speaker_segments = self.rttm_parser.merge_adjacent_segments(
                            speaker_segments
                        )
                    
                    if self.min_segment_duration > 0:
                        speaker_segments = self.rttm_parser.filter_short_segments(
                            speaker_segments, self.min_segment_duration
                        )
                
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

    def _prepare_diarization(self, audio_file: AudioFile, 
                           temp_dir: Path) -> Tuple[Path, DictConfig]:
        """
        Prepare manifest and configuration for diarization.
        
        Args:
            audio_file: Audio file to process
            temp_dir: Temporary directory for outputs
            
        Returns:
            Tuple of (manifest_path, configuration)
        """
        # Create manifest file
        manifest_path = temp_dir / "input_manifest.json"
        manifest_entry = self.config_builder.create_manifest_entry(
            audio_file.path, audio_file.duration or 100.0
        )
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest_entry, f)
            f.write('\n')
            
        # Build configuration
        cfg = self.config_builder.build_config(
            manifest_path, temp_dir, audio_file.path, 
            audio_file.duration or 100.0
        )
        
        return manifest_path, cfg
    
    def _initialize_diarizer(self, cfg: DictConfig) -> ClusteringDiarizer:
        """
        Initialize the NeMo diarizer with configuration.
        
        Args:
            cfg: Diarization configuration
            
        Returns:
            Initialized ClusteringDiarizer
        """
        logger.debug(f"Initializing ClusteringDiarizer with config keys: {list(cfg.keys())}")
        
        diarizer = ClusteringDiarizer(cfg=cfg)
        diarizer.to(self.device)
        
        # Set verbose attribute if not present
        if not hasattr(diarizer, 'verbose'):
            diarizer.verbose = True
            
        return diarizer
    
    def _parse_results(self, temp_dir: Path, 
                      audio_file: AudioFile) -> List[Tuple[float, float, str]]:
        """
        Parse diarization results from temporary directory.
        
        Args:
            temp_dir: Temporary directory with results
            audio_file: Original audio file
            
        Returns:
            List of speaker segments
        """
        # Find RTTM file
        rttm_file = self._find_rttm_file(temp_dir, audio_file)
        
        if not rttm_file:
            logger.warning("No RTTM file found in output directory")
            self._log_temp_dir_contents(temp_dir)
            return []
            
        # Parse RTTM file
        try:
            segments = self.rttm_parser.parse_rttm_file(rttm_file)
            
            # Check for single speaker warning
            if segments:
                unique_speakers = set(s[2] for s in segments)
                if len(unique_speakers) == 1:
                    logger.warning(
                        f"Only one speaker detected in {audio_file.path.name}. "
                        "This might indicate: 1) Single speaker audio, "
                        "2) Similar voices, 3) Need for parameter tuning"
                    )
                    
            return segments
            
        except Exception as e:
            logger.error(f"Failed to parse RTTM file: {e}")
            return []
    
    def _find_rttm_file(self, temp_dir: Path, 
                       audio_file: AudioFile) -> Optional[Path]:
        """
        Find the RTTM output file.
        
        Args:
            temp_dir: Directory to search
            audio_file: Original audio file
            
        Returns:
            Path to RTTM file or None
        """
        pred_rttms_dir = temp_dir / "pred_rttms"
        
        if not pred_rttms_dir.exists():
            return None
            
        rttm_files = list(pred_rttms_dir.glob("*.rttm"))
        logger.debug(f"Found {len(rttm_files)} RTTM files")
        
        # Try different naming conventions
        for possible_name in [
            f"{audio_file.path.stem}.rttm",
            "input_manifest.rttm"
        ]:
            possible_file = pred_rttms_dir / possible_name
            if possible_file.exists():
                return possible_file
                
        # Use the only file if there's just one
        if len(rttm_files) == 1:
            return rttm_files[0]
            
        return None
    
    def _log_temp_dir_contents(self, temp_dir: Path) -> None:
        """Log contents of temporary directory for debugging."""
        logger.debug("Temporary directory contents:")
        for root, dirs, files in os.walk(temp_dir):
            level = root.replace(str(temp_dir), '').count(os.sep)
            indent = ' ' * 2 * level
            logger.debug(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                logger.debug(f"{subindent}{file}")
    
    def get_speaker_mapping(self, speaker_segments: List[Tuple[float, float, str]],
                          speaker_mapping: Optional[Dict[str, str]] = None) -> List[Tuple[float, float, str]]:
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
