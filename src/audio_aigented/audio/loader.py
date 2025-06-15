"""
Audio file loading and preprocessing module.

This module handles loading audio files, validation, resampling, and preparation
for ASR processing with NVIDIA NeMo compatibility.
"""

import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from ..models.schemas import AudioFile, ProcessingConfig

logger = logging.getLogger(__name__)


class AudioLoader:
    """
    Handles loading and preprocessing of audio files for ASR processing.
    
    Provides methods to load, validate, and prepare audio files with proper
    resampling and format conversion for NVIDIA NeMo compatibility.
    """

    def __init__(self, config: ProcessingConfig) -> None:
        """
        Initialize the AudioLoader.
        
        Args:
            config: Processing configuration containing audio settings
        """
        self.config = config
        self.target_sample_rate = config.audio["sample_rate"]
        # Reduce max duration to 10 seconds for better GPU memory management
        self.max_duration = config.audio.get("max_duration", 10.0)

        logger.info(f"AudioLoader initialized with sample rate: {self.target_sample_rate}")

    def discover_audio_files(self, input_dir: Path | None = None) -> list[Path]:
        """
        Discover all .wav audio files in the input directory.
        
        Args:
            input_dir: Optional input directory. If None, uses config default.
            
        Returns:
            List of Path objects for discovered audio files
        """
        search_dir = input_dir or self.config.input_dir

        if not search_dir.exists():
            logger.warning(f"Input directory does not exist: {search_dir}")
            return []

        # Find all .wav files
        audio_files = list(search_dir.glob("*.wav"))

        if not audio_files:
            logger.warning(f"No .wav files found in {search_dir}")

        logger.info(f"Discovered {len(audio_files)} audio files in {search_dir}")
        return audio_files

    def load_audio_file(self, file_path: Path) -> AudioFile:
        """
        Load and validate an audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            AudioFile instance with metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            # Get audio info without loading the full file
            info = sf.info(str(file_path))

            audio_file = AudioFile(
                path=file_path,
                sample_rate=info.samplerate,
                duration=info.duration,
                channels=info.channels,
                format=info.format.lower()
            )

            logger.debug(f"Loaded audio file info: {file_path.name} "
                        f"({info.duration:.2f}s, {info.samplerate}Hz, {info.channels}ch)")

            return audio_file

        except Exception as e:
            logger.error(f"Failed to load audio file {file_path}: {e}")
            raise ValueError(f"Unsupported audio file format: {file_path}")

    def load_audio_data(self, audio_file: AudioFile) -> tuple[np.ndarray, int]:
        """
        Load audio data and resample if necessary.
        
        Args:
            audio_file: AudioFile instance
            
        Returns:
            Tuple of (audio_data, sample_rate)
            
        Raises:
            ValueError: If audio loading fails
        """
        try:
            # Load audio data
            audio_data, original_sr = librosa.load(
                str(audio_file.path),
                sr=None,  # Keep original sample rate initially
                mono=True  # Convert to mono
            )

            # Resample if necessary
            if original_sr != self.target_sample_rate:
                logger.debug(f"Resampling {audio_file.path.name} from "
                           f"{original_sr}Hz to {self.target_sample_rate}Hz")
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=original_sr,
                    target_sr=self.target_sample_rate
                )

            # Normalize audio data
            audio_data = self._normalize_audio(audio_data)

            return audio_data, self.target_sample_rate

        except Exception as e:
            logger.error(f"Failed to load audio data from {audio_file.path}: {e}")
            raise ValueError(f"Failed to load audio data: {e}")

    def segment_audio(self, audio_data: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        """
        Segment long audio into chunks for processing.
        
        Args:
            audio_data: Audio data array
            sample_rate: Sample rate of the audio
            
        Returns:
            List of audio segments
        """
        duration = len(audio_data) / sample_rate

        # If audio is shorter than max duration, return as single segment
        if duration <= self.max_duration:
            return [audio_data]

        # Calculate segment parameters
        segment_samples = int(self.max_duration * sample_rate)
        overlap_samples = int(0.1 * sample_rate)  # 100ms overlap

        segments = []
        start = 0

        while start < len(audio_data):
            end = min(start + segment_samples, len(audio_data))
            segment = audio_data[start:end]

            # Only add segment if it's long enough
            if len(segment) > overlap_samples:
                segments.append(segment)

            start = end - overlap_samples

            # Break if we've reached the end
            if end >= len(audio_data):
                break

        logger.debug(f"Segmented audio into {len(segments)} chunks")
        return segments

    def process_audio_files(self, audio_files: list[Path]) -> list[AudioFile]:
        """
        Process multiple audio files and return AudioFile instances.
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            List of validated AudioFile instances
        """
        processed_files = []

        for file_path in tqdm(audio_files, desc="Loading audio files"):
            try:
                audio_file = self.load_audio_file(file_path)
                processed_files.append(audio_file)

            except Exception as e:
                logger.error(f"Skipping {file_path}: {e}")
                continue

        logger.info(f"Successfully processed {len(processed_files)} audio files")
        return processed_files

    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio data to [-1, 1] range.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Normalized audio data
        """
        # Reason: Prevent clipping and ensure consistent amplitude
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val

        return audio_data

    def validate_audio_file(self, audio_file: AudioFile) -> bool:
        """
        Validate audio file for processing compatibility.
        
        Args:
            audio_file: AudioFile instance to validate
            
        Returns:
            True if file is valid for processing
        """
        try:
            # Check file exists
            if not audio_file.path.exists():
                logger.error(f"File does not exist: {audio_file.path}")
                return False

            # Check duration
            if audio_file.duration is None or audio_file.duration <= 0:
                logger.error(f"Invalid duration: {audio_file.duration}")
                return False

            # Check sample rate
            if audio_file.sample_rate is None or audio_file.sample_rate <= 0:
                logger.error(f"Invalid sample rate: {audio_file.sample_rate}")
                return False

            # Check if file is too short (minimum 0.1 seconds)
            if audio_file.duration < 0.1:
                logger.warning(f"Audio file is very short: {audio_file.duration}s")

            return True

        except Exception as e:
            logger.error(f"Validation failed for {audio_file.path}: {e}")
            return False
