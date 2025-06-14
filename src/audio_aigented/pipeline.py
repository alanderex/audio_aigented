"""
Main transcription pipeline orchestration.

This module coordinates all components of the audio transcription system,
providing a unified interface for processing audio files through the complete
ASR pipeline with error handling, logging, and progress tracking.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

from .audio.loader import AudioLoader
from .config.manager import ConfigManager
from .formatting.formatter import OutputFormatter
from .models.schemas import ProcessingConfig, TranscriptionResult, PipelineStatus, AudioFile
from .output.writer import FileWriter
from .transcription.asr import ASRTranscriber

logger = logging.getLogger(__name__)


class TranscriptionPipeline:
    """
    Main orchestrator for the audio transcription pipeline.
    
    Coordinates audio loading, ASR processing, formatting, and output writing
    with comprehensive error handling and progress tracking.
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None, config_path: Optional[Path] = None) -> None:
        """
        Initialize the transcription pipeline.
        
        Args:
            config: Optional ProcessingConfig instance
            config_path: Optional path to configuration file
        """
        # Load configuration
        if config is None:
            config_manager = ConfigManager(config_path)
            self.config = config_manager.load_config()
        else:
            self.config = config
            
        # Initialize components
        self.audio_loader = AudioLoader(self.config)
        self.asr_transcriber = ASRTranscriber(self.config)
        self.file_writer = FileWriter(self.config)
        
        # Initialize diarizer if enabled
        self.diarizer = None
        if self.config.processing.get("enable_diarization", True):
            try:
                from .diarization.diarizer import NeMoDiarizer
                self.diarizer = NeMoDiarizer()
                logger.info("Speaker diarization enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize diarizer: {e}")
                logger.info("Continuing without speaker diarization")
                self.config.processing["enable_diarization"] = False
        else:
            logger.info("Speaker diarization disabled")
        
        # Pipeline state
        self.status = PipelineStatus()
        
        # Setup logging
        self._setup_logging()
        
        logger.info("TranscriptionPipeline initialized successfully")
        
    def _setup_logging(self) -> None:
        """Setup logging configuration based on config settings."""
        log_level = self.config.processing.get("log_level", "INFO")
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
    def process_directory(self, input_dir: Optional[Path] = None) -> List[TranscriptionResult]:
        """
        Process all audio files in a directory.
        
        Args:
            input_dir: Optional input directory. If None, uses config default.
            
        Returns:
            List of TranscriptionResult instances
        """
        search_dir = input_dir or self.config.input_dir
        logger.info(f"Starting directory processing: {search_dir}")
        
        # Discover audio files
        audio_file_paths = self.audio_loader.discover_audio_files(search_dir)
        
        if not audio_file_paths:
            logger.warning("No audio files found to process")
            return []
            
        # Process all files
        return self.process_files(audio_file_paths)
        
    def process_files(self, audio_file_paths: List[Path]) -> List[TranscriptionResult]:
        """
        Process a list of audio files.
        
        Args:
            audio_file_paths: List of audio file paths to process
            
        Returns:
            List of TranscriptionResult instances
        """
        logger.info(f"Processing {len(audio_file_paths)} audio files")
        
        # Initialize pipeline status
        self.status = PipelineStatus(total_files=len(audio_file_paths))
        
        results = []
        
        # Process each file
        for file_path in tqdm(audio_file_paths, desc="Processing audio files"):
            try:
                self.status.current_file = file_path.name
                
                result = self.process_single_file(file_path)
                results.append(result)
                
                self.status.completed_files += 1
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                self.status.failed_files += 1
                
                # Create empty result for failed processing
                try:
                    audio_file = AudioFile(path=file_path)
                    empty_result = TranscriptionResult(
                        audio_file=audio_file,
                        segments=[],
                        full_text="",
                        processing_time=0.0,
                        metadata={"error": str(e)}
                    )
                    results.append(empty_result)
                except Exception:
                    # Skip if we can't even create an empty result
                    pass
                    
        # Write all results
        self.file_writer.write_batch_results(results)
        
        # Create summary report
        if results:
            self.file_writer.create_summary_report(results)
            
        # Log final statistics
        self._log_final_statistics(results)
        
        return results
        
    def process_single_file(self, file_path: Path) -> TranscriptionResult:
        """
        Process a single audio file through the complete pipeline.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            TranscriptionResult instance
        """
        logger.info(f"Processing file: {file_path.name}")
        start_time = time.time()
        
        try:
            # Stage 1: Load audio file
            audio_file = self.audio_loader.load_audio_file(file_path)
            
            # Validate audio file
            if not self.audio_loader.validate_audio_file(audio_file):
                raise ValueError(f"Audio file validation failed: {file_path}")
                
            # Stage 2: Load audio data
            audio_data, sample_rate = self.audio_loader.load_audio_data(audio_file)
            
            # Stage 3: Transcribe audio
            result = self.asr_transcriber.transcribe_audio_file(audio_file, audio_data)

            # Stage 4: Perform diarization (if enabled)
            if self.config.processing.get("enable_diarization", True) and self.diarizer is not None:
                try:
                    speaker_segments = self.diarizer.diarize(audio_file)
                    
                    if speaker_segments:
                        # Assign speaker IDs to transcription segments
                        self._assign_speakers_to_segments(result.segments, speaker_segments)
                        
                        # Add speaker information to metadata
                        result.metadata["speaker_segments"] = [
                            {"start": start, "end": end, "speaker": speaker}
                            for start, end, speaker in speaker_segments
                        ]
                        result.metadata["num_speakers"] = len(set(speaker for _, _, speaker in speaker_segments))
                        
                        logger.info(f"Diarization completed for {file_path.name}: "
                                  f"identified {result.metadata['num_speakers']} speakers")
                    else:
                        logger.warning(f"No speaker segments detected for {file_path.name}")
                        
                except Exception as e:
                    logger.warning(f"Diarization failed for {file_path.name}: {e}")
                    logger.info("Continuing without speaker identification")
            else:
                logger.debug("Diarization disabled - skipping speaker identification")
            
            # Stage 5: Write output
            created_files = self.file_writer.write_transcription_result(result)
            
            # Add file paths to metadata
            result.metadata["output_files"] = [str(f) for f in created_files]
            
            total_time = time.time() - start_time
            logger.info(f"Successfully processed {file_path.name} in {total_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed for {file_path}: {e}")
            raise
            
    def process_single_file_with_caching(self, file_path: Path) -> TranscriptionResult:
        """
        Process a single file with caching support.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            TranscriptionResult instance
        """
        # Check if caching is enabled
        if not self.config.processing.get("enable_caching", True):
            return self.process_single_file(file_path)
            
        # Generate cache key based on file path and modification time
        cache_key = self._generate_cache_key(file_path)
        cache_path = self.config.cache_dir / f"{cache_key}.json"
        
        # Try to load from cache
        if cache_path.exists():
            try:
                cached_result = self._load_cached_result(cache_path, file_path)
                if cached_result:
                    logger.info(f"Loaded cached result for {file_path.name}")
                    return cached_result
            except Exception as e:
                logger.warning(f"Failed to load cached result: {e}")
                
        # Process normally and cache result
        result = self.process_single_file(file_path)
        
        try:
            self._save_cached_result(result, cache_path)
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
            
        return result
        
    def _generate_cache_key(self, file_path: Path) -> str:
        """
        Generate a cache key for a file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Cache key string
        """
        import hashlib
        
        # Use file path and modification time for cache key
        stat = file_path.stat()
        key_data = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(key_data.encode()).hexdigest()
        
    def _load_cached_result(self, cache_path: Path, file_path: Path) -> Optional[TranscriptionResult]:
        """
        Load cached transcription result.
        
        Args:
            cache_path: Path to cache file
            file_path: Original audio file path
            
        Returns:
            TranscriptionResult if successful, None otherwise
        """
        import json
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Reconstruct TranscriptionResult
            # This is a simplified version - in production you'd want
            # more robust serialization/deserialization
            audio_file = AudioFile(path=file_path)
            
            # Note: This is a basic implementation
            # For production, you'd want proper Pydantic serialization
            return None  # Disable caching for now - requires more complex implementation
            
        except Exception:
            return None
            
    def _save_cached_result(self, result: TranscriptionResult, cache_path: Path) -> None:
        """
        Save transcription result to cache.
        
        Args:
            result: TranscriptionResult to cache
            cache_path: Path to save cache file
        """
        # Ensure cache directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # For now, skip caching implementation
        # In production, you'd serialize the result properly
        pass
        
    def _log_final_statistics(self, results: List[TranscriptionResult]) -> None:
        """
        Log final processing statistics.
        
        Args:
            results: List of all processing results
        """
        successful = sum(1 for r in results if r.full_text.strip())
        failed = len(results) - successful
        
        total_duration = sum(r.audio_file.duration or 0 for r in results if r.audio_file.duration)
        total_processing_time = sum(r.processing_time or 0 for r in results if r.processing_time)
        
        avg_speed_ratio = total_duration / total_processing_time if total_processing_time > 0 else 0
        
        logger.info("=" * 50)
        logger.info("PIPELINE PROCESSING COMPLETED")
        logger.info(f"Total files: {len(results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total audio duration: {total_duration:.2f}s")
        logger.info(f"Total processing time: {total_processing_time:.2f}s")
        logger.info(f"Average speed ratio: {avg_speed_ratio:.2f}x")
        logger.info("=" * 50)
        
    def _assign_speakers_to_segments(self, transcription_segments, speaker_segments):
        """
        Assign speaker IDs to transcription segments based on temporal overlap.
        
        Args:
            transcription_segments: List of AudioSegment instances from transcription
            speaker_segments: List of (start, end, speaker_id) tuples from diarization
        """
        logger.debug(f"Assigning speakers to {len(transcription_segments)} transcription segments "
                    f"using {len(speaker_segments)} speaker segments")
        
        for segment in transcription_segments:
            # Find the speaker segment with the most overlap
            best_speaker = None
            max_overlap = 0.0
            
            for speaker_start, speaker_end, speaker_id in speaker_segments:
                # Calculate overlap between transcription segment and speaker segment
                overlap_start = max(segment.start_time, speaker_start)
                overlap_end = min(segment.end_time, speaker_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_speaker = speaker_id
            
            # Assign speaker if we found a significant overlap
            if best_speaker and max_overlap > 0:
                segment.speaker_id = best_speaker
                logger.debug(f"Assigned speaker {best_speaker} to segment "
                           f"[{segment.start_time:.2f}s - {segment.end_time:.2f}s] "
                           f"with {max_overlap:.2f}s overlap")
            else:
                logger.debug(f"No speaker assigned to segment "
                           f"[{segment.start_time:.2f}s - {segment.end_time:.2f}s]")
        
    @property
    def pipeline_status(self) -> PipelineStatus:
        """Get current pipeline processing status."""
        return self.status
        
    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats."""
        return ["wav"]  # Currently only supporting WAV files