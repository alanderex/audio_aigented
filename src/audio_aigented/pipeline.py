"""
Main transcription pipeline orchestration.

This module coordinates all components of the audio transcription system,
providing a unified interface for processing audio files through the complete
ASR pipeline with error handling, logging, and progress tracking.
"""

import logging
import time
from pathlib import Path

from tqdm import tqdm

from .audio.loader import AudioLoader
from .cache.manager import CacheManager
from .config.manager import ConfigManager
from .models.schemas import (
    AudioFile,
    PipelineStatus,
    ProcessingConfig,
    TranscriptionResult,
)
from .output.writer import FileWriter
from .transcription.asr import ASRTranscriber

logger = logging.getLogger(__name__)


class TranscriptionPipeline:
    """
    Main orchestrator for the audio transcription pipeline.
    
    Coordinates audio loading, ASR processing, formatting, and output writing
    with comprehensive error handling and progress tracking.
    """

    def __init__(self, config: ProcessingConfig | None = None, config_path: Path | None = None) -> None:
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
        
        # Initialize cache manager
        cache_enabled = self.config.processing.get("enable_caching", True)
        self.cache_manager = CacheManager(self.config.cache_dir, enabled=cache_enabled)

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

    def process_directory(self, input_dir: Path | None = None) -> list[TranscriptionResult]:
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

    def process_files(self, audio_file_paths: list[Path]) -> list[TranscriptionResult]:
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
                
                # Check cache first
                cached_result = self.cache_manager.get_cached_result(file_path)
                if cached_result:
                    results.append(cached_result)
                    self.status.completed_files += 1
                    continue

                result = self.process_single_file(file_path)
                results.append(result)
                
                # Cache the result
                self.cache_manager.cache_result(file_path, result)

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

            # Stage 3: Perform diarization FIRST (if enabled)
            speaker_segments = None
            if self.config.processing.get("enable_diarization", True) and self.diarizer is not None:
                try:
                    logger.info(f"Performing speaker diarization for {file_path.name}")
                    speaker_segments = self.diarizer.diarize(audio_file)

                    if speaker_segments:
                        # Log detailed speaker segment information
                        unique_speakers = set(speaker for _, _, speaker in speaker_segments)
                        logger.info(f"Diarization found {len(speaker_segments)} segments with {len(unique_speakers)} unique speakers: {unique_speakers}")

                        # Log first few speaker segments for debugging
                        for i, (start, end, speaker) in enumerate(speaker_segments[:5]):
                            logger.debug(f"Speaker segment {i}: {speaker} [{start:.2f}s - {end:.2f}s]")
                    else:
                        logger.warning(f"No speaker segments detected for {file_path.name}")

                except Exception as e:
                    logger.warning(f"Diarization failed for {file_path.name}: {e}")
                    logger.info("Continuing without speaker identification")
                    speaker_segments = None
            else:
                logger.debug("Diarization disabled - will assign single speaker")

            # Stage 4: Transcribe audio with speaker information
            if speaker_segments:
                # Transcribe with diarization - process each speaker segment
                logger.info(f"Transcribing {len(speaker_segments)} speaker segments")
                result = self._transcribe_with_speakers(audio_file, audio_data, speaker_segments)
            else:
                # Transcribe without diarization - assume single speaker
                logger.info("Transcribing as single speaker")
                result = self.asr_transcriber.transcribe_audio_file(audio_file, audio_data)

                # Assign default speaker to all segments
                for segment in result.segments:
                    segment.speaker_id = "SPEAKER_00"

                result.metadata["num_speakers"] = 1
                result.metadata["speaker_segments"] = [
                    {"start": 0.0, "end": audio_file.duration or 0.0, "speaker": "SPEAKER_00"}
                ]

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

    def clear_cache(self) -> int:
        """
        Clear all cached transcription results.
        
        Returns:
            Number of cache files cleared
        """
        return self.cache_manager.clear_cache()
    
    @property
    def cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return {
            "cache_size": self.cache_manager.cache_size,
            "cache_size_bytes": self.cache_manager.cache_size_bytes
        }

    def _log_final_statistics(self, results: list[TranscriptionResult]) -> None:
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

    def _transcribe_with_speakers(self, audio_file: AudioFile, audio_data, speaker_segments: list[tuple[float, float, str]]) -> TranscriptionResult:
        """
        Transcribe audio with speaker diarization by processing each speaker segment.
        
        Args:
            audio_file: The audio file being processed
            audio_data: The loaded audio data
            speaker_segments: List of (start_time, end_time, speaker_id) tuples
            
        Returns:
            TranscriptionResult with speaker-attributed segments
        """
        all_segments = []
        full_text_parts = []

        # Group consecutive segments by speaker for more efficient processing
        grouped_segments = []
        current_group = None

        for start, end, speaker in speaker_segments:
            if current_group is None or current_group['speaker'] != speaker:
                if current_group:
                    grouped_segments.append(current_group)
                current_group = {
                    'speaker': speaker,
                    'start': start,
                    'end': end,
                    'segments': [(start, end)]
                }
            else:
                # Extend current group
                current_group['end'] = end
                current_group['segments'].append((start, end))

        if current_group:
            grouped_segments.append(current_group)

        logger.info(f"Processing {len(grouped_segments)} speaker groups from {len(speaker_segments)} segments")

        # Process each speaker group
        for group in grouped_segments:
            speaker_id = group['speaker']
            group_start = group['start']
            group_end = group['end']

            logger.debug(f"Transcribing {speaker_id} segment: {group_start:.2f}s - {group_end:.2f}s")

            # Extract audio segment for this speaker group
            start_sample = int(group_start * audio_file.sample_rate)
            end_sample = int(group_end * audio_file.sample_rate)
            segment_audio = audio_data[start_sample:end_sample]

            # Create temporary audio file for this segment
            import tempfile

            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, segment_audio, audio_file.sample_rate)

                # Create AudioFile object for the segment
                segment_audio_file = AudioFile(
                    path=Path(tmp_file.name),
                    duration=group_end - group_start,
                    sample_rate=audio_file.sample_rate,
                    channels=audio_file.channels
                )

                try:
                    # Transcribe this segment
                    segment_result = self.asr_transcriber.transcribe_audio_file(segment_audio_file, segment_audio)

                    # Adjust timestamps and assign speaker
                    for seg in segment_result.segments:
                        seg.start_time += group_start
                        seg.end_time += group_start
                        seg.speaker_id = speaker_id
                        all_segments.append(seg)

                    if segment_result.full_text:
                        full_text_parts.append(segment_result.full_text)

                finally:
                    # Clean up temporary file
                    Path(tmp_file.name).unlink(missing_ok=True)

        # Sort segments by start time
        all_segments.sort(key=lambda s: s.start_time)

        # Create final result
        result = TranscriptionResult(
            audio_file=audio_file,
            segments=all_segments,
            full_text=' '.join(full_text_parts),
            processing_time=0.0,  # Will be updated by caller
            metadata={
                'num_speakers': len(set(s['speaker'] for s in grouped_segments)),
                'speaker_segments': [
                    {"start": start, "end": end, "speaker": speaker}
                    for start, end, speaker in speaker_segments
                ]
            }
        )

        return result

    @property
    def pipeline_status(self) -> PipelineStatus:
        """Get current pipeline processing status."""
        return self.status

    def get_supported_formats(self) -> list[str]:
        """Get list of supported audio formats."""
        return ["wav"]  # Currently only supporting WAV files
