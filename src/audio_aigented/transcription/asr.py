"""
Automatic Speech Recognition (ASR) using NVIDIA NeMo.

This module provides ASR functionality using NVIDIA NeMo's pre-trained models
with GPU acceleration and efficient batch processing.
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
from tqdm import tqdm

# Set PyTorch CUDA memory allocation configuration for better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# NeMo imports - these will be available after installation
try:
    import nemo.collections.asr as nemo_asr
    from nemo.core.classes import typecheck
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    nemo_asr = None

from ..models.schemas import AudioFile, AudioSegment, TranscriptionResult, ProcessingConfig

logger = logging.getLogger(__name__)


class ASRTranscriber:
    """
    NVIDIA NeMo-based automatic speech recognition transcriber.
    
    Handles loading pre-trained models, GPU acceleration, and batch processing
    for efficient speech-to-text conversion.
    """
    
    def __init__(self, config: ProcessingConfig) -> None:
        """
        Initialize the ASR transcriber.
        
        Args:
            config: Processing configuration containing transcription settings
            
        Raises:
            ImportError: If NVIDIA NeMo is not available
            RuntimeError: If GPU is requested but not available
        """
        if not NEMO_AVAILABLE:
            raise ImportError(
                "NVIDIA NeMo is not available. Please install with: "
                "pip install nemo-toolkit[asr]"
            )
            
        self.config = config
        self.transcription_config = config.transcription
        self.device = self._setup_device()
        self.model: Optional[Any] = None
        self.model_name = self.transcription_config["model_name"]
        self.enable_confidence = self.transcription_config.get("enable_confidence_scores", True)
        
        logger.info(f"ASRTranscriber initialized with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
    def _setup_device(self) -> str:
        """
        Setup and validate the processing device.
        
        Returns:
            Device string ('cuda' or 'cpu')
            
        Raises:
            RuntimeError: If CUDA is requested but not available
        """
        requested_device = self.transcription_config.get("device", "cuda")
        
        if requested_device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return "cpu"
            else:
                # Log GPU information
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"CUDA available with {gpu_count} GPU(s). Using: {gpu_name}")
                return "cuda"
        else:
            return "cpu"
            
    def load_model(self) -> None:
        """
        Load the NVIDIA NeMo ASR model.
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            logger.info(f"Loading NeMo ASR model: {self.model_name}")
            start_time = time.time()
            
            # Load pre-trained model
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.model_name,
                map_location=self.device
            )
            
            # Move model to device
            if self.device == "cuda":
                self.model = self.model.cuda()
            else:
                self.model = self.model.cpu()
                
            # Set to evaluation mode
            self.model.eval()
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
            # Cache model info - DIAGNOSTIC: Convert all values to strings for Pydantic compatibility
            self._model_info = {
                "name": self.model_name,
                "device": self.device,
                "load_time": f"{load_time:.2f}",  # Convert float to string
                "sample_rate": str(getattr(self.model, 'sample_rate', 16000))  # Convert int to string
            }
            
            logger.info(f"DIAGNOSTIC - Model info types: {[(k, type(v)) for k, v in self._model_info.items()]}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
            
    def transcribe_audio_file(self, audio_file: AudioFile, audio_data: np.ndarray) -> TranscriptionResult:
        """
        Transcribe a complete audio file.
        
        Args:
            audio_file: AudioFile instance with metadata
            audio_data: Audio data array
            
        Returns:
            TranscriptionResult with complete transcription
        """
        if self.model is None:
            self.load_model()
            
        logger.info(f"Transcribing audio file: {audio_file.path.name}")
        
        # DIAGNOSTIC: Log audio data size and memory usage
        audio_size_mb = audio_data.nbytes / (1024 * 1024)
        duration = len(audio_data) / self.config.audio["sample_rate"]
        logger.info(f"DIAGNOSTIC - Audio size: {audio_size_mb:.2f} MB, Duration: {duration:.2f}s")
        
        # Clear CUDA cache before processing
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.info("DIAGNOSTIC - Cleared CUDA cache before processing")
        
        start_time = time.time()
        
        try:
            # Get transcription segments with memory-efficient processing
            segments = self._transcribe_segments_chunked(audio_data)
            
            # Create full text
            full_text = ' '.join(segment.text for segment in segments)
            
            processing_time = time.time() - start_time
            
            # Create transcription result
            result = TranscriptionResult(
                audio_file=audio_file,
                segments=segments,
                full_text=full_text,
                processing_time=processing_time,
                model_info=self._model_info.copy(),
                metadata={
                    "segments_count": len(segments),
                    "average_confidence": self._calculate_average_confidence(segments),
                    "processing_speed_ratio": audio_file.duration / processing_time if audio_file.duration else 0
                }
            )
            
            logger.info(f"Transcription completed in {processing_time:.2f}s "
                       f"(speed ratio: {result.metadata['processing_speed_ratio']:.2f}x)")
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed for {audio_file.path}: {e}")
            raise
            
    def _transcribe_segments_chunked(self, audio_data: np.ndarray) -> List[AudioSegment]:
        """
        Transcribe audio data using memory-efficient chunking.
        
        Args:
            audio_data: Audio data array
            
        Returns:
            List of AudioSegment instances
        """
        # Use the audio loader's segmentation for memory efficiency
        from ..audio.loader import AudioLoader
        audio_loader = AudioLoader(self.config)
        audio_chunks = audio_loader.segment_audio(audio_data, self.config.audio["sample_rate"])
        
        logger.info(f"Processing {len(audio_chunks)} audio chunks for memory efficiency")
        
        segments = []
        current_time = 0.0
        
        for i, chunk in enumerate(audio_chunks):
            try:
                # Clear GPU cache between chunks
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                # Process single chunk
                chunk_segments = self._transcribe_single_chunk(chunk, current_time)
                segments.extend(chunk_segments)
                
                # Update time offset for next chunk
                chunk_duration = len(chunk) / self.config.audio["sample_rate"]
                current_time += chunk_duration
                
                logger.debug(f"Processed chunk {i+1}/{len(audio_chunks)}")
                
            except Exception as e:
                logger.error(f"Failed to process chunk {i+1}: {e}")
                # Continue with next chunk rather than failing completely
                continue
                
        return segments
    
    def _transcribe_single_chunk(self, audio_chunk: np.ndarray, start_time_offset: float) -> List[AudioSegment]:
        """
        Transcribe a single audio chunk.
        
        Args:
            audio_chunk: Single audio chunk array
            start_time_offset: Time offset for this chunk
            
        Returns:
            List of AudioSegment instances for this chunk
        """
        try:
            # DIAGNOSTIC: Log chunk details
            chunk_size_mb = audio_chunk.nbytes / (1024 * 1024)
            logger.debug(f"DIAGNOSTIC - Processing chunk: {chunk_size_mb:.2f} MB")
            
            # Get transcription from NeMo model
            transcription = self.model.transcribe([audio_chunk])
            
            # DIAGNOSTIC: Log transcription result type and content
            logger.debug(f"DIAGNOSTIC - Transcription result type: {type(transcription)}")
            logger.debug(f"DIAGNOSTIC - Transcription result: {transcription}")
            
            # Handle different NeMo result formats
            text = self._extract_text_from_result(transcription)
                
            # Create segment with proper timing
            if text and text.strip():
                chunk_duration = len(audio_chunk) / self.config.audio["sample_rate"]
                segment = AudioSegment(
                    text=text.strip(),
                    start_time=start_time_offset,
                    end_time=start_time_offset + chunk_duration,
                    confidence=self._estimate_confidence(text) if self.enable_confidence else None
                )
                return [segment]
            else:
                return []
            
        except Exception as e:
            logger.error(f"Chunk transcription failed: {e}")
            return []
            
    def _extract_text_from_result(self, transcription_result) -> str:
        """
        Extract text from NeMo transcription result, handling different formats.
        
        Args:
            transcription_result: Result from NeMo model.transcribe()
            
        Returns:
            Extracted text string
        """
        try:
            # Handle list of results
            if isinstance(transcription_result, list):
                if len(transcription_result) == 0:
                    return ""
                result = transcription_result[0]
            else:
                result = transcription_result
                
            # Handle NeMo Hypothesis objects
            if hasattr(result, 'text'):
                return result.text
            elif hasattr(result, 'hyp'):
                # Some NeMo models return objects with 'hyp' attribute
                hyp = result.hyp
                if hasattr(hyp, 'text'):
                    return hyp.text
                else:
                    return str(hyp)
            elif isinstance(result, str):
                return result
            else:
                # Fallback to string conversion
                return str(result)
                
        except Exception as e:
            logger.warning(f"Failed to extract text from transcription result: {e}")
            return ""
            
    def _estimate_confidence(self, text: str) -> float:
        """
        Estimate confidence score for transcribed text.
        
        Args:
            text: Transcribed text
            
        Returns:
            Estimated confidence score (0.0 to 1.0)
        """
        # Reason: Simple heuristic-based confidence estimation
        # In production, this would use model-specific confidence scores
        if not text or not text.strip():
            return 0.0
            
        # Basic heuristics for confidence estimation
        confidence = 0.8  # Base confidence
        
        # Adjust based on text characteristics
        words = text.split()
        if len(words) > 0:
            # Longer utterances typically have lower confidence
            length_factor = max(0.5, 1.0 - (len(words) / 100))
            confidence *= length_factor
            
        # Adjust for special characters (might indicate unclear speech)
        special_chars = sum(1 for c in text if not c.isalnum() and c != ' ')
        if special_chars > len(text) * 0.1:  # More than 10% special chars
            confidence *= 0.9
            
        return min(1.0, max(0.0, confidence))
        
    def _calculate_average_confidence(self, segments: List[AudioSegment]) -> Optional[float]:
        """
        Calculate average confidence score across segments.
        
        Args:
            segments: List of audio segments
            
        Returns:
            Average confidence score or None if no confidence scores
        """
        if not segments:
            return None
            
        confidences = [seg.confidence for seg in segments if seg.confidence is not None]
        
        if not confidences:
            return None
            
        return sum(confidences) / len(confidences)
        
    def transcribe_batch(self, audio_files: List[Tuple[AudioFile, np.ndarray]]) -> List[TranscriptionResult]:
        """
        Transcribe multiple audio files in batch.
        
        Args:
            audio_files: List of (AudioFile, audio_data) tuples
            
        Returns:
            List of TranscriptionResult instances
        """
        if self.model is None:
            self.load_model()
            
        results = []
        
        for audio_file, audio_data in tqdm(audio_files, desc="Transcribing audio files"):
            try:
                result = self.transcribe_audio_file(audio_file, audio_data)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to transcribe {audio_file.path}: {e}")
                # Create empty result for failed transcription
                empty_result = TranscriptionResult(
                    audio_file=audio_file,
                    segments=[],
                    full_text="",
                    processing_time=0.0,
                    model_info=self._model_info.copy() if hasattr(self, '_model_info') else {},
                    metadata={"error": str(e)}
                )
                results.append(empty_result)
                
        logger.info(f"Batch transcription completed: {len(results)} files processed")
        return results
        
    @property
    def is_model_loaded(self) -> bool:
        """Check if the ASR model is loaded."""
        return self.model is not None
        
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if hasattr(self, '_model_info'):
            return self._model_info.copy()
        return {}