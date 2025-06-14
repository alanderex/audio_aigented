"""
Pydantic data models for the audio transcription pipeline.

This module defines the core data structures used throughout the system
for type safety, validation, and serialization.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator, ConfigDict


class AudioFile(BaseModel):
    """
    Represents an audio file to be processed.
    
    Attributes:
        path: Path to the audio file
        sample_rate: Sample rate of the audio file
        duration: Duration in seconds
        channels: Number of audio channels
        format: Audio format (e.g., 'wav', 'mp3')
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    path: Path
    sample_rate: Optional[int] = None
    duration: Optional[float] = None
    channels: Optional[int] = None
    format: Optional[str] = None
    
    @validator('path')
    def validate_path_exists(cls, v: Path) -> Path:
        """Validate that the audio file exists."""
        if not v.exists():
            raise ValueError(f"Audio file does not exist: {v}")
        return v


class AudioSegment(BaseModel):
    """
    Represents a segment of transcribed audio with timing information.
    
    Attributes:
        text: Transcribed text content
        start_time: Start time in seconds
        end_time: End time in seconds
        confidence: Confidence score (0.0 to 1.0)
        speaker_id: Optional speaker identifier
    """
    text: str = Field(..., description="Transcribed text content")
    start_time: float = Field(..., ge=0.0, description="Start time in seconds")
    end_time: float = Field(..., gt=0.0, description="End time in seconds")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    speaker_id: Optional[str] = Field(None, description="Speaker identifier")
    
    @validator('end_time')
    def validate_end_after_start(cls, v: float, values: Dict[str, Any]) -> float:
        """Ensure end_time is after start_time."""
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError("end_time must be greater than start_time")
        return v


class TranscriptionResult(BaseModel):
    """
    Complete transcription result for an audio file.
    
    Attributes:
        audio_file: Information about the source audio file
        segments: List of transcribed segments with timing
        full_text: Complete transcription text
        processing_time: Time taken for processing in seconds
        model_info: Information about the ASR model used
        timestamp: When the transcription was created
        metadata: Additional processing metadata
    """
    audio_file: AudioFile
    segments: List[AudioSegment] = Field(default_factory=list)
    full_text: str = Field(default="", description="Complete transcription")
    processing_time: Optional[float] = Field(None, ge=0.0, description="Processing time in seconds")
    model_info: Dict[str, str] = Field(default_factory=dict, description="ASR model information")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('full_text', always=True)
    def generate_full_text(cls, v: str, values: Dict[str, Any]) -> str:
        """Generate full text from segments if not provided."""
        if not v and 'segments' in values and values['segments']:
            return ' '.join(segment.text for segment in values['segments'])
        return v


class ProcessingConfig(BaseModel):
    """
    Configuration for audio processing pipeline.
    
    Attributes:
        input_dir: Directory containing input audio files
        output_dir: Directory for output files
        cache_dir: Directory for caching models and intermediate results
        audio: Audio processing configuration
        transcription: ASR model configuration
        output: Output formatting configuration
        processing: General processing options
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Directory paths
    input_dir: Path = Field(default=Path("./inputs"))
    output_dir: Path = Field(default=Path("./outputs"))
    cache_dir: Path = Field(default=Path("./cache"))
    
    # Audio processing settings
    audio: Dict[str, Any] = Field(default_factory=lambda: {
        "sample_rate": 16000,
        "batch_size": 8,
        "max_duration": 30.0,  # Max segment duration in seconds
    })
    
    # Transcription settings
    transcription: Dict[str, Any] = Field(default_factory=lambda: {
        "model_name": "stt_en_conformer_ctc_large",
        "device": "cuda",
        "enable_confidence_scores": True,
        "language": "en",
    })
    
    # Output settings
    output: Dict[str, Any] = Field(default_factory=lambda: {
        "formats": ["json", "txt", "attributed_txt"],
        "include_timestamps": True,
        "include_confidence": True,
        "pretty_json": True,
    })
    
    # Processing settings
    processing: Dict[str, Any] = Field(default_factory=lambda: {
        "enable_caching": True,
        "parallel_workers": 1,
        "log_level": "INFO",
        "enable_diarization": True,
    })
    
    @validator('input_dir', 'output_dir', 'cache_dir')
    def ensure_directories_exist(cls, v: Path) -> Path:
        """Ensure directories exist, create if they don't."""
        v.mkdir(parents=True, exist_ok=True)
        return v


class PipelineStatus(BaseModel):
    """
    Status information for pipeline processing.
    
    Attributes:
        total_files: Total number of files to process
        completed_files: Number of completed files
        failed_files: Number of failed files
        current_file: Currently processing file
        start_time: Pipeline start time
        estimated_completion: Estimated completion time
    """
    total_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    current_file: Optional[str] = None
    start_time: datetime = Field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate processing progress as percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.completed_files / self.total_files) * 100.0
    
    @property
    def is_complete(self) -> bool:
        """Check if processing is complete."""
        return self.completed_files + self.failed_files >= self.total_files