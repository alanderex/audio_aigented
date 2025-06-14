"""
Unit tests for data models and schemas.

Tests the Pydantic models for validation, serialization, and edge cases.
"""

import pytest
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.audio_aigented.models.schemas import (
    AudioFile,
    AudioSegment,
    TranscriptionResult,
    ProcessingConfig,
    PipelineStatus
)


class TestAudioFile:
    """Tests for AudioFile model."""
    
    def test_audio_file_creation_success(self):
        """Test successful AudioFile creation with valid data."""
        # Create a temporary file for testing
        with NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
        try:
            audio_file = AudioFile(
                path=tmp_path,
                sample_rate=16000,
                duration=10.5,
                channels=1,
                format='wav'
            )
            
            assert audio_file.path == tmp_path
            assert audio_file.sample_rate == 16000
            assert audio_file.duration == 10.5
            assert audio_file.channels == 1
            assert audio_file.format == 'wav'
            
        finally:
            # Clean up
            if tmp_path.exists():
                tmp_path.unlink()
                
    def test_audio_file_nonexistent_path_failure(self):
        """Test AudioFile creation fails with nonexistent file."""
        nonexistent_path = Path('/nonexistent/file.wav')
        
        with pytest.raises(ValueError, match="Audio file does not exist"):
            AudioFile(path=nonexistent_path)
            
    def test_audio_file_minimal_creation(self):
        """Test AudioFile creation with minimal data (edge case)."""
        with NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
        try:
            audio_file = AudioFile(path=tmp_path)
            
            assert audio_file.path == tmp_path
            assert audio_file.sample_rate is None
            assert audio_file.duration is None
            assert audio_file.channels is None
            assert audio_file.format is None
            
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestAudioSegment:
    """Tests for AudioSegment model."""
    
    def test_audio_segment_creation_success(self):
        """Test successful AudioSegment creation with valid data."""
        segment = AudioSegment(
            text="Hello world",
            start_time=0.0,
            end_time=2.5,
            confidence=0.95,
            speaker_id="speaker_1"
        )
        
        assert segment.text == "Hello world"
        assert segment.start_time == 0.0
        assert segment.end_time == 2.5
        assert segment.confidence == 0.95
        assert segment.speaker_id == "speaker_1"
        
    def test_audio_segment_end_before_start_failure(self):
        """Test AudioSegment creation fails when end_time <= start_time."""
        with pytest.raises(ValueError, match="end_time must be greater than start_time"):
            AudioSegment(
                text="Invalid timing",
                start_time=5.0,
                end_time=3.0  # End before start
            )
            
    def test_audio_segment_edge_case_minimal(self):
        """Test AudioSegment with minimal required data (edge case)."""
        segment = AudioSegment(
            text="",
            start_time=0.0,
            end_time=0.1
        )
        
        assert segment.text == ""
        assert segment.confidence is None
        assert segment.speaker_id is None


class TestTranscriptionResult:
    """Tests for TranscriptionResult model."""
    
    def test_transcription_result_creation_success(self):
        """Test successful TranscriptionResult creation."""
        with NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
        try:
            audio_file = AudioFile(path=tmp_path, duration=10.0)
            
            segments = [
                AudioSegment(text="Hello", start_time=0.0, end_time=1.0),
                AudioSegment(text="world", start_time=1.0, end_time=2.0)
            ]
            
            result = TranscriptionResult(
                audio_file=audio_file,
                segments=segments,
                processing_time=5.0,
                model_info={"name": "test_model"}
            )
            
            assert result.audio_file == audio_file
            assert len(result.segments) == 2
            assert result.full_text == "Hello world"  # Auto-generated
            assert result.processing_time == 5.0
            assert result.model_info["name"] == "test_model"
            assert isinstance(result.timestamp, datetime)
            
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
                
    def test_transcription_result_full_text_generation(self):
        """Test automatic full_text generation from segments (edge case)."""
        with NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
        try:
            audio_file = AudioFile(path=tmp_path)
            
            segments = [
                AudioSegment(text="First segment", start_time=0.0, end_time=1.0),
                AudioSegment(text="Second segment", start_time=1.0, end_time=2.0)
            ]
            
            result = TranscriptionResult(
                audio_file=audio_file,
                segments=segments
            )
            
            assert result.full_text == "First segment Second segment"
            
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
                
    def test_transcription_result_empty_segments(self):
        """Test TranscriptionResult with empty segments (edge case)."""
        with NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
        try:
            audio_file = AudioFile(path=tmp_path)
            
            result = TranscriptionResult(
                audio_file=audio_file,
                segments=[]
            )
            
            assert result.full_text == ""
            assert len(result.segments) == 0
            
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestProcessingConfig:
    """Tests for ProcessingConfig model."""
    
    def test_processing_config_creation_success(self):
        """Test successful ProcessingConfig creation with defaults."""
        config = ProcessingConfig()
        
        assert config.input_dir == Path("./inputs")
        assert config.output_dir == Path("./outputs")
        assert config.cache_dir == Path("./cache")
        assert config.audio["sample_rate"] == 16000
        assert config.transcription["model_name"] == "stt_en_conformer_ctc_large"
        assert config.transcription["device"] == "cuda"
        
    def test_processing_config_custom_values(self):
        """Test ProcessingConfig with custom values."""
        config = ProcessingConfig(
            input_dir=Path("./custom_input"),
            audio={"sample_rate": 22050, "batch_size": 16}
        )
        
        assert config.input_dir == Path("./custom_input")
        assert config.audio["sample_rate"] == 22050
        assert config.audio["batch_size"] == 16
        
    def test_processing_config_directory_creation(self):
        """Test that ProcessingConfig creates directories (edge case)."""
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "test_input"
            output_path = temp_path / "test_output"
            cache_path = temp_path / "test_cache"
            
            config = ProcessingConfig(
                input_dir=input_path,
                output_dir=output_path,
                cache_dir=cache_path
            )
            
            # Directories should be created by the validator
            assert input_path.exists()
            assert output_path.exists()
            assert cache_path.exists()


class TestPipelineStatus:
    """Tests for PipelineStatus model."""
    
    def test_pipeline_status_creation_success(self):
        """Test successful PipelineStatus creation."""
        status = PipelineStatus(
            total_files=10,
            completed_files=7,
            failed_files=2
        )
        
        assert status.total_files == 10
        assert status.completed_files == 7
        assert status.failed_files == 2
        assert isinstance(status.start_time, datetime)
        
    def test_pipeline_status_progress_calculation(self):
        """Test progress percentage calculation."""
        status = PipelineStatus(
            total_files=100,
            completed_files=25,
            failed_files=5
        )
        
        assert status.progress_percentage == 25.0
        
    def test_pipeline_status_completion_check(self):
        """Test completion status check."""
        status = PipelineStatus(
            total_files=10,
            completed_files=8,
            failed_files=2
        )
        
        assert status.is_complete is True
        
    def test_pipeline_status_zero_files_edge_case(self):
        """Test PipelineStatus with zero files (edge case)."""
        status = PipelineStatus(total_files=0)
        
        assert status.progress_percentage == 0.0
        assert status.is_complete is True
        
    def test_pipeline_status_incomplete_failure(self):
        """Test PipelineStatus when not complete (failure case)."""
        status = PipelineStatus(
            total_files=10,
            completed_files=5,
            failed_files=2
        )
        
        assert status.is_complete is False
        assert status.progress_percentage == 50.0