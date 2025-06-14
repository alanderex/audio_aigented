"""
Integration tests for pipeline with speaker diarization.

Tests the complete pipeline flow with diarization enabled.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.audio_aigented.pipeline import TranscriptionPipeline
from src.audio_aigented.models.schemas import ProcessingConfig, AudioSegment


class TestPipelineDiarization:
    """Test pipeline integration with speaker diarization."""
    
    @pytest.fixture
    def config_with_diarization(self):
        """Create config with diarization enabled."""
        return ProcessingConfig(
            processing={
                "enable_diarization": True,
                "log_level": "DEBUG"
            }
        )
    
    @pytest.fixture
    def config_without_diarization(self):
        """Create config with diarization disabled."""
        return ProcessingConfig(
            processing={
                "enable_diarization": False,
                "log_level": "DEBUG"
            }
        )
    
    @patch('src.audio_aigented.diarization.diarizer.ClusteringDiarizer')
    @patch('pathlib.Path.exists')
    def test_pipeline_initialization_with_diarization(self, mock_exists, mock_clustering_diarizer, config_with_diarization):
        """Test pipeline initialization with diarization enabled."""
        # Arrange
        mock_exists.return_value = True
        mock_diarizer_instance = Mock()
        mock_clustering_diarizer.return_value = mock_diarizer_instance
        
        # Act
        pipeline = TranscriptionPipeline(config=config_with_diarization)
        
        # Assert
        assert pipeline.diarizer is not None
        assert pipeline.config.processing["enable_diarization"] is True
    
    def test_pipeline_initialization_without_diarization(self, config_without_diarization):
        """Test pipeline initialization with diarization disabled."""
        # Act
        pipeline = TranscriptionPipeline(config=config_without_diarization)
        
        # Assert
        assert pipeline.diarizer is None
        assert pipeline.config.processing["enable_diarization"] is False
    
    @patch('src.audio_aigented.diarization.diarizer.ClusteringDiarizer')
    @patch('pathlib.Path.exists')
    def test_pipeline_initialization_diarization_failure(self, mock_exists, mock_clustering_diarizer, config_with_diarization):
        """Test pipeline initialization when diarization fails to initialize."""
        # Arrange
        mock_exists.return_value = True
        mock_clustering_diarizer.side_effect = Exception("Failed to load model")
        
        # Act
        pipeline = TranscriptionPipeline(config=config_with_diarization)
        
        # Assert
        assert pipeline.diarizer is None
        assert pipeline.config.processing["enable_diarization"] is False
    
    def test_assign_speakers_to_segments(self, config_with_diarization):
        """Test speaker assignment to transcription segments."""
        # Arrange
        pipeline = TranscriptionPipeline(config=config_with_diarization)
        
        # Create mock transcription segments
        segments = [
            AudioSegment(text="Hello", start_time=0.0, end_time=2.0),
            AudioSegment(text="world", start_time=3.0, end_time=5.0),
            AudioSegment(text="how are you", start_time=6.0, end_time=8.0)
        ]
        
        # Create mock speaker segments
        speaker_segments = [
            (0.0, 4.0, "SPEAKER_00"),  # Covers first two segments
            (5.0, 10.0, "SPEAKER_01")  # Covers third segment
        ]
        
        # Act
        pipeline._assign_speakers_to_segments(segments, speaker_segments)
        
        # Assert
        assert segments[0].speaker_id == "SPEAKER_00"  # Full overlap
        assert segments[1].speaker_id == "SPEAKER_00"  # Partial overlap with SPEAKER_00
        assert segments[2].speaker_id == "SPEAKER_01"  # Partial overlap with SPEAKER_01
    
    def test_assign_speakers_no_overlap(self, config_with_diarization):
        """Test speaker assignment when there's no temporal overlap."""
        # Arrange
        pipeline = TranscriptionPipeline(config=config_with_diarization)
        
        # Create mock transcription segments
        segments = [
            AudioSegment(text="Hello", start_time=10.0, end_time=12.0),
        ]
        
        # Create mock speaker segments (no overlap)
        speaker_segments = [
            (0.0, 5.0, "SPEAKER_00"),
        ]
        
        # Act
        pipeline._assign_speakers_to_segments(segments, speaker_segments)
        
        # Assert
        assert segments[0].speaker_id is None  # No overlap, no assignment
    
    def test_assign_speakers_multiple_overlaps(self, config_with_diarization):
        """Test speaker assignment when segment overlaps with multiple speakers."""
        # Arrange
        pipeline = TranscriptionPipeline(config=config_with_diarization)
        
        # Create mock transcription segments
        segments = [
            AudioSegment(text="Hello world", start_time=2.0, end_time=6.0),  # Overlaps both speakers
        ]
        
        # Create mock speaker segments
        speaker_segments = [
            (0.0, 4.0, "SPEAKER_00"),   # 2 seconds overlap (2.0-4.0)
            (3.0, 8.0, "SPEAKER_01")    # 3 seconds overlap (3.0-6.0) - should win
        ]
        
        # Act
        pipeline._assign_speakers_to_segments(segments, speaker_segments)
        
        # Assert
        # Should assign to SPEAKER_01 due to longer overlap (3s vs 2s)
        assert segments[0].speaker_id == "SPEAKER_01"


if __name__ == "__main__":
    pytest.main([__file__])