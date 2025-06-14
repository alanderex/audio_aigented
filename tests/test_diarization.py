"""
Unit tests for speaker diarization functionality.

Tests the NeMo-based speaker diarization implementation.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from src.audio_aigented.diarization.diarizer import NeMoDiarizer
from src.audio_aigented.models.schemas import AudioFile


class TestNeMoDiarizer:
    """Test NeMo speaker diarization functionality."""
    
    @pytest.fixture
    def sample_audio_file(self, tmp_path):
        """Create a sample audio file for testing."""
        audio_path = tmp_path / "test_audio.wav"
        audio_path.touch()  # Create empty file
        
        return AudioFile(
            path=audio_path,
            sample_rate=16000,
            duration=10.0,
            channels=1,
            format="wav"
        )
    
    @patch('src.audio_aigented.diarization.diarizer.OmegaConf.load')
    @patch('pathlib.Path.exists')
    def test_diarizer_initialization(self, mock_exists, mock_omega_load):
        """Test NeMo diarizer initialization."""
        # Arrange
        mock_exists.return_value = True  # Mock config file exists
        mock_config = Mock()
        mock_omega_load.return_value = mock_config
        
        # Act
        diarizer = NeMoDiarizer()
        
        # Assert
        assert diarizer.device in ["cuda", "cpu"]
        assert hasattr(diarizer, 'base_cfg')
        assert diarizer.base_cfg == mock_config
        mock_omega_load.assert_called_once()
    
    @patch('src.audio_aigented.diarization.diarizer.ClusteringDiarizer')
    @patch('src.audio_aigented.diarization.diarizer.OmegaConf.load')
    @patch('src.audio_aigented.diarization.diarizer.OmegaConf.create')
    @patch('pathlib.Path.exists')
    @patch('tempfile.TemporaryDirectory')
    def test_diarize_success(self, mock_tempdir, mock_exists, mock_omega_create, mock_omega_load, mock_clustering_diarizer, sample_audio_file):
        """Test successful speaker diarization."""
        # Arrange
        mock_exists.return_value = True  # Mock config file exists
        mock_tempdir.return_value.__enter__.return_value = "/tmp/test_dir"
        
        # Mock configuration loading
        mock_base_config = Mock()
        mock_omega_load.return_value = mock_base_config
        mock_runtime_config = Mock()
        mock_omega_create.return_value = mock_runtime_config
        
        mock_diarizer_instance = Mock()
        mock_clustering_diarizer.return_value = mock_diarizer_instance
        
        # Mock speaker segments
        mock_segment_1 = Mock()
        mock_segment_1.start = 0.0
        mock_segment_1.end = 5.0
        
        mock_segment_2 = Mock()
        mock_segment_2.start = 5.0
        mock_segment_2.end = 10.0
        
        mock_diarizer_instance.speaker_segs = {
            "SPEAKER_00": [mock_segment_1],
            "SPEAKER_01": [mock_segment_2]
        }
        
        diarizer = NeMoDiarizer()
        
        # Act
        speaker_segments = diarizer.diarize(sample_audio_file)
        
        # Assert
        assert len(speaker_segments) == 2
        # Results should be sorted by start time
        assert speaker_segments[0] == (0.0, 5.0, "SPEAKER_00")
        assert speaker_segments[1] == (5.0, 10.0, "SPEAKER_01")
        
        # Verify diarizer was called correctly
        mock_diarizer_instance.diarize.assert_called_once_with(paths2audio_files=[str(sample_audio_file.path)])
        mock_clustering_diarizer.assert_called_once()
    
    @patch('src.audio_aigented.diarization.diarizer.ClusteringDiarizer')
    @patch('src.audio_aigented.diarization.diarizer.OmegaConf.load')
    @patch('src.audio_aigented.diarization.diarizer.OmegaConf.create')
    @patch('pathlib.Path.exists')
    @patch('tempfile.TemporaryDirectory')
    def test_diarize_failure(self, mock_tempdir, mock_exists, mock_omega_create, mock_omega_load, mock_clustering_diarizer, sample_audio_file):
        """Test diarization failure handling."""
        # Arrange
        mock_exists.return_value = True  # Mock config file exists
        mock_tempdir.return_value.__enter__.return_value = "/tmp/test_dir"
        
        # Mock configuration loading
        mock_base_config = Mock()
        mock_omega_load.return_value = mock_base_config
        mock_runtime_config = Mock()
        mock_omega_create.return_value = mock_runtime_config
        
        mock_diarizer_instance = Mock()
        mock_clustering_diarizer.return_value = mock_diarizer_instance
        mock_diarizer_instance.diarize.side_effect = Exception("Diarization failed")
        
        diarizer = NeMoDiarizer()
        
        # Act
        speaker_segments = diarizer.diarize(sample_audio_file)
        
        # Assert
        assert speaker_segments == []
    
    @patch('src.audio_aigented.diarization.diarizer.ClusteringDiarizer')
    @patch('src.audio_aigented.diarization.diarizer.OmegaConf.load')
    @patch('src.audio_aigented.diarization.diarizer.OmegaConf.create')
    @patch('pathlib.Path.exists')
    @patch('tempfile.TemporaryDirectory')
    def test_diarize_empty_results(self, mock_tempdir, mock_exists, mock_omega_create, mock_omega_load, mock_clustering_diarizer, sample_audio_file):
        """Test diarization with no speaker segments found."""
        # Arrange
        mock_exists.return_value = True  # Mock config file exists
        mock_tempdir.return_value.__enter__.return_value = "/tmp/test_dir"
        
        # Mock configuration loading
        mock_base_config = Mock()
        mock_omega_load.return_value = mock_base_config
        mock_runtime_config = Mock()
        mock_omega_create.return_value = mock_runtime_config
        
        mock_diarizer_instance = Mock()
        mock_clustering_diarizer.return_value = mock_diarizer_instance
        mock_diarizer_instance.speaker_segs = {}
        
        diarizer = NeMoDiarizer()
        
        # Act
        speaker_segments = diarizer.diarize(sample_audio_file)
        
        # Assert
        assert speaker_segments == []
    
    @patch('pathlib.Path.exists')
    def test_diarizer_config_file_not_found(self, mock_exists):
        """Test diarizer initialization with missing config file."""
        # Arrange
        mock_exists.return_value = False  # Mock config file doesn't exist
        
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            NeMoDiarizer()
    
    def test_get_speaker_mapping(self):
        """Test speaker mapping functionality."""
        # Create a minimal diarizer instance without full initialization
        with patch('pathlib.Path.exists', return_value=True), \
             patch('src.audio_aigented.diarization.diarizer.ClusteringDiarizer'):
            diarizer = NeMoDiarizer()
        
        # Test data
        speaker_segments = [
            (0.0, 5.0, "SPEAKER_00"),
            (5.0, 10.0, "SPEAKER_01"),
            (10.0, 15.0, "SPEAKER_00")
        ]
        
        speaker_mapping = {
            "SPEAKER_00": "Alice",
            "SPEAKER_01": "Bob"
        }
        
        # Act
        mapped_segments = diarizer.get_speaker_mapping(speaker_segments, speaker_mapping)
        
        # Assert
        expected = [
            (0.0, 5.0, "Alice"),
            (5.0, 10.0, "Bob"),
            (10.0, 15.0, "Alice")
        ]
        assert mapped_segments == expected
    
    def test_get_speaker_mapping_no_mapping(self):
        """Test speaker mapping with no mapping provided."""
        # Create a minimal diarizer instance without full initialization
        with patch('pathlib.Path.exists', return_value=True), \
             patch('src.audio_aigented.diarization.diarizer.ClusteringDiarizer'):
            diarizer = NeMoDiarizer()
        
        # Test data
        speaker_segments = [
            (0.0, 5.0, "SPEAKER_00"),
            (5.0, 10.0, "SPEAKER_01")
        ]
        
        # Act
        mapped_segments = diarizer.get_speaker_mapping(speaker_segments, None)
        
        # Assert
        assert mapped_segments == speaker_segments


if __name__ == "__main__":
    pytest.main([__file__])