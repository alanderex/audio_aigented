"""
Unit tests for audio loading and preprocessing.

Tests the AudioLoader class for file discovery, loading, validation, and preprocessing.
"""

import pytest
import numpy as np
import tempfile
import wave
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.audio_aigented.audio.loader import AudioLoader
from src.audio_aigented.models.schemas import ProcessingConfig, AudioFile


def create_mock_wav_file(path: Path, duration: float = 1.0, sample_rate: int = 16000) -> None:
    """
    Create a mock WAV file for testing.
    
    Args:
        path: Path where to create the file
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
    """
    # Generate simple sine wave
    frames = int(duration * sample_rate)
    audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, frames))  # 440 Hz tone
    
    # Convert to 16-bit integers
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Write WAV file
    with wave.open(str(path), 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())


class TestAudioLoader:
    """Tests for AudioLoader class."""
    
    def test_audio_loader_initialization_success(self):
        """Test successful AudioLoader initialization."""
        config = ProcessingConfig()
        loader = AudioLoader(config)
        
        assert loader.config == config
        assert loader.target_sample_rate == 16000
        assert loader.max_duration == 30.0
        
    def test_discover_audio_files_success(self):
        """Test successful audio file discovery."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test WAV files
            wav1 = temp_path / "test1.wav"
            wav2 = temp_path / "test2.wav"
            txt_file = temp_path / "not_audio.txt"
            
            create_mock_wav_file(wav1)
            create_mock_wav_file(wav2)
            txt_file.write_text("not an audio file")
            
            config = ProcessingConfig(input_dir=temp_path)
            loader = AudioLoader(config)
            
            discovered_files = loader.discover_audio_files()
            
            # Should find only WAV files
            assert len(discovered_files) == 2
            assert wav1 in discovered_files
            assert wav2 in discovered_files
            assert txt_file not in discovered_files
            
    def test_discover_audio_files_empty_directory(self):
        """Test audio file discovery in empty directory (edge case)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            config = ProcessingConfig(input_dir=temp_path)
            loader = AudioLoader(config)
            
            discovered_files = loader.discover_audio_files()
            
            assert len(discovered_files) == 0
            
    def test_discover_audio_files_nonexistent_directory_failure(self):
        """Test audio file discovery with nonexistent directory (failure case)."""
        nonexistent_path = Path("/nonexistent/directory")
        
        config = ProcessingConfig(input_dir=nonexistent_path)
        loader = AudioLoader(config)
        
        discovered_files = loader.discover_audio_files()
        
        # Should return empty list, not crash
        assert len(discovered_files) == 0
        
    @patch('soundfile.info')
    def test_load_audio_file_success(self, mock_sf_info):
        """Test successful audio file loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file = temp_path / "test.wav"
            create_mock_wav_file(audio_file, duration=5.0)
            
            # Mock soundfile.info response
            mock_info = MagicMock()
            mock_info.samplerate = 16000
            mock_info.duration = 5.0
            mock_info.channels = 1
            mock_info.format = 'WAV'
            mock_sf_info.return_value = mock_info
            
            config = ProcessingConfig()
            loader = AudioLoader(config)
            
            audio_file_obj = loader.load_audio_file(audio_file)
            
            assert isinstance(audio_file_obj, AudioFile)
            assert audio_file_obj.path == audio_file
            assert audio_file_obj.sample_rate == 16000
            assert audio_file_obj.duration == 5.0
            assert audio_file_obj.channels == 1
            assert audio_file_obj.format == 'wav'
            
    def test_load_audio_file_nonexistent_failure(self):
        """Test loading nonexistent audio file (failure case)."""
        nonexistent_file = Path("/nonexistent/file.wav")
        
        config = ProcessingConfig()
        loader = AudioLoader(config)
        
        with pytest.raises(FileNotFoundError):
            loader.load_audio_file(nonexistent_file)
            
    @patch('soundfile.info')  
    def test_load_audio_file_invalid_format_failure(self, mock_sf_info):
        """Test loading invalid audio file format (failure case)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            invalid_file = temp_path / "invalid.wav"
            invalid_file.write_text("This is not a valid audio file")
            
            # Mock soundfile.info to raise exception
            mock_sf_info.side_effect = Exception("Invalid format")
            
            config = ProcessingConfig()
            loader = AudioLoader(config)
            
            with pytest.raises(ValueError, match="Unsupported audio file format"):
                loader.load_audio_file(invalid_file)
                
    @patch('librosa.load')
    def test_load_audio_data_success(self, mock_librosa_load):
        """Test successful audio data loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file_path = temp_path / "test.wav"
            create_mock_wav_file(audio_file_path)
            
            audio_file = AudioFile(path=audio_file_path, sample_rate=16000, duration=1.0)
            
            # Mock librosa.load response
            mock_audio_data = np.random.randn(16000).astype(np.float32)
            mock_librosa_load.return_value = (mock_audio_data, 16000)
            
            config = ProcessingConfig()
            loader = AudioLoader(config)
            
            audio_data, sample_rate = loader.load_audio_data(audio_file)
            
            assert isinstance(audio_data, np.ndarray)
            assert sample_rate == 16000
            assert len(audio_data) == 16000
            
    @patch('librosa.load')
    @patch('librosa.resample')
    def test_load_audio_data_with_resampling(self, mock_resample, mock_librosa_load):
        """Test audio data loading with resampling (edge case)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file_path = temp_path / "test.wav"
            create_mock_wav_file(audio_file_path)
            
            audio_file = AudioFile(path=audio_file_path, sample_rate=44100, duration=1.0)
            
            # Mock librosa.load response (44.1kHz)
            mock_audio_data = np.random.randn(44100).astype(np.float32)
            mock_librosa_load.return_value = (mock_audio_data, 44100)
            
            # Mock librosa.resample response (16kHz)
            resampled_data = np.random.randn(16000).astype(np.float32)
            mock_resample.return_value = resampled_data
            
            config = ProcessingConfig()
            loader = AudioLoader(config)
            
            audio_data, sample_rate = loader.load_audio_data(audio_file)
            
            # Should have called resample
            mock_resample.assert_called_once()
            assert sample_rate == 16000
            assert len(audio_data) == 16000
            
    def test_segment_audio_short_file(self):
        """Test audio segmentation with short file (edge case)."""
        config = ProcessingConfig()
        loader = AudioLoader(config)
        
        # Create short audio (less than max_duration)
        short_audio = np.random.randn(8000).astype(np.float32)  # 0.5 seconds at 16kHz
        
        segments = loader.segment_audio(short_audio, 16000)
        
        # Should return single segment
        assert len(segments) == 1
        assert len(segments[0]) == 8000
        
    def test_segment_audio_long_file(self):
        """Test audio segmentation with long file."""
        config = ProcessingConfig()
        loader = AudioLoader(config)
        
        # Create long audio (more than max_duration of 30s)
        long_audio = np.random.randn(800000).astype(np.float32)  # 50 seconds at 16kHz
        
        segments = loader.segment_audio(long_audio, 16000)
        
        # Should return multiple segments
        assert len(segments) > 1
        
        # Each segment should be approximately max_duration or less
        for segment in segments:
            segment_duration = len(segment) / 16000
            assert segment_duration <= 30.1  # Allow small tolerance
            
    def test_process_audio_files_success(self):
        """Test processing multiple audio files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            files = []
            for i in range(3):
                file_path = temp_path / f"test{i}.wav"
                create_mock_wav_file(file_path)
                files.append(file_path)
                
            config = ProcessingConfig()
            loader = AudioLoader(config)
            
            with patch('soundfile.info') as mock_sf_info:
                mock_info = MagicMock()
                mock_info.samplerate = 16000
                mock_info.duration = 1.0
                mock_info.channels = 1
                mock_info.format = 'WAV'
                mock_sf_info.return_value = mock_info
                
                processed_files = loader.process_audio_files(files)
                
                assert len(processed_files) == 3
                for audio_file in processed_files:
                    assert isinstance(audio_file, AudioFile)
                    
    def test_process_audio_files_with_failures(self):
        """Test processing audio files with some failures (failure case)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mix of valid and invalid files
            valid_file = temp_path / "valid.wav"
            invalid_file = temp_path / "invalid.wav"
            
            create_mock_wav_file(valid_file)
            invalid_file.write_text("not audio")
            
            files = [valid_file, invalid_file]
            
            config = ProcessingConfig()
            loader = AudioLoader(config)
            
            with patch('soundfile.info') as mock_sf_info:
                def side_effect(path):
                    if 'valid' in str(path):
                        mock_info = MagicMock()
                        mock_info.samplerate = 16000
                        mock_info.duration = 1.0
                        mock_info.channels = 1
                        mock_info.format = 'WAV'
                        return mock_info
                    else:
                        raise Exception("Invalid format")
                        
                mock_sf_info.side_effect = side_effect
                
                processed_files = loader.process_audio_files(files)
                
                # Should only process valid file
                assert len(processed_files) == 1
                assert processed_files[0].path == valid_file
                
    def test_validate_audio_file_success(self):
        """Test successful audio file validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file_path = temp_path / "test.wav"
            create_mock_wav_file(audio_file_path)
            
            audio_file = AudioFile(
                path=audio_file_path,
                sample_rate=16000,
                duration=5.0,
                channels=1
            )
            
            config = ProcessingConfig()
            loader = AudioLoader(config)
            
            is_valid = loader.validate_audio_file(audio_file)
            
            assert is_valid is True
            
    def test_validate_audio_file_invalid_duration_failure(self):
        """Test audio file validation with invalid duration (failure case)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file_path = temp_path / "test.wav"
            create_mock_wav_file(audio_file_path)
            
            audio_file = AudioFile(
                path=audio_file_path,
                sample_rate=16000,
                duration=0.0,  # Invalid duration
                channels=1
            )
            
            config = ProcessingConfig()
            loader = AudioLoader(config)
            
            is_valid = loader.validate_audio_file(audio_file)
            
            assert is_valid is False
            
    def test_validate_audio_file_very_short_warning(self):
        """Test audio file validation with very short duration (edge case)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file_path = temp_path / "test.wav"
            create_mock_wav_file(audio_file_path, duration=0.05)  # 50ms
            
            audio_file = AudioFile(
                path=audio_file_path,
                sample_rate=16000,
                duration=0.05,
                channels=1
            )
            
            config = ProcessingConfig()
            loader = AudioLoader(config)
            
            is_valid = loader.validate_audio_file(audio_file)
            
            # Should still be valid but log warning
            assert is_valid is True
            
    def test_normalize_audio_success(self):
        """Test audio normalization."""
        config = ProcessingConfig()
        loader = AudioLoader(config)
        
        # Create audio data with values outside [-1, 1]
        audio_data = np.array([0.5, 1.5, -2.0, 0.0, 1.0], dtype=np.float32)
        
        normalized = loader._normalize_audio(audio_data)
        
        # Should be normalized to [-1, 1] range
        assert np.max(np.abs(normalized)) <= 1.0
        assert np.min(normalized) >= -1.0
        assert np.max(normalized) <= 1.0
        
    def test_normalize_audio_zero_amplitude_edge_case(self):
        """Test audio normalization with zero amplitude (edge case)."""
        config = ProcessingConfig()
        loader = AudioLoader(config)
        
        # Create silent audio
        silent_audio = np.zeros(1000, dtype=np.float32)
        
        normalized = loader._normalize_audio(silent_audio)
        
        # Should remain zero
        assert np.all(normalized == 0.0)