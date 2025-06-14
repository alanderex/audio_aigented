"""
Unit tests for attributed text file output functionality.

Tests the file writing capabilities for the attributed text format.
"""

import pytest
from datetime import datetime
from pathlib import Path

from src.audio_aigented.output.writer import FileWriter
from src.audio_aigented.models.schemas import ProcessingConfig, TranscriptionResult, AudioFile, AudioSegment


class TestAttributedFileOutput:
    """Test attributed text file output functionality."""
    
    @pytest.fixture
    def config(self, tmp_path):
        """Create test configuration."""
        return ProcessingConfig(
            input_dir=tmp_path / "inputs",
            output_dir=tmp_path / "outputs",
            cache_dir=tmp_path / "cache",
            output={
                "formats": ["attributed_txt"],
                "include_timestamps": True,
                "include_confidence": True,
                "pretty_json": True,
            }
        )
    
    @pytest.fixture
    def file_writer(self, config):
        """Create FileWriter instance for testing."""
        return FileWriter(config)
    
    @pytest.fixture
    def sample_audio_file(self, tmp_path):
        """Create a sample audio file for testing."""
        audio_path = tmp_path / "inputs" / "test_audio.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.touch()  # Create empty file
        
        return AudioFile(
            path=audio_path,
            sample_rate=16000,
            duration=6.0,
            channels=1,
            format="wav"
        )
    
    @pytest.fixture
    def sample_transcription_result(self, sample_audio_file):
        """Create sample transcription result with speakers."""
        segments = [
            AudioSegment(
                text="Hello there!",
                start_time=0.0,
                end_time=1.5,
                confidence=0.95,
                speaker_id="SPEAKER_00"
            ),
            AudioSegment(
                text="Hi, how are you?",
                start_time=1.5,
                end_time=3.0,
                confidence=0.92,
                speaker_id="SPEAKER_01"
            ),
            AudioSegment(
                text="I'm doing well, thanks!",
                start_time=3.0,
                end_time=6.0,
                confidence=0.89,
                speaker_id="SPEAKER_00"
            )
        ]
        
        return TranscriptionResult(
            audio_file=sample_audio_file,
            segments=segments,
            full_text="Hello there! Hi, how are you? I'm doing well, thanks!",
            processing_time=1.2,
            timestamp=datetime.now()
        )
    
    def test_write_attributed_text_output(self, file_writer, sample_transcription_result):
        """Test writing attributed text output to file."""
        # Act
        created_files = file_writer.write_transcription_result(sample_transcription_result)
        
        # Assert
        assert len(created_files) == 1
        
        attributed_file = created_files[0]
        assert attributed_file.name == "transcript_attributed.txt"
        assert attributed_file.exists()
        
        # Check file content
        content = attributed_file.read_text(encoding='utf-8')
        expected_lines = [
            "SPEAKER_00: Hello there!",
            "SPEAKER_01: Hi, how are you?",
            "SPEAKER_00: I'm doing well, thanks!"
        ]
        expected_content = "\n".join(expected_lines)
        
        assert content == expected_content
        
    def test_write_attributed_text_with_multiple_formats(self, tmp_path):
        """Test writing attributed text along with other formats."""
        # Arrange
        config = ProcessingConfig(
            output_dir=tmp_path / "outputs",
            output={
                "formats": ["json", "txt", "attributed_txt"],
                "include_timestamps": True,
                "include_confidence": True,
                "pretty_json": True,
            }
        )
        
        file_writer = FileWriter(config)
        
        # Create sample audio file
        audio_path = tmp_path / "test_audio.wav"
        audio_path.touch()
        
        audio_file = AudioFile(path=audio_path, sample_rate=16000, duration=3.0)
        
        segments = [
            AudioSegment(
                text="Test message.",
                start_time=0.0,
                end_time=3.0,
                confidence=0.9,
                speaker_id="SPEAKER_00"
            )
        ]
        
        result = TranscriptionResult(
            audio_file=audio_file,
            segments=segments,
            full_text="Test message.",
            processing_time=0.5,
            timestamp=datetime.now()
        )
        
        # Act
        created_files = file_writer.write_transcription_result(result)
        
        # Assert
        assert len(created_files) == 3
        
        file_names = [f.name for f in created_files]
        assert "transcript.json" in file_names
        assert "transcript.txt" in file_names
        assert "transcript_attributed.txt" in file_names
        
        # Check attributed file content
        attributed_file = next(f for f in created_files if f.name == "transcript_attributed.txt")
        content = attributed_file.read_text(encoding='utf-8')
        assert content == "SPEAKER_00: Test message."
        
    def test_write_attributed_text_empty_result(self, file_writer, sample_audio_file):
        """Test writing attributed text with empty transcription result."""
        # Arrange
        empty_result = TranscriptionResult(
            audio_file=sample_audio_file,
            segments=[],
            full_text="Empty transcription result.",
            processing_time=0.1,
            timestamp=datetime.now()
        )
        
        # Act
        created_files = file_writer.write_transcription_result(empty_result)
        
        # Assert
        assert len(created_files) == 1
        
        attributed_file = created_files[0]
        content = attributed_file.read_text(encoding='utf-8')
        assert content == "UNKNOWN_SPEAKER: Empty transcription result."


if __name__ == "__main__":
    pytest.main([__file__])