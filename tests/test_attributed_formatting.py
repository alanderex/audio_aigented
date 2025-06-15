"""
Unit tests for attributed text formatting functionality.

Tests the theater play style speaker attribution format.
"""

import pytest
from datetime import datetime

from src.audio_aigented.formatting.formatter import OutputFormatter
from src.audio_aigented.models.schemas import TranscriptionResult, AudioFile, AudioSegment


class TestAttributedTextFormatting:
    """Test attributed text formatting functionality."""
    
    @pytest.fixture
    def formatter(self):
        """Create OutputFormatter instance for testing."""
        return OutputFormatter(
            include_timestamps=True,
            include_confidence=True,
            pretty_json=True
        )
    
    @pytest.fixture
    def sample_audio_file(self, tmp_path):
        """Create a sample audio file for testing."""
        audio_path = tmp_path / "test_audio.wav"
        audio_path.touch()  # Create empty file
        
        return AudioFile(
            path=audio_path,
            sample_rate=16000,
            duration=10.5,
            channels=1,
            format="wav"
        )
    
    @pytest.fixture
    def sample_segments_with_speakers(self):
        """Create sample segments with speaker information."""
        return [
            AudioSegment(
                text="Hello, how are you today?",
                start_time=0.0,
                end_time=2.5,
                confidence=0.95,
                speaker_id="SPEAKER_00"
            ),
            AudioSegment(
                text="I'm doing great, thank you!",
                start_time=2.5,
                end_time=5.0,
                confidence=0.92,
                speaker_id="SPEAKER_01"
            ),
            AudioSegment(
                text="That's wonderful to hear.",
                start_time=5.0,
                end_time=7.5,
                confidence=0.88,
                speaker_id="SPEAKER_00"
            ),
            AudioSegment(
                text="How about you?",
                start_time=7.5,
                end_time=8.5,
                confidence=0.91,
                speaker_id="SPEAKER_00"
            ),
            AudioSegment(
                text="I'm having a fantastic day!",
                start_time=8.5,
                end_time=10.5,
                confidence=0.94,
                speaker_id="SPEAKER_01"
            )
        ]
    
    def test_format_as_attributed_text_with_speakers(self, formatter, sample_audio_file, sample_segments_with_speakers):
        """Test attributed text formatting with speaker segments."""
        # Arrange
        result = TranscriptionResult(
            audio_file=sample_audio_file,
            segments=sample_segments_with_speakers,
            full_text="Hello, how are you today? I'm doing great, thank you! That's wonderful to hear. How about you? I'm having a fantastic day!",
            processing_time=1.2,
            timestamp=datetime.now()
        )
        
        # Act
        attributed_text = formatter.format_as_attributed_text(result)
        
        # Assert
        expected_lines = [
            "SPEAKER_00: Hello, how are you today?",
            "SPEAKER_01: I'm doing great, thank you!",
            "SPEAKER_00: That's wonderful to hear. How about you?",
            "SPEAKER_01: I'm having a fantastic day!"
        ]
        expected_output = "\n".join(expected_lines)
        
        assert attributed_text == expected_output
        assert "SPEAKER_00:" in attributed_text
        assert "SPEAKER_01:" in attributed_text
        
    def test_format_as_attributed_text_without_speakers(self, formatter, sample_audio_file):
        """Test attributed text formatting without speaker information."""
        # Arrange
        segments = [
            AudioSegment(
                text="This is some text without speaker information.",
                start_time=0.0,
                end_time=3.0,
                confidence=0.95,
                speaker_id=None
            )
        ]
        
        result = TranscriptionResult(
            audio_file=sample_audio_file,
            segments=segments,
            full_text="This is some text without speaker information.",
            processing_time=0.8,
            timestamp=datetime.now()
        )
        
        # Act
        attributed_text = formatter.format_as_attributed_text(result)
        
        # Assert
        assert attributed_text == "UNKNOWN_SPEAKER: This is some text without speaker information."
        
    def test_format_as_attributed_text_no_segments(self, formatter, sample_audio_file):
        """Test attributed text formatting with no segments (fallback to full text)."""
        # Arrange
        result = TranscriptionResult(
            audio_file=sample_audio_file,
            segments=[],
            full_text="This is the full transcription text.",
            processing_time=0.5,
            timestamp=datetime.now()
        )
        
        # Act
        attributed_text = formatter.format_as_attributed_text(result)
        
        # Assert
        assert attributed_text == "UNKNOWN_SPEAKER: This is the full transcription text."
        
    def test_format_as_attributed_text_empty_segments(self, formatter, sample_audio_file):
        """Test attributed text formatting with empty text segments."""
        # Arrange
        segments = [
            AudioSegment(
                text="",
                start_time=0.0,
                end_time=1.0,
                confidence=0.5,
                speaker_id="SPEAKER_00"
            ),
            AudioSegment(
                text="Valid text here.",
                start_time=1.0,
                end_time=2.0,
                confidence=0.9,
                speaker_id="SPEAKER_01"
            )
        ]
        
        result = TranscriptionResult(
            audio_file=sample_audio_file,
            segments=segments,
            full_text="Valid text here.",
            processing_time=0.3,
            timestamp=datetime.now()
        )
        
        # Act
        attributed_text = formatter.format_as_attributed_text(result)
        
        # Assert
        assert attributed_text == "SPEAKER_01: Valid text here."
        assert "SPEAKER_00:" not in attributed_text  # Empty text should be filtered out
        
    def test_format_as_attributed_text_single_speaker(self, formatter, sample_audio_file):
        """Test attributed text formatting with single speaker multiple segments."""
        # Arrange
        segments = [
            AudioSegment(
                text="First sentence.",
                start_time=0.0,
                end_time=2.0,
                confidence=0.95,
                speaker_id="SPEAKER_00"
            ),
            AudioSegment(
                text="Second sentence.",
                start_time=2.0,
                end_time=4.0,
                confidence=0.92,
                speaker_id="SPEAKER_00"
            ),
            AudioSegment(
                text="Third sentence.",
                start_time=4.0,
                end_time=6.0,
                confidence=0.88,
                speaker_id="SPEAKER_00"
            )
        ]
        
        result = TranscriptionResult(
            audio_file=sample_audio_file,
            segments=segments,
            full_text="First sentence. Second sentence. Third sentence.",
            processing_time=1.0,
            timestamp=datetime.now()
        )
        
        # Act
        attributed_text = formatter.format_as_attributed_text(result)
        
        # Assert
        assert attributed_text == "SPEAKER_00: First sentence. Second sentence. Third sentence."
        
    def test_format_as_attributed_text_alternating_speakers(self, formatter, sample_audio_file):
        """Test attributed text formatting with alternating speakers."""
        # Arrange
        segments = [
            AudioSegment(text="Question?", start_time=0.0, end_time=1.0, speaker_id="SPEAKER_A"),
            AudioSegment(text="Answer.", start_time=1.0, end_time=2.0, speaker_id="SPEAKER_B"),
            AudioSegment(text="Follow up?", start_time=2.0, end_time=3.0, speaker_id="SPEAKER_A"),
            AudioSegment(text="Response.", start_time=3.0, end_time=4.0, speaker_id="SPEAKER_B"),
        ]
        
        result = TranscriptionResult(
            audio_file=sample_audio_file,
            segments=segments,
            full_text="Question? Answer. Follow up? Response.",
            processing_time=0.8,
            timestamp=datetime.now()
        )
        
        # Act
        attributed_text = formatter.format_as_attributed_text(result)
        
        # Assert
        expected_lines = [
            "SPEAKER_A: Question?",
            "SPEAKER_B: Answer.",
            "SPEAKER_A: Follow up?",
            "SPEAKER_B: Response."
        ]
        expected_output = "\n".join(expected_lines)
        
        assert attributed_text == expected_output


if __name__ == "__main__":
    pytest.main([__file__])