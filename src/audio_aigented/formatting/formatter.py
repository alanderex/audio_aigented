"""
Output formatting for transcription results.

This module handles formatting transcription results into structured JSON
and human-readable text formats with proper timestamps and metadata.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List

from ..models.schemas import TranscriptionResult, AudioSegment

logger = logging.getLogger(__name__)


class OutputFormatter:
    """
    Formats transcription results for output in various formats.
    
    Handles conversion to JSON and text formats with configurable
    options for timestamps, confidence scores, and formatting.
    """
    
    def __init__(self, include_timestamps: bool = True, include_confidence: bool = True, pretty_json: bool = True) -> None:
        """
        Initialize the output formatter.
        
        Args:
            include_timestamps: Include timing information in output
            include_confidence: Include confidence scores in output
            pretty_json: Format JSON with indentation
        """
        self.include_timestamps = include_timestamps
        self.include_confidence = include_confidence
        self.pretty_json = pretty_json
        
    def format_as_json(self, result: TranscriptionResult) -> str:
        """
        Format transcription result as JSON.
        
        Args:
            result: TranscriptionResult to format
            
        Returns:
            JSON string representation
        """
        # Create output dictionary
        output = {
            "audio_file": {
                "path": str(result.audio_file.path),
                "duration": result.audio_file.duration,
                "sample_rate": result.audio_file.sample_rate,
                "channels": result.audio_file.channels,
                "format": result.audio_file.format
            },
            "transcription": {
                "full_text": result.full_text,
                "segments": self._format_segments_for_json(result.segments)
            },
            "processing": {
                "processing_time": result.processing_time,
                "timestamp": result.timestamp.isoformat(),
                "model_info": result.model_info
            },
            "metadata": result.metadata
        }
        
        # Format JSON
        if self.pretty_json:
            return json.dumps(output, indent=2, ensure_ascii=False)
        else:
            return json.dumps(output, ensure_ascii=False)
            
    def format_as_text(self, result: TranscriptionResult) -> str:
        """
        Format transcription result as human-readable text.
        
        Args:
            result: TranscriptionResult to format
            
        Returns:
            Formatted text string
        """
        lines = []
        
        # Header
        lines.append("=" * 60)
        lines.append(f"AUDIO TRANSCRIPTION REPORT")
        lines.append("=" * 60)
        lines.append(f"File: {result.audio_file.path.name}")
        lines.append(f"Duration: {result.audio_file.duration:.2f} seconds")
        lines.append(f"Processed: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if result.processing_time:
            speed_ratio = result.metadata.get('processing_speed_ratio', 0)
            lines.append(f"Processing Time: {result.processing_time:.2f}s (Speed: {speed_ratio:.2f}x)")
            
        lines.append("-" * 60)
        
        # Full transcription
        lines.append("FULL TRANSCRIPTION:")
        lines.append("")
        lines.append(result.full_text)
        lines.append("")
        
        # Detailed segments (if timestamps enabled)
        if self.include_timestamps and result.segments:
            lines.append("-" * 60)
            lines.append("DETAILED SEGMENTS:")
            lines.append("")
            
            for i, segment in enumerate(result.segments, 1):
                timestamp_str = f"[{self._format_timestamp(segment.start_time)} - {self._format_timestamp(segment.end_time)}]"
                confidence_str = f" (confidence: {segment.confidence:.2f})" if self.include_confidence and segment.confidence else ""
                
                lines.append(f"{i:2d}. {timestamp_str}{confidence_str}")
                lines.append(f"    {segment.text}")
                lines.append("")
                
        # Statistics
        if result.segments:
            lines.append("-" * 60)
            lines.append("STATISTICS:")
            avg_confidence = result.metadata.get('average_confidence')
            if avg_confidence and self.include_confidence:
                lines.append(f"Average Confidence: {avg_confidence:.2f}")
            lines.append(f"Total Segments: {len(result.segments)}")
            
        lines.append("=" * 60)
        
        return "\n".join(lines)
        
    def _format_segments_for_json(self, segments: List[AudioSegment]) -> List[Dict[str, Any]]:
        """
        Format segments for JSON output.
        
        Args:
            segments: List of AudioSegment instances
            
        Returns:
            List of formatted segment dictionaries
        """
        formatted_segments = []
        
        for segment in segments:
            seg_dict = {
                "text": segment.text
            }
            
            if self.include_timestamps:
                seg_dict.update({
                    "start_time": segment.start_time,
                    "end_time": segment.end_time
                })
                
            if self.include_confidence and segment.confidence is not None:
                seg_dict["confidence"] = segment.confidence
                
            if segment.speaker_id:
                seg_dict["speaker_id"] = segment.speaker_id
                
            formatted_segments.append(seg_dict)
            
        return formatted_segments
        
    def _format_timestamp(self, seconds: float) -> str:
        """
        Format timestamp in MM:SS.mmm format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes:02d}:{remaining_seconds:06.3f}"