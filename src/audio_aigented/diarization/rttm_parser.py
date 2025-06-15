"""
RTTM (Rich Transcription Time Marked) file parser.

This module handles parsing of RTTM format files produced by speaker diarization
systems, with robust handling of various edge cases.
"""

import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


class RTTMParser:
    """
    Parser for RTTM (Rich Transcription Time Marked) format files.
    
    Handles parsing of speaker diarization output with support for
    filenames containing spaces and various RTTM format variations.
    """

    def __init__(self, collar: float = 0.25) -> None:
        """
        Initialize the RTTM parser.
        
        Args:
            collar: Time collar in seconds for segment boundaries
        """
        self.collar = collar

    def parse_rttm_file(self, rttm_path: Path) -> List[Tuple[float, float, str]]:
        """
        Parse an RTTM file and extract speaker segments.
        
        Args:
            rttm_path: Path to the RTTM file
            
        Returns:
            List of (start_time, end_time, speaker_id) tuples, sorted by start time
            
        Raises:
            FileNotFoundError: If RTTM file doesn't exist
            ValueError: If RTTM file is malformed
        """
        if not rttm_path.exists():
            raise FileNotFoundError(f"RTTM file not found: {rttm_path}")
            
        logger.info(f"Parsing RTTM file: {rttm_path}")
        
        segments = []
        
        try:
            with open(rttm_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            logger.debug(f"RTTM file has {len(lines)} lines")
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                # Log first few lines for debugging
                if line_num <= 3:
                    logger.debug(f"RTTM line {line_num}: {line}")
                    
                segment = self._parse_rttm_line(line, line_num)
                if segment:
                    segments.append(segment)
                    
        except Exception as e:
            logger.error(f"Failed to parse RTTM file: {e}")
            raise ValueError(f"Failed to parse RTTM file: {e}")
            
        # Sort segments by start time
        segments.sort(key=lambda x: x[0])
        
        # Log summary
        if segments:
            unique_speakers = set(speaker for _, _, speaker in segments)
            logger.info(f"Parsed {len(segments)} segments with {len(unique_speakers)} speakers: {unique_speakers}")
        else:
            logger.warning("No valid segments found in RTTM file")
            
        return segments

    def _parse_rttm_line(self, line: str, line_num: int) -> Tuple[float, float, str] | None:
        """
        Parse a single RTTM line.
        
        RTTM format: SPEAKER filename 1 start_time duration <NA> <NA> speaker_id <NA> <NA>
        
        Args:
            line: RTTM line to parse
            line_num: Line number for error reporting
            
        Returns:
            (start_time, end_time, speaker_id) tuple or None if invalid
        """
        if not line.startswith("SPEAKER"):
            return None
            
        try:
            # Split the line into parts
            parts = line.split()
            
            # Find the numeric fields - the "1" channel field helps us locate the right position
            # Look for pattern: ... 1 float float ...
            channel_idx = None
            
            for i, part in enumerate(parts):
                if part == "1" and i > 0 and i < len(parts) - 4:
                    # Check if the next two fields are valid floats
                    try:
                        float(parts[i + 1])  # start_time
                        float(parts[i + 2])  # duration
                        channel_idx = i
                        break
                    except (ValueError, IndexError):
                        continue
                        
            if channel_idx is None:
                logger.warning(f"Could not find valid time fields in RTTM line {line_num}: {line}")
                return None
                
            # Extract fields based on the channel index
            start_time = float(parts[channel_idx + 1])
            duration = float(parts[channel_idx + 2])
            end_time = start_time + duration
            
            # Speaker ID is typically at position channel_idx + 5
            if channel_idx + 5 < len(parts):
                speaker_id = parts[channel_idx + 5]
            else:
                logger.warning(f"Could not find speaker ID in RTTM line {line_num}")
                return None
                
            # Validate values
            if start_time < 0 or duration < 0:
                logger.warning(f"Invalid time values at line {line_num}: start={start_time}, duration={duration}")
                return None
                
            # Apply collar if needed (expand segment boundaries slightly)
            if self.collar > 0:
                start_time = max(0, start_time - self.collar)
                end_time = end_time + self.collar
                
            return (start_time, end_time, speaker_id)
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse RTTM line {line_num}: {line} - Error: {e}")
            return None

    def merge_adjacent_segments(self, segments: List[Tuple[float, float, str]], 
                               max_gap: float = 0.5) -> List[Tuple[float, float, str]]:
        """
        Merge adjacent segments from the same speaker.
        
        Args:
            segments: List of (start, end, speaker_id) tuples
            max_gap: Maximum gap in seconds to merge segments
            
        Returns:
            Merged segments list
        """
        if not segments:
            return segments
            
        # Sort by start time
        segments = sorted(segments, key=lambda x: x[0])
        
        merged = []
        current_start, current_end, current_speaker = segments[0]
        
        for start, end, speaker in segments[1:]:
            # Check if we can merge with current segment
            if speaker == current_speaker and start - current_end <= max_gap:
                # Extend current segment
                current_end = max(current_end, end)
            else:
                # Save current segment and start new one
                merged.append((current_start, current_end, current_speaker))
                current_start, current_end, current_speaker = start, end, speaker
                
        # Don't forget the last segment
        merged.append((current_start, current_end, current_speaker))
        
        logger.debug(f"Merged {len(segments)} segments into {len(merged)} segments")
        return merged

    def filter_short_segments(self, segments: List[Tuple[float, float, str]], 
                             min_duration: float = 0.1) -> List[Tuple[float, float, str]]:
        """
        Filter out segments shorter than minimum duration.
        
        Args:
            segments: List of (start, end, speaker_id) tuples
            min_duration: Minimum segment duration in seconds
            
        Returns:
            Filtered segments list
        """
        filtered = []
        
        for start, end, speaker in segments:
            duration = end - start
            if duration >= min_duration:
                filtered.append((start, end, speaker))
            else:
                logger.debug(f"Filtered out short segment: {speaker} [{start:.2f}-{end:.2f}] ({duration:.2f}s)")
                
        if len(filtered) < len(segments):
            logger.info(f"Filtered {len(segments) - len(filtered)} short segments")
            
        return filtered