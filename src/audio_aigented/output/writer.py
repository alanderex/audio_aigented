"""
File output and writing module.

This module handles writing transcription results to disk in various formats
with proper directory structure and file naming conventions.
"""

import logging
from pathlib import Path

from ..formatting.formatter import OutputFormatter
from ..models.schemas import ProcessingConfig, TranscriptionResult

logger = logging.getLogger(__name__)


class FileWriter:
    """
    Handles writing transcription results to disk.
    
    Creates output directories per audio file and writes JSON and TXT
    formats according to configuration settings.
    """

    def __init__(self, config: ProcessingConfig) -> None:
        """
        Initialize the file writer.
        
        Args:
            config: Processing configuration containing output settings
        """
        self.config = config
        self.output_dir = config.output_dir
        self.output_formats = config.output.get("formats", ["json", "txt"])

        # Initialize formatter with config settings
        self.formatter = OutputFormatter(
            include_timestamps=config.output.get("include_timestamps", True),
            include_confidence=config.output.get("include_confidence", True),
            pretty_json=config.output.get("pretty_json", True)
        )

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"FileWriter initialized with output dir: {self.output_dir}")
        logger.info(f"Output formats: {self.output_formats}")

    def write_transcription_result(self, result: TranscriptionResult) -> list[Path]:
        """
        Write transcription result to disk.
        
        Args:
            result: TranscriptionResult to write
            
        Returns:
            List of created file paths
        """
        # Create output directory for this audio file
        audio_filename = result.audio_file.path.stem  # filename without extension
        file_output_dir = self.output_dir / audio_filename
        file_output_dir.mkdir(parents=True, exist_ok=True)

        created_files = []

        # Write each requested format
        for format_type in self.output_formats:
            try:
                if format_type.lower() == "json":
                    json_path = self._write_json_output(result, file_output_dir)
                    created_files.append(json_path)

                elif format_type.lower() == "txt":
                    txt_path = self._write_text_output(result, file_output_dir)
                    created_files.append(txt_path)

                elif format_type.lower() == "attributed_txt":
                    attributed_txt_path = self._write_attributed_text_output(result, file_output_dir)
                    created_files.append(attributed_txt_path)

                else:
                    logger.warning(f"Unsupported output format: {format_type}")

            except Exception as e:
                logger.error(f"Failed to write {format_type} output for {audio_filename}: {e}")

        logger.info(f"Created {len(created_files)} output files for {audio_filename}")
        return created_files

    def _write_json_output(self, result: TranscriptionResult, output_dir: Path) -> Path:
        """
        Write JSON format output.
        
        Args:
            result: TranscriptionResult to write
            output_dir: Directory to write to
            
        Returns:
            Path to created JSON file
        """
        json_path = output_dir / "transcript.json"

        try:
            json_content = self.formatter.format_as_json(result)

            with open(json_path, 'w', encoding='utf-8') as f:
                f.write(json_content)

            logger.debug(f"JSON output written to {json_path}")
            return json_path

        except Exception as e:
            logger.error(f"Failed to write JSON output: {e}")
            raise

    def _write_text_output(self, result: TranscriptionResult, output_dir: Path) -> Path:
        """
        Write text format output.
        
        Args:
            result: TranscriptionResult to write
            output_dir: Directory to write to
            
        Returns:
            Path to created text file
        """
        txt_path = output_dir / "transcript.txt"

        try:
            text_content = self.formatter.format_as_text(result)

            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text_content)

            logger.debug(f"Text output written to {txt_path}")
            return txt_path

        except Exception as e:
            logger.error(f"Failed to write text output: {e}")
            raise

    def _write_attributed_text_output(self, result: TranscriptionResult, output_dir: Path) -> Path:
        """
        Write attributed text format output (theater play style).
        
        Args:
            result: TranscriptionResult to write
            output_dir: Directory to write to
            
        Returns:
            Path to created attributed text file
        """
        attributed_txt_path = output_dir / "transcript_attributed.txt"

        try:
            attributed_content = self.formatter.format_as_attributed_text(result)

            with open(attributed_txt_path, 'w', encoding='utf-8') as f:
                f.write(attributed_content)

            logger.debug(f"Attributed text output written to {attributed_txt_path}")
            return attributed_txt_path

        except Exception as e:
            logger.error(f"Failed to write attributed text output: {e}")
            raise

    def write_batch_results(self, results: list[TranscriptionResult]) -> list[list[Path]]:
        """
        Write multiple transcription results to disk.
        
        Args:
            results: List of TranscriptionResult instances
            
        Returns:
            List of lists containing created file paths for each result
        """
        all_created_files = []

        for result in results:
            try:
                created_files = self.write_transcription_result(result)
                all_created_files.append(created_files)

            except Exception as e:
                logger.error(f"Failed to write result for {result.audio_file.path}: {e}")
                all_created_files.append([])  # Empty list for failed writes

        successful_writes = sum(1 for files in all_created_files if files)
        logger.info(f"Batch write completed: {successful_writes}/{len(results)} files written successfully")

        return all_created_files

    def create_summary_report(self, results: list[TranscriptionResult]) -> Path:
        """
        Create a summary report for all processed files.
        
        Args:
            results: List of all TranscriptionResult instances
            
        Returns:
            Path to created summary report
        """
        summary_path = self.output_dir / "processing_summary.txt"

        try:
            lines = []
            lines.append("AUDIO TRANSCRIPTION PROCESSING SUMMARY")
            lines.append("=" * 50)
            lines.append(f"Total files processed: {len(results)}")
            lines.append(f"Generated on: {results[0].timestamp.strftime('%Y-%m-%d %H:%M:%S') if results else 'N/A'}")
            lines.append("")

            # Statistics
            successful_results = [r for r in results if r.full_text.strip()]
            failed_results = len(results) - len(successful_results)

            lines.append("PROCESSING STATISTICS:")
            lines.append(f"  Successful: {len(successful_results)}")
            lines.append(f"  Failed: {failed_results}")

            if successful_results:
                total_duration = sum(r.audio_file.duration or 0 for r in successful_results)
                total_processing_time = sum(r.processing_time or 0 for r in successful_results)
                avg_speed_ratio = total_duration / total_processing_time if total_processing_time > 0 else 0

                lines.append(f"  Total audio duration: {total_duration:.2f} seconds")
                lines.append(f"  Total processing time: {total_processing_time:.2f} seconds")
                lines.append(f"  Average speed ratio: {avg_speed_ratio:.2f}x")

            lines.append("")

            # File listing
            lines.append("PROCESSED FILES:")
            for i, result in enumerate(results, 1):
                status = "✓" if result.full_text.strip() else "✗"
                duration = f"{result.audio_file.duration:.2f}s" if result.audio_file.duration else "N/A"
                lines.append(f"  {i:2d}. {status} {result.audio_file.path.name} ({duration})")

            lines.append("")
            lines.append("=" * 50)

            # Write summary
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))

            logger.info(f"Summary report created: {summary_path}")
            return summary_path

        except Exception as e:
            logger.error(f"Failed to create summary report: {e}")
            raise

    def get_output_directory(self, audio_filename: str) -> Path:
        """
        Get the output directory path for a specific audio file.
        
        Args:
            audio_filename: Name of the audio file (without extension)
            
        Returns:
            Path to the output directory for this file
        """
        return self.output_dir / audio_filename

    def cleanup_empty_directories(self) -> None:
        """
        Clean up any empty output directories.
        """
        try:
            for item in self.output_dir.iterdir():
                if item.is_dir() and not any(item.iterdir()):
                    item.rmdir()
                    logger.debug(f"Removed empty directory: {item}")

        except Exception as e:
            logger.warning(f"Failed to cleanup empty directories: {e}")
