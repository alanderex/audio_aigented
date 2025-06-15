"""
Main CLI entry point for the Audio Transcription Pipeline.

This module provides a command-line interface for running the ASR transcription
system with various options for input directories, configuration, and output formats.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click

from src.audio_aigented.pipeline import TranscriptionPipeline
from src.audio_aigented.config.manager import ConfigManager


@click.command()
@click.option(
    '--input-dir', '-i',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help='Directory containing .wav audio files to process'
)
@click.option(
    '--config', '-c',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help='Path to configuration YAML file'
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help='Output directory for transcription results'
)
@click.option(
    '--log-level', '-l',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR'], case_sensitive=False),
    default='INFO',
    help='Logging level'
)
@click.option(
    '--device',
    type=click.Choice(['cuda', 'cpu'], case_sensitive=False),
    help='Device for ASR processing (cuda/cpu)'
)
@click.option(
    '--model-name',
    type=str,
    help='NVIDIA NeMo model name to use for transcription'
)
@click.option(
    '--formats',
    type=str,
    help='Output formats (comma-separated: json,txt)'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Show what would be processed without actually processing'
)
@click.option(
    '--enable-diarization/--disable-diarization',
    default=True,
    help='Enable speaker diarization (default: enabled)'
)
@click.version_option(version="0.1.0", prog_name="audio-aigented")
def cli(
    input_dir: Optional[Path],
    config: Optional[Path],
    output_dir: Optional[Path],
    log_level: str,
    device: Optional[str],
    model_name: Optional[str],
    formats: Optional[str],
    dry_run: bool,
    enable_diarization: bool
) -> None:
    """
    Audio Transcription Pipeline using NVIDIA NeMo.
    
    Process .wav audio files through automatic speech recognition (ASR)
    and generate structured transcription outputs.
    
    Examples:
        audio-transcribe -i ./my_audio_files
        audio-transcribe -i ./audio -o ./results -c ./my_config.yaml
        audio-transcribe --input-dir ./podcasts --device cuda --log-level DEBUG
    """
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config_manager = ConfigManager(config)
        pipeline_config = config_manager.load_config()
        
        # Override config with CLI arguments
        if input_dir:
            pipeline_config.input_dir = input_dir
        if output_dir:
            pipeline_config.output_dir = output_dir
        if device:
            pipeline_config.transcription["device"] = device.lower()
        if model_name:
            pipeline_config.transcription["model_name"] = model_name
        if formats:
            pipeline_config.output["formats"] = [f.strip() for f in formats.split(',')]
            
        # Update log level and diarization in config
        pipeline_config.processing["log_level"] = log_level
        pipeline_config.processing["enable_diarization"] = enable_diarization
        
        # Initialize pipeline
        pipeline = TranscriptionPipeline(pipeline_config)
        
        # Show configuration
        click.echo("🎙️  Audio Transcription Pipeline")
        click.echo("=" * 40)
        click.echo(f"Input Directory: {pipeline_config.input_dir}")
        click.echo(f"Output Directory: {pipeline_config.output_dir}")
        click.echo(f"ASR Model: {pipeline_config.transcription['model_name']}")
        click.echo(f"Device: {pipeline_config.transcription['device']}")
        click.echo(f"Output Formats: {', '.join(pipeline_config.output['formats'])}")
        click.echo("=" * 40)
        
        # Discover files
        audio_files = pipeline.audio_loader.discover_audio_files()
        
        if not audio_files:
            click.echo("❌ No .wav files found in input directory")
            sys.exit(1)
            
        click.echo(f"📁 Found {len(audio_files)} audio files to process")
        
        # Handle dry run
        if dry_run:
            click.echo("\n🔍 Files that would be processed:")
            for i, file_path in enumerate(audio_files, 1):
                click.echo(f"  {i:2d}. {file_path.name}")
            click.echo(f"\n✅ Dry run complete. {len(audio_files)} files would be processed.")
            return
            
        # Process files
        click.echo("\n🚀 Starting transcription...")
        results = pipeline.process_files(audio_files)
        
        # Show results
        successful = sum(1 for r in results if r.full_text.strip())
        failed = len(results) - successful
        
        click.echo(f"\n✅ Processing complete!")
        click.echo(f"   Successful: {successful}")
        click.echo(f"   Failed: {failed}")
        click.echo(f"   Output directory: {pipeline_config.output_dir}")
        
        if failed > 0:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
