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
@click.option(
    '--vocabulary-file',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help='Path to custom vocabulary file for improved accuracy'
)
@click.option(
    '--beam-size',
    type=int,
    help='Beam search width for decoding (default: 4)'
)
@click.option(
    '--clear-cache',
    is_flag=True,
    help='Clear the cache before processing'
)
@click.option(
    '--create-context-templates',
    is_flag=True,
    help='Create context template files for each audio file'
)
@click.option(
    '--content-dir',
    multiple=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help='Directory containing companion content files. Can be specified multiple times.'
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
    enable_diarization: bool,
    vocabulary_file: Optional[Path],
    beam_size: Optional[int],
    clear_cache: bool,
    create_context_templates: bool,
    content_dir: tuple[Path, ...]
) -> None:
    """
    Audio Transcription Pipeline using NVIDIA NeMo.
    
    Process audio files through automatic speech recognition (ASR)
    and generate structured transcription outputs.
    
    Supported formats: .wav, .mp3, .m4a, .flac
    
    Examples:
        audio-transcribe -i ./my_audio_files
        audio-transcribe -i ./audio -o ./results -c ./my_config.yaml
        audio-transcribe --input-dir ./podcasts --device cuda --log-level DEBUG
    """
    # Setup logging with centralized configuration
    from src.audio_aigented.utils.logging_config import configure_logging
    import tqdm
    
    # Configure third-party library logging first
    configure_logging(log_level)
    
    # Create a tqdm-aware logging handler for our application
    class TqdmLoggingHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except Exception:
                self.handleError(record)
    
    # Configure logging with tqdm handler
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add tqdm-aware handler
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    root_logger.addHandler(handler)
    
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
        
        # Update enhanced transcription options
        if vocabulary_file:
            pipeline_config.transcription["vocabulary_file"] = str(vocabulary_file)
        if beam_size is not None:
            pipeline_config.transcription["beam_size"] = beam_size
        
        # Initialize pipeline
        pipeline = TranscriptionPipeline(pipeline_config)
        
        # Store content directories in pipeline config for per-file processing
        if content_dir:
            # Convert tuple to list and store in config
            pipeline_config.processing["content_directories"] = [str(d) for d in content_dir]
            click.echo(f"\n📁 Will search for companion content files in {len(content_dir)} directories:")
            for dir_path in content_dir:
                click.echo(f"   - {dir_path}")
        
        # Clear cache if requested
        if clear_cache:
            click.echo("🧹 Clearing cache...")
            count = pipeline.cache_manager.clear_cache()
            click.echo(f"✅ Cleared {count} cached files")
            click.echo("")
            
            # If no input directory specified, just clear cache and exit
            if not input_dir and not pipeline_config.input_dir:
                click.echo("Cache cleared. No input directory specified, exiting.")
                return
        
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
        
        # Handle context template creation
        if create_context_templates:
            click.echo("\n📝 Creating context templates...")
            created_count = 0
            for file_path in audio_files:
                try:
                    template_path = pipeline.context_manager.save_context_template(file_path)
                    click.echo(f"   ✅ Created: {template_path.name}")
                    created_count += 1
                except Exception as e:
                    click.echo(f"   ❌ Failed for {file_path.name}: {e}")
            click.echo(f"\nCreated {created_count} context template files")
            click.echo("Edit these files to provide custom vocabulary, speaker names, and corrections.")
            return
        
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
