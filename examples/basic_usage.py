"""
Basic usage example for the Audio Transcription Pipeline.

This script demonstrates how to use the audio transcription system
to process audio files and generate transcriptions.
"""

import logging
from pathlib import Path

from src.audio_aigented.pipeline import TranscriptionPipeline
from src.audio_aigented.config.manager import ConfigManager
from src.audio_aigented.models.schemas import ProcessingConfig


def main():
    """
    Main example function demonstrating basic usage.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🎙️  Audio Transcription Pipeline - Basic Usage Example")
    print("=" * 60)
    
    # Method 1: Using default configuration
    print("\n📝 Method 1: Using default configuration")
    try:
        # Initialize pipeline with default settings
        pipeline = TranscriptionPipeline()
        
        print(f"   Input directory: {pipeline.config.input_dir}")
        print(f"   Output directory: {pipeline.config.output_dir}")
        print(f"   ASR model: {pipeline.config.transcription['model_name']}")
        print(f"   Device: {pipeline.config.transcription['device']}")
        
        # Check for audio files
        audio_files = pipeline.audio_loader.discover_audio_files()
        print(f"   Found {len(audio_files)} audio files")
        
        if audio_files:
            print("   Audio files found:")
            for i, file_path in enumerate(audio_files[:5], 1):  # Show first 5
                print(f"     {i}. {file_path.name}")
            if len(audio_files) > 5:
                print(f"     ... and {len(audio_files) - 5} more")
                
            # Process files (uncomment to actually run transcription)
            # results = pipeline.process_files(audio_files)
            # print(f"   Transcription completed: {len(results)} files processed")
        else:
            print("   ⚠️  No audio files found. Place .wav files in ./inputs/ directory")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        
    # Method 2: Using custom configuration
    print("\n📝 Method 2: Using custom configuration")
    try:
        # Create custom configuration
        custom_config = ProcessingConfig(
            input_dir=Path("./examples/sample_audio"),
            output_dir=Path("./examples/outputs"),
            audio={
                "sample_rate": 16000,
                "batch_size": 4
            },
            transcription={
                "model_name": "stt_en_conformer_ctc_large",
                "device": "cuda",  # Change to "cpu" if no GPU
                "enable_confidence_scores": True
            },
            output={
                "formats": ["json", "txt"],
                "include_timestamps": True,
                "pretty_json": True
            }
        )
        
        # Initialize pipeline with custom config
        pipeline = TranscriptionPipeline(config=custom_config)
        
        print(f"   Custom input directory: {custom_config.input_dir}")
        print(f"   Custom output directory: {custom_config.output_dir}")
        print(f"   Custom batch size: {custom_config.audio['batch_size']}")
        
        # Check for audio files in custom directory
        audio_files = pipeline.audio_loader.discover_audio_files()
        print(f"   Found {len(audio_files)} audio files in custom directory")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        
    # Method 3: Using configuration file
    print("\n📝 Method 3: Using configuration file")
    try:
        # Load configuration from file
        config_manager = ConfigManager(Path("config/default.yaml"))
        config = config_manager.load_config()
        
        # Initialize pipeline
        pipeline = TranscriptionPipeline(config=config)
        
        print(f"   Loaded from: {config_manager.config_path}")
        print(f"   Input directory: {config.input_dir}")
        print(f"   ASR model: {config.transcription['model_name']}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        
    # Method 4: Processing a single file
    print("\n📝 Method 4: Processing a single file")
    try:
        pipeline = TranscriptionPipeline()
        
        # Example of processing a single file (if it exists)
        example_file = Path("./inputs/example.wav")
        if example_file.exists():
            print(f"   Processing: {example_file.name}")
            
            # Process single file (uncomment to actually run)
            # result = pipeline.process_single_file(example_file)
            # print(f"   Transcription: {result.full_text[:100]}...")
            # print(f"   Processing time: {result.processing_time:.2f}s")
            # print(f"   Output files: {result.metadata.get('output_files', [])}")
        else:
            print(f"   ⚠️  Example file not found: {example_file}")
            print("   Create an example.wav file in ./inputs/ to test single file processing")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        
    print("\n" + "=" * 60)
    print("✅ Example completed!")
    print("\nTo run actual transcription:")
    print("1. Place .wav files in ./inputs/ directory")
    print("2. Run: python examples/basic_usage.py")
    print("3. Or use CLI: python main.py --input-dir ./inputs")
    print("\nFor more options, run: python main.py --help")


def demonstrate_config_customization():
    """
    Demonstrate how to customize configuration for specific use cases.
    """
    print("\n🔧 Configuration Customization Examples")
    print("-" * 40)
    
    # Example 1: High-quality transcription
    print("\n1. High-quality transcription setup:")
    hq_config = ProcessingConfig(
        audio={"sample_rate": 16000, "batch_size": 1},  # Process one at a time
        transcription={
            "model_name": "stt_en_conformer_ctc_large",  # Use large model
            "device": "cuda",
            "enable_confidence_scores": True
        },
        output={
            "formats": ["json"],
            "include_timestamps": True,
            "include_confidence": True,
            "pretty_json": True
        }
    )
    print(f"   Model: {hq_config.transcription['model_name']}")
    print(f"   Batch size: {hq_config.audio['batch_size']}")
    print(f"   Includes confidence scores: {hq_config.transcription['enable_confidence_scores']}")
    
    # Example 2: Fast processing
    print("\n2. Fast processing setup:")
    fast_config = ProcessingConfig(
        audio={"sample_rate": 16000, "batch_size": 8},  # Process multiple files
        transcription={
            "model_name": "stt_en_conformer_ctc_medium",  # Use medium model
            "device": "cuda",
            "enable_confidence_scores": False  # Skip confidence for speed
        },
        output={
            "formats": ["txt"],  # Only text output
            "include_timestamps": False,
            "pretty_json": False
        }
    )
    print(f"   Model: {fast_config.transcription['model_name']}")
    print(f"   Batch size: {fast_config.audio['batch_size']}")
    print(f"   Output formats: {fast_config.output['formats']}")
    
    # Example 3: CPU-only processing
    print("\n3. CPU-only processing setup:")
    cpu_config = ProcessingConfig(
        transcription={
            "model_name": "stt_en_conformer_ctc_small",  # Use smaller model
            "device": "cpu",
            "enable_confidence_scores": True
        },
        processing={
            "parallel_workers": 1,  # Single worker for CPU
            "log_level": "DEBUG"
        }
    )
    print(f"   Device: {cpu_config.transcription['device']}")
    print(f"   Model: {cpu_config.transcription['model_name']}")
    print(f"   Workers: {cpu_config.processing['parallel_workers']}")


if __name__ == "__main__":
    main()
    demonstrate_config_customization()