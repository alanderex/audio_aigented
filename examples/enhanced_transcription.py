#!/usr/bin/env python3
"""
Example of using enhanced transcription with custom vocabulary.

This script demonstrates how to:
1. Use a custom vocabulary file
2. Apply contextual biasing for technical terms
3. Configure advanced decoding parameters
4. Post-process transcriptions with corrections
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.audio_aigented.transcription.enhanced_asr import EnhancedASRTranscriber
from src.audio_aigented.transcription.vocabulary import VocabularyManager
from src.audio_aigented.config.manager import ConfigManager
from src.audio_aigented.audio.loader import AudioLoader
from src.audio_aigented.models.schemas import AudioFile


def create_custom_vocabulary():
    """Create a custom vocabulary for your domain."""
    vocab_manager = VocabularyManager()
    
    # Add common corrections for your domain
    corrections = {
        "python": "Python",
        "java script": "JavaScript",
        "type script": "TypeScript",
        "git hub": "GitHub",
        "docker": "Docker",
        "kubernetes": "Kubernetes",
        "machine learning": "machine learning",
        "a i": "AI",
        "m l": "ML",
        "l l m": "LLM",
        "g p t": "GPT",
        "api": "API",
    }
    vocab_manager.add_corrections(corrections)
    
    # Add technical terms to boost
    technical_terms = [
        "PyTorch", "TensorFlow", "NumPy", "Pandas", "Scikit-learn",
        "FastAPI", "Django", "Flask", "React", "Vue", "Angular",
        "PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "Kafka",
        "AWS", "Azure", "GCP", "Lambda", "EC2", "S3",
        "CI/CD", "DevOps", "GitOps", "MLOps", "DataOps",
        "REST", "GraphQL", "gRPC", "WebSocket", "HTTP",
        "async", "await", "callback", "promise", "observable",
        "microservices", "serverless", "containerization",
        "neural network", "deep learning", "transformer",
        "BERT", "GPT", "LSTM", "CNN", "RNN", "GAN",
    ]
    vocab_manager.add_terms(technical_terms)
    
    # Add contextual phrases
    phrases = [
        "continuous integration",
        "continuous deployment",
        "infrastructure as code",
        "test driven development",
        "domain driven design",
        "event driven architecture",
        "service oriented architecture",
        "representational state transfer",
        "single page application",
        "progressive web app",
        "natural language processing",
        "computer vision",
        "reinforcement learning",
        "transfer learning",
        "federated learning",
        "edge computing",
        "quantum computing",
    ]
    for phrase in phrases:
        vocab_manager.contextual_phrases.append(phrase)
    
    return vocab_manager


def main():
    """Demonstrate enhanced transcription with custom vocabulary."""
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Configure for enhanced transcription
    config.transcription["beam_size"] = 8  # Larger beam for better accuracy
    config.transcription["lm_weight"] = 0.5  # Enable language model scoring
    
    # Create enhanced transcriber
    transcriber = EnhancedASRTranscriber(config)
    
    # Create and set custom vocabulary
    vocab_manager = create_custom_vocabulary()
    transcriber.vocab_manager = vocab_manager
    
    # Save vocabulary for reuse
    vocab_file = Path("custom_tech_vocabulary.txt")
    vocab_manager.save_to_file(vocab_file)
    print(f"Saved custom vocabulary to: {vocab_file}")
    
    # Example: Add vocabulary from sample text
    sample_text = """
    We're building a microservices architecture using Kubernetes and Docker.
    The API is built with FastAPI and uses PostgreSQL for the database.
    We're implementing CI/CD with GitHub Actions and deploying to AWS Lambda.
    The machine learning pipeline uses PyTorch and is trained on GPU clusters.
    """
    transcriber.add_vocabulary_from_text(sample_text)
    
    # Display decoding configuration
    print("\nDecoding Configuration:")
    for key, value in transcriber.get_decoding_info().items():
        print(f"  {key}: {value}")
    
    # Example usage with an audio file
    audio_file = Path("example_audio.wav")
    if audio_file.exists():
        # Load audio
        audio_loader = AudioLoader(config)
        audio_data, _ = audio_loader.load_audio_file(audio_file)
        
        # Create AudioFile instance
        audio_file_obj = AudioFile(
            path=audio_file,
            duration=len(audio_data) / config.audio["sample_rate"],
            sample_rate=config.audio["sample_rate"],
            channels=1,
            format="wav"
        )
        
        # Transcribe with enhanced settings
        result = transcriber.transcribe_audio_file(audio_file_obj, audio_data)
        
        print(f"\nTranscription Result:")
        print(f"Text: {result.full_text}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Vocabulary corrections applied: {result.metadata.get('vocabulary_corrections_applied', False)}")
    else:
        print(f"\nTo test transcription, place an audio file at: {audio_file}")
    
    print("\nUsage from command line:")
    print(f"  python main.py --vocabulary-file {vocab_file} --beam-size 8")


if __name__ == "__main__":
    main()