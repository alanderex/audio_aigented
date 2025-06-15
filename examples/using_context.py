#!/usr/bin/env python3
"""
Example: Using context features for improved transcription.

This example demonstrates how to use:
1. Global vocabulary files
2. Per-file context for custom terms and speaker names
3. Enhanced transcription features
"""

import json
from pathlib import Path
from src.audio_aigented.pipeline import TranscriptionPipeline
from src.audio_aigented.config.manager import ConfigManager

def create_sample_vocabulary_file():
    """Create a sample global vocabulary file."""
    vocab_content = """# Global Vocabulary File
# Technical terms and acronyms for AI/ML domain

# Technical terms for boosting
neural_network
transformer
embedding
attention_mechanism
backpropagation
gradient_descent
overfitting
regularization

# Acronyms
AI:Artificial Intelligence
ML:Machine Learning
NLP:Natural Language Processing
ASR:Automatic Speech Recognition
TTS:Text to Speech
GPU:Graphics Processing Unit
CUDA:Compute Unified Device Architecture

# Common corrections
neuro network -> neural network
back prop -> backpropagation
overfit -> overfitting

# Contextual phrases
"deep learning model"
"natural language processing"
"speech recognition system"
"""
    
    vocab_file = Path("vocabulary.txt")
    vocab_file.write_text(vocab_content)
    print(f"Created global vocabulary file: {vocab_file}")
    return vocab_file

def create_sample_context_file(audio_file: Path):
    """Create a sample context file for a specific audio."""
    context = {
        "vocabulary": [
            "PyTorch",
            "TensorFlow",
            "Hugging Face",
            "BERT",
            "GPT-3",
            "fine-tuning",
            "zero-shot learning"
        ],
        "corrections": {
            "pie torch": "PyTorch",
            "tensor flow": "TensorFlow",
            "hugging face": "Hugging Face",
            "bert": "BERT",
            "gpt three": "GPT-3"
        },
        "speakers": {
            "SPEAKER_00": "Dr. Sarah Chen",
            "SPEAKER_01": "Prof. Michael Brown",
            "SPEAKER_02": "Alex Johnson"
        },
        "topic": "Deep Learning Workshop - Neural Network Architectures",
        "acronyms": {
            "RNN": "Recurrent Neural Network",
            "CNN": "Convolutional Neural Network",
            "GAN": "Generative Adversarial Network"
        },
        "phrases": [
            "transformer architecture",
            "attention is all you need",
            "pre-trained models",
            "transfer learning"
        ],
        "notes": "Technical workshop with three speakers discussing state-of-the-art models"
    }
    
    context_file = audio_file.with_suffix(audio_file.suffix + '.context.json')
    with open(context_file, 'w') as f:
        json.dump(context, f, indent=2)
    
    print(f"Created context file: {context_file}")
    return context_file

def main():
    """Demonstrate context-enhanced transcription."""
    
    # Create sample files
    vocab_file = create_sample_vocabulary_file()
    
    # Setup configuration with enhanced features
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Enable enhanced transcription features
    config.transcription["vocabulary_file"] = str(vocab_file)
    config.transcription["beam_size"] = 8  # Larger beam for better accuracy
    config.transcription["enable_file_context"] = True
    
    # Initialize pipeline
    pipeline = TranscriptionPipeline(config)
    
    # Example: Process audio with context
    audio_file = Path("./inputs/workshop_recording.wav")
    
    if audio_file.exists():
        # Create context file for this audio
        context_file = create_sample_context_file(audio_file)
        
        print("\nProcessing audio with enhanced context...")
        result = pipeline.process_single_file(audio_file)
        
        # Show results with speaker names
        print("\nTranscription with speaker names:")
        for segment in result.segments[:5]:  # Show first 5 segments
            speaker_name = segment.metadata.get('speaker_name', segment.speaker_id)
            print(f"{speaker_name}: {segment.text}")
            
        # Clean up example files
        vocab_file.unlink()
        context_file.unlink()
    else:
        print(f"\nAudio file not found: {audio_file}")
        print("This example requires an audio file to demonstrate context features.")
        
        # Show what context file would look like
        print("\nExample context file structure:")
        print(json.dumps(create_sample_context_file(Path("example.wav")), indent=2))

if __name__ == "__main__":
    main()