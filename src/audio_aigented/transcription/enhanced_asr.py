"""
Enhanced ASR with custom vocabulary and advanced decoding options.

This module extends the basic ASR functionality with support for:
- Custom vocabulary and contextual biasing
- N-gram language models
- Advanced beam search decoding
- Post-processing corrections
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .asr import ASRTranscriber
from .vocabulary import VocabularyManager
from ..models.schemas import AudioFile, TranscriptionResult

logger = logging.getLogger(__name__)


class EnhancedASRTranscriber(ASRTranscriber):
    """
    Enhanced ASR transcriber with custom vocabulary support.
    
    Extends the base ASR transcriber with additional features for
    improving transcription accuracy through custom vocabulary,
    language models, and advanced decoding strategies.
    """
    
    def __init__(self, config: dict) -> None:
        """
        Initialize the enhanced ASR transcriber.
        
        Args:
            config: Processing configuration
        """
        super().__init__(config)
        
        # Initialize vocabulary manager
        self.vocab_manager = VocabularyManager()
        
        # Load custom vocabulary if specified
        vocab_file = config.transcription.get("vocabulary_file")
        if vocab_file:
            self.vocab_manager.load_from_file(Path(vocab_file))
            
        # Advanced decoding parameters
        self.beam_size = config.transcription.get("beam_size", 4)
        self.lm_weight = config.transcription.get("lm_weight", 0.0)
        self.word_insertion_penalty = config.transcription.get("word_insertion_penalty", 0.0)
        self.blank_penalty = config.transcription.get("blank_penalty", 0.0)
        
        # Language model path (optional)
        self.lm_path = config.transcription.get("language_model_path")
        
        logger.info("Enhanced ASR transcriber initialized with custom vocabulary support")
        
    def load_model(self) -> None:
        """
        Load the ASR model with enhanced configuration.
        
        Overrides base method to add custom vocabulary and decoding parameters.
        """
        # Load base model
        super().load_model()
        
        # Configure beam search decoding if supported
        if hasattr(self.model, 'change_decoding_strategy'):
            try:
                # Configure CTC beam search with custom parameters
                decoding_config = {
                    'strategy': 'beam',
                    'beam': {
                        'beam_size': self.beam_size,
                        'beam_alpha': self.lm_weight,  # LM weight
                        'beam_beta': self.word_insertion_penalty,
                        'blank_penalty': self.blank_penalty,
                    }
                }
                
                # Add language model if available
                if self.lm_path and Path(self.lm_path).exists():
                    decoding_config['beam']['lm_path'] = self.lm_path
                    logger.info(f"Using language model: {self.lm_path}")
                    
                # Add custom vocabulary as hot words
                vocab_hints = self.vocab_manager.get_vocabulary_hints()
                if vocab_hints:
                    # Some models support hot words/contextual biasing
                    if hasattr(self.model, 'set_hot_words'):
                        self.model.set_hot_words(vocab_hints[:100])  # Limit to top 100
                        logger.info(f"Set {len(vocab_hints[:100])} hot words for contextual biasing")
                        
                self.model.change_decoding_strategy(decoding_config)
                logger.info(f"Configured beam search decoding with beam_size={self.beam_size}")
                
            except Exception as e:
                logger.warning(f"Could not configure advanced decoding: {e}")
                logger.info("Using default decoding strategy")
                
    def transcribe_audio_file(self, audio_file: AudioFile, audio_data: np.ndarray) -> TranscriptionResult:
        """
        Transcribe audio file with vocabulary post-processing.
        
        Args:
            audio_file: AudioFile instance
            audio_data: Audio data array
            
        Returns:
            TranscriptionResult with vocabulary corrections applied
        """
        # Get base transcription
        result = super().transcribe_audio_file(audio_file, audio_data)
        
        # Apply vocabulary post-processing to all segments
        for segment in result.segments:
            original_text = segment.text
            segment.text = self.vocab_manager.post_process_text(segment.text)
            
            # Log significant corrections
            if segment.text != original_text:
                logger.debug(f"Corrected: '{original_text}' -> '{segment.text}'")
                
        # Regenerate full text from corrected segments
        result.full_text = ' '.join(segment.text for segment in result.segments)
        
        # Add vocabulary info to metadata
        result.metadata["vocabulary_corrections_applied"] = True
        result.metadata["vocabulary_terms_count"] = len(self.vocab_manager.get_vocabulary_hints())
        
        return result
        
    def add_vocabulary_from_text(self, text: str, extract_technical: bool = True) -> None:
        """
        Extract and add vocabulary from sample text.
        
        Args:
            text: Sample text containing domain-specific terms
            extract_technical: Whether to extract technical-looking terms
        """
        if not text:
            return
            
        # Extract potential technical terms
        if extract_technical:
            import re
            
            # Find capitalized terms (potential acronyms)
            acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
            for acronym in set(acronyms):
                if len(acronym) <= 10:  # Reasonable acronym length
                    self.vocab_manager.technical_terms.add(acronym)
                    
            # Find camelCase or PascalCase terms (common in tech)
            camel_terms = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text)
            for term in set(camel_terms):
                self.vocab_manager.technical_terms.add(term)
                
            # Find hyphenated technical terms
            hyphenated = re.findall(r'\b\w+(?:-\w+)+\b', text)
            for term in set(hyphenated):
                self.vocab_manager.technical_terms.add(term.lower())
                
        logger.info(f"Extracted {len(self.vocab_manager.technical_terms)} technical terms from text")
        
    def set_vocabulary_corrections(self, corrections: dict[str, str]) -> None:
        """
        Set custom vocabulary corrections.
        
        Args:
            corrections: Dictionary of {incorrect: correct} mappings
        """
        self.vocab_manager.add_corrections(corrections)
        logger.info(f"Added {len(corrections)} vocabulary corrections")
        
    def save_vocabulary(self, vocab_file: Path) -> None:
        """
        Save current vocabulary to file.
        
        Args:
            vocab_file: Path to save vocabulary
        """
        self.vocab_manager.save_to_file(vocab_file)
        
    def get_decoding_info(self) -> dict[str, Any]:
        """
        Get information about current decoding configuration.
        
        Returns:
            Dictionary with decoding parameters
        """
        return {
            "beam_size": self.beam_size,
            "lm_weight": self.lm_weight,
            "word_insertion_penalty": self.word_insertion_penalty,
            "blank_penalty": self.blank_penalty,
            "language_model": self.lm_path is not None,
            "vocabulary_terms": len(self.vocab_manager.get_vocabulary_hints()),
        }