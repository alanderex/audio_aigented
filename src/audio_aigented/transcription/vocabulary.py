"""
Custom vocabulary and contextual biasing for ASR.

This module provides functionality to improve transcription accuracy
for domain-specific terms, technical vocabulary, and contextual phrases.
"""

import logging
from pathlib import Path
from typing import Optional

import re

logger = logging.getLogger(__name__)


class VocabularyManager:
    """
    Manages custom vocabulary and contextual biasing for ASR.
    
    Provides methods to load, manage, and apply custom vocabulary
    to improve transcription accuracy for domain-specific terms.
    """
    
    def __init__(self) -> None:
        """Initialize the vocabulary manager."""
        self.custom_terms: dict[str, str] = {}
        self.acronyms: dict[str, str] = {}
        self.technical_terms: set[str] = set()
        self.contextual_phrases: list[str] = []
        
    def load_from_file(self, vocab_file: Path) -> None:
        """
        Load custom vocabulary from a file.
        
        File format (one per line):
        - term -> replacement (for corrections)
        - ACRONYM:expansion (for acronyms)
        - technical_term (for boosting)
        - "contextual phrase" (for multi-word terms)
        
        Args:
            vocab_file: Path to vocabulary file
        """
        if not vocab_file.exists():
            logger.warning(f"Vocabulary file not found: {vocab_file}")
            return
            
        logger.info(f"Loading custom vocabulary from: {vocab_file}")
        
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                # Handle corrections: "term -> replacement"
                if ' -> ' in line:
                    term, replacement = line.split(' -> ', 1)
                    self.custom_terms[term.lower()] = replacement
                    
                # Handle acronyms: "ACRONYM:expansion"
                elif ':' in line and line.split(':')[0].isupper():
                    acronym, expansion = line.split(':', 1)
                    self.acronyms[acronym] = expansion
                    
                # Handle contextual phrases (quoted)
                elif line.startswith('"') and line.endswith('"'):
                    phrase = line[1:-1]
                    self.contextual_phrases.append(phrase)
                    
                # Handle technical terms (single words for boosting)
                else:
                    self.technical_terms.add(line.lower())
                    
        logger.info(f"Loaded {len(self.custom_terms)} corrections, "
                   f"{len(self.acronyms)} acronyms, "
                   f"{len(self.technical_terms)} technical terms, "
                   f"{len(self.contextual_phrases)} contextual phrases")
                   
    def add_terms(self, terms: list[str]) -> None:
        """
        Add technical terms to boost during transcription.
        
        Args:
            terms: List of technical terms to add
        """
        for term in terms:
            self.technical_terms.add(term.lower())
            
    def add_corrections(self, corrections: dict[str, str]) -> None:
        """
        Add correction mappings.
        
        Args:
            corrections: Dictionary of {incorrect: correct} mappings
        """
        for incorrect, correct in corrections.items():
            self.custom_terms[incorrect.lower()] = correct
            
    def post_process_text(self, text: str) -> str:
        """
        Apply post-processing corrections to transcribed text.
        
        Args:
            text: Raw transcribed text
            
        Returns:
            Corrected text with custom vocabulary applied
        """
        if not text:
            return text
            
        original_text = text
        
        # Apply custom term corrections (case-insensitive)
        for term, replacement in self.custom_terms.items():
            # Use word boundaries to avoid partial replacements
            pattern = r'\b' + re.escape(term) + r'\b'
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            
        # Apply acronym expansions (case-sensitive)
        for acronym, expansion in self.acronyms.items():
            pattern = r'\b' + re.escape(acronym) + r'\b'
            text = re.sub(pattern, f"{acronym} ({expansion})", text)
            
        # Apply contextual phrase corrections
        for phrase in self.contextual_phrases:
            # Look for similar phrases and correct them
            pattern = self._create_fuzzy_pattern(phrase)
            if pattern:
                text = re.sub(pattern, phrase, text, flags=re.IGNORECASE)
                
        if text != original_text:
            logger.debug("Applied vocabulary corrections to transcription")
            
        return text
        
    def _create_fuzzy_pattern(self, phrase: str) -> Optional[str]:
        """
        Create a regex pattern that matches similar phrases.
        
        Args:
            phrase: Target phrase
            
        Returns:
            Regex pattern or None
        """
        # Simple fuzzy matching: allow optional spaces and common substitutions
        words = phrase.split()
        if len(words) < 2:
            return None
            
        # Build pattern allowing flexible spacing
        pattern_parts = []
        for word in words:
            pattern_parts.append(re.escape(word))
            
        # Join with flexible whitespace
        pattern = r'\s*'.join(pattern_parts)
        return pattern
        
    def get_vocabulary_hints(self) -> list[str]:
        """
        Get all vocabulary terms as hints for the ASR model.
        
        Returns:
            List of all vocabulary terms
        """
        hints = []
        
        # Add technical terms
        hints.extend(self.technical_terms)
        
        # Add custom term targets
        hints.extend(self.custom_terms.values())
        
        # Add acronyms
        hints.extend(self.acronyms.keys())
        
        # Add contextual phrases
        hints.extend(self.contextual_phrases)
        
        return hints
        
    def save_to_file(self, vocab_file: Path) -> None:
        """
        Save current vocabulary to a file.
        
        Args:
            vocab_file: Path to save vocabulary file
        """
        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write("# Custom Vocabulary File\n")
            f.write("# Format: term -> replacement, ACRONYM:expansion, or \"phrase\"\n\n")
            
            # Write corrections
            if self.custom_terms:
                f.write("# Corrections\n")
                for term, replacement in sorted(self.custom_terms.items()):
                    f.write(f"{term} -> {replacement}\n")
                f.write("\n")
                
            # Write acronyms
            if self.acronyms:
                f.write("# Acronyms\n")
                for acronym, expansion in sorted(self.acronyms.items()):
                    f.write(f"{acronym}:{expansion}\n")
                f.write("\n")
                
            # Write contextual phrases
            if self.contextual_phrases:
                f.write("# Contextual Phrases\n")
                for phrase in sorted(self.contextual_phrases):
                    f.write(f'"{phrase}"\n')
                f.write("\n")
                
            # Write technical terms
            if self.technical_terms:
                f.write("# Technical Terms\n")
                for term in sorted(self.technical_terms):
                    f.write(f"{term}\n")
                    
        logger.info(f"Saved vocabulary to: {vocab_file}")