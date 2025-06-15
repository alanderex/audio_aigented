"""
Per-file context management for improved transcription accuracy.

This module provides functionality to load and apply file-specific
context, vocabulary, and metadata to improve transcription quality.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from .content_analyzer import ContentAnalyzer

logger = logging.getLogger(__name__)


class FileContextManager:
    """
    Manages per-file context for transcription improvements.
    
    Supports loading context from sidecar files that provide:
    - Custom vocabulary specific to the audio file
    - Speaker information and names
    - Topic/domain hints
    - Expected phrases or terminology
    """
    
    def __init__(self) -> None:
        """Initialize the file context manager."""
        self.contexts: Dict[Path, Dict[str, Any]] = {}
        self.content_analyzer = ContentAnalyzer()
        
    def load_context_for_file(self, audio_file: Path) -> Optional[Dict[str, Any]]:
        """
        Load context for a specific audio file.
        
        Looks for context in order of priority:
        1. <audio_file>.context.json
        2. <audio_file>.txt (simple vocabulary list)
        3. <directory>/.context/<audio_file_name>.json
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            Context dictionary or None if no context found
        """
        # Check for sidecar JSON file
        context_json = audio_file.with_suffix(audio_file.suffix + '.context.json')
        if context_json.exists():
            return self._load_json_context(context_json)
            
        # Check for simple text vocabulary file
        context_txt = audio_file.with_suffix(audio_file.suffix + '.txt')
        if context_txt.exists():
            return self._load_text_context(context_txt)
            
        # Check for context in .context directory
        context_dir = audio_file.parent / '.context'
        if context_dir.exists():
            dir_context = context_dir / f"{audio_file.stem}.json"
            if dir_context.exists():
                return self._load_json_context(dir_context)
                
        return None
        
    def _load_json_context(self, context_file: Path) -> Dict[str, Any]:
        """
        Load context from JSON file.
        
        Expected format:
        {
            "vocabulary": ["term1", "term2", ...],
            "corrections": {"incorrect": "correct", ...},
            "speakers": {"SPEAKER_00": "John Doe", ...},
            "topic": "Technical presentation about AI",
            "acronyms": {"AI": "Artificial Intelligence", ...},
            "phrases": ["machine learning", "neural networks", ...],
            "notes": "Additional context notes"
        }
        
        Args:
            context_file: Path to JSON context file
            
        Returns:
            Context dictionary
        """
        try:
            with open(context_file, 'r', encoding='utf-8') as f:
                context = json.load(f)
                
            logger.info(f"Loaded JSON context from: {context_file}")
            
            # Validate and normalize context
            normalized = {
                'vocabulary': context.get('vocabulary', []),
                'corrections': context.get('corrections', {}),
                'speakers': context.get('speakers', {}),
                'topic': context.get('topic', ''),
                'acronyms': context.get('acronyms', {}),
                'phrases': context.get('phrases', []),
                'notes': context.get('notes', ''),
                'source_file': str(context_file)
            }
            
            return normalized
            
        except Exception as e:
            logger.error(f"Failed to load JSON context from {context_file}: {e}")
            return {}
            
    def _load_text_context(self, context_file: Path) -> Dict[str, Any]:
        """
        Load simple vocabulary from text file.
        
        Format: One term per line, # for comments
        
        Args:
            context_file: Path to text context file
            
        Returns:
            Context dictionary with vocabulary
        """
        try:
            vocabulary = []
            with open(context_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        vocabulary.append(line)
                        
            logger.info(f"Loaded {len(vocabulary)} terms from: {context_file}")
            
            return {
                'vocabulary': vocabulary,
                'corrections': {},
                'speakers': {},
                'topic': '',
                'acronyms': {},
                'phrases': [],
                'notes': f'Loaded from {context_file}',
                'source_file': str(context_file)
            }
            
        except Exception as e:
            logger.error(f"Failed to load text context from {context_file}: {e}")
            return {}
            
    def create_enhanced_vocabulary(self, base_vocab_manager, context: Dict[str, Any]) -> None:
        """
        Enhance vocabulary manager with file-specific context.
        
        Args:
            base_vocab_manager: Base VocabularyManager instance
            context: File-specific context dictionary
        """
        # Add vocabulary terms
        if context.get('vocabulary'):
            base_vocab_manager.add_terms(context['vocabulary'])
            
        # Add corrections
        if context.get('corrections'):
            base_vocab_manager.add_corrections(context['corrections'])
            
        # Add acronyms
        if context.get('acronyms'):
            base_vocab_manager.acronyms.update(context['acronyms'])
            
        # Add contextual phrases
        if context.get('phrases'):
            base_vocab_manager.contextual_phrases.extend(context['phrases'])
            
        logger.info("Enhanced vocabulary with file-specific context")
        
    def apply_speaker_names(self, segments: list, context: Dict[str, Any]) -> None:
        """
        Apply speaker names from context to segments.
        
        Args:
            segments: List of AudioSegment objects
            context: Context with speaker information
        """
        speakers = context.get('speakers', {})
        if not speakers:
            return
            
        for segment in segments:
            if segment.speaker_id in speakers:
                # Store original ID and add speaker name
                segment.metadata = segment.metadata or {}
                segment.metadata['original_speaker_id'] = segment.speaker_id
                segment.metadata['speaker_name'] = speakers[segment.speaker_id]
                
        logger.info(f"Applied {len(speakers)} speaker names from context")
        
    def save_context_template(self, audio_file: Path) -> Path:
        """
        Create a context template file for an audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Path to created template file
        """
        template = {
            "vocabulary": [
                "example_term1",
                "example_term2"
            ],
            "corrections": {
                "mistranscribed": "correct_term"
            },
            "speakers": {
                "SPEAKER_00": "Speaker Name",
                "SPEAKER_01": "Another Speaker"
            },
            "topic": "Brief description of the audio content",
            "acronyms": {
                "AI": "Artificial Intelligence",
                "ML": "Machine Learning"
            },
            "phrases": [
                "important multi-word phrase",
                "technical terminology"
            ],
            "notes": "Additional context or instructions for transcription"
        }
        
        template_path = audio_file.with_suffix(audio_file.suffix + '.context.json')
        
        with open(template_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2)
            
        logger.info(f"Created context template: {template_path}")
        return template_path
    
    def load_raw_content_files(self, content_files: List[Path]) -> Dict[str, Any]:
        """
        Load and analyze raw content files to extract context.
        
        Args:
            content_files: List of paths to content files (txt, html, md)
            
        Returns:
            Merged context dictionary from all content files
        """
        merged_context = {
            'vocabulary': [],
            'corrections': {},
            'speakers': {},
            'topic': '',
            'acronyms': {},
            'phrases': [],
            'notes': 'Context extracted from raw content files',
            'source_file': 'multiple'
        }
        
        for content_file in content_files:
            if not content_file.exists():
                logger.warning(f"Content file not found: {content_file}")
                continue
                
            logger.info(f"Analyzing content file: {content_file}")
            analyzed = self.content_analyzer.analyze_content_file(content_file)
            
            if analyzed:
                merged_context = self.content_analyzer.merge_with_context(
                    merged_context, analyzed
                )
                
        logger.info(f"Extracted context from {len(content_files)} content files")
        return merged_context
    
    def enhance_context_with_raw_content(self, base_context: Dict[str, Any], 
                                       content_files: List[Path]) -> Dict[str, Any]:
        """
        Enhance existing context with information from raw content files.
        
        Args:
            base_context: Existing context dictionary
            content_files: List of content files to analyze
            
        Returns:
            Enhanced context dictionary
        """
        if not content_files:
            return base_context
            
        # Analyze content files
        content_context = self.load_raw_content_files(content_files)
        
        # Merge with base context
        enhanced = self.content_analyzer.merge_with_context(base_context, content_context)
        
        logger.info("Enhanced context with raw content analysis")
        return enhanced