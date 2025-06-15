"""
Content analyzer for extracting context from raw text and HTML files.

This module analyzes text documents to extract vocabulary, technical terms,
and context that can improve transcription accuracy.
"""

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple
from html.parser import HTMLParser
from html import unescape

logger = logging.getLogger(__name__)


class HTMLTextExtractor(HTMLParser):
    """Extract text content from HTML."""
    
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.in_script = False
        self.in_style = False
        
    def handle_starttag(self, tag, attrs):
        if tag == 'script':
            self.in_script = True
        elif tag == 'style':
            self.in_style = True
            
    def handle_endtag(self, tag):
        if tag == 'script':
            self.in_script = False
        elif tag == 'style':
            self.in_style = False
            
    def handle_data(self, data):
        if not self.in_script and not self.in_style:
            self.text_parts.append(data)
            
    def get_text(self):
        return ' '.join(self.text_parts)


class ContentAnalyzer:
    """
    Analyzes raw content to extract vocabulary and context.
    
    Supports:
    - Plain text files
    - HTML documents
    - Markdown files
    - Meeting agendas
    - Technical documentation
    """
    
    def __init__(self):
        """Initialize the content analyzer."""
        # Common words to exclude from technical term extraction
        self.common_words = self._load_common_words()
        
    def analyze_content_file(self, content_file: Path) -> Dict[str, any]:
        """
        Analyze a content file and extract context information.
        
        Args:
            content_file: Path to content file (txt, html, md)
            
        Returns:
            Dictionary with extracted context
        """
        if not content_file.exists():
            logger.warning(f"Content file not found: {content_file}")
            return {}
            
        # Read content based on file type
        suffix = content_file.suffix.lower()
        
        try:
            if suffix in ['.html', '.htm']:
                content = self._read_html_file(content_file)
            elif suffix == '.md':
                content = self._read_markdown_file(content_file)
            else:
                # Default to plain text
                content = content_file.read_text(encoding='utf-8')
                
            logger.info(f"Analyzing content from: {content_file}")
            
            # Extract various types of information
            extracted = {
                'vocabulary': [],
                'technical_terms': [],
                'acronyms': {},
                'proper_names': [],
                'phrases': [],
                'numbers_and_ids': [],
                'topic_hints': self._extract_topic_hints(content),
                'source_file': str(content_file)
            }
            
            # Extract technical terms and vocabulary
            extracted['technical_terms'] = self._extract_technical_terms(content)
            extracted['vocabulary'] = self._extract_vocabulary(content)
            
            # Extract acronyms
            extracted['acronyms'] = self._extract_acronyms(content)
            
            # Extract proper names
            extracted['proper_names'] = self._extract_proper_names(content)
            
            # Extract important phrases
            extracted['phrases'] = self._extract_key_phrases(content)
            
            # Extract numbers, IDs, codes
            extracted['numbers_and_ids'] = self._extract_identifiers(content)
            
            # Log summary
            logger.info(f"Extracted: {len(extracted['vocabulary'])} vocabulary terms, "
                       f"{len(extracted['technical_terms'])} technical terms, "
                       f"{len(extracted['acronyms'])} acronyms, "
                       f"{len(extracted['proper_names'])} proper names")
            
            return extracted
            
        except Exception as e:
            logger.error(f"Failed to analyze content file {content_file}: {e}")
            return {}
            
    def _read_html_file(self, html_file: Path) -> str:
        """Extract text content from HTML file."""
        html_content = html_file.read_text(encoding='utf-8')
        
        # Use custom HTML parser
        parser = HTMLTextExtractor()
        parser.feed(html_content)
        text = parser.get_text()
        
        # Unescape HTML entities
        text = unescape(text)
        
        return text
        
    def _read_markdown_file(self, md_file: Path) -> str:
        """Extract text content from Markdown file."""
        content = md_file.read_text(encoding='utf-8')
        
        # Remove markdown formatting
        # Remove code blocks
        content = re.sub(r'```[\s\S]*?```', '', content)
        content = re.sub(r'`[^`]+`', lambda m: m.group(0)[1:-1], content)
        
        # Remove links but keep text
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
        
        # Remove images
        content = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', content)
        
        # Remove markdown symbols
        content = re.sub(r'[#*_~]', '', content)
        
        return content
        
    def _extract_technical_terms(self, content: str) -> List[str]:
        """Extract technical terms from content."""
        terms = set()
        
        # Find CamelCase terms (common in tech)
        camel_case = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', content)
        terms.update(camel_case)
        
        # Find snake_case terms
        snake_case = re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', content)
        terms.update(snake_case)
        
        # Find hyphenated terms
        hyphenated = re.findall(r'\b\w+(?:-\w+)+\b', content)
        terms.update(hyphenated)
        
        # Find terms with dots (e.g., file.extension, package.module)
        dotted = re.findall(r'\b\w+(?:\.\w+)+\b', content)
        terms.update(dotted)
        
        # Find potential technical terms (capitalized words not at sentence start)
        # This regex is complex but helps find mid-sentence capitalized terms
        technical_pattern = r'(?<=[a-z]\s)[A-Z]\w+'
        potential_terms = re.findall(technical_pattern, content)
        terms.update(potential_terms)
        
        return sorted(list(terms))
        
    def _extract_vocabulary(self, content: str) -> List[str]:
        """Extract domain-specific vocabulary."""
        # Tokenize and count words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        word_counts = Counter(words)
        
        # Find less common words (likely domain-specific)
        vocabulary = []
        for word, count in word_counts.items():
            # Include if not too common and appears multiple times
            if word not in self.common_words and count >= 2 and len(word) > 4:
                vocabulary.append(word)
                
        return sorted(vocabulary)[:100]  # Limit to top 100
        
    def _extract_acronyms(self, content: str) -> Dict[str, str]:
        """Extract acronyms and their potential expansions."""
        acronyms = {}
        
        # Pattern 1: "Full Name (ACRONYM)"
        pattern1 = r'([A-Z][a-zA-Z\s]+)\s*\(([A-Z]{2,})\)'
        matches1 = re.findall(pattern1, content)
        for full, abbr in matches1:
            acronyms[abbr] = full.strip()
            
        # Pattern 2: "ACRONYM (Full Name)"
        pattern2 = r'\b([A-Z]{2,})\s*\(([A-Z][a-zA-Z\s]+)\)'
        matches2 = re.findall(pattern2, content)
        for abbr, full in matches2:
            acronyms[abbr] = full.strip()
            
        # Pattern 3: "ACRONYM - Full Name" or "ACRONYM: Full Name"
        pattern3 = r'\b([A-Z]{2,})\s*[-:]\s*([A-Z][a-zA-Z\s]+)'
        matches3 = re.findall(pattern3, content)
        for abbr, full in matches3:
            if len(full) < 100:  # Avoid false positives
                acronyms[abbr] = full.strip()
                
        # Find standalone acronyms (all caps, 2-10 letters)
        standalone = re.findall(r'\b[A-Z]{2,10}\b', content)
        for acronym in set(standalone):
            if acronym not in acronyms and acronym not in ['THE', 'AND', 'FOR', 'NOT']:
                acronyms[acronym] = ""  # No expansion found
                
        return acronyms
        
    def _extract_proper_names(self, content: str) -> List[str]:
        """Extract likely proper names (people, places, products)."""
        names = set()
        
        # Pattern for names: Capitalized words in sequence
        # e.g., "John Smith", "New York", "Microsoft Office"
        name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        potential_names = re.findall(name_pattern, content)
        
        # Filter out common words and single words
        for name in potential_names:
            parts = name.split()
            if len(parts) >= 2 or (len(parts) == 1 and parts[0] not in self.common_words):
                names.add(name)
                
        return sorted(list(names))[:50]  # Limit to top 50
        
    def _extract_key_phrases(self, content: str) -> List[str]:
        """Extract important multi-word phrases."""
        phrases = []
        
        # Find quoted phrases
        quoted = re.findall(r'"([^"]+)"', content)
        phrases.extend([p for p in quoted if 2 <= len(p.split()) <= 5])
        
        # Find phrases in headings (if markdown/html indicators present)
        heading_pattern = r'^#+\s*(.+)$|<h[1-6]>([^<]+)</h[1-6]>'
        headings = re.findall(heading_pattern, content, re.MULTILINE)
        for heading in headings:
            text = heading[0] or heading[1]
            if text and 2 <= len(text.split()) <= 5:
                phrases.append(text.strip())
                
        # Find repeated multi-word sequences
        # This is a simple approach - could be enhanced with NLP
        words = content.split()
        for i in range(len(words) - 2):
            two_word = ' '.join(words[i:i+2])
            three_word = ' '.join(words[i:i+3])
            
            # Check if phrase appears multiple times
            if content.count(two_word) >= 3 and len(two_word) > 10:
                phrases.append(two_word)
            if content.count(three_word) >= 2 and len(three_word) > 15:
                phrases.append(three_word)
                
        # Deduplicate and limit
        return list(set(phrases))[:30]
        
    def _extract_identifiers(self, content: str) -> List[str]:
        """Extract numbers, IDs, version numbers, etc."""
        identifiers = set()
        
        # Version numbers (e.g., v1.2.3, 2.0, v2)
        versions = re.findall(r'\bv?\d+(?:\.\d+)*\b', content)
        identifiers.update(versions)
        
        # IDs and codes (mix of letters and numbers)
        ids = re.findall(r'\b[A-Z0-9]{4,}(?:[-_][A-Z0-9]+)*\b', content)
        identifiers.update(ids)
        
        # Ticket/issue numbers (e.g., JIRA-1234, #123)
        tickets = re.findall(r'\b[A-Z]+-\d+\b|#\d+\b', content)
        identifiers.update(tickets)
        
        return sorted(list(identifiers))[:20]
        
    def _extract_topic_hints(self, content: str) -> str:
        """Extract a brief topic description from content."""
        # Try to find title or first heading
        lines = content.strip().split('\n')
        
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line and not line.startswith('#'):
                # Clean up and truncate
                topic = re.sub(r'[#*_]', '', line)
                topic = ' '.join(topic.split())[:100]
                if len(topic) > 10:
                    return topic
                    
        return "Document content for context enhancement"
        
    def _load_common_words(self) -> Set[str]:
        """Load common English words to exclude."""
        # A minimal set of common words
        # In production, this could load from a file
        return {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
            'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one',
            'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out',
            'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when',
            'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
            'take', 'people', 'into', 'year', 'your', 'good', 'some',
            'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look',
            'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after',
            'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
            'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day',
            'most', 'us', 'is', 'was', 'are', 'been', 'has', 'had', 'were',
            'said', 'did', 'getting', 'made', 'find', 'where', 'much', 'too',
            'very', 'still', 'being', 'going', 'why', 'before', 'never',
            'here', 'more', 'always', 'those', 'tell', 'really', 'thing',
            'nothing', 'sure', 'right', 'mean', 'down', 'such', 'through'
        }
        
    def merge_with_context(self, base_context: Dict[str, any], 
                          analyzed_content: Dict[str, any]) -> Dict[str, any]:
        """
        Merge analyzed content with existing context.
        
        Args:
            base_context: Existing context dictionary
            analyzed_content: Newly analyzed content
            
        Returns:
            Merged context dictionary
        """
        merged = base_context.copy()
        
        # Merge vocabulary (unique terms)
        existing_vocab = set(merged.get('vocabulary', []))
        new_vocab = set(analyzed_content.get('vocabulary', []))
        merged['vocabulary'] = sorted(list(existing_vocab | new_vocab))
        
        # Merge technical terms
        tech_terms = set(analyzed_content.get('technical_terms', []))
        existing_vocab.update(tech_terms)
        merged['vocabulary'] = sorted(list(existing_vocab))
        
        # Merge acronyms
        merged_acronyms = merged.get('acronyms', {})
        new_acronyms = analyzed_content.get('acronyms', {})
        for acronym, expansion in new_acronyms.items():
            if acronym not in merged_acronyms or not merged_acronyms[acronym]:
                merged_acronyms[acronym] = expansion
        merged['acronyms'] = merged_acronyms
        
        # Add proper names to vocabulary
        proper_names = analyzed_content.get('proper_names', [])
        if proper_names:
            existing_vocab.update(proper_names)
            merged['vocabulary'] = sorted(list(existing_vocab))
            
        # Merge phrases
        existing_phrases = set(merged.get('phrases', []))
        new_phrases = set(analyzed_content.get('phrases', []))
        merged['phrases'] = sorted(list(existing_phrases | new_phrases))
        
        # Update topic if more specific
        if analyzed_content.get('topic_hints'):
            if not merged.get('topic') or len(merged['topic']) < 10:
                merged['topic'] = analyzed_content['topic_hints']
                
        # Add identifiers to vocabulary
        identifiers = analyzed_content.get('numbers_and_ids', [])
        if identifiers:
            existing_vocab.update(identifiers)
            merged['vocabulary'] = sorted(list(existing_vocab))
            
        # Add source information
        if 'notes' in merged:
            merged['notes'] += f"\nEnhanced with content from: {analyzed_content.get('source_file', 'unknown')}"
        else:
            merged['notes'] = f"Enhanced with content from: {analyzed_content.get('source_file', 'unknown')}"
            
        return merged