"""
Audio Transcription Pipeline using NVIDIA NeMo

A modular, GPU-accelerated audio processing pipeline for speech recognition.
"""

__version__ = "0.1.0"
__author__ = "Audio AI Team"

from .pipeline import TranscriptionPipeline

__all__ = ["TranscriptionPipeline"]