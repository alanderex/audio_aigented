"""
Centralized logging configuration for the audio transcription pipeline.

This module configures logging levels for various third-party libraries
to reduce verbose output during normal operation.
"""

import logging
import os


def configure_logging(log_level: str = "INFO") -> None:
    """
    Configure logging for the entire application.

    Args:
        log_level: Desired logging level for the application
    """
    # Set environment variables to reduce verbosity
    os.environ["NEMO_TESTING"] = "True"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Reduce tokenizer warnings

    # Configure third-party library logging to ERROR level
    third_party_loggers = [
        "nemo",
        "nemo.collections",
        "nemo.collections.asr",
        "nemo.collections.common",
        "nemo.core",
        "nemo.utils",
        "nemo_logging",
        "transformers",
        "torch",
        "torch.distributed",
        "pytorch_lightning",
        "pytorch_lightning.utilities",
        "hydra",
        "hydra.core",
        "urllib3",
        "filelock",
        "datasets",
        "huggingface_hub",
        "tqdm",
        "matplotlib",
        "numba",
    ]

    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    # Special handling for NeMo's custom logger
    try:
        import nemo.utils.logging as nemo_logging

        # Disable NeMo's rank logging prefix
        nemo_logging.set_log_level(logging.ERROR)
    except ImportError:
        pass

    # Configure our application logging
    app_loggers = [
        "src.audio_aigented",
        "__main__",
    ]

    for logger_name in app_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))


def suppress_nemo_output():
    """
    Additional function to suppress NeMo output during model operations.
    Call this before loading NeMo models.
    """
    import sys
    from io import StringIO

    # Create a context manager for suppressing output
    class SuppressOutput:
        def __enter__(self):
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr

    return SuppressOutput()
