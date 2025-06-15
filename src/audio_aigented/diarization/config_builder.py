"""
Configuration builder for NVIDIA NeMo diarization.

This module provides a clean interface for building and validating
diarization configurations with sensible defaults.
"""

import logging
from pathlib import Path
from typing import Dict, Any

from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)


class DiarizationConfigBuilder:
    """
    Builder for creating validated NeMo diarization configurations.
    
    Handles the complexity of NeMo configuration requirements and provides
    sensible defaults for common use cases.
    """

    def __init__(self, base_config_path: Path | None = None) -> None:
        """
        Initialize the configuration builder.
        
        Args:
            base_config_path: Path to base configuration YAML file
        """
        self.base_config_path = base_config_path
        self.device = "cuda" if self._cuda_available() else "cpu"
        
    def build_config(self, manifest_path: Path, output_dir: Path, 
                    audio_file_path: Path, duration: float) -> DictConfig:
        """
        Build a complete diarization configuration.
        
        Args:
            manifest_path: Path to manifest JSON file
            output_dir: Directory for diarization outputs
            audio_file_path: Path to the audio file being processed
            duration: Duration of the audio file in seconds
            
        Returns:
            Complete OmegaConf configuration for diarization
        """
        # Load base config if provided
        if self.base_config_path and self.base_config_path.exists():
            base_cfg = OmegaConf.load(self.base_config_path)
            cfg_dict = OmegaConf.to_container(base_cfg, resolve=True)
        else:
            # Use minimal default configuration
            cfg_dict = self._get_default_config()
            
        # Update with runtime parameters
        cfg_dict['diarizer']['manifest_filepath'] = str(manifest_path)
        cfg_dict['diarizer']['out_dir'] = str(output_dir)
        
        # Add device configuration
        cfg_dict = self._add_device_config(cfg_dict)
        
        # Validate and fix common issues
        cfg_dict = self._validate_and_fix_config(cfg_dict)
        
        # Create OmegaConf
        cfg = OmegaConf.create(cfg_dict)
        
        logger.debug(f"Built diarization config with keys: {list(cfg.keys())}")
        return cfg

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get a minimal working default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'verbose': True,
            'device': self.device,
            'diarizer': {
                'vad': {
                    'model_path': 'vad_multilingual_marblenet',
                    'window_length_in_sec': 0.15,
                    'shift_length_in_sec': 0.01,
                    'parameters': {
                        'onset': 0.8,
                        'offset': 0.6,
                        'min_duration_on': 0.1,
                        'min_duration_off': 0.1,
                        'filter_speech_first': True,
                        'smoothing': False  # Avoid smoothing config issues
                    }
                },
                'speaker_embeddings': {
                    'model_path': 'titanet_large',
                    'parameters': {
                        'window_length_in_sec': [1.5, 1.25, 1.0, 0.75, 0.5],
                        'shift_length_in_sec': [0.75, 0.625, 0.5, 0.375, 0.25],
                        'multiscale_weights': [1, 1, 1, 1, 1],
                        'save_embeddings': False
                    }
                },
                'clustering': {
                    'parameters': {
                        'oracle_num_speakers': None,
                        'enhanced_count_thres': 80,
                        'max_rp_threshold': 0.25,
                        'sparse_search_volume': 30
                    }
                },
                'msdd_model': {
                    'model_path': 'diar_msdd_telephonic',
                    'parameters': {
                        'use_speaker_model_from_ckpt': True,
                        'infer_batch_size': 25,
                        'sigmoid_threshold': [0.7],
                        'seq_eval_mode': False,
                        'split_infer': True
                    }
                },
                'collar': 0.25,
                'ignore_overlap': True
            }
        }

    def _add_device_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add device configuration to all relevant sections.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Updated configuration with device settings
        """
        # Add top-level device if not present
        if 'device' not in config:
            config['device'] = self.device
            
        # Add device to diarizer section
        if 'diarizer' in config and 'device' not in config['diarizer']:
            config['diarizer']['device'] = self.device
            
        # Add device to sub-components
        for component in ['speaker_embeddings', 'vad', 'msdd_model']:
            if component in config.get('diarizer', {}):
                if 'device' not in config['diarizer'][component]:
                    config['diarizer'][component]['device'] = self.device
                    
        return config

    def _validate_and_fix_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration and fix common issues.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated and fixed configuration
        """
        # Ensure required top-level keys
        if 'verbose' not in config:
            config['verbose'] = True
            
        # Ensure diarizer section exists
        if 'diarizer' not in config:
            config['diarizer'] = {}
            
        diarizer = config['diarizer']
        
        # Add missing collar parameter
        if 'collar' not in diarizer:
            diarizer['collar'] = 0.25
            logger.debug("Added missing collar parameter")
            
        # Add missing ignore_overlap parameter
        if 'ignore_overlap' not in diarizer:
            diarizer['ignore_overlap'] = True
            logger.debug("Added missing ignore_overlap parameter")
            
        # Fix VAD smoothing parameter
        if 'vad' in diarizer and 'parameters' in diarizer['vad']:
            vad_params = diarizer['vad']['parameters']
            if 'smoothing' not in vad_params:
                vad_params['smoothing'] = False
                logger.debug("Added missing VAD smoothing parameter")
                
        return config

    def _cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def create_manifest_entry(self, audio_path: Path, duration: float) -> Dict[str, Any]:
        """
        Create a manifest entry for the audio file.
        
        Args:
            audio_path: Path to audio file
            duration: Duration in seconds
            
        Returns:
            Manifest entry dictionary
        """
        return {
            "audio_filepath": str(audio_path),
            "duration": duration if duration else 100.0,  # Default fallback
            "label": "infer",
            "text": "-",
            "offset": 0.0
        }