"""
Configuration management using OmegaConf for YAML-based settings.

This module provides centralized configuration management with validation,
defaults, and easy access to settings throughout the application.
"""

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from ..models.schemas import ProcessingConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages application configuration using OmegaConf and Pydantic validation.
    
    Provides methods to load, validate, and access configuration settings
    from YAML files with proper type checking and defaults.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to configuration file. If None, uses default.
        """
        self._config_path = config_path or Path("config/default.yaml")
        self._config: DictConfig | None = None
        self._validated_config: ProcessingConfig | None = None

    def load_config(self, config_path: Path | None = None) -> ProcessingConfig:
        """
        Load and validate configuration from YAML file.
        
        Args:
            config_path: Optional path to configuration file
            
        Returns:
            Validated ProcessingConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If config validation fails
        """
        if config_path:
            self._config_path = config_path

        # Load default configuration first
        default_config = self._get_default_config()

        # Load user configuration if file exists
        user_config = {}
        if self._config_path.exists():
            logger.info(f"Loading configuration from {self._config_path}")
            user_config = OmegaConf.load(self._config_path)
        else:
            logger.warning(f"Config file not found: {self._config_path}. Using defaults.")

        # Merge configurations (user overrides defaults)
        self._config = OmegaConf.merge(default_config, user_config)

        # Convert to regular dict for Pydantic validation
        config_dict = OmegaConf.to_container(self._config, resolve=True)

        try:
            # Validate with Pydantic
            self._validated_config = ProcessingConfig(**config_dict)
            logger.info("Configuration loaded and validated successfully")
            return self._validated_config

        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

    def get_config(self) -> ProcessingConfig:
        """
        Get the current validated configuration.
        
        Returns:
            Current ProcessingConfig instance
            
        Raises:
            RuntimeError: If configuration hasn't been loaded yet
        """
        if self._validated_config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._validated_config

    def save_config(self, config: ProcessingConfig, output_path: Path | None = None) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: ProcessingConfig instance to save
            output_path: Optional output path. If None, uses current config path.
        """
        save_path = output_path or self._config_path

        # Convert Pydantic model to dict
        config_dict = config.model_dump()

        # Convert Path objects to strings for YAML serialization
        config_dict = self._paths_to_strings(config_dict)

        # Create OmegaConf and save
        omega_config = OmegaConf.create(config_dict)

        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            OmegaConf.save(omega_config, f)

        logger.info(f"Configuration saved to {save_path}")

    def update_config(self, updates: dict[str, Any]) -> ProcessingConfig:
        """
        Update current configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            
        Returns:
            Updated ProcessingConfig instance
        """
        if self._validated_config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")

        # Get current config as dict
        current_dict = self._validated_config.model_dump()

        # Apply updates using OmegaConf merge
        current_omega = OmegaConf.create(current_dict)
        updates_omega = OmegaConf.create(updates)
        merged = OmegaConf.merge(current_omega, updates_omega)

        # Validate updated configuration
        updated_dict = OmegaConf.to_container(merged, resolve=True)
        self._validated_config = ProcessingConfig(**updated_dict)

        logger.info("Configuration updated successfully")
        return self._validated_config

    def _get_default_config(self) -> DictConfig:
        """
        Get default configuration as OmegaConf DictConfig.
        
        Returns:
            Default configuration
        """
        default_config = ProcessingConfig()
        config_dict = default_config.model_dump()
        config_dict = self._paths_to_strings(config_dict)
        return OmegaConf.create(config_dict)

    def _paths_to_strings(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Convert Path objects to strings recursively for YAML serialization.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Dictionary with Path objects converted to strings
        """
        result = {}
        for key, value in config_dict.items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, dict):
                result[key] = self._paths_to_strings(value)
            else:
                result[key] = value
        return result

    @property
    def config_path(self) -> Path:
        """Get the current configuration file path."""
        return self._config_path

    @property
    def is_loaded(self) -> bool:
        """Check if configuration has been loaded."""
        return self._validated_config is not None


def load_config(config_path: Path | None = None) -> ProcessingConfig:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Validated ProcessingConfig instance
    """
    manager = ConfigManager(config_path)
    return manager.load_config()
