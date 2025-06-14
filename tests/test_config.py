"""
Unit tests for configuration management.

Tests the ConfigManager class for loading, validation, and updating configurations.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.audio_aigented.config.manager import ConfigManager, load_config
from src.audio_aigented.models.schemas import ProcessingConfig


class TestConfigManager:
    """Tests for ConfigManager class."""
    
    def test_config_manager_load_default_success(self):
        """Test successful loading of default configuration."""
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        assert isinstance(config, ProcessingConfig)
        assert config.input_dir == Path("./inputs")
        assert config.output_dir == Path("./outputs")
        assert config.audio["sample_rate"] == 16000
        assert config.transcription["model_name"] == "stt_en_conformer_ctc_large"
        
    def test_config_manager_load_custom_config_success(self):
        """Test loading custom configuration file."""
        # Create temporary config file
        config_content = """
input_dir: "./custom_inputs"
output_dir: "./custom_outputs"
audio:
  sample_rate: 22050
  batch_size: 16
transcription:
  model_name: "custom_model"
  device: "cpu"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            tmp_file.write(config_content)
            tmp_path = Path(tmp_file.name)
            
        try:
            config_manager = ConfigManager(tmp_path)
            config = config_manager.load_config()
            
            assert config.input_dir == Path("./custom_inputs")
            assert config.output_dir == Path("./custom_outputs")
            assert config.audio["sample_rate"] == 22050
            assert config.audio["batch_size"] == 16
            assert config.transcription["model_name"] == "custom_model"
            assert config.transcription["device"] == "cpu"
            
        finally:
            # Clean up
            if tmp_path.exists():
                tmp_path.unlink()
                
    def test_config_manager_nonexistent_file_uses_defaults(self):
        """Test that nonexistent config file falls back to defaults (edge case)."""
        nonexistent_path = Path("/nonexistent/config.yaml")
        config_manager = ConfigManager(nonexistent_path)
        
        # Should not raise error, should use defaults
        config = config_manager.load_config()
        
        assert isinstance(config, ProcessingConfig)
        assert config.input_dir == Path("./inputs")  # Default value
        
    def test_config_manager_invalid_yaml_failure(self):
        """Test that invalid YAML configuration fails (failure case)."""
        # Create invalid YAML file
        invalid_yaml = """
input_dir: "./test"
audio:
  sample_rate: invalid_number
  nested:
    - item1
    - item2
    invalid: [unclosed list
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            tmp_file.write(invalid_yaml)
            tmp_path = Path(tmp_file.name)
            
        try:
            config_manager = ConfigManager(tmp_path)
            
            # Should raise an error due to invalid YAML
            with pytest.raises(Exception):
                config_manager.load_config()
                
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
                
    def test_config_manager_get_config_before_load_failure(self):
        """Test getting config before loading fails (failure case)."""
        config_manager = ConfigManager()
        
        with pytest.raises(RuntimeError, match="Configuration not loaded"):
            config_manager.get_config()
            
    def test_config_manager_update_config_success(self):
        """Test successful configuration update."""
        config_manager = ConfigManager()
        config_manager.load_config()
        
        updates = {
            "audio": {"sample_rate": 32000},
            "transcription": {"device": "cpu"}
        }
        
        updated_config = config_manager.update_config(updates)
        
        assert updated_config.audio["sample_rate"] == 32000
        assert updated_config.transcription["device"] == "cpu"
        # Other values should remain unchanged
        assert updated_config.transcription["model_name"] == "stt_en_conformer_ctc_large"
        
    def test_config_manager_save_config_success(self):
        """Test successful configuration saving."""
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        # Modify config
        config.audio["sample_rate"] = 48000
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as tmp_file:
            save_path = Path(tmp_file.name)
            
        try:
            config_manager.save_config(config, save_path)
            
            # Verify file was created and has content
            assert save_path.exists()
            
            # Load saved config and verify
            new_manager = ConfigManager(save_path)
            loaded_config = new_manager.load_config()
            
            assert loaded_config.audio["sample_rate"] == 48000
            
        finally:
            if save_path.exists():
                save_path.unlink()
                
    def test_config_manager_is_loaded_property(self):
        """Test is_loaded property (edge case)."""
        config_manager = ConfigManager()
        
        assert config_manager.is_loaded is False
        
        config_manager.load_config()
        
        assert config_manager.is_loaded is True
        
    def test_config_manager_config_path_property(self):
        """Test config_path property."""
        custom_path = Path("./custom_config.yaml")
        config_manager = ConfigManager(custom_path)
        
        assert config_manager.config_path == custom_path


class TestLoadConfigFunction:
    """Tests for the standalone load_config function."""
    
    def test_load_config_function_success(self):
        """Test successful config loading via convenience function."""
        config = load_config()
        
        assert isinstance(config, ProcessingConfig)
        assert config.input_dir == Path("./inputs")
        
    def test_load_config_function_with_path(self):
        """Test config loading with custom path."""
        config_content = """
input_dir: "./function_test"
audio:
  sample_rate: 8000
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            tmp_file.write(config_content)
            tmp_path = Path(tmp_file.name)
            
        try:
            config = load_config(tmp_path)
            
            assert config.input_dir == Path("./function_test")
            assert config.audio["sample_rate"] == 8000
            
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
                
    def test_load_config_function_nonexistent_path_failure(self):
        """Test config loading function with nonexistent path (failure handled gracefully)."""
        nonexistent_path = Path("/nonexistent/path.yaml")
        
        # Should not raise error, should use defaults
        config = load_config(nonexistent_path)
        
        assert isinstance(config, ProcessingConfig)
        assert config.input_dir == Path("./inputs")  # Default value


class TestConfigValidation:
    """Tests for configuration validation edge cases."""
    
    def test_config_with_missing_sections(self):
        """Test configuration with missing sections uses defaults (edge case)."""
        partial_config = """
input_dir: "./partial"
# Missing audio, transcription, output, processing sections
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            tmp_file.write(partial_config)
            tmp_path = Path(tmp_file.name)
            
        try:
            config_manager = ConfigManager(tmp_path)
            config = config_manager.load_config()
            
            # Should have custom input_dir but default other values
            assert config.input_dir == Path("./partial")
            assert config.audio["sample_rate"] == 16000  # Default
            assert config.transcription["model_name"] == "stt_en_conformer_ctc_large"  # Default
            
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
                
    def test_config_with_invalid_types_failure(self):
        """Test configuration with invalid data types fails (failure case)."""
        invalid_config = """
input_dir: 123  # Should be string/path
audio:
  sample_rate: "not_a_number"  # Should be integer
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            tmp_file.write(invalid_config)
            tmp_path = Path(tmp_file.name)
            
        try:
            config_manager = ConfigManager(tmp_path)
            
            # Should raise validation error
            with pytest.raises(Exception):  # Could be ValidationError or other
                config_manager.load_config()
                
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
                
    def test_config_paths_to_strings_conversion(self):
        """Test internal path to string conversion (edge case)."""
        config_manager = ConfigManager()
        
        # Test the internal method
        test_dict = {
            "path_value": Path("./test/path"),
            "string_value": "regular_string",
            "nested": {
                "nested_path": Path("./nested/path"),
                "nested_string": "nested_value"
            },
            "number": 42
        }
        
        result = config_manager._paths_to_strings(test_dict)
        
        assert result["path_value"] == "./test/path"
        assert result["string_value"] == "regular_string"
        assert result["nested"]["nested_path"] == "./nested/path"
        assert result["nested"]["nested_string"] == "nested_value"
        assert result["number"] == 42