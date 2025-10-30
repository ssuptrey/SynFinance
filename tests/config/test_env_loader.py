"""
Tests for EnvLoader

Week 7 Day 2: Configuration Management System
"""

import os
import pytest
import tempfile
from pathlib import Path
import time
import gc

from src.config.env_loader import EnvLoader


class TestEnvLoader:
    """Test EnvLoader functionality"""
    
    @pytest.fixture
    def temp_env_file(self):
        """Create temporary .env file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            env_path = Path(f.name)
        
        yield env_path
        
        # Cleanup - force garbage collection and add small delay for Windows
        gc.collect()
        time.sleep(0.01)
        
        try:
            if env_path.exists():
                env_path.unlink()
        except PermissionError:
            # On Windows, if file is still locked, try again after a short delay
            time.sleep(0.1)
            try:
                env_path.unlink(missing_ok=True)
            except PermissionError:
                pass  # Ignore if still can't delete - OS will clean up temp files
    
    @pytest.fixture
    def sample_env_content(self):
        """Sample .env file content"""
        return """
# Database settings
DB_HOST=localhost
DB_PORT=5432
DB_NAME=synfinance
DB_USER=admin
DB_PASSWORD=secret123

# Server settings
SERVER_HOST=0.0.0.0
SERVER_PORT=8000

# Feature flags
ENABLE_DEBUG=true
ENABLE_CACHE=false

# Numeric values
MAX_CONNECTIONS=100
TIMEOUT=30.5

# List values
ALLOWED_HOSTS=localhost,127.0.0.1,example.com

# Quoted values
API_KEY="sk-1234567890abcdef"
JWT_SECRET='very-secret-key-12345'

# Variable expansion
DATA_DIR=/var/data
LOG_DIR=${DATA_DIR}/logs
BACKUP_DIR=${DATA_DIR}/backups

# Default values
OPTIONAL_VAR=${MISSING_VAR:-default_value}
"""
    
    def test_load_env_file(self, temp_env_file, sample_env_content):
        """Test loading .env file"""
        # Write content
        with open(temp_env_file, 'w') as f:
            f.write(sample_env_content)
        
        # Load
        loader = EnvLoader(str(temp_env_file))
        variables = loader.load_env_file()
        
        # Verify
        assert len(variables) > 0
        assert variables["DB_HOST"] == "localhost"
        assert variables["DB_PORT"] == "5432"
        assert variables["DB_NAME"] == "synfinance"
        assert variables["DB_PASSWORD"] == "secret123"
    
    def test_get_methods(self, temp_env_file, sample_env_content):
        """Test get methods with type conversion"""
        # Write and load
        with open(temp_env_file, 'w') as f:
            f.write(sample_env_content)
        
        loader = EnvLoader(str(temp_env_file))
        loader.load_env_file()
        
        # String
        assert loader.get("DB_HOST") == "localhost"
        assert loader.get("MISSING", "default") == "default"
        
        # Integer
        assert loader.get_int("DB_PORT") == 5432
        assert loader.get_int("MAX_CONNECTIONS") == 100
        assert loader.get_int("MISSING", 999) == 999
        
        # Float
        assert loader.get_float("TIMEOUT") == 30.5
        assert loader.get_float("MISSING", 1.5) == 1.5
        
        # Boolean
        assert loader.get_bool("ENABLE_DEBUG") is True
        assert loader.get_bool("ENABLE_CACHE") is False
        assert loader.get_bool("MISSING", True) is True
        
        # List
        hosts = loader.get_list("ALLOWED_HOSTS")
        assert len(hosts) == 3
        assert "localhost" in hosts
        assert "127.0.0.1" in hosts
        assert "example.com" in hosts
    
    def test_quoted_values(self, temp_env_file, sample_env_content):
        """Test handling of quoted values"""
        with open(temp_env_file, 'w') as f:
            f.write(sample_env_content)
        
        loader = EnvLoader(str(temp_env_file))
        loader.load_env_file()
        
        # Double quotes removed
        assert loader.get("API_KEY") == "sk-1234567890abcdef"
        
        # Single quotes removed
        assert loader.get("JWT_SECRET") == "very-secret-key-12345"
    
    def test_variable_expansion(self, temp_env_file, sample_env_content):
        """Test variable expansion"""
        with open(temp_env_file, 'w') as f:
            f.write(sample_env_content)
        
        loader = EnvLoader(str(temp_env_file))
        loader.load_env_file()
        
        # Variable expansion
        assert loader.get("DATA_DIR") == "/var/data"
        assert loader.get("LOG_DIR") == "/var/data/logs"
        assert loader.get("BACKUP_DIR") == "/var/data/backups"
        
        # Default value expansion
        assert loader.get("OPTIONAL_VAR") == "default_value"
    
    def test_require_method(self, temp_env_file):
        """Test require method"""
        # Create minimal env file
        with open(temp_env_file, 'w') as f:
            f.write("REQUIRED_VAR=value123\n")
        
        loader = EnvLoader(str(temp_env_file))
        loader.load_env_file()
        
        # Exists
        assert loader.require("REQUIRED_VAR") == "value123"
        
        # Missing
        with pytest.raises(ValueError, match="Required environment variable not set"):
            loader.require("MISSING_VAR")
    
    def test_validate_required(self, temp_env_file):
        """Test validate_required method"""
        # Create env file
        with open(temp_env_file, 'w') as f:
            f.write("VAR1=value1\nVAR2=value2\n")
        
        loader = EnvLoader(str(temp_env_file))
        loader.load_env_file()
        
        # All present
        loader.validate_required(["VAR1", "VAR2"])
        
        # Some missing
        with pytest.raises(ValueError, match="Required environment variables not set"):
            loader.validate_required(["VAR1", "VAR2", "VAR3"])
    
    def test_substitute_in_config(self, temp_env_file):
        """Test configuration substitution"""
        # Create env file
        with open(temp_env_file, 'w') as f:
            f.write("API_URL=https://api.example.com\nAPI_KEY=secret123\n")
        
        loader = EnvLoader(str(temp_env_file))
        loader.load_env_file()
        
        # Test substitution
        config = {
            "api": {
                "url": "${API_URL}",
                "key": "${API_KEY}",
                "timeout": 30
            },
            "features": ["feature1", "feature2"]
        }
        
        result = loader.substitute_in_config(config)
        
        assert result["api"]["url"] == "https://api.example.com"
        assert result["api"]["key"] == "secret123"
        assert result["api"]["timeout"] == 30
        assert result["features"] == ["feature1", "feature2"]
    
    def test_create_env_template(self, temp_env_file):
        """Test .env template creation"""
        loader = EnvLoader()
        
        template_path = temp_env_file.parent / "template.env"
        loader.create_env_template(str(template_path), include_values=False)
        
        assert template_path.exists()
        
        # Read and verify
        with open(template_path, 'r') as f:
            content = f.read()
        
        assert "SYNFINANCE_ENV" in content
        assert "SYNFINANCE_DB_HOST" in content
        assert "SYNFINANCE_JWT_SECRET" in content
        
        # Cleanup
        template_path.unlink()
    
    def test_empty_env_file(self, temp_env_file):
        """Test handling of empty .env file"""
        # Create empty file
        temp_env_file.touch()
        
        loader = EnvLoader(str(temp_env_file))
        variables = loader.load_env_file()
        
        assert len(variables) == 0
        assert loader.is_loaded() is True
    
    def test_missing_env_file(self):
        """Test handling of missing .env file"""
        loader = EnvLoader("nonexistent.env")
        variables = loader.load_env_file()
        
        assert len(variables) == 0
        # File doesn't exist, so is_loaded should be False
        assert loader.is_loaded() is False
    
    def test_invalid_lines(self, temp_env_file):
        """Test handling of invalid lines"""
        # Create file with invalid lines
        with open(temp_env_file, 'w') as f:
            f.write("VALID=value\n")
            f.write("INVALID_NO_EQUALS\n")
            f.write("# Comment\n")
            f.write("\n")
            f.write("ANOTHER_VALID=another\n")
        
        loader = EnvLoader(str(temp_env_file))
        variables = loader.load_env_file()
        
        # Should load valid ones and skip invalid
        assert len(variables) == 2
        assert variables["VALID"] == "value"
        assert variables["ANOTHER_VALID"] == "another"
    
    def test_from_env_file(self, temp_env_file):
        """Test from_env_file class method"""
        # Create file
        with open(temp_env_file, 'w') as f:
            f.write("TEST_VAR=test_value\n")
        
        loader = EnvLoader.from_env_file(str(temp_env_file))
        
        assert loader.is_loaded() is True
        assert loader.get("TEST_VAR") == "test_value"
    
    def test_get_all(self, temp_env_file):
        """Test get_all method"""
        # Create file
        with open(temp_env_file, 'w') as f:
            f.write("VAR1=value1\nVAR2=value2\nVAR3=value3\n")
        
        loader = EnvLoader(str(temp_env_file))
        loader.load_env_file()
        
        all_vars = loader.get_all()
        
        assert len(all_vars) == 3
        assert all_vars["VAR1"] == "value1"
        assert all_vars["VAR2"] == "value2"
        assert all_vars["VAR3"] == "value3"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
