"""
Tests for ConfigManager

Week 7 Day 2: Configuration Management System
"""

import json
import os
import pytest
import tempfile
from pathlib import Path
import yaml

from src.config.config_manager import (
    ConfigManager,
    AppConfig,
    ServerConfig,
    DatabaseConfig,
    CacheConfig,
    GenerationConfig,
    MLConfig,
    MonitoringConfig,
    SecurityConfig,
    Environment,
    LogLevel,
)


class TestPydanticModels:
    """Test Pydantic configuration models"""
    
    def test_server_config_defaults(self):
        """Test ServerConfig default values"""
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.workers == 4
        assert config.timeout == 60
        assert config.reload is False
        assert config.log_level == LogLevel.INFO
    
    def test_server_config_validation(self):
        """Test ServerConfig validation"""
        # Valid port
        config = ServerConfig(port=8080)
        assert config.port == 8080
        
        # Invalid port (too low)
        with pytest.raises(Exception):
            ServerConfig(port=80)
        
        # Invalid port (too high)
        with pytest.raises(Exception):
            ServerConfig(port=70000)
    
    def test_database_config_url(self):
        """Test DatabaseConfig URL generation"""
        config = DatabaseConfig(
            host="db.example.com",
            port=5432,
            name="testdb",
            user="testuser",
            password="testpass"
        )
        assert config.url == "postgresql://testuser:testpass@db.example.com:5432/testdb"
        
        # Without password
        config_no_pass = DatabaseConfig(
            host="localhost",
            name="testdb",
            user="user"
        )
        assert config_no_pass.url == "postgresql://user@localhost:5432/testdb"
    
    def test_cache_config_backend_validation(self):
        """Test CacheConfig backend validation"""
        # Valid backends
        config_memory = CacheConfig(backend="memory")
        assert config_memory.backend == "memory"
        
        config_redis = CacheConfig(backend="redis")
        assert config_redis.backend == "redis"
        
        # Invalid backend
        with pytest.raises(Exception):
            CacheConfig(backend="invalid")
    
    def test_generation_config_rate_validation(self):
        """Test GenerationConfig rate validation"""
        # Valid rates
        config = GenerationConfig(fraud_rate=0.02, anomaly_rate=0.05)
        assert config.fraud_rate == 0.02
        assert config.anomaly_rate == 0.05
        
        # Invalid: rates sum > 1.0
        with pytest.raises(Exception):
            GenerationConfig(fraud_rate=0.6, anomaly_rate=0.6)
    
    def test_ml_config_defaults(self):
        """Test MLConfig default values"""
        config = MLConfig()
        assert config.model_path == "models/"
        assert config.fraud_threshold == 0.5
        assert config.enable_gpu is False
        assert config.feature_engineering["total_features"] == 69
    
    def test_security_config_jwt_validation(self):
        """Test SecurityConfig JWT validation"""
        # JWT enabled but no secret - should raise error
        with pytest.raises(Exception):
            SecurityConfig(jwt_enabled=True, jwt_secret="")
    
    def test_security_config_ssl_validation(self):
        """Test SecurityConfig SSL validation"""
        # SSL enabled but no cert paths - should raise error
        with pytest.raises(Exception):
            SecurityConfig(ssl_enabled=True)
    
    def test_app_config_defaults(self):
        """Test AppConfig default values"""
        config = AppConfig()
        assert config.environment == Environment.DEVELOPMENT
        assert config.app_name == "SynFinance"
        assert config.version == "1.0.0"
        assert isinstance(config.server, ServerConfig)
        assert isinstance(config.database, DatabaseConfig)


class TestConfigManager:
    """Test ConfigManager functionality"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()
            yield config_dir
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration"""
        return {
            "app_name": "TestApp",
            "version": "1.0.0",
            "debug": True,
            "server": {
                "host": "localhost",
                "port": 8080,
                "workers": 2
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "testdb",
                "user": "testuser",
                "password": "testpass"
            }
        }
    
    def test_config_manager_singleton(self):
        """Test ConfigManager is singleton"""
        manager1 = ConfigManager()
        manager2 = ConfigManager()
        assert manager1 is manager2
    
    def test_load_yaml_config(self, temp_config_dir, sample_config):
        """Test loading YAML configuration"""
        # Create config file
        config_file = temp_config_dir / "test.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        # Load config
        manager = ConfigManager()
        manager._config_dir = temp_config_dir
        config = manager.load_config(config_path=config_file)
        
        assert config.app_name == "TestApp"
        assert config.server.host == "localhost"
        assert config.server.port == 8080
    
    def test_environment_detection(self):
        """Test environment detection from ENV"""
        # Set environment variable
        os.environ["SYNFINANCE_ENV"] = "production"
        
        manager = ConfigManager()
        env = manager._get_environment()
        assert env == Environment.PRODUCTION
        
        # Cleanup
        del os.environ["SYNFINANCE_ENV"]
    
    def test_config_merge(self, temp_config_dir):
        """Test configuration merging (base + environment)"""
        # Create base config
        base_config = {
            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 4
            }
        }
        base_file = temp_config_dir / "default.yaml"
        with open(base_file, 'w') as f:
            yaml.dump(base_config, f)
        
        # Create env config
        env_config = {
            "server": {
                "port": 9000,
                "workers": 2
            }
        }
        env_file = temp_config_dir / "development.yaml"
        with open(env_file, 'w') as f:
            yaml.dump(env_config, f)
        
        # Load config
        manager = ConfigManager()
        manager._config_dir = temp_config_dir
        config = manager.load_config(config_path=env_file)
        
        # Check merged values
        assert config.server.host == "0.0.0.0"  # From base
        assert config.server.port == 9000  # From env (overridden)
        assert config.server.workers == 2  # From env (overridden)
    
    def test_env_variable_override(self, temp_config_dir, sample_config):
        """Test environment variable overrides"""
        # Set environment variables
        os.environ["SYNFINANCE_SERVER_PORT"] = "7000"
        os.environ["SYNFINANCE_DB_PASSWORD"] = "secret"
        
        # Create config file
        config_file = temp_config_dir / "test.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        # Load config
        manager = ConfigManager()
        manager._config_dir = temp_config_dir
        config = manager.load_config(config_path=config_file)
        
        # Check overrides
        assert config.server.port == 7000
        assert config.database.password == "secret"
        
        # Cleanup
        del os.environ["SYNFINANCE_SERVER_PORT"]
        del os.environ["SYNFINANCE_DB_PASSWORD"]
    
    def test_validate_config(self, temp_config_dir, sample_config):
        """Test configuration validation"""
        # Valid config
        valid_file = temp_config_dir / "valid.yaml"
        with open(valid_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        manager = ConfigManager()
        manager._config_dir = temp_config_dir
        assert manager.validate_config(valid_file) is True
        
        # Invalid config
        invalid_config = {"server": {"port": 80}}  # Port too low
        invalid_file = temp_config_dir / "invalid.yaml"
        with open(invalid_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        assert manager.validate_config(invalid_file) is False
    
    def test_export_config_yaml(self, temp_config_dir, sample_config):
        """Test exporting configuration as YAML"""
        # Load config
        config_file = temp_config_dir / "test.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        manager = ConfigManager()
        manager._config_dir = temp_config_dir
        manager.load_config(config_path=config_file)
        
        # Export
        export_file = temp_config_dir / "export.yaml"
        manager.export_config(export_file, format="yaml")
        
        # Verify
        assert export_file.exists()
        with open(export_file, 'r') as f:
            exported_data = yaml.safe_load(f)
        assert "server" in exported_data
        assert "database" in exported_data
    
    def test_export_config_json(self, temp_config_dir, sample_config):
        """Test exporting configuration as JSON"""
        # Load config
        config_file = temp_config_dir / "test.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        manager = ConfigManager()
        manager._config_dir = temp_config_dir
        manager.load_config(config_path=config_file)
        
        # Export
        export_file = temp_config_dir / "export.json"
        manager.export_config(export_file, format="json")
        
        # Verify
        assert export_file.exists()
        with open(export_file, 'r') as f:
            exported_data = json.load(f)
        assert "server" in exported_data
        assert "database" in exported_data
    
    def test_production_validation(self, temp_config_dir):
        """Test production environment validations"""
        # Production config with reload enabled (should raise error)
        prod_config = {
            "server": {
                "reload": True  # Not allowed in production
            }
        }
        
        config_file = temp_config_dir / "production.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(prod_config, f)
        
        # Set environment to production
        import os
        old_env = os.environ.get('SYNFINANCE_ENV')
        os.environ['SYNFINANCE_ENV'] = 'production'
        
        try:
            # Reset singleton to pick up new environment
            ConfigManager._instance = None
            ConfigManager._config = None
            
            manager = ConfigManager()
            manager._config_dir = temp_config_dir
            
            with pytest.raises(ValueError, match="Server reload must be disabled"):
                manager.load_config(config_path=config_file)
        finally:
            # Restore original environment
            if old_env:
                os.environ['SYNFINANCE_ENV'] = old_env
            else:
                os.environ.pop('SYNFINANCE_ENV', None)
            
            # Reset singleton again to restore original state
            ConfigManager._instance = None
            ConfigManager._config = None
    
    def test_get_config(self):
        """Test get_config convenience function"""
        from src.config.config_manager import get_config
        
        config = get_config()
        assert isinstance(config, AppConfig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
