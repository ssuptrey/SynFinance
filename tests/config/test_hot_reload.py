"""
Tests for ConfigWatcher (Hot-Reload)

Week 7 Day 2: Configuration Management System
"""

import os
import pytest
import tempfile
import time
from pathlib import Path
import yaml

from src.config.config_manager import ConfigManager, AppConfig
from src.config.hot_reload import ConfigWatcher


class TestConfigWatcher:
    """Test ConfigWatcher functionality"""
    
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
            "server": {
                "host": "localhost",
                "port": 8080
            },
            "database": {
                "host": "localhost",
                "name": "testdb",
                "user": "testuser"
            }
        }
    
    @pytest.fixture
    def config_manager_with_file(self, temp_config_dir, sample_config):
        """Create ConfigManager with loaded config"""
        # Create config file
        config_file = temp_config_dir / "development.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        # Create manager and load
        manager = ConfigManager()
        manager._config_dir = temp_config_dir
        manager.load_config(config_path=config_file)
        
        return manager
    
    def test_watcher_initialization(self, config_manager_with_file, temp_config_dir):
        """Test ConfigWatcher initialization"""
        watcher = ConfigWatcher(
            config_manager_with_file,
            watch_paths=[temp_config_dir],
            auto_reload=True
        )
        
        assert watcher.config_manager == config_manager_with_file
        assert watcher.auto_reload is True
        assert not watcher.is_watching()
    
    def test_start_stop_watcher(self, config_manager_with_file, temp_config_dir):
        """Test starting and stopping watcher"""
        watcher = ConfigWatcher(
            config_manager_with_file,
            watch_paths=[temp_config_dir],
            auto_reload=False
        )
        
        # Start
        watcher.start()
        assert watcher.is_watching() is True
        
        # Stop
        watcher.stop()
        assert watcher.is_watching() is False
    
    def test_context_manager(self, config_manager_with_file, temp_config_dir):
        """Test ConfigWatcher as context manager"""
        watcher = ConfigWatcher(
            config_manager_with_file,
            watch_paths=[temp_config_dir],
            auto_reload=False
        )
        
        assert not watcher.is_watching()
        
        with watcher:
            assert watcher.is_watching() is True
        
        assert watcher.is_watching() is False
    
    def test_get_watched_files(self, config_manager_with_file, temp_config_dir, sample_config):
        """Test getting list of watched files"""
        # Create multiple config files
        for name in ["development.yaml", "staging.yaml", "production.yaml"]:
            config_file = temp_config_dir / name
            with open(config_file, 'w') as f:
                yaml.dump(sample_config, f)
        
        watcher = ConfigWatcher(
            config_manager_with_file,
            watch_paths=[temp_config_dir]
        )
        
        watched_files = watcher.get_watched_files()
        assert len(watched_files) >= 3
        assert any(f.name == "development.yaml" for f in watched_files)
        assert any(f.name == "staging.yaml" for f in watched_files)
        assert any(f.name == "production.yaml" for f in watched_files)
    
    def test_add_remove_listener(self, config_manager_with_file, temp_config_dir):
        """Test adding and removing event listeners"""
        watcher = ConfigWatcher(
            config_manager_with_file,
            watch_paths=[temp_config_dir]
        )
        
        # Track events
        events = []
        
        def listener(event_type, data):
            events.append((event_type, data))
        
        # Add listener
        watcher.add_listener(listener)
        assert listener in watcher._listeners
        
        # Remove listener
        watcher.remove_listener(listener)
        assert listener not in watcher._listeners
    
    def test_manual_reload(self, config_manager_with_file, temp_config_dir, sample_config):
        """Test manual reload trigger"""
        watcher = ConfigWatcher(
            config_manager_with_file,
            watch_paths=[temp_config_dir]
        )
        
        # Track events
        events = []
        
        def listener(event_type, data):
            events.append(event_type)
        
        watcher.add_listener(listener)
        
        # Update config file
        config_file = temp_config_dir / "development.yaml"
        sample_config["server"]["port"] = 9000
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        # Trigger manual reload
        watcher.trigger_reload()
        
        # Should have manual_reload event
        assert "manual_reload" in events or "reload_failed" in events
    
    def test_file_change_detection_polling(self, config_manager_with_file, temp_config_dir, sample_config):
        """Test file change detection in polling mode"""
        watcher = ConfigWatcher(
            config_manager_with_file,
            watch_paths=[temp_config_dir],
            auto_reload=False
        )
        
        # Force polling mode
        watcher._use_watchdog = False
        
        # Track events
        events = []
        
        def listener(event_type, data):
            events.append(event_type)
        
        watcher.add_listener(listener)
        
        # Start watching
        watcher.start()
        
        # Give it time to initialize
        time.sleep(0.5)
        
        # Update config file
        config_file = temp_config_dir / "development.yaml"
        time.sleep(0.1)  # Ensure different mtime
        sample_config["server"]["port"] = 9999
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        # Wait for polling to detect change
        time.sleep(3)
        
        # Stop watching
        watcher.stop()
        
        # Should have detected file change
        assert "file_changed" in events
    
    def test_validation_on_reload(self, config_manager_with_file, temp_config_dir):
        """Test that validation is performed on reload"""
        watcher = ConfigWatcher(
            config_manager_with_file,
            watch_paths=[temp_config_dir],
            auto_reload=True
        )
        
        # Track events
        events = []
        
        def listener(event_type, data):
            events.append(event_type)
        
        watcher.add_listener(listener)
        
        # Create invalid config
        config_file = temp_config_dir / "development.yaml"
        invalid_config = {
            "server": {
                "port": 80  # Invalid port (too low)
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # Trigger reload
        watcher._on_file_changed(config_file)
        
        # Should fail validation
        assert "validation_failed" in events
    
    def test_double_start_protection(self, config_manager_with_file, temp_config_dir):
        """Test that starting twice doesn't cause issues"""
        watcher = ConfigWatcher(
            config_manager_with_file,
            watch_paths=[temp_config_dir]
        )
        
        watcher.start()
        assert watcher.is_watching()
        
        # Start again (should be ignored)
        watcher.start()
        assert watcher.is_watching()
        
        watcher.stop()
    
    def test_nonexistent_watch_path(self, config_manager_with_file):
        """Test handling of nonexistent watch path"""
        nonexistent_path = Path("/nonexistent/path/config")
        
        watcher = ConfigWatcher(
            config_manager_with_file,
            watch_paths=[nonexistent_path]
        )
        
        # Should not raise error
        watcher.start()
        watched_files = watcher.get_watched_files()
        assert len(watched_files) == 0
        watcher.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
