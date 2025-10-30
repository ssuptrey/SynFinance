"""
Configuration Manager with Environment-Based Configs

Comprehensive configuration system with:
- Multi-environment support (dev/staging/prod)
- Pydantic validation
- YAML/JSON/ENV loading
- Type checking and defaults
- Configuration versioning

Week 7 Day 2: Configuration Management System
"""

import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Supported environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class LogLevel(str, Enum):
    """Supported log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# =============================================================================
# Configuration Models
# =============================================================================

class ServerConfig(BaseModel):
    """Server configuration"""
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1024, le=65535, description="Server port")
    workers: int = Field(default=4, ge=1, le=32, description="Number of worker processes")
    timeout: int = Field(default=60, ge=1, description="Request timeout in seconds")
    reload: bool = Field(default=False, description="Enable auto-reload (dev only)")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    max_request_size: int = Field(default=10485760, ge=1024, description="Max request size in bytes (default 10MB)")
    
    @field_validator('reload')
    @classmethod
    def validate_reload(cls, v: bool, info) -> bool:
        """Ensure reload is only enabled in development"""
        # Note: info.data is not available during construction
        return v
    
    class Config:
        use_enum_values = True


class DatabaseConfig(BaseModel):
    """Database configuration"""
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1024, le=65535, description="Database port")
    name: str = Field(default="synfinance", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(default="", description="Database password")
    pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, le=100, description="Max pool overflow")
    pool_timeout: int = Field(default=30, ge=1, description="Pool connection timeout")
    pool_recycle: int = Field(default=3600, ge=60, description="Pool recycle time in seconds")
    echo: bool = Field(default=False, description="Echo SQL queries")
    ssl_mode: str = Field(default="prefer", description="SSL mode")
    retry_attempts: int = Field(default=3, ge=1, description="Connection retry attempts")
    retry_delay: int = Field(default=1, ge=0, description="Retry delay in seconds")
    
    @property
    def url(self) -> str:
        """Get database URL"""
        password = f":{self.password}" if self.password else ""
        return f"postgresql://{self.user}{password}@{self.host}:{self.port}/{self.name}"


class CacheConfig(BaseModel):
    """Cache configuration"""
    enabled: bool = Field(default=True, description="Enable caching")
    backend: str = Field(default="memory", description="Cache backend (memory/redis)")
    host: str = Field(default="localhost", description="Redis host (if using Redis)")
    port: int = Field(default=6379, ge=1024, le=65535, description="Redis port")
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    max_size: int = Field(default=10000, ge=100, description="Max cache entries")
    ttl: int = Field(default=3600, ge=60, description="Default TTL in seconds")
    eviction_policy: str = Field(default="lru", description="Eviction policy (lru/lfu/fifo)")
    
    @field_validator('backend')
    @classmethod
    def validate_backend(cls, v: str) -> str:
        """Validate cache backend"""
        allowed = ["memory", "redis"]
        if v not in allowed:
            raise ValueError(f"Backend must be one of {allowed}")
        return v


class GenerationConfig(BaseModel):
    """Transaction generation configuration"""
    num_customers: int = Field(default=1000, ge=10, le=1000000, description="Number of customers")
    num_transactions: int = Field(default=10000, ge=100, le=10000000, description="Number of transactions")
    fraud_rate: float = Field(default=0.02, ge=0.0, le=1.0, description="Fraud rate (0.0-1.0)")
    anomaly_rate: float = Field(default=0.05, ge=0.0, le=1.0, description="Anomaly rate (0.0-1.0)")
    start_date: str = Field(default="2024-01-01", description="Start date (YYYY-MM-DD)")
    end_date: str = Field(default="2024-12-31", description="End date (YYYY-MM-DD)")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    parallel: bool = Field(default=True, description="Enable parallel generation")
    batch_size: int = Field(default=1000, ge=100, le=100000, description="Batch size for parallel processing")
    
    @model_validator(mode='after')
    def validate_rates(self) -> 'GenerationConfig':
        """Validate fraud and anomaly rates don't exceed 100%"""
        if self.fraud_rate + self.anomaly_rate > 1.0:
            raise ValueError("fraud_rate + anomaly_rate cannot exceed 1.0")
        return self


class MLConfig(BaseModel):
    """Machine learning configuration"""
    model_path: str = Field(default="models/", description="Path to model files")
    fraud_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Fraud detection threshold")
    anomaly_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Anomaly detection threshold")
    batch_size: int = Field(default=100, ge=1, le=10000, description="Prediction batch size")
    enable_gpu: bool = Field(default=False, description="Enable GPU acceleration")
    num_threads: int = Field(default=4, ge=1, le=32, description="Number of threads for inference")
    cache_predictions: bool = Field(default=True, description="Cache predictions")
    model_refresh_interval: int = Field(default=3600, ge=60, description="Model refresh interval in seconds")
    feature_engineering: Dict[str, Any] = Field(
        default={
            "enabled": True,
            "fraud_features": True,
            "anomaly_features": True,
            "interaction_features": True,
            "total_features": 69
        },
        description="Feature engineering settings"
    )


class MonitoringConfig(BaseModel):
    """Monitoring configuration"""
    enabled: bool = Field(default=True, description="Enable monitoring")
    prometheus_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    prometheus_port: int = Field(default=9090, ge=1024, le=65535, description="Prometheus port")
    grafana_enabled: bool = Field(default=True, description="Enable Grafana dashboards")
    grafana_port: int = Field(default=3000, ge=1024, le=65535, description="Grafana port")
    export_interval: int = Field(default=60, ge=1, description="Metrics export interval in seconds")
    retention_days: int = Field(default=30, ge=1, le=365, description="Metrics retention in days")
    alerting_enabled: bool = Field(default=True, description="Enable alerting")
    alert_webhook: Optional[str] = Field(default=None, description="Alert webhook URL")
    log_metrics: bool = Field(default=True, description="Log metrics to file")
    metrics_file: str = Field(default="logs/metrics.log", description="Metrics log file path")


class SecurityConfig(BaseModel):
    """Security configuration"""
    api_key_enabled: bool = Field(default=False, description="Enable API key authentication")
    api_keys: List[str] = Field(default=[], description="Valid API keys")
    jwt_enabled: bool = Field(default=False, description="Enable JWT authentication")
    jwt_secret: str = Field(default="", description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiry: int = Field(default=3600, ge=60, description="JWT expiry in seconds")
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, ge=1, description="Rate limit requests per window")
    rate_limit_window: int = Field(default=60, ge=1, description="Rate limit window in seconds")
    cors_enabled: bool = Field(default=True, description="Enable CORS")
    allowed_hosts: List[str] = Field(default=["*"], description="Allowed hosts")
    ssl_enabled: bool = Field(default=False, description="Enable SSL/TLS")
    ssl_cert_path: Optional[str] = Field(default=None, description="SSL certificate path")
    ssl_key_path: Optional[str] = Field(default=None, description="SSL private key path")
    
    @field_validator('jwt_secret')
    @classmethod
    def validate_jwt_secret(cls, v: str, info) -> str:
        """Validate JWT secret is set if JWT is enabled"""
        # Note: We can't check jwt_enabled here during construction
        # Validation will be done in model_validator
        return v
    
    @model_validator(mode='after')
    def validate_security_settings(self) -> 'SecurityConfig':
        """Validate security settings"""
        if self.jwt_enabled and not self.jwt_secret:
            raise ValueError("JWT secret must be set when JWT is enabled")
        if self.ssl_enabled and (not self.ssl_cert_path or not self.ssl_key_path):
            raise ValueError("SSL certificate and key paths must be set when SSL is enabled")
        return self


class AppConfig(BaseModel):
    """Application configuration"""
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Environment")
    debug: bool = Field(default=False, description="Debug mode")
    app_name: str = Field(default="SynFinance", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    server: ServerConfig = Field(default_factory=ServerConfig, description="Server configuration")
    database: DatabaseConfig = Field(default_factory=DatabaseConfig, description="Database configuration")
    cache: CacheConfig = Field(default_factory=CacheConfig, description="Cache configuration")
    generation: GenerationConfig = Field(default_factory=GenerationConfig, description="Generation configuration")
    ml: MLConfig = Field(default_factory=MLConfig, description="ML configuration")
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig, description="Monitoring configuration")
    security: SecurityConfig = Field(default_factory=SecurityConfig, description="Security configuration")
    
    @model_validator(mode='after')
    def validate_environment_settings(self) -> 'AppConfig':
        """Validate environment-specific settings"""
        # Production validations
        if self.environment == Environment.PRODUCTION:
            if self.debug:
                logger.warning("Debug mode should not be enabled in production")
            if self.server.reload:
                raise ValueError("Server reload must be disabled in production")
            if not self.security.api_key_enabled and not self.security.jwt_enabled:
                logger.warning("No authentication enabled in production")
        
        # Development defaults
        if self.environment == Environment.DEVELOPMENT:
            self.debug = True
            self.server.log_level = LogLevel.DEBUG
        
        return self
    
    class Config:
        use_enum_values = True


# =============================================================================
# Configuration Manager
# =============================================================================

class ConfigManager:
    """
    Configuration Manager
    
    Manages application configuration with:
    - Multi-environment support
    - YAML/JSON/ENV loading
    - Pydantic validation
    - Hot-reloading
    - Configuration versioning
    """
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[AppConfig] = None
    
    def __new__(cls) -> 'ConfigManager':
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize configuration manager"""
        if self._config is None:
            self._config_dir = Path("config")
            self._environment = self._get_environment()
            self._version = "1.0.0"
            logger.info(f"Initializing ConfigManager for environment: {self._environment}")
    
    def _get_environment(self) -> Environment:
        """Get current environment from ENV"""
        env_str = os.getenv("SYNFINANCE_ENV", "development").lower()
        try:
            return Environment(env_str)
        except ValueError:
            logger.warning(f"Invalid environment '{env_str}', defaulting to development")
            return Environment.DEVELOPMENT
    
    def load_config(
        self,
        config_path: Optional[Union[str, Path]] = None,
        environment: Optional[Environment] = None
    ) -> AppConfig:
        """
        Load configuration from file
        
        Args:
            config_path: Path to config file (optional, auto-detects based on environment)
            environment: Override environment
            
        Returns:
            Loaded and validated configuration
        """
        if environment:
            self._environment = environment
        
        # Determine config file path
        if config_path is None:
            config_path = self._config_dir / f"{self._environment.value}.yaml"
        else:
            config_path = Path(config_path)
        
        logger.info(f"Loading configuration from: {config_path}")
        
        # Load base config
        base_config_path = self._config_dir / "default.yaml"
        config_data = {}
        
        if base_config_path.exists():
            logger.debug(f"Loading base config from: {base_config_path}")
            config_data = self._load_yaml(base_config_path)
        
        # Override with environment-specific config
        if config_path.exists():
            logger.debug(f"Loading environment config from: {config_path}")
            env_config = self._load_yaml(config_path)
            config_data = self._merge_configs(config_data, env_config)
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")
        
        # Override with environment variables
        config_data = self._apply_env_overrides(config_data)
        
        # Set environment
        config_data['environment'] = self._environment.value
        
        # Validate and create config
        try:
            self._config = AppConfig(**config_data)
            logger.info(f"Configuration loaded successfully for {self._environment.value}")
            return self._config
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML file"""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading YAML from {path}: {e}")
            return {}
    
    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON file"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON from {path}: {e}")
            return {}
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configurations"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        # Common overrides
        env_mappings = {
            "SYNFINANCE_DEBUG": ("debug", bool),
            "SYNFINANCE_SERVER_HOST": ("server.host", str),
            "SYNFINANCE_SERVER_PORT": ("server.port", int),
            "SYNFINANCE_DB_HOST": ("database.host", str),
            "SYNFINANCE_DB_PORT": ("database.port", int),
            "SYNFINANCE_DB_NAME": ("database.name", str),
            "SYNFINANCE_DB_USER": ("database.user", str),
            "SYNFINANCE_DB_PASSWORD": ("database.password", str),
            "SYNFINANCE_REDIS_HOST": ("cache.host", str),
            "SYNFINANCE_REDIS_PORT": ("cache.port", int),
            "SYNFINANCE_JWT_SECRET": ("security.jwt_secret", str),
            "SYNFINANCE_API_KEYS": ("security.api_keys", list),
        }
        
        for env_var, (config_path, value_type) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(config, config_path, value, value_type)
        
        return config
    
    def _set_nested_value(
        self,
        config: Dict[str, Any],
        path: str,
        value: str,
        value_type: type
    ) -> None:
        """Set nested configuration value"""
        keys = path.split('.')
        current = config
        
        # Navigate to nested dict
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set value with type conversion
        final_key = keys[-1]
        try:
            if value_type == bool:
                current[final_key] = value.lower() in ('true', '1', 'yes', 'on')
            elif value_type == int:
                current[final_key] = int(value)
            elif value_type == float:
                current[final_key] = float(value)
            elif value_type == list:
                current[final_key] = value.split(',')
            else:
                current[final_key] = value
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to set {path}={value}: {e}")
    
    def get_config(self) -> AppConfig:
        """Get current configuration"""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def reload_config(self) -> AppConfig:
        """Reload configuration"""
        logger.info("Reloading configuration")
        return self.load_config()
    
    def validate_config(self, config_path: Union[str, Path]) -> bool:
        """
        Validate configuration file
        
        Args:
            config_path: Path to config file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            config_data = self._load_yaml(Path(config_path))
            AppConfig(**config_data)
            logger.info(f"Configuration validation passed: {config_path}")
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def export_config(self, output_path: Union[str, Path], format: str = "yaml") -> None:
        """
        Export current configuration
        
        Args:
            output_path: Output file path
            format: Output format (yaml/json)
        """
        if self._config is None:
            raise ValueError("No configuration loaded")
        
        output_path = Path(output_path)
        # Use mode='python' to serialize Enums as values
        config_dict = self._config.model_dump(mode='python')
        
        # Convert Enum values to strings
        config_dict = self._convert_enums_to_str(config_dict)
        
        if format == "yaml":
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        elif format == "json":
            with open(output_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Configuration exported to: {output_path}")
    
    def _convert_enums_to_str(self, obj: Any) -> Any:
        """Recursively convert Enum objects to their values"""
        from enum import Enum
        
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, dict):
            return {k: self._convert_enums_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_enums_to_str(item) for item in obj]
        else:
            return obj
    
    def get_environment(self) -> Environment:
        """Get current environment"""
        return self._environment
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self._environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self._environment == Environment.DEVELOPMENT
    
    @property
    def version(self) -> str:
        """Get configuration version"""
        return self._version


# Singleton instance
config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Get application configuration (convenience function)"""
    return config_manager.get_config()
