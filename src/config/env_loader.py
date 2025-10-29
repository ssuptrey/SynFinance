"""
Environment Variable Loader

Manages environment variables with:
- .env file loading
- Environment variable substitution
- Secret management integration
- Validation of required variables
- Default value handling

Week 7 Day 2: Configuration Management System
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EnvLoader:
    """
    Environment Variable Loader
    
    Loads and manages environment variables with:
    - .env file support
    - Variable substitution in configs
    - Secret management
    - Validation
    """
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize environment loader
        
        Args:
            env_file: Path to .env file (default: .env in project root)
        """
        self.env_file = Path(env_file) if env_file else Path(".env")
        self._variables: Dict[str, str] = {}
        self._loaded = False
    
    def load_env_file(self, env_file: Optional[Path] = None) -> Dict[str, str]:
        """
        Load environment variables from .env file
        
        Args:
            env_file: Path to .env file (optional)
            
        Returns:
            Dictionary of loaded variables
        """
        if env_file:
            self.env_file = env_file
        
        if not self.env_file.exists():
            logger.warning(f".env file not found: {self.env_file}")
            return {}
        
        logger.info(f"Loading environment variables from: {self.env_file}")
        variables = {}
        
        try:
            with open(self.env_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse KEY=VALUE
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes if present
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        
                        # Expand variables
                        value = self._expand_variables(value, variables)
                        
                        variables[key] = value
                        
                        # Set in os.environ
                        os.environ[key] = value
                        logger.debug(f"Loaded: {key}={value[:20]}..." if len(value) > 20 else f"Loaded: {key}={value}")
                    else:
                        logger.warning(f"Invalid line {line_num} in {self.env_file}: {line}")
        
        except Exception as e:
            logger.error(f"Error loading .env file: {e}")
            raise
        
        self._variables = variables
        self._loaded = True
        logger.info(f"Loaded {len(variables)} environment variables")
        
        return variables
    
    def _expand_variables(self, value: str, variables: Dict[str, str]) -> str:
        """
        Expand variable references in value
        
        Supports:
        - $VAR
        - ${VAR}
        - ${VAR:-default}
        
        Args:
            value: Value to expand
            variables: Available variables
            
        Returns:
            Expanded value
        """
        # Pattern: ${VAR} or ${VAR:-default}
        pattern = r'\$\{([^}:]+)(?::-([^}]+))?\}'
        
        def replace(match):
            var_name = match.group(1)
            default_value = match.group(2) or ''
            
            # Look in loaded variables first, then os.environ
            return variables.get(var_name) or os.getenv(var_name, default_value)
        
        expanded = re.sub(pattern, replace, value)
        
        # Pattern: $VAR (simple form)
        pattern_simple = r'\$([A-Z_][A-Z0-9_]*)'
        
        def replace_simple(match):
            var_name = match.group(1)
            return variables.get(var_name) or os.getenv(var_name, '')
        
        expanded = re.sub(pattern_simple, replace_simple, expanded)
        
        return expanded
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get environment variable
        
        Args:
            key: Variable name
            default: Default value if not found
            
        Returns:
            Variable value or default
        """
        return os.getenv(key, default)
    
    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """Get environment variable as integer"""
        value = self.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            logger.warning(f"Invalid integer value for {key}: {value}")
            return default
    
    def get_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """Get environment variable as float"""
        value = self.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            logger.warning(f"Invalid float value for {key}: {value}")
            return default
    
    def get_bool(self, key: str, default: Optional[bool] = None) -> Optional[bool]:
        """Get environment variable as boolean"""
        value = self.get(key)
        if value is None:
            return default
        return value.lower() in ('true', '1', 'yes', 'on')
    
    def get_list(self, key: str, separator: str = ',', default: Optional[List[str]] = None) -> Optional[List[str]]:
        """Get environment variable as list"""
        value = self.get(key)
        if value is None:
            return default or []
        return [item.strip() for item in value.split(separator) if item.strip()]
    
    def require(self, key: str) -> str:
        """
        Get required environment variable
        
        Args:
            key: Variable name
            
        Returns:
            Variable value
            
        Raises:
            ValueError: If variable not set
        """
        value = self.get(key)
        if value is None:
            raise ValueError(f"Required environment variable not set: {key}")
        return value
    
    def validate_required(self, required_vars: List[str]) -> None:
        """
        Validate that all required variables are set
        
        Args:
            required_vars: List of required variable names
            
        Raises:
            ValueError: If any required variable is missing
        """
        missing = []
        for var in required_vars:
            if not self.get(var):
                missing.append(var)
        
        if missing:
            raise ValueError(f"Required environment variables not set: {', '.join(missing)}")
    
    def substitute_in_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Substitute environment variables in configuration
        
        Recursively processes configuration dict and substitutes
        strings like ${VAR} with environment variable values.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with substituted values
        """
        if isinstance(config, dict):
            return {
                key: self.substitute_in_config(value)
                for key, value in config.items()
            }
        elif isinstance(config, list):
            return [self.substitute_in_config(item) for item in config]
        elif isinstance(config, str):
            return self._expand_variables(config, self._variables)
        else:
            return config
    
    def create_env_template(self, output_path: str, include_values: bool = False) -> None:
        """
        Create .env template file
        
        Args:
            output_path: Path to output file
            include_values: Include current values (default: False)
        """
        template_vars = [
            ("SYNFINANCE_ENV", "development", "Environment (development/staging/production/test)"),
            ("SYNFINANCE_DEBUG", "false", "Enable debug mode"),
            ("SYNFINANCE_SERVER_HOST", "0.0.0.0", "Server host"),
            ("SYNFINANCE_SERVER_PORT", "8000", "Server port"),
            ("SYNFINANCE_DB_HOST", "localhost", "Database host"),
            ("SYNFINANCE_DB_PORT", "5432", "Database port"),
            ("SYNFINANCE_DB_NAME", "synfinance", "Database name"),
            ("SYNFINANCE_DB_USER", "postgres", "Database user"),
            ("SYNFINANCE_DB_PASSWORD", "", "Database password"),
            ("SYNFINANCE_REDIS_HOST", "localhost", "Redis host"),
            ("SYNFINANCE_REDIS_PORT", "6379", "Redis port"),
            ("SYNFINANCE_REDIS_PASSWORD", "", "Redis password (if required)"),
            ("SYNFINANCE_JWT_SECRET", "", "JWT secret key (generate with: openssl rand -hex 32)"),
            ("SYNFINANCE_API_KEYS", "", "Comma-separated list of API keys"),
            ("SYNFINANCE_ALERT_WEBHOOK", "", "Alert webhook URL (optional)"),
        ]
        
        with open(output_path, 'w') as f:
            f.write("# SynFinance Environment Configuration\n")
            f.write("# Copy this file to .env and fill in the values\n\n")
            
            for var_name, default_value, description in template_vars:
                f.write(f"# {description}\n")
                if include_values:
                    current_value = self.get(var_name, default_value)
                    f.write(f"{var_name}={current_value}\n\n")
                else:
                    f.write(f"# {var_name}={default_value}\n\n")
        
        logger.info(f"Created .env template: {output_path}")
    
    def is_loaded(self) -> bool:
        """Check if environment variables have been loaded"""
        return self._loaded
    
    def get_all(self) -> Dict[str, str]:
        """Get all loaded variables"""
        return self._variables.copy()
    
    @staticmethod
    def from_env_file(env_file: str) -> 'EnvLoader':
        """
        Create EnvLoader and load from file
        
        Args:
            env_file: Path to .env file
            
        Returns:
            Loaded EnvLoader instance
        """
        loader = EnvLoader(env_file)
        loader.load_env_file()
        return loader


# Try to load .env file on import
try:
    from dotenv import load_dotenv
    # Load .env file if python-dotenv is installed
    load_dotenv()
    logger.debug("Loaded .env file with python-dotenv")
except ImportError:
    logger.debug("python-dotenv not installed, using custom loader")
