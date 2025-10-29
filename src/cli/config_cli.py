"""
Configuration CLI

Command-line interface for configuration management:
- Validate configuration files
- Display current configuration
- Update configuration values
- Compare configurations
- Export/import configurations

Week 7 Day 2: Configuration Management System
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.config.config_manager import ConfigManager, Environment, AppConfig
from src.config.env_loader import EnvLoader

logger = logging.getLogger(__name__)


class ConfigCLI:
    """
    Configuration Command-Line Interface
    
    Provides commands for managing configuration:
    - validate: Validate configuration files
    - show: Display current configuration
    - set: Update configuration values
    - diff: Compare configurations
    - export: Export configuration
    - import: Import configuration
    """
    
    def __init__(self):
        """Initialize configuration CLI"""
        self.config_manager = ConfigManager()
        self.env_loader = EnvLoader()
    
    def validate(self, config_path: str) -> bool:
        """
        Validate configuration file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            True if valid, False otherwise
        """
        print(f"Validating configuration: {config_path}")
        
        try:
            if self.config_manager.validate_config(config_path):
                print("Configuration is valid")
                return True
            else:
                print("Configuration validation failed")
                return False
        except Exception as e:
            print(f"Validation error: {e}")
            return False
    
    def show(
        self,
        environment: Optional[str] = None,
        format: str = "yaml",
        section: Optional[str] = None
    ) -> None:
        """
        Display current configuration
        
        Args:
            environment: Environment to show (default: current)
            format: Output format (yaml/json)
            section: Specific section to show (optional)
        """
        try:
            # Load config for specified environment
            if environment:
                env_enum = Environment(environment)
                config = self.config_manager.load_config(environment=env_enum)
            else:
                config = self.config_manager.get_config()
            
            # Get config dict
            config_dict = config.model_dump()
            
            # Extract specific section if requested
            if section:
                if section in config_dict:
                    config_dict = {section: config_dict[section]}
                else:
                    print(f"Section '{section}' not found")
                    print(f"Available sections: {', '.join(config_dict.keys())}")
                    return
            
            # Format output
            if format == "json":
                output = json.dumps(config_dict, indent=2)
            else:
                output = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
            
            print(output)
        
        except Exception as e:
            print(f"Error displaying configuration: {e}")
    
    def set_value(self, key: str, value: str, environment: Optional[str] = None) -> None:
        """
        Update configuration value
        
        Args:
            key: Configuration key (dot-separated path)
            value: New value
            environment: Environment to update (default: current)
        """
        try:
            # Determine config file to update
            if environment:
                env_enum = Environment(environment)
                config_file = Path("config") / f"{environment}.yaml"
            else:
                env_enum = self.config_manager.get_environment()
                config_file = Path("config") / f"{env_enum.value}.yaml"
            
            if not config_file.exists():
                print(f"Config file not found: {config_file}")
                return
            
            # Load current config
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f) or {}
            
            # Update value
            keys = key.split('.')
            current = config_data
            
            # Navigate to nested dict
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Set value with type conversion
            final_key = keys[-1]
            current[final_key] = self._convert_value(value)
            
            # Validate updated config
            print("Validating updated configuration...")
            if not self._validate_dict(config_data):
                print("Updated configuration is invalid")
                return
            
            # Save config
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            
            print(f"✅ Updated {key} = {value} in {config_file}")
        
        except Exception as e:
            print(f"Error updating configuration: {e}")
    
    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type"""
        # Try boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Try JSON (for lists/dicts)
        if value.startswith('[') or value.startswith('{'):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # Return as string
        return value
    
    def _validate_dict(self, config_dict: Dict[str, Any]) -> bool:
        """Validate configuration dictionary"""
        try:
            AppConfig(**config_dict)
            return True
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    def diff(self, env1: str, env2: str) -> None:
        """
        Compare two environment configurations
        
        Args:
            env1: First environment
            env2: Second environment
        """
        try:
            # Load configs
            config1 = self.config_manager.load_config(environment=Environment(env1))
            config2 = self.config_manager.load_config(environment=Environment(env2))
            
            # Get dicts
            dict1 = config1.model_dump()
            dict2 = config2.model_dump()
            
            # Find differences
            differences = self._find_differences(dict1, dict2)
            
            if not differences:
                print(f"No differences between {env1} and {env2}")
                return
            
            print(f"Differences between {env1} and {env2}:\n")
            print(f"{'Key':<40} {env1:<20} {env2:<20}")
            print("-" * 80)
            
            for key, (val1, val2) in differences.items():
                val1_str = str(val1)[:18] if val1 is not None else "N/A"
                val2_str = str(val2)[:18] if val2 is not None else "N/A"
                print(f"{key:<40} {val1_str:<20} {val2_str:<20}")
        
        except Exception as e:
            print(f"Error comparing configurations: {e}")
    
    def _find_differences(
        self,
        dict1: Dict[str, Any],
        dict2: Dict[str, Any],
        prefix: str = ""
    ) -> Dict[str, tuple]:
        """Recursively find differences between two dicts"""
        differences = {}
        
        # Check all keys in dict1
        for key, value1 in dict1.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if key not in dict2:
                differences[full_key] = (value1, None)
            elif isinstance(value1, dict) and isinstance(dict2[key], dict):
                # Recursive comparison for nested dicts
                nested_diff = self._find_differences(value1, dict2[key], full_key)
                differences.update(nested_diff)
            elif value1 != dict2[key]:
                differences[full_key] = (value1, dict2[key])
        
        # Check for keys only in dict2
        for key, value2 in dict2.items():
            if key not in dict1:
                full_key = f"{prefix}.{key}" if prefix else key
                differences[full_key] = (None, value2)
        
        return differences
    
    def export(self, output_path: str, format: str = "yaml") -> None:
        """
        Export current configuration
        
        Args:
            output_path: Output file path
            format: Output format (yaml/json)
        """
        try:
            self.config_manager.export_config(output_path, format)
            print(f"Configuration exported to: {output_path}")
        except Exception as e:
            print(f"Error exporting configuration: {e}")
    
    def import_config(self, input_path: str) -> None:
        """
        Import configuration from file
        
        Args:
            input_path: Input file path
        """
        try:
            input_file = Path(input_path)
            if not input_file.exists():
                print(f"File not found: {input_path}")
                return
            
            # Load config data
            if input_file.suffix == '.json':
                with open(input_file, 'r') as f:
                    config_data = json.load(f)
            else:
                with open(input_file, 'r') as f:
                    config_data = yaml.safe_load(f)
            
            # Validate
            print("Validating imported configuration...")
            if not self._validate_dict(config_data):
                print("Imported configuration is invalid")
                return
            
            # Determine target environment
            environment = config_data.get('environment', 'development')
            target_file = Path("config") / f"{environment}.yaml"
            
            # Confirm overwrite
            if target_file.exists():
                response = input(f"{target_file} already exists. Overwrite? (y/N): ")
                if response.lower() != 'y':
                    print("Import cancelled")
                    return
            
            # Save config
            with open(target_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            
            print(f"Configuration imported to: {target_file}")
        
        except Exception as e:
            print(f"Error importing configuration: {e}")
    
    def create_env_template(self, output_path: str = ".env.example") -> None:
        """
        Create .env template file
        
        Args:
            output_path: Output file path (default: .env.example)
        """
        try:
            self.env_loader.create_env_template(output_path, include_values=False)
            print(f"Created .env template: {output_path}")
        except Exception as e:
            print(f"Error creating .env template: {e}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="SynFinance Configuration Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration file')
    validate_parser.add_argument('config_path', help='Path to configuration file')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Display configuration')
    show_parser.add_argument('--env', dest='environment', help='Environment to show')
    show_parser.add_argument('--format', choices=['yaml', 'json'], default='yaml', help='Output format')
    show_parser.add_argument('--section', help='Specific section to show')
    
    # Set command
    set_parser = subparsers.add_parser('set', help='Update configuration value')
    set_parser.add_argument('key', help='Configuration key (dot-separated)')
    set_parser.add_argument('value', help='New value')
    set_parser.add_argument('--env', dest='environment', help='Environment to update')
    
    # Diff command
    diff_parser = subparsers.add_parser('diff', help='Compare configurations')
    diff_parser.add_argument('env1', help='First environment')
    diff_parser.add_argument('env2', help='Second environment')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export configuration')
    export_parser.add_argument('output_path', help='Output file path')
    export_parser.add_argument('--format', choices=['yaml', 'json'], default='yaml', help='Output format')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import configuration')
    import_parser.add_argument('input_path', help='Input file path')
    
    # Env-template command
    env_parser = subparsers.add_parser('env-template', help='Create .env template')
    env_parser.add_argument('--output', default='.env.example', help='Output file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s: %(message)s'
    )
    
    # Execute command
    cli = ConfigCLI()
    
    try:
        if args.command == 'validate':
            success = cli.validate(args.config_path)
            sys.exit(0 if success else 1)
        
        elif args.command == 'show':
            cli.show(args.environment, args.format, args.section)
        
        elif args.command == 'set':
            cli.set_value(args.key, args.value, args.environment)
        
        elif args.command == 'diff':
            cli.diff(args.env1, args.env2)
        
        elif args.command == 'export':
            cli.export(args.output_path, args.format)
        
        elif args.command == 'import':
            cli.import_config(args.input_path)
        
        elif args.command == 'env-template':
            cli.create_env_template(args.output)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
