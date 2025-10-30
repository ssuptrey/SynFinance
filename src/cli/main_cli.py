"""
SynFinance CLI - Main Command Line Interface

Comprehensive CLI for data generation, model training, database operations,
and system management.

Week 7 Day 6: CLI Tools
"""

import click
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """
    SynFinance - Synthetic Financial Transaction Generator
    
    A comprehensive fraud detection system for generating synthetic
    financial data and training machine learning models.
    """
    pass


# Import command groups
from src.cli.generate_commands import generate
from src.cli.model_commands import model
from src.cli.database_commands import database
from src.cli.system_commands import system

# Register command groups
cli.add_command(generate)
cli.add_command(model)
cli.add_command(database)
cli.add_command(system)


if __name__ == '__main__':
    cli()
