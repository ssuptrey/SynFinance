"""
System Management CLI Commands

Commands for system health, monitoring, and maintenance.

Week 7 Day 6: CLI Tools
"""

import click
from rich.console import Console
from rich.table import Table
import sys

console = Console()


@click.group()
def system():
    """System management and health check commands"""
    pass


@system.command()
def health():
    """Check system health"""
    console.print("[bold blue]System Health Check[/bold blue]\n")
    
    from src.database import get_db_manager
    from src.observability import get_logger
    import psutil
    
    logger = get_logger("health_check")
    
    table = Table(title="System Health")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")
    
    # Database
    try:
        db_manager = get_db_manager()
        db_healthy = db_manager.health_check()
        table.add_row("Database", "OK" if db_healthy else "FAIL", "Connection pool active")
    except Exception as e:
        table.add_row("Database", "FAIL", str(e))
    
    # Memory
    memory = psutil.virtual_memory()
    table.add_row("Memory", "OK", f"{memory.percent}% used ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)")
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    table.add_row("CPU", "OK", f"{cpu_percent}% used")
    
    # Disk
    disk = psutil.disk_usage('/')
    table.add_row("Disk", "OK", f"{disk.percent}% used ({disk.used/1024**3:.1f}GB / {disk.total/1024**3:.1f}GB)")
    
    console.print(table)


@system.command()
def info():
    """Display system information"""
    console.print("[bold blue]System Information[/bold blue]\n")
    
    import platform
    import psutil
    
    table = Table(show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Python Version", sys.version.split()[0])
    table.add_row("Platform", platform.platform())
    table.add_row("Processor", platform.processor())
    table.add_row("CPU Cores", str(psutil.cpu_count()))
    table.add_row("Total Memory", f"{psutil.virtual_memory().total / 1024**3:.1f} GB")
    table.add_row("SynFinance Version", "1.0.0")
    
    console.print(table)


@system.command()
@click.option('--component', '-c', type=click.Choice(['cache', 'logs', 'all']), default='all')
def clean(component):
    """Clean system caches and temporary files"""
    console.print(f"[bold blue]Cleaning {component}...[/bold blue]")
    
    try:
        import shutil
        from pathlib import Path
        
        if component in ['cache', 'all']:
            cache_dirs = [
                Path('__pycache__'),
                Path('src/__pycache__'),
                Path('.pytest_cache')
            ]
            
            for cache_dir in cache_dirs:
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                    console.print(f"Cleaned {cache_dir}")
        
        if component in ['logs', 'all']:
            log_files = Path('.').glob('*.log')
            for log_file in log_files:
                log_file.unlink()
                console.print(f"Removed {log_file}")
        
        console.print("[green]Cleaning complete[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@system.command()
def config():
    """Display current configuration"""
    console.print("[bold blue]System Configuration[/bold blue]\n")
    
    from src.database import DatabaseConfig
    import os
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Setting")
    table.add_column("Value")
    
    # Database config
    db_config = DatabaseConfig.from_env()
    table.add_row("DB Host", db_config.host)
    table.add_row("DB Port", str(db_config.port))
    table.add_row("DB Name", db_config.database)
    table.add_row("DB Pool Size", str(db_config.pool_size))
    table.add_row("DB Max Overflow", str(db_config.max_overflow))
    
    # Environment
    table.add_row("Environment", os.getenv("ENVIRONMENT", "development"))
    
    console.print(table)


@system.command()
@click.option('--output', '-o', default='system_metrics.json', help='Output file')
def metrics(output):
    """Export system metrics"""
    console.print("[bold blue]Collecting system metrics...[/bold blue]")
    
    try:
        import json
        import psutil
        from datetime import datetime
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': psutil.cpu_percent(interval=1),
                'count': psutil.cpu_count()
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'used': psutil.disk_usage('/').used,
                'percent': psutil.disk_usage('/').percent
            }
        }
        
        with open(output, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        console.print(f"[green]Metrics exported to {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@system.command()
def version():
    """Display version information"""
    console.print("[bold cyan]SynFinance v1.0.0[/bold cyan]")
    console.print("Synthetic Financial Transaction Generator")
    console.print("Week 7 Complete - Production Ready")
