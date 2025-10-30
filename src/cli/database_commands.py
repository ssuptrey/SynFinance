"""
Database Management CLI Commands

Commands for database operations, migrations, and maintenance.

Week 7 Day 6: CLI Tools
"""

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def database():
    """Database management commands"""
    pass


@database.command()
def init():
    """Initialize database tables"""
    console.print("[bold blue]Initializing database...[/bold blue]")
    
    try:
        from src.database import get_db_manager
        
        db_manager = get_db_manager()
        db_manager.create_all_tables()
        
        console.print("[green]Database initialized successfully[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@database.command()
@click.confirmation_option(prompt='Are you sure you want to drop all tables?')
def drop():
    """Drop all database tables (DANGEROUS)"""
    console.print("[bold red]Dropping all tables...[/bold red]")
    
    try:
        from src.database import get_db_manager
        
        db_manager = get_db_manager()
        db_manager.drop_all_tables()
        
        console.print("[yellow]All tables dropped[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@database.command()
def status():
    """Check database connection status"""
    console.print("[bold blue]Checking database status...[/bold blue]")
    
    try:
        from src.database import get_db_manager
        
        db_manager = get_db_manager()
        
        # Health check
        is_healthy = db_manager.health_check()
        
        if is_healthy:
            console.print("[green]Database is healthy[/green]")
        else:
            console.print("[red]Database is not healthy[/red]")
        
        # Pool status
        pool_status = db_manager.get_pool_status()
        
        table = Table(title="Connection Pool Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in pool_status.items():
            table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@database.command()
@click.option('--table', '-t', help='Table name to query')
@click.option('--limit', '-l', default=10, help='Number of rows to show')
def query(table, limit):
    """Query database tables"""
    console.print(f"[bold blue]Querying {table or 'all tables'}...[/bold blue]")
    
    try:
        from src.database import get_db_manager, Transaction, Customer, Merchant
        
        db_manager = get_db_manager()
        
        with db_manager.session_scope() as session:
            if table == 'transactions' or not table:
                results = session.query(Transaction).limit(limit).all()
                console.print(f"\nTransactions: {len(results)} rows")
                for tx in results[:5]:
                    console.print(f"  {tx.transaction_id}: ${tx.amount} - Fraud: {tx.is_fraud}")
            
            if table == 'customers' or not table:
                results = session.query(Customer).limit(limit).all()
                console.print(f"\nCustomers: {len(results)} rows")
                for cust in results[:5]:
                    console.print(f"  {cust.customer_id}: {cust.first_name} {cust.last_name}")
            
            if table == 'merchants' or not table:
                results = session.query(Merchant).limit(limit).all()
                console.print(f"\nMerchants: {len(results)} rows")
                for merch in results[:5]:
                    console.print(f"  {merch.merchant_id}: {merch.name} ({merch.category})")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@database.command()
@click.option('--file', '-f', required=True, type=click.Path(exists=True), help='Data file to load')
@click.option('--table', '-t', required=True, type=click.Choice(['transactions', 'customers', 'merchants']), help='Target table')
def load(file, table):
    """Load data from file into database"""
    console.print(f"[bold blue]Loading {file} into {table}...[/bold blue]")
    
    try:
        import pandas as pd
        from src.database import get_db_manager, session_scope
        from src.database.repositories import TransactionRepository, CustomerRepository, MerchantRepository
        
        # Load data
        df = pd.read_csv(file)
        console.print(f"Loaded {len(df):,} rows from file")
        
        # Insert into database
        with session_scope() as session:
            if table == 'transactions':
                repo = TransactionRepository(session)
                data = df.to_dict('records')
                count = repo.bulk_create(data)
            elif table == 'customers':
                repo = CustomerRepository(session)
                data = df.to_dict('records')
                count = repo.bulk_create(data)
            elif table == 'merchants':
                repo = MerchantRepository(session)
                data = df.to_dict('records')
                count = repo.bulk_create(data)
        
        console.print(f"[green]Loaded {count:,} rows into {table}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()
