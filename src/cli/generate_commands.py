"""
Data Generation CLI Commands

Commands for generating synthetic financial data.

Week 7 Day 6: CLI Tools
"""

import click
from rich.console import Console
from rich.progress import track
from pathlib import Path
import json

console = Console()


@click.group()
def generate():
    """Generate synthetic financial data"""
    pass


@generate.command()
@click.option('--count', '-n', default=1000, help='Number of transactions to generate')
@click.option('--fraud-rate', '-f', default=0.02, help='Fraud rate (0.0-1.0)')
@click.option('--anomaly-rate', '-a', default=0.05, help='Anomaly rate (0.0-1.0)')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', '-fmt', type=click.Choice(['csv', 'json', 'parquet']), default='csv', help='Output format')
@click.option('--seed', type=int, help='Random seed for reproducibility')
def transactions(count, fraud_rate, anomaly_rate, output, format, seed):
    """Generate synthetic transactions"""
    from src.data_generator import SyntheticDataGenerator
    
    console.print(f"[bold blue]Generating {count:,} transactions...[/bold blue]")
    console.print(f"Fraud rate: {fraud_rate:.2%}, Anomaly rate: {anomaly_rate:.2%}")
    
    try:
        generator = SyntheticDataGenerator(
            fraud_rate=fraud_rate,
            anomaly_rate=anomaly_rate,
            random_seed=seed
        )
        
        # Generate with progress bar
        transactions_list = []
        for i in track(range(count), description="Generating"):
            tx = generator.generate_transaction()
            transactions_list.append(tx.to_dict())
        
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame(transactions_list)
        
        # Save to file
        if output:
            output_path = Path(output)
            if format == 'csv':
                df.to_csv(output_path, index=False)
            elif format == 'json':
                df.to_json(output_path, orient='records', indent=2)
            elif format == 'parquet':
                df.to_parquet(output_path, index=False)
            
            console.print(f"[green]Saved {count:,} transactions to {output_path}[/green]")
        else:
            console.print(df.head(10))
        
        # Print statistics
        fraud_count = df['is_fraud'].sum()
        anomaly_count = df['is_anomaly'].sum()
        
        console.print(f"\n[bold]Statistics:[/bold]")
        console.print(f"Total transactions: {count:,}")
        console.print(f"Fraud transactions: {fraud_count:,} ({fraud_count/count:.2%})")
        console.print(f"Anomaly transactions: {anomaly_count:,} ({anomaly_count/count:.2%})")
        console.print(f"Normal transactions: {count - fraud_count:,}")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@generate.command()
@click.option('--count', '-n', default=100, help='Number of customers to generate')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def customers(count, output):
    """Generate synthetic customer profiles"""
    from src.customer_generator import CustomerGenerator
    
    console.print(f"[bold blue]Generating {count:,} customers...[/bold blue]")
    
    try:
        generator = CustomerGenerator()
        
        customers_list = []
        for i in track(range(count), description="Generating"):
            customer = generator.generate_customer()
            customers_list.append(customer.to_dict())
        
        import pandas as pd
        df = pd.DataFrame(customers_list)
        
        if output:
            output_path = Path(output)
            df.to_csv(output_path, index=False)
            console.print(f"[green]Saved {count:,} customers to {output_path}[/green]")
        else:
            console.print(df.head(10))
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@generate.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), help='Input transactions file')
@click.option('--output', '-o', type=click.Path(), help='Output features file')
def features(input, output):
    """Generate ML features from transactions"""
    from src.generators.feature_generator import FeatureGenerator
    import pandas as pd
    
    console.print(f"[bold blue]Generating features from {input}...[/bold blue]")
    
    try:
        # Load transactions
        df = pd.read_csv(input)
        console.print(f"Loaded {len(df):,} transactions")
        
        # Generate features
        generator = FeatureGenerator()
        features_list = []
        
        for _, tx in track(df.iterrows(), total=len(df), description="Extracting features"):
            features = generator.generate_features(tx)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        if output:
            output_path = Path(output)
            features_df.to_csv(output_path, index=False)
            console.print(f"[green]Saved {len(features_df):,} feature sets to {output_path}[/green]")
        else:
            console.print(features_df.head(10))
        
        console.print(f"\nGenerated {len(features_df.columns)} features")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@generate.command()
@click.option('--count', '-n', default=10000, help='Number of transactions')
@click.option('--output', '-o', default='output/complete_dataset.csv', help='Output file')
@click.option('--with-features', is_flag=True, help='Include ML features')
@click.option('--with-predictions', is_flag=True, help='Include model predictions')
def dataset(count, output, with_features, with_predictions):
    """Generate complete dataset with transactions, features, and predictions"""
    console.print(f"[bold blue]Generating complete dataset ({count:,} transactions)...[/bold blue]")
    
    try:
        from src.data_generator import SyntheticDataGenerator
        import pandas as pd
        
        # Generate transactions
        generator = SyntheticDataGenerator()
        transactions_list = []
        
        for i in track(range(count), description="Generating transactions"):
            tx = generator.generate_transaction()
            transactions_list.append(tx.to_dict())
        
        df = pd.DataFrame(transactions_list)
        
        # Generate features if requested
        if with_features:
            from src.generators.feature_generator import FeatureGenerator
            console.print("[yellow]Generating features...[/yellow]")
            feature_gen = FeatureGenerator()
            
            features_list = []
            for _, tx in track(df.iterrows(), total=len(df), description="Extracting features"):
                features = feature_gen.generate_features(tx)
                features_list.append(features)
            
            features_df = pd.DataFrame(features_list)
            df = pd.concat([df, features_df], axis=1)
        
        # Generate predictions if requested
        if with_predictions:
            console.print("[yellow]Generating predictions...[/yellow]")
            # This would use the ML model
            console.print("[yellow]Prediction generation requires trained model (skipped)[/yellow]")
        
        # Save
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        console.print(f"[green]Complete dataset saved to {output_path}[/green]")
        console.print(f"Total columns: {len(df.columns)}")
        console.print(f"Total rows: {len(df):,}")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()
