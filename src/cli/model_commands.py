"""
Model Management CLI Commands

Commands for training, evaluating, and deploying ML models.

Week 7 Day 6: CLI Tools
"""

import click
from rich.console import Console
from rich.table import Table
from pathlib import Path
import json

console = Console()


@click.group()
def model():
    """Train and manage ML models"""
    pass


@model.command()
@click.option('--data', '-d', required=True, type=click.Path(exists=True), help='Training data file')
@click.option('--model-type', '-m', type=click.Choice(['random_forest', 'xgboost', 'logistic']), default='random_forest', help='Model type')
@click.option('--output', '-o', default='models/fraud_detector.pkl', help='Output model file')
@click.option('--test-size', default=0.2, help='Test set size (0.0-1.0)')
@click.option('--cv-folds', default=5, help='Cross-validation folds')
def train(data, model_type, output, test_size, cv_folds):
    """Train fraud detection model"""
    console.print(f"[bold blue]Training {model_type} model...[/bold blue]")
    
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        import xgboost as xgb
        import pickle
        
        # Load data
        console.print(f"Loading data from {data}...")
        df = pd.read_csv(data)
        console.print(f"Loaded {len(df):,} samples")
        
        # Prepare features and target
        if 'is_fraud' not in df.columns:
            console.print("[red]Error: 'is_fraud' column not found in data[/red]")
            raise click.Abort()
        
        # Remove non-feature columns
        exclude_cols = ['transaction_id', 'customer_id', 'merchant_id', 'timestamp', 'is_fraud', 'fraud_type']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].select_dtypes(include=['number']).fillna(0)
        y = df['is_fraud']
        
        console.print(f"Using {len(feature_cols)} features")
        console.print(f"Fraud rate: {y.mean():.2%}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        console.print(f"Train size: {len(X_train):,}, Test size: {len(X_test):,}")
        
        # Train model
        console.print(f"Training {model_type} model...")
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == 'xgboost':
            model = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:  # logistic
            model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        console.print(f"\n[bold]Training Results:[/bold]")
        console.print(f"Train accuracy: {train_score:.4f}")
        console.print(f"Test accuracy: {test_score:.4f}")
        
        # Cross-validation
        console.print(f"\nPerforming {cv_folds}-fold cross-validation...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, n_jobs=-1)
        console.print(f"CV scores: {cv_scores}")
        console.print(f"CV mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Save model
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        
        console.print(f"\n[green]Model saved to {output_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@model.command()
@click.option('--model', '-m', required=True, type=click.Path(exists=True), help='Model file')
@click.option('--data', '-d', required=True, type=click.Path(exists=True), help='Test data file')
@click.option('--output', '-o', help='Output report file (JSON)')
def evaluate(model, data, output):
    """Evaluate model performance"""
    console.print(f"[bold blue]Evaluating model {model}...[/bold blue]")
    
    try:
        import pandas as pd
        import pickle
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        # Load model
        with open(model, 'rb') as f:
            clf = pickle.load(f)
        
        # Load data
        df = pd.read_csv(data)
        
        # Prepare features
        exclude_cols = ['transaction_id', 'customer_id', 'merchant_id', 'timestamp', 'is_fraud', 'fraud_type']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].select_dtypes(include=['number']).fillna(0)
        y = df['is_fraud']
        
        # Predict
        console.print("Making predictions...")
        y_pred = clf.predict(X)
        y_prob = clf.predict_proba(X)[:, 1] if hasattr(clf, 'predict_proba') else y_pred
        
        # Metrics
        report = classification_report(y, y_pred, output_dict=True)
        cm = confusion_matrix(y, y_pred)
        roc_auc = roc_auc_score(y, y_prob)
        
        # Display results
        console.print("\n[bold]Classification Report:[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Class")
        table.add_column("Precision")
        table.add_column("Recall")
        table.add_column("F1-Score")
        table.add_column("Support")
        
        for label in ['0', '1']:
            if label in report:
                metrics = report[label]
                table.add_row(
                    f"Class {label}",
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['f1-score']:.4f}",
                    f"{int(metrics['support'])}"
                )
        
        console.print(table)
        
        console.print(f"\n[bold]Overall Metrics:[/bold]")
        console.print(f"Accuracy: {report['accuracy']:.4f}")
        console.print(f"ROC-AUC: {roc_auc:.4f}")
        
        console.print(f"\n[bold]Confusion Matrix:[/bold]")
        console.print(f"TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
        console.print(f"FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
        
        # Save report
        if output:
            results = {
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'roc_auc': float(roc_auc)
            }
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            console.print(f"\n[green]Report saved to {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@model.command()
@click.option('--model', '-m', required=True, type=click.Path(exists=True), help='Model file')
@click.option('--data', '-d', required=True, type=click.Path(exists=True), help='Data to predict')
@click.option('--output', '-o', help='Output predictions file')
def predict(model, data, output):
    """Make predictions on new data"""
    console.print(f"[bold blue]Making predictions...[/bold blue]")
    
    try:
        import pandas as pd
        import pickle
        
        # Load model
        with open(model, 'rb') as f:
            clf = pickle.load(f)
        
        # Load data
        df = pd.read_csv(data)
        
        # Prepare features
        exclude_cols = ['transaction_id', 'customer_id', 'merchant_id', 'timestamp', 'is_fraud', 'fraud_type']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].select_dtypes(include=['number']).fillna(0)
        
        # Predict
        console.print(f"Predicting {len(X):,} samples...")
        predictions = clf.predict(X)
        probabilities = clf.predict_proba(X)[:, 1] if hasattr(clf, 'predict_proba') else predictions
        
        # Add to dataframe
        df['fraud_prediction'] = predictions
        df['fraud_probability'] = probabilities
        
        # Statistics
        fraud_count = predictions.sum()
        console.print(f"\n[bold]Prediction Results:[/bold]")
        console.print(f"Total predictions: {len(predictions):,}")
        console.print(f"Predicted fraud: {fraud_count:,} ({fraud_count/len(predictions):.2%})")
        console.print(f"Predicted normal: {len(predictions) - fraud_count:,}")
        
        # Save
        if output:
            df.to_csv(output, index=False)
            console.print(f"\n[green]Predictions saved to {output}[/green]")
        else:
            console.print("\nSample predictions:")
            console.print(df[['transaction_id', 'fraud_prediction', 'fraud_probability']].head(10))
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@model.command()
def list():
    """List available models"""
    console.print("[bold blue]Available Models:[/bold blue]\n")
    
    models_dir = Path("models")
    if not models_dir.exists():
        console.print("[yellow]No models directory found[/yellow]")
        return
    
    model_files = list(models_dir.glob("*.pkl"))
    
    if not model_files:
        console.print("[yellow]No models found[/yellow]")
        return
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model Name")
    table.add_column("Size")
    table.add_column("Modified")
    
    for model_file in model_files:
        stat = model_file.stat()
        table.add_row(
            model_file.name,
            f"{stat.st_size / 1024:.1f} KB",
            f"{stat.st_mtime}"
        )
    
    console.print(table)
