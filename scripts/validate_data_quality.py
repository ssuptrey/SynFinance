"""
Data Quality Validation for ML Datasets

This script performs comprehensive data quality validation on ML-ready datasets,
including correlation analysis, missing value detection, and outlier identification.

Usage:
    python validate_data_quality.py <dataset_path> [--output-dir output/quality]

Author: SynFinance Development Team
Version: 0.5.0
Date: October 26, 2025
"""

import sys
import os
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV or JSON."""
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.json'):
        return pd.DataFrame(json.load(open(filepath)))
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def analyze_missing_values(df: pd.DataFrame) -> Dict:
    """Analyze missing values in dataset."""
    print("\n" + "=" * 60)
    print("MISSING VALUE ANALYSIS")
    print("=" * 60)
    
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    
    results = {
        'total_samples': len(df),
        'total_features': len(df.columns),
        'features_with_missing': {},
        'summary': {}
    }
    
    if missing_counts.sum() == 0:
        print("✓ No missing values found")
        results['summary']['has_missing'] = False
    else:
        print(f"⚠ Found missing values in {(missing_counts > 0).sum()} features:")
        for col in missing_counts[missing_counts > 0].index:
            count = missing_counts[col]
            pct = missing_pct[col]
            print(f"  - {col}: {count} ({pct:.2f}%)")
            results['features_with_missing'][col] = {
                'count': int(count),
                'percentage': float(pct)
            }
        results['summary']['has_missing'] = True
        results['summary']['total_missing'] = int(missing_counts.sum())
    
    return results


def analyze_correlations(df: pd.DataFrame, threshold: float = 0.8) -> Dict:
    """Analyze feature correlations."""
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Remove ID columns
    exclude_cols = [col for col in numeric_df.columns if 'id' in col.lower()]
    numeric_df = numeric_df.drop(columns=exclude_cols, errors='ignore')
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Find high correlations
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': float(corr_matrix.iloc[i, j])
                })
    
    results = {
        'num_features_analyzed': len(numeric_df.columns),
        'high_correlation_threshold': threshold,
        'high_correlation_pairs': high_corr_pairs,
        'correlation_matrix': corr_matrix.to_dict()
    }
    
    if high_corr_pairs:
        print(f"⚠ Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > {threshold}):")
        for pair in high_corr_pairs[:10]:  # Show first 10
            print(f"  - {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
    else:
        print(f"✓ No highly correlated features found (threshold: {threshold})")
    
    return results, corr_matrix


def detect_outliers_iqr(df: pd.DataFrame, multiplier: float = 1.5) -> Dict:
    """Detect outliers using IQR method."""
    print("\n" + "=" * 60)
    print("OUTLIER DETECTION (IQR METHOD)")
    print("=" * 60)
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    exclude_cols = [col for col in numeric_df.columns if 'id' in col.lower() or 'is_' in col.lower()]
    numeric_df = numeric_df.drop(columns=exclude_cols, errors='ignore')
    
    results = {
        'method': 'IQR',
        'multiplier': multiplier,
        'features_with_outliers': {}
    }
    
    total_outliers = 0
    for col in numeric_df.columns:
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = ((numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)).sum()
        
        if outliers > 0:
            outlier_pct = (outliers / len(numeric_df)) * 100
            results['features_with_outliers'][col] = {
                'count': int(outliers),
                'percentage': float(outlier_pct),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'Q1': float(Q1),
                'Q3': float(Q3),
                'IQR': float(IQR)
            }
            total_outliers += outliers
    
    if results['features_with_outliers']:
        print(f"⚠ Found outliers in {len(results['features_with_outliers'])} features:")
        for col, info in list(results['features_with_outliers'].items())[:10]:
            print(f"  - {col}: {info['count']} ({info['percentage']:.2f}%)")
    else:
        print("✓ No outliers detected")
    
    results['total_outliers'] = total_outliers
    return results


def analyze_feature_distributions(df: pd.DataFrame) -> Dict:
    """Analyze feature distributions."""
    print("\n" + "=" * 60)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    numeric_df = df.select_dtypes(include=[np.number])
    exclude_cols = [col for col in numeric_df.columns if 'id' in col.lower()]
    numeric_df = numeric_df.drop(columns=exclude_cols, errors='ignore')
    
    results = {
        'features': {}
    }
    
    low_variance_features = []
    
    for col in numeric_df.columns:
        stats = {
            'mean': float(numeric_df[col].mean()),
            'std': float(numeric_df[col].std()),
            'min': float(numeric_df[col].min()),
            'max': float(numeric_df[col].max()),
            'median': float(numeric_df[col].median()),
            'variance': float(numeric_df[col].var()),
            'skewness': float(numeric_df[col].skew()),
            'kurtosis': float(numeric_df[col].kurtosis())
        }
        
        # Check for low variance
        if stats['variance'] < 0.01:
            low_variance_features.append(col)
        
        results['features'][col] = stats
    
    print(f"✓ Analyzed {len(numeric_df.columns)} numeric features")
    
    if low_variance_features:
        print(f"⚠ Found {len(low_variance_features)} low-variance features:")
        for col in low_variance_features[:10]:
            print(f"  - {col}: variance={results['features'][col]['variance']:.6f}")
    
    results['low_variance_features'] = low_variance_features
    return results


def check_class_balance(df: pd.DataFrame, target_col: str = 'is_fraud') -> Dict:
    """Check class balance in target variable."""
    print("\n" + "=" * 60)
    print("CLASS BALANCE ANALYSIS")
    print("=" * 60)
    
    if target_col not in df.columns:
        print(f"⚠ Target column '{target_col}' not found")
        return {}
    
    class_counts = df[target_col].value_counts()
    total = len(df)
    
    results = {
        'target_column': target_col,
        'total_samples': total,
        'class_distribution': {}
    }
    
    print(f"Target: {target_col}")
    print(f"Total samples: {total}")
    print("\nClass distribution:")
    
    for class_val, count in class_counts.items():
        pct = (count / total) * 100
        print(f"  - Class {class_val}: {count} ({pct:.2f}%)")
        results['class_distribution'][str(class_val)] = {
            'count': int(count),
            'percentage': float(pct)
        }
    
    # Calculate imbalance ratio
    if len(class_counts) == 2:
        imbalance_ratio = class_counts.max() / class_counts.min()
        results['imbalance_ratio'] = float(imbalance_ratio)
        
        if imbalance_ratio > 10:
            print(f"\n⚠ Severely imbalanced (ratio: {imbalance_ratio:.2f}:1)")
        elif imbalance_ratio > 3:
            print(f"\n⚠ Moderately imbalanced (ratio: {imbalance_ratio:.2f}:1)")
        else:
            print(f"\n✓ Well balanced (ratio: {imbalance_ratio:.2f}:1)")
    
    return results


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, output_path: str):
    """Plot correlation heatmap."""
    plt.figure(figsize=(14, 12))
    
    # Select top features by variance for readability
    if len(corr_matrix) > 30:
        # Calculate variance for each feature
        variances = corr_matrix.var()
        top_features = variances.nlargest(30).index
        corr_matrix = corr_matrix.loc[top_features, top_features]
    
    sns.heatmap(
        corr_matrix,
        cmap='coolwarm',
        center=0,
        annot=False,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    
    plt.title('Feature Correlation Heatmap (Top 30 by Variance)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved correlation heatmap to {output_path}")
    plt.close()


def plot_feature_distributions(df: pd.DataFrame, output_path: str, num_features: int = 12):
    """Plot distribution of top features."""
    numeric_df = df.select_dtypes(include=[np.number])
    exclude_cols = [col for col in numeric_df.columns if 'id' in col.lower() or 'is_' in col.lower()]
    numeric_df = numeric_df.drop(columns=exclude_cols, errors='ignore')
    
    # Select features with highest variance
    variances = numeric_df.var()
    top_features = variances.nlargest(num_features).index
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(top_features):
        axes[idx].hist(numeric_df[col], bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(col, fontsize=10)
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Feature Distributions (Top 12 by Variance)', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved feature distributions to {output_path}")
    plt.close()


def main():
    """Main data quality validation pipeline."""
    parser = argparse.ArgumentParser(description='Validate ML dataset quality')
    parser.add_argument('dataset_path', help='Path to dataset file (CSV or JSON)')
    parser.add_argument('--output-dir', default='output/quality_validation',
                       help='Output directory for results')
    parser.add_argument('--correlation-threshold', type=float, default=0.8,
                       help='Threshold for high correlation detection')
    parser.add_argument('--iqr-multiplier', type=float, default=1.5,
                       help='IQR multiplier for outlier detection')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("SYNFINANCE DATA QUALITY VALIDATION")
    print("=" * 60)
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("\nLoading dataset...")
    df = load_dataset(args.dataset_path)
    print(f"✓ Loaded dataset: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Run all analyses
    all_results = {
        'dataset_path': args.dataset_path,
        'dataset_shape': {'rows': int(df.shape[0]), 'columns': int(df.shape[1])},
        'analyses': {}
    }
    
    # 1. Missing values
    all_results['analyses']['missing_values'] = analyze_missing_values(df)
    
    # 2. Correlations
    corr_results, corr_matrix = analyze_correlations(df, args.correlation_threshold)
    all_results['analyses']['correlations'] = corr_results
    
    # 3. Outliers
    all_results['analyses']['outliers'] = detect_outliers_iqr(df, args.iqr_multiplier)
    
    # 4. Distributions
    all_results['analyses']['distributions'] = analyze_feature_distributions(df)
    
    # 5. Class balance
    all_results['analyses']['class_balance'] = check_class_balance(df)
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_correlation_heatmap(corr_matrix, f'{args.output_dir}/correlation_heatmap.png')
    plot_feature_distributions(df, f'{args.output_dir}/feature_distributions.png')
    
    # Save results
    output_file = f'{args.output_dir}/quality_validation_report.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Saved validation report to {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    print(f"\nDataset: {df.shape[0]} samples × {df.shape[1]} features")
    
    if all_results['analyses']['missing_values']['summary'].get('has_missing'):
        print(f"⚠ Missing values: {all_results['analyses']['missing_values']['summary']['total_missing']}")
    else:
        print("✓ No missing values")
    
    high_corr = len(all_results['analyses']['correlations']['high_correlation_pairs'])
    if high_corr > 0:
        print(f"⚠ High correlations: {high_corr} pairs")
    else:
        print("✓ No high correlations")
    
    total_outliers = all_results['analyses']['outliers']['total_outliers']
    if total_outliers > 0:
        print(f"⚠ Outliers detected: {total_outliers}")
    else:
        print("✓ No outliers detected")
    
    low_var = len(all_results['analyses']['distributions']['low_variance_features'])
    if low_var > 0:
        print(f"⚠ Low-variance features: {low_var}")
    else:
        print("✓ No low-variance features")
    
    print(f"\nAll results saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
