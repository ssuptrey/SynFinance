"""
Demo: Dataset Comparison System
================================

Demonstrates the statistical dataset comparison capabilities of SynFinance:
1. Multi-dataset generation with different characteristics
2. Statistical comparison (KS test, Chi-Square, Kruskal-Wallis)
3. Effect size calculation (Cohen's d)
4. Similarity scoring
5. Fraud pattern comparison
6. Comparison visualizations
7. Automated recommendations

This script generates multiple datasets with varying fraud rates and transaction
patterns, then performs comprehensive statistical comparisons.

Author: SynFinance Development Team
Date: November 2, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Import comparison module
from src.reporting import DatasetComparator, ComparisonResult


def generate_dataset(n_rows=1000, fraud_rate=0.05, amount_mean=8.0, amount_std=1.5, 
                     category_distribution=None, seed=None):
    """
    Generate a synthetic transaction dataset with specific characteristics.
    
    Args:
        n_rows: Number of transactions
        fraud_rate: Proportion of fraudulent transactions (0.0 to 1.0)
        amount_mean: Mean of log-normal distribution for amounts
        amount_std: Standard deviation of log-normal distribution
        category_distribution: Dictionary of category probabilities
        seed: Random seed for reproducibility
    
    Returns:
        pandas.DataFrame: Generated transaction dataset
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Default category distribution
    if category_distribution is None:
        category_distribution = {
            'Electronics': 0.3,
            'Groceries': 0.25,
            'Travel': 0.2,
            'Entertainment': 0.15,
            'Healthcare': 0.1
        }
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, periods=n_rows)
    
    # Generate transaction amounts (log-normal distribution)
    amounts = np.random.lognormal(mean=amount_mean, sigma=amount_std, size=n_rows)
    
    # Generate categories
    categories = list(category_distribution.keys())
    probabilities = list(category_distribution.values())
    category_values = np.random.choice(categories, size=n_rows, p=probabilities)
    
    # Generate fraud data
    fraud_mask = np.random.random(n_rows) < fraud_rate
    fraud_types = np.where(
        fraud_mask,
        np.random.choice(
            ['Card Skimming', 'Identity Theft', 'Account Takeover'],
            size=n_rows,
            p=[0.5, 0.3, 0.2]
        ),
        None
    )
    
    # Generate customer and merchant IDs
    customer_ids = np.random.randint(1000, 2000, size=n_rows)
    merchant_ids = np.random.randint(500, 700, size=n_rows)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Transaction_ID': range(1, n_rows + 1),
        'Transaction_Date': dates,
        'Transaction_Amount': amounts,
        'Category': category_values,
        'Customer_ID': customer_ids,
        'Merchant_ID': merchant_ids,
        'Fraud_Type': fraud_types,
        'Is_Fraud': fraud_mask.astype(int)
    })
    
    return data


def scenario_1_fraud_rate_comparison():
    """
    Scenario 1: Compare datasets with different fraud rates.
    
    This demonstrates how the comparison tool detects differences in fraud
    patterns between datasets with low vs. high fraud rates.
    """
    print("\n" + "="*70)
    print("SCENARIO 1: Fraud Rate Comparison")
    print("="*70)
    print("\nComparing datasets with different fraud rates (2% vs 8%)")
    
    # Generate datasets
    print("\n[*] Generating datasets...")
    low_fraud = generate_dataset(n_rows=1000, fraud_rate=0.02, seed=42)
    high_fraud = generate_dataset(n_rows=1000, fraud_rate=0.08, seed=43)
    
    print(f"   • Low Fraud Dataset: {len(low_fraud):,} transactions, "
          f"{low_fraud['Is_Fraud'].sum()} frauds ({low_fraud['Is_Fraud'].mean():.1%})")
    print(f"   • High Fraud Dataset: {len(high_fraud):,} transactions, "
          f"{high_fraud['Is_Fraud'].sum()} frauds ({high_fraud['Is_Fraud'].mean():.1%})")
    
    # Perform comparison
    print("\n[*] Performing statistical comparison...")
    comparator = DatasetComparator(significance_level=0.05)
    result = comparator.compare_datasets(
        datasets=[low_fraud, high_fraud],
        names=['Low Fraud (2%)', 'High Fraud (8%)']
    )
    
    # Print results
    print_comparison_results(result)
    
    # Generate visualizations
    output_dir = Path('output/comparison_demo/scenario1_fraud_rate')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[*] Generating comparison visualizations...")
    comparator.generate_comparison_visualizations(result, output_dir)
    print(f"   [OK] Saved 3 charts to: {output_dir}/")
    
    # Fraud pattern comparison
    print("\n[*] Comparing fraud patterns...")
    fraud_result = comparator.compare_fraud_patterns(
        datasets=[low_fraud, high_fraud],
        names=['Low Fraud (2%)', 'High Fraud (8%)']
    )
    print(f"   [OK] Fraud comparison complete")
    for name, stats in fraud_result.items():
        print(f"   - {name}: {stats['total_fraud']} frauds ({stats['fraud_rate']:.1%})")
    
    return result


def scenario_2_amount_distribution_comparison():
    """
    Scenario 2: Compare datasets with different transaction amount distributions.
    
    This demonstrates detection of differences in transaction amounts between
    normal and high-value transaction datasets.
    """
    print("\n" + "="*70)
    print("SCENARIO 2: Transaction Amount Distribution Comparison")
    print("="*70)
    print("\nComparing datasets with different amount distributions")
    
    # Generate datasets
    print("\n[*] Generating datasets...")
    normal_amounts = generate_dataset(n_rows=1000, amount_mean=8.0, amount_std=1.5, seed=44)
    high_amounts = generate_dataset(n_rows=1000, amount_mean=9.5, amount_std=1.5, seed=45)
    
    print(f"   • Normal Amounts: Mean = Rs.{normal_amounts['Transaction_Amount'].mean():,.2f}, "
          f"Median = Rs.{normal_amounts['Transaction_Amount'].median():,.2f}")
    print(f"   • High Amounts: Mean = Rs.{high_amounts['Transaction_Amount'].mean():,.2f}, "
          f"Median = Rs.{high_amounts['Transaction_Amount'].median():,.2f}")
    
    # Perform comparison
    print("\n[*] Performing statistical comparison...")
    comparator = DatasetComparator(significance_level=0.05)
    result = comparator.compare_datasets(
        datasets=[normal_amounts, high_amounts],
        names=['Normal Amounts', 'High Amounts']
    )
    
    # Print results
    print_comparison_results(result)
    
    # Generate visualizations
    output_dir = Path('output/comparison_demo/scenario2_amounts')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[*] Generating comparison visualizations...")
    comparator.generate_comparison_visualizations(result, output_dir)
    print(f"   [OK] Saved 3 charts to: {output_dir}/")
    
    return result


def scenario_3_category_distribution_comparison():
    """
    Scenario 3: Compare datasets with different category distributions.
    
    This demonstrates detection of differences in categorical data using
    Chi-Square tests.
    """
    print("\n" + "="*70)
    print("SCENARIO 3: Category Distribution Comparison")
    print("="*70)
    print("\nComparing datasets with different category distributions")
    
    # Define different category distributions
    electronics_heavy = {
        'Electronics': 0.5,
        'Groceries': 0.2,
        'Travel': 0.15,
        'Entertainment': 0.1,
        'Healthcare': 0.05
    }
    
    groceries_heavy = {
        'Electronics': 0.2,
        'Groceries': 0.5,
        'Travel': 0.15,
        'Entertainment': 0.1,
        'Healthcare': 0.05
    }
    
    # Generate datasets
    print("\n[*] Generating datasets...")
    dataset1 = generate_dataset(n_rows=1000, category_distribution=electronics_heavy, seed=46)
    dataset2 = generate_dataset(n_rows=1000, category_distribution=groceries_heavy, seed=47)
    
    print(f"   • Electronics-Heavy: Top category = {dataset1['Category'].value_counts().index[0]} "
          f"({dataset1['Category'].value_counts().iloc[0] / len(dataset1):.1%})")
    print(f"   • Groceries-Heavy: Top category = {dataset2['Category'].value_counts().index[0]} "
          f"({dataset2['Category'].value_counts().iloc[0] / len(dataset2):.1%})")
    
    # Perform comparison
    print("\n[*] Performing statistical comparison...")
    comparator = DatasetComparator(significance_level=0.05)
    result = comparator.compare_datasets(
        datasets=[dataset1, dataset2],
        names=['Electronics-Heavy', 'Groceries-Heavy']
    )
    
    # Print results
    print_comparison_results(result)
    
    # Generate visualizations
    output_dir = Path('output/comparison_demo/scenario3_categories')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[*] Generating comparison visualizations...")
    comparator.generate_comparison_visualizations(result, output_dir)
    print(f"   [OK] Saved 3 charts to: {output_dir}/")
    
    return result


def scenario_4_multi_dataset_comparison():
    """
    Scenario 4: Compare 3+ datasets simultaneously.
    
    This demonstrates multi-dataset comparison using Kruskal-Wallis test
    for numeric fields.
    """
    print("\n" + "="*70)
    print("SCENARIO 4: Multi-Dataset Comparison (3+ datasets)")
    print("="*70)
    print("\nComparing 3 datasets with low/medium/high fraud rates")
    
    # Generate datasets
    print("\n[*] Generating datasets...")
    low_fraud = generate_dataset(n_rows=800, fraud_rate=0.02, seed=48)
    medium_fraud = generate_dataset(n_rows=800, fraud_rate=0.05, seed=49)
    high_fraud = generate_dataset(n_rows=800, fraud_rate=0.10, seed=50)
    
    print(f"   • Low Fraud: {low_fraud['Is_Fraud'].mean():.1%} fraud rate")
    print(f"   • Medium Fraud: {medium_fraud['Is_Fraud'].mean():.1%} fraud rate")
    print(f"   • High Fraud: {high_fraud['Is_Fraud'].mean():.1%} fraud rate")
    
    # Perform comparison
    print("\n[*] Performing statistical comparison (Kruskal-Wallis test)...")
    comparator = DatasetComparator(significance_level=0.05)
    result = comparator.compare_datasets(
        datasets=[low_fraud, medium_fraud, high_fraud],
        names=['Low Fraud (2%)', 'Medium Fraud (5%)', 'High Fraud (10%)']
    )
    
    # Print results
    print_comparison_results(result)
    
    # Generate visualizations
    output_dir = Path('output/comparison_demo/scenario4_multi')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[*] Generating comparison visualizations...")
    comparator.generate_comparison_visualizations(result, output_dir)
    print(f"   [OK] Saved 3 charts to: {output_dir}/")
    
    return result


def scenario_5_identical_datasets():
    """
    Scenario 5: Compare identical datasets (sanity check).
    
    This demonstrates that identical datasets produce 100% similarity
    with no significant differences.
    """
    print("\n" + "="*70)
    print("SCENARIO 5: Identical Dataset Comparison (Sanity Check)")
    print("="*70)
    print("\nComparing a dataset with itself (should show 100% similarity)")
    
    # Generate dataset
    print("\n[*] Generating dataset...")
    dataset = generate_dataset(n_rows=1000, seed=51)
    
    # Perform comparison
    print("\n[*] Performing statistical comparison...")
    comparator = DatasetComparator(significance_level=0.05)
    result = comparator.compare_datasets(
        datasets=[dataset, dataset.copy()],
        names=['Dataset A', 'Dataset B (identical)']
    )
    
    # Print results
    print_comparison_results(result)
    
    return result


def print_comparison_results(result: ComparisonResult):
    """Print formatted comparison results."""
    print("\n" + "-"*70)
    print("COMPARISON RESULTS")
    print("-"*70)
    
    print(f"\n[DATA] Overview:")
    print(f"   • Datasets Compared: {', '.join(result.dataset_names)}")
    print(f"   • Total Fields: {result.total_fields}")
    print(f"   • Compared Fields: {len(result.field_comparisons)}")
    print(f"   • Significant Differences: {result.significant_differences}")
    print(f"   • Similarity Score: {result.similarity_score:.1%}")
    
    if result.field_comparisons:
        print(f"\n[LIST] Field-by-Field Comparison:")
        
        # Convert dict to list and sort by significance (significant first, then by p-value)
        field_list = list(result.field_comparisons.values())
        sorted_comparisons = sorted(
            field_list,
            key=lambda x: (not x.is_significant, x.p_value if x.p_value is not None else 1.0)
        )
        
        for comp in sorted_comparisons[:10]:  # Show top 10
            status = "[WARN] SIGNIFICANT" if comp.is_significant else "[OK] Similar"
            print(f"\n   {comp.field_name} ({comp.statistical_test}):")
            print(f"      Status: {status}")
            if comp.p_value is not None:
                print(f"      P-value: {comp.p_value:.4f}")
            if comp.effect_size is not None:
                print(f"      Effect Size (Cohen's d): {comp.effect_size:.3f}")
    
    if result.recommendations:
        print(f"\n[TIP] Recommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"   {i}. {rec}")


def print_summary_statistics():
    """Print summary of all scenarios."""
    print("\n" + "="*70)
    print("DEMONSTRATION SUMMARY")
    print("="*70)
    
    print(f"\n[OK] Completed 5 Comparison Scenarios:")
    print(f"   1. Fraud Rate Comparison (2% vs 8%)")
    print(f"   2. Transaction Amount Distribution")
    print(f"   3. Category Distribution (Chi-Square test)")
    print(f"   4. Multi-Dataset Comparison (3 datasets)")
    print(f"   5. Identical Dataset Sanity Check")
    
    print(f"\n[FILES] Generated Output:")
    output_base = Path('output/comparison_demo')
    if output_base.exists():
        total_files = sum(1 for _ in output_base.rglob('*.png'))
        print(f"   • {total_files} visualization charts")
        print(f"   • Saved to: {output_base}/")
    
    print(f"\n[TOOL] Statistical Tests Demonstrated:")
    print(f"   • Kolmogorov-Smirnov Test (2-sample numeric)")
    print(f"   • Kruskal-Wallis Test (3+ sample numeric)")
    print(f"   • Chi-Square Test (categorical data)")
    print(f"   • Cohen's d (effect size calculation)")
    
    print(f"\n[TIP] Key Insights:")
    print(f"   • Similarity scores range from 0% (completely different) to 100% (identical)")
    print(f"   • P-values < 0.05 indicate statistically significant differences")
    print(f"   • Effect sizes show practical significance (small/medium/large)")
    print(f"   • Automated recommendations guide further analysis")
    
    print(f"\n[OK] Demonstration Complete!")


def main():
    """Main demonstration function."""
    print("="*70)
    print("SynFinance Dataset Comparison System - Demonstration")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Version: 2.17.0")
    
    # Run all scenarios
    print("\n[LIST] Running 5 comparison scenarios...")
    
    try:
        # Scenario 1: Fraud rate comparison
        result1 = scenario_1_fraud_rate_comparison()
        
        # Scenario 2: Amount distribution comparison
        result2 = scenario_2_amount_distribution_comparison()
        
        # Scenario 3: Category distribution comparison
        result3 = scenario_3_category_distribution_comparison()
        
        # Scenario 4: Multi-dataset comparison
        result4 = scenario_4_multi_dataset_comparison()
        
        # Scenario 5: Identical datasets (sanity check)
        result5 = scenario_5_identical_datasets()
        
        # Print summary
        print_summary_statistics()
        
    except Exception as e:
        print(f"\n[ERROR] Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
