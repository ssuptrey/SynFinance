"""
Dataset Comparison Module

This module provides comprehensive dataset comparison capabilities with statistical
testing, distribution analysis, and visualization generation.

Author: SynFinance Development Team
Date: November 2, 2025
Version: 2.17.0
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency, ttest_ind
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class FieldComparison:
    """Results of comparing a single field across datasets."""
    field_name: str
    data_type: str
    statistical_test: str
    test_statistic: float
    p_value: float
    is_significant: bool
    effect_size: Optional[float] = None
    dataset_means: Optional[Dict[str, float]] = None
    dataset_stds: Optional[Dict[str, float]] = None
    recommendation: Optional[str] = None


@dataclass
class ComparisonResult:
    """Complete comparison results for multiple datasets."""
    dataset_names: List[str]
    total_fields: int
    compared_fields: int
    field_comparisons: Dict[str, FieldComparison] = field(default_factory=dict)
    overall_statistics: Dict[str, Any] = field(default_factory=dict)
    significant_differences: List[str] = field(default_factory=list)
    similarity_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)


class DatasetComparator:
    """
    Compare multiple datasets with statistical testing and visualization.
    
    Features:
    - Distribution comparison (KS test, chi-square)
    - Statistical metrics comparison (means, variances)
    - Effect size calculation (Cohen's d)
    - Fraud pattern comparison
    - Quality metrics comparison
    - Visualization generation
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize the dataset comparator.
        
        Args:
            significance_level: P-value threshold for statistical significance (default: 0.05)
        """
        self.significance_level = significance_level
    
    def compare_datasets(
        self,
        datasets: List[pd.DataFrame],
        names: Optional[List[str]] = None,
        fields_to_compare: Optional[List[str]] = None
    ) -> ComparisonResult:
        """
        Compare multiple datasets across all fields.
        
        Args:
            datasets: List of DataFrames to compare
            names: Optional names for each dataset
            fields_to_compare: Optional list of specific fields to compare
            
        Returns:
            ComparisonResult with complete comparison analysis
        """
        if len(datasets) < 2:
            raise ValueError("At least 2 datasets required for comparison")
        
        # Generate default names if not provided
        if names is None:
            names = [f"Dataset_{i+1}" for i in range(len(datasets))]
        
        if len(names) != len(datasets):
            raise ValueError("Number of names must match number of datasets")
        
        # Find common fields across all datasets
        common_fields = set(datasets[0].columns)
        for df in datasets[1:]:
            common_fields = common_fields.intersection(set(df.columns))
        
        if fields_to_compare:
            common_fields = common_fields.intersection(set(fields_to_compare))
        
        common_fields = sorted(list(common_fields))
        
        # Initialize result
        result = ComparisonResult(
            dataset_names=names,
            total_fields=len(datasets[0].columns),
            compared_fields=len(common_fields)
        )
        
        # Compare each field
        for field in common_fields:
            field_data = [df[field] for df in datasets]
            comparison = self._compare_field(field, field_data, names)
            result.field_comparisons[field] = comparison
            
            if comparison.is_significant:
                result.significant_differences.append(field)
        
        # Calculate overall statistics
        result.overall_statistics = self._calculate_overall_statistics(datasets, names)
        
        # Calculate similarity score
        result.similarity_score = self._calculate_similarity_score(result)
        
        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)
        
        return result
    
    def _compare_field(
        self,
        field_name: str,
        field_data: List[pd.Series],
        dataset_names: List[str]
    ) -> FieldComparison:
        """
        Compare a single field across datasets.
        
        Args:
            field_name: Name of the field
            field_data: List of Series containing field data from each dataset
            dataset_names: Names of datasets
            
        Returns:
            FieldComparison object
        """
        # Determine data type
        first_series = field_data[0]
        is_numeric = pd.api.types.is_numeric_dtype(first_series)
        is_categorical = pd.api.types.is_categorical_dtype(first_series) or \
                        pd.api.types.is_object_dtype(first_series)
        
        if is_numeric:
            return self._compare_numeric_field(field_name, field_data, dataset_names)
        elif is_categorical:
            return self._compare_categorical_field(field_name, field_data, dataset_names)
        else:
            # Default: treat as categorical
            return self._compare_categorical_field(field_name, field_data, dataset_names)
    
    def _compare_numeric_field(
        self,
        field_name: str,
        field_data: List[pd.Series],
        dataset_names: List[str]
    ) -> FieldComparison:
        """
        Compare numeric field using Kolmogorov-Smirnov test.
        
        Args:
            field_name: Name of the field
            field_data: List of numeric Series
            dataset_names: Names of datasets
            
        Returns:
            FieldComparison object
        """
        # Remove NaN values
        clean_data = [series.dropna() for series in field_data]
        
        # For 2 datasets, use KS test
        if len(clean_data) == 2:
            statistic, p_value = ks_2samp(clean_data[0], clean_data[1])
            test_name = "Kolmogorov-Smirnov"
        else:
            # For >2 datasets, use Kruskal-Wallis test
            statistic, p_value = stats.kruskal(*clean_data)
            test_name = "Kruskal-Wallis"
        
        is_significant = p_value < self.significance_level
        
        # Calculate effect size (Cohen's d for 2 datasets)
        effect_size = None
        if len(clean_data) == 2:
            effect_size = self._cohens_d(clean_data[0], clean_data[1])
        
        # Calculate means and stds
        dataset_means = {name: float(data.mean()) for name, data in zip(dataset_names, clean_data)}
        dataset_stds = {name: float(data.std()) for name, data in zip(dataset_names, clean_data)}
        
        # Generate recommendation
        recommendation = self._generate_field_recommendation(
            field_name, is_significant, effect_size, test_name
        )
        
        return FieldComparison(
            field_name=field_name,
            data_type="numeric",
            statistical_test=test_name,
            test_statistic=float(statistic),
            p_value=float(p_value),
            is_significant=is_significant,
            effect_size=effect_size,
            dataset_means=dataset_means,
            dataset_stds=dataset_stds,
            recommendation=recommendation
        )
    
    def _compare_categorical_field(
        self,
        field_name: str,
        field_data: List[pd.Series],
        dataset_names: List[str]
    ) -> FieldComparison:
        """
        Compare categorical field using chi-square test.
        
        Args:
            field_name: Name of the field
            field_data: List of categorical Series
            dataset_names: Names of datasets
            
        Returns:
            FieldComparison object
        """
        # Create contingency table
        try:
            # Get value counts for each dataset
            value_counts = [series.value_counts() for series in field_data]
            
            # Find all unique values across datasets
            all_values = set()
            for vc in value_counts:
                all_values.update(vc.index)
            all_values = sorted(list(all_values))
            
            # Build contingency table
            contingency = []
            for vc in value_counts:
                row = [vc.get(val, 0) for val in all_values]
                contingency.append(row)
            
            # Perform chi-square test
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            
            is_significant = p_value < self.significance_level
            
            recommendation = self._generate_field_recommendation(
                field_name, is_significant, None, "Chi-Square"
            )
            
            return FieldComparison(
                field_name=field_name,
                data_type="categorical",
                statistical_test="Chi-Square",
                test_statistic=float(chi2),
                p_value=float(p_value),
                is_significant=is_significant,
                recommendation=recommendation
            )
            
        except Exception as e:
            # Fallback if chi-square fails
            return FieldComparison(
                field_name=field_name,
                data_type="categorical",
                statistical_test="Chi-Square (Failed)",
                test_statistic=0.0,
                p_value=1.0,
                is_significant=False,
                recommendation=f"Could not perform test: {str(e)}"
            )
    
    def _cohens_d(self, series1: pd.Series, series2: pd.Series) -> float:
        """
        Calculate Cohen's d effect size.
        
        Args:
            series1: First series
            series2: Second series
            
        Returns:
            Cohen's d value
        """
        n1, n2 = len(series1), len(series2)
        var1, var2 = series1.var(), series2.var()
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return float((series1.mean() - series2.mean()) / pooled_std)
    
    def _calculate_overall_statistics(
        self,
        datasets: List[pd.DataFrame],
        names: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate overall statistics for all datasets.
        
        Args:
            datasets: List of DataFrames
            names: Dataset names
            
        Returns:
            Dictionary of overall statistics
        """
        stats_dict = {}
        
        for name, df in zip(names, datasets):
            stats_dict[name] = {
                'total_records': len(df),
                'total_fields': len(df.columns),
                'missing_values': int(df.isnull().sum().sum()),
                'completeness': float(1 - (df.isnull().sum().sum() / (len(df) * len(df.columns))))
            }
            
            # Add fraud statistics if available
            if 'Fraud_Type' in df.columns:
                fraud_count = df['Fraud_Type'].notna().sum()
                stats_dict[name]['fraud_count'] = int(fraud_count)
                stats_dict[name]['fraud_rate'] = float(fraud_count / len(df))
            
            # Add amount statistics if available
            if 'Transaction_Amount' in df.columns:
                stats_dict[name]['total_amount'] = float(df['Transaction_Amount'].sum())
                stats_dict[name]['mean_amount'] = float(df['Transaction_Amount'].mean())
                stats_dict[name]['median_amount'] = float(df['Transaction_Amount'].median())
        
        return stats_dict
    
    def _calculate_similarity_score(self, result: ComparisonResult) -> float:
        """
        Calculate overall similarity score (0-1, higher = more similar).
        
        Args:
            result: ComparisonResult object
            
        Returns:
            Similarity score between 0 and 1
        """
        if result.compared_fields == 0:
            return 0.0
        
        # Count non-significant differences (similar fields)
        similar_fields = result.compared_fields - len(result.significant_differences)
        
        # Similarity score is ratio of similar fields
        return similar_fields / result.compared_fields
    
    def _generate_field_recommendation(
        self,
        field_name: str,
        is_significant: bool,
        effect_size: Optional[float],
        test_name: str
    ) -> str:
        """Generate recommendation for a field comparison."""
        if not is_significant:
            return f"No significant difference detected ({test_name})"
        
        if effect_size is not None:
            if abs(effect_size) < 0.2:
                magnitude = "negligible"
            elif abs(effect_size) < 0.5:
                magnitude = "small"
            elif abs(effect_size) < 0.8:
                magnitude = "medium"
            else:
                magnitude = "large"
            
            return f"Significant difference with {magnitude} effect size (d={effect_size:.3f})"
        else:
            return f"Significant difference detected ({test_name})"
    
    def _generate_recommendations(self, result: ComparisonResult) -> List[str]:
        """Generate overall recommendations based on comparison results."""
        recommendations = []
        
        # Similarity assessment
        if result.similarity_score >= 0.9:
            recommendations.append("Datasets are highly similar (>90% fields match)")
        elif result.similarity_score >= 0.7:
            recommendations.append("Datasets are moderately similar (70-90% fields match)")
        else:
            recommendations.append(f"Datasets show substantial differences ({result.similarity_score:.1%} similarity)")
        
        # Significant differences
        if len(result.significant_differences) > 0:
            recommendations.append(f"Review {len(result.significant_differences)} fields with significant differences")
        
        # Field-specific recommendations
        for field, comparison in result.field_comparisons.items():
            if comparison.is_significant and comparison.effect_size and abs(comparison.effect_size) > 0.8:
                recommendations.append(f"Large effect detected in '{field}' - investigate distribution differences")
        
        return recommendations
    
    def compare_fraud_patterns(
        self,
        datasets: List[pd.DataFrame],
        names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare fraud patterns across datasets.
        
        Args:
            datasets: List of DataFrames with fraud data
            names: Optional dataset names
            
        Returns:
            Dictionary with fraud comparison results
        """
        if names is None:
            names = [f"Dataset_{i+1}" for i in range(len(datasets))]
        
        fraud_comparison = {}
        
        for name, df in zip(names, datasets):
            if 'Fraud_Type' not in df.columns:
                continue
            
            fraud_data = df[df['Fraud_Type'].notna()]
            
            fraud_comparison[name] = {
                'total_fraud': len(fraud_data),
                'fraud_rate': len(fraud_data) / len(df) if len(df) > 0 else 0,
                'pattern_distribution': fraud_data['Fraud_Type'].value_counts().to_dict() if len(fraud_data) > 0 else {}
            }
            
            if 'Fraud_Confidence' in fraud_data.columns and len(fraud_data) > 0:
                fraud_comparison[name]['avg_confidence'] = float(fraud_data['Fraud_Confidence'].mean())
                fraud_comparison[name]['high_confidence_count'] = int((fraud_data['Fraud_Confidence'] > 0.8).sum())
        
        return fraud_comparison
    
    def generate_comparison_visualizations(
        self,
        result: ComparisonResult,
        datasets: List[pd.DataFrame],
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, plt.Figure]:
        """
        Generate comparison visualizations.
        
        Args:
            result: ComparisonResult object
            datasets: List of DataFrames
            output_dir: Optional directory to save figures
            
        Returns:
            Dictionary of figure objects
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        figures = {}
        
        # 1. Similarity score chart
        fig, ax = plt.subplots(figsize=(8, 6))
        similar_count = result.compared_fields - len(result.significant_differences)
        different_count = len(result.significant_differences)
        
        ax.bar(['Similar', 'Different'], [similar_count, different_count], 
               color=['#27AE60', '#E74C3C'])
        ax.set_ylabel('Number of Fields')
        ax.set_title('Dataset Similarity Overview')
        ax.text(0, similar_count, str(similar_count), ha='center', va='bottom', fontweight='bold')
        ax.text(1, different_count, str(different_count), ha='center', va='bottom', fontweight='bold')
        
        figures['similarity_overview'] = fig
        if output_dir:
            fig.savefig(output_dir / 'similarity_overview.png', dpi=100, bbox_inches='tight')
        
        # 2. P-value distribution
        p_values = [comp.p_value for comp in result.field_comparisons.values()]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(p_values, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(self.significance_level, color='red', linestyle='--', 
                   label=f'Significance Level ({self.significance_level})')
        ax.set_xlabel('P-Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of P-Values Across All Field Comparisons')
        ax.legend()
        
        figures['pvalue_distribution'] = fig
        if output_dir:
            fig.savefig(output_dir / 'pvalue_distribution.png', dpi=100, bbox_inches='tight')
        
        # 3. Effect sizes (for numeric comparisons)
        effect_sizes = {name: comp.effect_size 
                       for name, comp in result.field_comparisons.items() 
                       if comp.effect_size is not None}
        
        if effect_sizes:
            fig, ax = plt.subplots(figsize=(12, 6))
            fields = list(effect_sizes.keys())
            values = list(effect_sizes.values())
            
            colors = ['#E74C3C' if abs(v) > 0.8 else '#F39C12' if abs(v) > 0.5 else '#27AE60' 
                     for v in values]
            
            ax.barh(fields, values, color=colors)
            ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel("Cohen's d (Effect Size)")
            ax.set_title("Effect Sizes for Numeric Field Comparisons")
            ax.grid(axis='x', alpha=0.3)
            
            figures['effect_sizes'] = fig
            if output_dir:
                fig.savefig(output_dir / 'effect_sizes.png', dpi=100, bbox_inches='tight')
        
        return figures
    
    def generate_comparison_report(
        self,
        result: ComparisonResult,
        datasets: List[pd.DataFrame],
        output_path: Union[str, Path],
        format: str = 'html'
    ):
        """
        Generate a comprehensive comparison report.
        
        Args:
            result: ComparisonResult object
            datasets: List of DataFrames
            output_path: Path to save report
            format: Report format ('html', 'excel')
        """
        output_path = Path(output_path)
        
        if format == 'html':
            from src.reporting.html_generator import HTMLReportGenerator
            
            # Prepare metrics for report
            metrics = {
                'datasets_compared': len(result.dataset_names),
                'total_fields': result.total_fields,
                'compared_fields': result.compared_fields,
                'significant_differences': len(result.significant_differences),
                'similarity_score': result.similarity_score
            }
            
            # Generate visualizations
            charts = self.generate_comparison_visualizations(result, datasets)
            
            # Generate HTML report
            generator = HTMLReportGenerator()
            
            # For now, use base template (we can create a specific comparison template later)
            html = generator.generate_executive_report(
                data=datasets[0],  # Use first dataset as reference
                metrics=metrics,
                charts=charts
            )
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html, encoding='utf-8')
            
        elif format == 'excel':
            from src.reporting.excel_generator import ExcelDashboardGenerator
            
            generator = ExcelDashboardGenerator()
            
            # Create comparison workbook
            # For now, use first dataset (we can enhance this later)
            generator.create_dashboard_workbook(datasets[0], output_path)


# Example usage
if __name__ == "__main__":
    # Create two sample datasets with slight differences
    np.random.seed(42)
    
    dataset1 = pd.DataFrame({
        'Transaction_Amount': np.random.lognormal(8, 1.5, 1000),
        'Category': np.random.choice(['Groceries', 'Dining', 'Shopping'], 1000),
        'Fraud_Type': [None] * 980 + ['Card Cloning'] * 20
    })
    
    np.random.seed(43)  # Different seed for variation
    dataset2 = pd.DataFrame({
        'Transaction_Amount': np.random.lognormal(8.2, 1.6, 1000),  # Slightly different parameters
        'Category': np.random.choice(['Groceries', 'Dining', 'Shopping'], 1000),
        'Fraud_Type': [None] * 970 + ['Card Cloning'] * 30  # Different fraud rate
    })
    
    # Compare datasets
    comparator = DatasetComparator()
    result = comparator.compare_datasets(
        datasets=[dataset1, dataset2],
        names=["Low Fraud", "High Fraud"]
    )
    
    print("=" * 60)
    print("DATASET COMPARISON RESULTS")
    print("=" * 60)
    print(f"\nDatasets Compared: {', '.join(result.dataset_names)}")
    print(f"Total Fields: {result.total_fields}")
    print(f"Compared Fields: {result.compared_fields}")
    print(f"Significant Differences: {len(result.significant_differences)}")
    print(f"Similarity Score: {result.similarity_score:.1%}")
    
    print("\n" + "=" * 60)
    print("FIELD-BY-FIELD COMPARISON")
    print("=" * 60)
    for field, comparison in result.field_comparisons.items():
        print(f"\n{field} ({comparison.data_type}):")
        print(f"  Test: {comparison.statistical_test}")
        print(f"  P-value: {comparison.p_value:.4f}")
        print(f"  Significant: {'Yes' if comparison.is_significant else 'No'}")
        if comparison.effect_size:
            print(f"  Effect Size: {comparison.effect_size:.3f}")
        print(f"  {comparison.recommendation}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    for i, rec in enumerate(result.recommendations, 1):
        print(f"{i}. {rec}")
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 60)
    figures = comparator.generate_comparison_visualizations(
        result, [dataset1, dataset2], output_dir="comparison_charts"
    )
    print(f"✅ Generated {len(figures)} comparison charts in 'comparison_charts/' directory")
    
    print("\n✅ Dataset comparison complete!")
