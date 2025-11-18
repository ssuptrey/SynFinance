"""
Tests for Dataset Comparator.

Author: SynFinance Development Team
Date: November 2, 2025
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import matplotlib
matplotlib.use('Agg')

from src.reporting.dataset_comparator import (
    DatasetComparator,
    ComparisonResult,
    FieldComparison
)


class TestDatasetComparator:
    """Test suite for dataset comparison."""
    
    @pytest.fixture
    def identical_datasets(self):
        """Create two identical datasets."""
        np.random.seed(42)
        data = pd.DataFrame({
            'Amount': np.random.lognormal(8, 1.5, 100),
            'Category': np.random.choice(['A', 'B', 'C'], 100),
            'Count': np.random.randint(1, 10, 100)
        })
        return [data.copy(), data.copy()]
    
    @pytest.fixture
    def different_datasets(self):
        """Create two different datasets."""
        np.random.seed(42)
        data1 = pd.DataFrame({
            'Amount': np.random.lognormal(8, 1.5, 100),
            'Category': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        np.random.seed(43)
        data2 = pd.DataFrame({
            'Amount': np.random.lognormal(9, 2.0, 100),  # Different distribution
            'Category': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        return [data1, data2]
    
    @pytest.fixture
    def comparator(self):
        """Create DatasetComparator instance."""
        return DatasetComparator()
    
    def test_comparator_initialization(self, comparator):
        """Test DatasetComparator initializes correctly."""
        assert comparator is not None
        assert comparator.significance_level == 0.05
    
    def test_custom_significance_level(self):
        """Test custom significance level."""
        comparator = DatasetComparator(significance_level=0.01)
        assert comparator.significance_level == 0.01
    
    def test_compare_identical_datasets(self, comparator, identical_datasets):
        """Test comparing identical datasets."""
        result = comparator.compare_datasets(
            datasets=identical_datasets,
            names=['Dataset1', 'Dataset2']
        )
        
        assert isinstance(result, ComparisonResult)
        assert result.similarity_score == 1.0  # 100% similar
        assert len(result.significant_differences) == 0
    
    def test_compare_different_datasets(self, comparator, different_datasets):
        """Test comparing different datasets."""
        result = comparator.compare_datasets(
            datasets=different_datasets,
            names=['Low', 'High']
        )
        
        assert isinstance(result, ComparisonResult)
        assert result.similarity_score < 1.0  # Not identical
        assert len(result.field_comparisons) > 0
    
    def test_compare_numeric_fields(self, comparator):
        """Test comparison of numeric fields."""
        data1 = pd.DataFrame({'Amount': np.random.normal(100, 10, 100)})
        data2 = pd.DataFrame({'Amount': np.random.normal(150, 10, 100)})
        
        result = comparator.compare_datasets([data1, data2])
        
        field_comp = result.field_comparisons['Amount']
        assert field_comp.data_type == 'numeric'
        assert field_comp.statistical_test == 'Kolmogorov-Smirnov'
        assert field_comp.effect_size is not None
    
    def test_compare_categorical_fields(self, comparator):
        """Test comparison of categorical fields."""
        data1 = pd.DataFrame({'Category': ['A'] * 50 + ['B'] * 50})
        data2 = pd.DataFrame({'Category': ['A'] * 30 + ['B'] * 70})
        
        result = comparator.compare_datasets([data1, data2])
        
        field_comp = result.field_comparisons['Category']
        assert field_comp.data_type == 'categorical'
        assert field_comp.statistical_test == 'Chi-Square'
    
    def test_compare_multiple_datasets(self, comparator):
        """Test comparing more than 2 datasets."""
        np.random.seed(42)
        datasets = [
            pd.DataFrame({'Amount': np.random.lognormal(8, 1.5, 100)})
            for _ in range(3)
        ]
        
        result = comparator.compare_datasets(datasets)
        
        assert len(result.dataset_names) == 3
        field_comp = result.field_comparisons['Amount']
        assert field_comp.statistical_test == 'Kruskal-Wallis'
    
    def test_default_dataset_names(self, comparator, identical_datasets):
        """Test automatic generation of dataset names."""
        result = comparator.compare_datasets(identical_datasets)
        
        assert 'Dataset_1' in result.dataset_names
        assert 'Dataset_2' in result.dataset_names
    
    def test_specific_fields_comparison(self, comparator):
        """Test comparing specific fields only."""
        data1 = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })
        data2 = data1.copy()
        
        result = comparator.compare_datasets(
            [data1, data2],
            fields_to_compare=['A', 'B']
        )
        
        assert 'A' in result.field_comparisons
        assert 'B' in result.field_comparisons
        assert 'C' not in result.field_comparisons
    
    def test_cohens_d_calculation(self, comparator):
        """Test Cohen's d effect size calculation."""
        series1 = pd.Series([1, 2, 3, 4, 5])
        series2 = pd.Series([6, 7, 8, 9, 10])
        
        effect_size = comparator._cohens_d(series1, series2)
        
        assert effect_size is not None
        assert abs(effect_size) > 0  # Should have large effect
    
    def test_overall_statistics(self, comparator, different_datasets):
        """Test calculation of overall statistics."""
        result = comparator.compare_datasets(different_datasets)
        
        assert 'Dataset_1' in result.overall_statistics
        assert 'Dataset_2' in result.overall_statistics
        assert 'total_records' in result.overall_statistics['Dataset_1']
        assert 'completeness' in result.overall_statistics['Dataset_1']
    
    def test_similarity_score_calculation(self, comparator):
        """Test similarity score calculation."""
        # Create datasets with known similarity
        data1 = pd.DataFrame({
            'A': [1] * 100,
            'B': [2] * 100,
            'C': [3] * 100
        })
        data2 = data1.copy()
        
        result = comparator.compare_datasets([data1, data2])
        
        assert result.similarity_score == 1.0  # Identical
    
    def test_recommendations_generation(self, comparator, different_datasets):
        """Test recommendation generation."""
        result = comparator.compare_datasets(different_datasets)
        
        assert len(result.recommendations) > 0
        assert isinstance(result.recommendations[0], str)
    
    def test_compare_fraud_patterns(self, comparator):
        """Test fraud pattern comparison."""
        data1 = pd.DataFrame({
            'Fraud_Type': [None] * 95 + ['Card Cloning'] * 5,
            'Fraud_Confidence': [None] * 95 + [0.8] * 5
        })
        data2 = pd.DataFrame({
            'Fraud_Type': [None] * 90 + ['Card Cloning'] * 10,
            'Fraud_Confidence': [None] * 90 + [0.9] * 10
        })
        
        fraud_comparison = comparator.compare_fraud_patterns(
            [data1, data2],
            names=['Low Fraud', 'High Fraud']
        )
        
        assert 'Low Fraud' in fraud_comparison
        assert 'High Fraud' in fraud_comparison
        assert fraud_comparison['High Fraud']['fraud_rate'] > fraud_comparison['Low Fraud']['fraud_rate']
    
    def test_generate_comparison_visualizations(self, comparator, different_datasets):
        """Test visualization generation."""
        result = comparator.compare_datasets(different_datasets)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            figures = comparator.generate_comparison_visualizations(
                result,
                different_datasets,
                output_dir=tmpdir
            )
            
            assert len(figures) > 0
            assert 'similarity_overview' in figures
            assert 'pvalue_distribution' in figures
            
            # Verify files were created
            output_dir = Path(tmpdir)
            assert (output_dir / 'similarity_overview.png').exists()
            assert (output_dir / 'pvalue_distribution.png').exists()


class TestDatasetComparatorEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def comparator(self):
        return DatasetComparator()
    
    def test_single_dataset_error(self, comparator):
        """Test error when comparing single dataset."""
        data = pd.DataFrame({'A': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="At least 2 datasets required"):
            comparator.compare_datasets([data])
    
    def test_mismatched_names_error(self, comparator):
        """Test error when names don't match dataset count."""
        data1 = pd.DataFrame({'A': [1, 2, 3]})
        data2 = pd.DataFrame({'A': [4, 5, 6]})
        
        with pytest.raises(ValueError, match="Number of names must match"):
            comparator.compare_datasets([data1, data2], names=['Only One'])
    
    def test_no_common_fields(self, comparator):
        """Test comparison with no common fields."""
        data1 = pd.DataFrame({'A': [1, 2, 3]})
        data2 = pd.DataFrame({'B': [4, 5, 6]})
        
        result = comparator.compare_datasets([data1, data2])
        
        assert result.compared_fields == 0
        assert len(result.field_comparisons) == 0
    
    def test_nan_values_handling(self, comparator):
        """Test handling of NaN values."""
        data1 = pd.DataFrame({'Amount': [1, 2, np.nan, 4, 5]})
        data2 = pd.DataFrame({'Amount': [6, np.nan, 8, 9, 10]})
        
        result = comparator.compare_datasets([data1, data2])
        
        # Should handle NaN gracefully
        assert 'Amount' in result.field_comparisons
        assert result.field_comparisons['Amount'].p_value is not None
    
    def test_zero_variance_fields(self, comparator):
        """Test comparison of fields with zero variance."""
        data1 = pd.DataFrame({'Constant': [5] * 100})
        data2 = pd.DataFrame({'Constant': [5] * 100})
        
        result = comparator.compare_datasets([data1, data2])
        
        assert 'Constant' in result.field_comparisons
        # Should not be significant (identical constants)
        assert not result.field_comparisons['Constant'].is_significant
    
    def test_small_sample_size(self, comparator):
        """Test comparison with very small sample."""
        data1 = pd.DataFrame({'Amount': [1, 2]})
        data2 = pd.DataFrame({'Amount': [3, 4]})
        
        result = comparator.compare_datasets([data1, data2])
        
        assert 'Amount' in result.field_comparisons
    
    def test_categorical_with_different_categories(self, comparator):
        """Test categorical comparison with different category sets."""
        data1 = pd.DataFrame({'Cat': ['A', 'B', 'C'] * 10})
        data2 = pd.DataFrame({'Cat': ['D', 'E', 'F'] * 10})
        
        result = comparator.compare_datasets([data1, data2])
        
        field_comp = result.field_comparisons['Cat']
        # Should detect significant difference
        assert field_comp.is_significant


class TestFieldComparison:
    """Test FieldComparison dataclass."""
    
    def test_field_comparison_creation(self):
        """Test creating FieldComparison object."""
        comp = FieldComparison(
            field_name='Amount',
            data_type='numeric',
            statistical_test='KS',
            test_statistic=0.5,
            p_value=0.01,
            is_significant=True,
            effect_size=0.8
        )
        
        assert comp.field_name == 'Amount'
        assert comp.is_significant
        assert comp.effect_size == 0.8


class TestComparisonResult:
    """Test ComparisonResult dataclass."""
    
    def test_comparison_result_creation(self):
        """Test creating ComparisonResult object."""
        result = ComparisonResult(
            dataset_names=['A', 'B'],
            total_fields=10,
            compared_fields=8
        )
        
        assert len(result.dataset_names) == 2
        assert result.total_fields == 10
        assert result.compared_fields == 8
        assert result.similarity_score == 0.0  # Default


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
