"""
Tests for HTML Report Generator.

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
import matplotlib.pyplot as plt

from src.reporting.html_generator import HTMLReportGenerator


class TestHTMLReportGenerator:
    """Test suite for HTML report generation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample transaction data."""
        np.random.seed(42)
        return pd.DataFrame({
            'Transaction_ID': range(1, 101),
            'Transaction_Amount': np.random.lognormal(8, 1.5, 100),
            'Category': np.random.choice(['Groceries', 'Dining', 'Shopping', 'Travel'], 100),
            'Customer_ID': np.random.randint(1, 20, 100),
            'Fraud_Type': [None] * 95 + ['Card Cloning'] * 5,
            'Fraud_Confidence': [None] * 95 + list(np.random.uniform(0.7, 0.95, 5))
        })
    
    @pytest.fixture
    def generator(self):
        """Create HTMLReportGenerator instance."""
        return HTMLReportGenerator()
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics."""
        return {
            'total_transactions': 100,
            'total_amount': 500000,
            'avg_amount': 5000,
            'fraud_rate': 0.05,
            'fraud_count': 5,
            'completeness': 0.98,
            'unique_customers': 19
        }
    
    def test_generator_initialization(self, generator):
        """Test HTMLReportGenerator initializes correctly."""
        assert generator is not None
        assert generator.env is not None
        assert generator.template_dir.exists()
    
    def test_custom_filters_registered(self, generator):
        """Test custom Jinja2 filters are registered."""
        assert 'number_format' in generator.env.filters
        assert 'percentage' in generator.env.filters
        assert 'currency' in generator.env.filters
        assert 'date_format' in generator.env.filters
    
    def test_number_format_filter(self, generator):
        """Test number formatting filter."""
        format_func = generator.env.filters['number_format']
        
        assert format_func(1000) == '1,000'
        assert format_func(1000.5, 2) == '1,000.50'
        assert format_func(1234567.89, 2) == '1,234,567.89'
    
    def test_percentage_filter(self, generator):
        """Test percentage formatting filter."""
        format_func = generator.env.filters['percentage']
        
        assert format_func(0.05, 1) == '5.0%'
        assert format_func(0.5, 2) == '50.00%'
        assert format_func(0.123, 2) == '12.30%'
    
    def test_currency_filter(self, generator):
        """Test currency formatting filter."""
        format_func = generator.env.filters['currency']
        
        assert format_func(5000) == '₹5,000.00'
        assert format_func(1234.56) == '₹1,234.56'
        assert format_func(1000000) == '₹1,000,000.00'
    
    def test_generate_executive_report(self, generator, sample_data, sample_metrics):
        """Test executive report generation."""
        html = generator.generate_executive_report(
            data=sample_data,
            metrics=sample_metrics
        )
        
        assert html is not None
        assert len(html) > 0
        assert 'Executive Summary' in html
        assert 'SynFinance' in html
        assert '100' in html  # Total transactions
    
    def test_executive_report_with_charts(self, generator, sample_data, sample_metrics):
        """Test executive report with embedded charts."""
        # Create simple chart
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['A', 'B', 'C'], [1, 2, 3])
        
        # Use expected chart name from template
        charts = {'distribution': fig}
        
        html = generator.generate_executive_report(
            data=sample_data,
            metrics=sample_metrics,
            charts=charts
        )
        
        assert 'data:image/png;base64,' in html
        plt.close(fig)
    
    def test_executive_report_with_findings(self, generator, sample_data, sample_metrics):
        """Test executive report with findings."""
        findings = [
            {
                'title': 'High Fraud Rate',
                'description': 'Fraud rate is above threshold',
                'severity': 'warning',
                'impact': 'Potential revenue loss'
            }
        ]
        
        html = generator.generate_executive_report(
            data=sample_data,
            metrics=sample_metrics,
            findings=findings
        )
        
        assert 'High Fraud Rate' in html
        assert 'alert-warning' in html
    
    def test_executive_report_with_recommendations(self, generator, sample_data, sample_metrics):
        """Test executive report with recommendations."""
        recommendations = [
            {'title': 'Improve Detection', 'description': 'Enhance fraud detection algorithms'},
            {'title': 'Monitor Trends', 'description': 'Track fraud patterns over time'}
        ]
        
        html = generator.generate_executive_report(
            data=sample_data,
            metrics=sample_metrics,
            recommendations=recommendations
        )
        
        assert 'Improve Detection' in html
        assert 'Monitor Trends' in html
    
    def test_save_report(self, generator, sample_data, sample_metrics):
        """Test saving report to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_report.html'
            
            html = generator.generate_executive_report(
                data=sample_data,
                metrics=sample_metrics,
                output_path=output_path
            )
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0
            
            # Verify content
            saved_content = output_path.read_text(encoding='utf-8')
            assert saved_content == html
    
    def test_figure_to_base64(self, generator):
        """Test matplotlib figure to base64 conversion."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        base64_str = generator._figure_to_base64(fig)
        
        assert base64_str is not None
        assert len(base64_str) > 0
        assert isinstance(base64_str, str)
    
    def test_generate_technical_report(self, generator, sample_data):
        """Test technical report generation."""
        statistics = {
            'mean_amount': sample_data['Transaction_Amount'].mean(),
            'median_amount': sample_data['Transaction_Amount'].median(),
            'std_amount': sample_data['Transaction_Amount'].std()
        }
        
        html = generator.generate_technical_report(
            data=sample_data,
            statistics=statistics
        )
        
        assert html is not None
        assert 'Technical Analysis' in html
    
    def test_generate_fraud_report(self, generator, sample_data):
        """Test fraud report generation."""
        fraud_statistics = {
            'total_patterns': 1,
            'avg_confidence': 0.85
        }
        
        html = generator.generate_fraud_report(
            data=sample_data,
            fraud_statistics=fraud_statistics
        )
        
        assert html is not None
        assert 'Fraud Detection' in html
    
    def test_generate_quality_report(self, generator, sample_data):
        """Test data quality report generation."""
        quality_metrics = {
            'field_variance': 0.85,
            'outlier_count': 3
        }
        
        html = generator.generate_quality_report(
            data=sample_data,
            quality_metrics=quality_metrics
        )
        
        assert html is not None
        assert 'Data Quality' in html


class TestHTMLReportEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def generator(self):
        return HTMLReportGenerator()
    
    def test_empty_dataframe(self, generator):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        metrics = {'total_transactions': 0}
        
        html = generator.generate_executive_report(
            data=empty_df,
            metrics=metrics
        )
        
        assert html is not None
        assert len(html) > 0
    
    def test_missing_optional_parameters(self, generator):
        """Test report generation with minimal parameters."""
        sample_data = pd.DataFrame({'Amount': [100, 200, 300]})
        
        html = generator.generate_executive_report(
            data=sample_data,
            metrics={}
        )
        
        assert html is not None
    
    def test_invalid_chart_object(self, generator):
        """Test handling of invalid chart object."""
        sample_data = pd.DataFrame({'Amount': [100, 200, 300]})
        charts = {'invalid': 'not_a_figure'}
        
        html = generator.generate_executive_report(
            data=sample_data,
            metrics={},
            charts=charts
        )
        
        # Should not include invalid chart
        assert html is not None
        assert 'data:image/png;base64,' not in html
    
    def test_large_dataset(self, generator):
        """Test report generation with large dataset."""
        large_data = pd.DataFrame({
            'Amount': np.random.lognormal(8, 1.5, 10000)
        })
        
        metrics = {
            'total_transactions': 10000,
            'total_amount': large_data['Amount'].sum()
        }
        
        html = generator.generate_executive_report(
            data=large_data,
            metrics=metrics
        )
        
        assert html is not None
        assert '10,000' in html


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
