"""
Tests for HTML Dashboard Generator

Tests dashboard generation, chart embedding, and HTML output.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile

from src.analytics.dashboard import HTMLDashboardGenerator
from src.analytics.advanced_analytics import ModelMetrics, FeatureImportanceResult


@pytest.fixture
def dashboard_generator():
    """Create dashboard generator"""
    return HTMLDashboardGenerator()


@pytest.fixture
def sample_model_metrics():
    """Create sample model metrics"""
    return ModelMetrics(
        accuracy=0.85,
        precision=0.80,
        recall=0.75,
        f1=0.77,
        roc_auc=0.88,
        average_precision=0.82,
        confusion_matrix=np.array([[80, 20], [25, 75]]),
        classification_report="Test report",
        fpr=np.linspace(0, 1, 100),
        tpr=np.linspace(0, 1, 100),
        roc_thresholds=np.linspace(0, 1, 100),
        precision_curve=np.linspace(0, 1, 100),
        recall_curve=np.linspace(0, 1, 100),
        pr_thresholds=np.linspace(0, 1, 100),
    )


@pytest.fixture
def sample_importance_results():
    """Create sample feature importance results"""
    return [
        FeatureImportanceResult(
            method='tree_based',
            feature_names=[f'feature_{i}' for i in range(10)],
            importances=np.random.rand(10),
        ),
        FeatureImportanceResult(
            method='permutation',
            feature_names=[f'feature_{i}' for i in range(10)],
            importances=np.random.rand(10),
            importances_std=np.random.rand(10) * 0.1,
        ),
    ]


@pytest.fixture
def sample_chart():
    """Create a simple matplotlib chart"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
    ax.set_title("Sample Chart")
    return fig


class TestHTMLDashboardGenerator:
    """Test HTML dashboard generator"""
    
    def test_initialization(self, dashboard_generator):
        """Test dashboard generator initialization"""
        assert dashboard_generator.template is not None
        assert len(dashboard_generator.template) > 0
    
    def test_figure_to_base64(self, dashboard_generator, sample_chart):
        """Test converting matplotlib figure to base64"""
        base64_str = dashboard_generator._figure_to_base64(sample_chart)
        
        assert base64_str.startswith('data:image/png;base64,')
        assert len(base64_str) > 100
        
        plt.close(sample_chart)
    
    def test_generate_navigation(self, dashboard_generator):
        """Test navigation generation"""
        sections = ["Overview", "Feature Importance", "Model Performance"]
        nav_html = dashboard_generator._generate_navigation(sections)
        
        assert '<div class="nav">' in nav_html
        assert 'Overview' in nav_html
        assert 'Feature Importance' in nav_html
        assert 'Model Performance' in nav_html
    
    def test_generate_stats_grid(self, dashboard_generator):
        """Test stats grid generation"""
        stats = {
            "Total Transactions": "1,000",
            "Fraud Rate": "5.2%",
            "Accuracy": "85.0%",
        }
        grid_html = dashboard_generator._generate_stats_grid(stats)
        
        assert '<div class="stats-grid">' in grid_html
        assert 'Total Transactions' in grid_html
        assert '1,000' in grid_html
        assert 'Fraud Rate' in grid_html
    
    def test_generate_chart(self, dashboard_generator, sample_chart):
        """Test chart HTML generation"""
        chart_html = dashboard_generator._generate_chart("Test Chart", sample_chart)
        
        assert '<div class="chart-container">' in chart_html
        assert 'Test Chart' in chart_html
        assert '<img src="data:image/png;base64,' in chart_html
        
        plt.close(sample_chart)
    
    def test_generate_table(self, dashboard_generator):
        """Test table generation"""
        headers = ['Feature', 'Importance', 'Rank']
        rows = [
            ['feature_0', 0.95, 1],  # Use high value for metric-good
            ['feature_1', 0.70, 2],  # Use medium value for metric-warning
            ['feature_2', 0.15, 3],  # Use low value for metric-bad
        ]
        table_html = dashboard_generator._generate_table(headers, rows)
        
        assert '<table>' in table_html
        assert 'Feature' in table_html
        assert 'feature_0' in table_html
        assert 'metric-good' in table_html  # High values should be green
        assert 'metric-warning' in table_html  # Medium values should be yellow
        assert 'metric-bad' in table_html  # Low values should be red
    
    def test_generate_overview_section(self, dashboard_generator, sample_model_metrics):
        """Test overview section generation"""
        dataset_info = {
            'total_transactions': 1000,
            'fraud_transactions': 52,
            'fraud_rate': 0.052,
            'total_features': 68,
        }
        
        html = dashboard_generator.generate_overview_section(dataset_info, sample_model_metrics)
        
        assert '<section class="section" id="overview">' in html
        assert 'Dataset Overview' in html
        assert '1,000' in html
        assert '5.20%' in html
    
    def test_generate_feature_importance_section(
        self, dashboard_generator, sample_importance_results, sample_chart
    ):
        """Test feature importance section generation"""
        charts = {'tree_based_importance': sample_chart}
        
        html = dashboard_generator.generate_feature_importance_section(
            sample_importance_results, charts
        )
        
        assert '<section class="section" id="feature-importance">' in html
        assert 'Feature Importance' in html
        assert 'tree_based' in html.lower()
        
        plt.close(sample_chart)
    
    def test_generate_model_performance_section(
        self, dashboard_generator, sample_model_metrics, sample_chart
    ):
        """Test model performance section generation"""
        charts = {
            'confusion_matrix': sample_chart,
            'roc_curve': sample_chart,
            'pr_curve': sample_chart,
        }
        
        html = dashboard_generator.generate_model_performance_section(
            sample_model_metrics, charts
        )
        
        assert '<section class="section" id="model-performance">' in html
        assert 'Model Performance' in html
        assert 'Confusion Matrix' in html
        
        plt.close(sample_chart)
    
    def test_generate_anomaly_section(self, dashboard_generator, sample_chart):
        """Test anomaly detection section generation"""
        anomaly_stats = {
            'total_anomalies': 150,
            'anomaly_rate': 0.15,
            'avg_severity': 0.65,
            'high_severity_count': 25,
            'anomaly_types': {
                'Behavioral': 50,
                'Geographic': 40,
                'Temporal': 35,
                'Amount': 25,
            }
        }
        charts = {'anomaly_distribution': sample_chart}
        
        html = dashboard_generator.generate_anomaly_section(anomaly_stats, charts)
        
        assert '<section class="section" id="anomaly-detection">' in html
        assert 'Anomaly Detection' in html
        assert 'Behavioral' in html
        
        plt.close(sample_chart)
    
    def test_generate_full_dashboard(
        self, dashboard_generator, sample_model_metrics, 
        sample_importance_results, sample_chart
    ):
        """Test full dashboard generation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dashboard.html"
            
            dataset_info = {
                'total_transactions': 1000,
                'fraud_transactions': 100,
                'fraud_rate': 0.1,
                'total_features': 68,
            }
            
            charts = {
                'tree_based_importance': sample_chart,
                'confusion_matrix': sample_chart,
                'roc_curve': sample_chart,
            }
            
            dashboard_generator.generate_dashboard(
                output_path=str(output_path),
                title="Test Dashboard",
                subtitle="Test Analysis",
                dataset_info=dataset_info,
                model_metrics=sample_model_metrics,
                importance_results=sample_importance_results,
                charts=charts,
            )
            
            assert output_path.exists()
            assert output_path.stat().st_size > 1000  # Should be reasonably sized
            
            # Read and check content
            html_content = output_path.read_text(encoding='utf-8')
            assert 'Test Dashboard' in html_content
            assert 'Dataset Overview' in html_content
            assert 'Feature Importance' in html_content
            
            plt.close(sample_chart)
    
    def test_generate_minimal_dashboard(self, dashboard_generator):
        """Test dashboard with minimal inputs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "minimal_dashboard.html"
            
            dashboard_generator.generate_dashboard(
                output_path=str(output_path),
                dataset_info={'total_transactions': 100}
            )
            
            assert output_path.exists()
            html_content = output_path.read_text(encoding='utf-8')
            assert 'Fraud Detection Analytics Dashboard' in html_content
