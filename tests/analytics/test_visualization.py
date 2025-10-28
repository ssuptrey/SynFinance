"""
Tests for Visualization Framework

Tests distribution plots, correlation heatmaps, feature importance charts,
ROC curves, and dashboard generation.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.analytics.visualization import VisualizationFramework


@pytest.fixture
def viz_framework():
    """Create visualization framework"""
    return VisualizationFramework()


@pytest.fixture
def sample_data():
    """Generate sample data"""
    np.random.seed(42)
    return {
        'normal': np.random.normal(0, 1, 1000),
        'skewed': np.random.exponential(2, 1000),
        'uniform': np.random.uniform(0, 10, 1000),
    }


@pytest.fixture
def feature_names():
    """Generate feature names"""
    return [f'feature_{i}' for i in range(10)]


class TestVisualizationFramework:
    """Test visualization framework"""
    
    def test_initialization(self):
        """Test framework initialization"""
        viz = VisualizationFramework(figsize=(10, 8))
        assert viz.figsize == (10, 8)
    
    def test_plot_distribution(self, viz_framework, sample_data):
        """Test distribution plot"""
        fig = viz_framework.plot_distribution(
            sample_data['normal'],
            title="Normal Distribution",
            kde=True,
            show_stats=True
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_distribution_without_kde(self, viz_framework, sample_data):
        """Test distribution plot without KDE"""
        fig = viz_framework.plot_distribution(
            sample_data['uniform'],
            kde=False,
            show_stats=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_boxplot(self, viz_framework, sample_data):
        """Test boxplot"""
        fig = viz_framework.plot_boxplot(
            sample_data,
            title="Feature Comparison"
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_violin(self, viz_framework, sample_data):
        """Test violin plot"""
        fig = viz_framework.plot_violin(
            sample_data,
            title="Distribution by Group"
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_correlation_heatmap(self, viz_framework, feature_names):
        """Test correlation heatmap"""
        # Create correlation matrix
        n = len(feature_names)
        corr_matrix = np.random.randn(n, n)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(corr_matrix, 1.0)
        corr_matrix = np.clip(corr_matrix, -1, 1)
        
        fig = viz_framework.plot_correlation_heatmap(
            corr_matrix,
            feature_names,
            annot=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_confusion_matrix(self, viz_framework):
        """Test confusion matrix plot"""
        cm = np.array([[80, 20], [10, 90]])
        
        fig = viz_framework.plot_confusion_matrix(
            cm,
            labels=['Normal', 'Fraud'],
            normalize=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_confusion_matrix_normalized(self, viz_framework):
        """Test normalized confusion matrix"""
        cm = np.array([[80, 20], [10, 90]])
        
        fig = viz_framework.plot_confusion_matrix(
            cm,
            normalize=True
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_feature_importance(self, viz_framework, feature_names):
        """Test feature importance plot"""
        importances = np.random.rand(len(feature_names))
        
        fig = viz_framework.plot_feature_importance(
            feature_names,
            importances,
            top_n=5
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_feature_importance_with_std(self, viz_framework, feature_names):
        """Test feature importance with error bars"""
        importances = np.random.rand(len(feature_names))
        std = np.random.rand(len(feature_names)) * 0.1
        
        fig = viz_framework.plot_feature_importance(
            feature_names,
            importances,
            std=std,
            top_n=10
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_roc_curve(self, viz_framework):
        """Test ROC curve plot"""
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr)  # Simulated ROC curve
        roc_auc = 0.85
        
        fig = viz_framework.plot_roc_curve(fpr, tpr, roc_auc)
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_precision_recall_curve(self, viz_framework):
        """Test precision-recall curve"""
        recall = np.linspace(0, 1, 100)
        precision = 1 - recall * 0.5  # Simulated PR curve
        avg_precision = 0.75
        
        fig = viz_framework.plot_precision_recall_curve(
            precision, recall, avg_precision
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_time_series(self, viz_framework):
        """Test time series plot"""
        timestamps = np.arange(100)
        values = np.cumsum(np.random.randn(100))
        
        fig = viz_framework.plot_time_series(
            timestamps, values, add_trend=True
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_time_series_without_trend(self, viz_framework):
        """Test time series without trend line"""
        timestamps = np.arange(50)
        values = np.random.randn(50)
        
        fig = viz_framework.plot_time_series(
            timestamps, values, add_trend=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_geographic_heatmap(self, viz_framework):
        """Test geographic heatmap"""
        cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata']
        values = np.array([100, 85, 70, 60, 50])
        
        fig = viz_framework.plot_geographic_heatmap(cities, values)
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_comparison(self, viz_framework):
        """Test model comparison plot"""
        metrics_dict = {
            'Model A': {
                'accuracy': 0.85,
                'precision': 0.80,
                'recall': 0.75,
                'f1': 0.77,
                'roc_auc': 0.88,
            },
            'Model B': {
                'accuracy': 0.90,
                'precision': 0.88,
                'recall': 0.82,
                'f1': 0.85,
                'roc_auc': 0.92,
            },
        }
        
        fig = viz_framework.plot_comparison(metrics_dict)
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_interactive_scatter_without_plotly(self, viz_framework):
        """Test interactive scatter (should warn if plotly unavailable)"""
        x = np.random.randn(100)
        y = np.random.randn(100)
        
        # This may return None or a plotly figure depending on availability
        result = viz_framework.plot_interactive_scatter(x, y)
        # Don't assert type as it depends on plotly availability
    
    def test_plot_interactive_correlation_without_plotly(self, viz_framework, feature_names):
        """Test interactive correlation heatmap"""
        n = len(feature_names)
        corr_matrix = np.random.randn(n, n)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix, 1.0)
        corr_matrix = np.clip(corr_matrix, -1, 1)
        
        result = viz_framework.plot_interactive_correlation(
            corr_matrix, feature_names
        )
        # Don't assert type as it depends on plotly availability
    
    def test_plot_interactive_feature_importance_without_plotly(self, viz_framework, feature_names):
        """Test interactive feature importance"""
        importances = np.random.rand(len(feature_names))
        
        result = viz_framework.plot_interactive_feature_importance(
            feature_names, importances
        )
        # Don't assert type as it depends on plotly availability
