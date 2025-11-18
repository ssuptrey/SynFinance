"""
Tests for statistical plots module.

Tests model diagnostics, performance curves, and statistical visualizations.
"""

import pytest
import numpy as np
import pandas as pd

# Configure matplotlib to use non-interactive backend for testing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.figure as mpl_figure

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.visualizations.statistical_plots import StatisticalPlots

# Check if statsmodels is available
try:
    import statsmodels
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# Fixtures

@pytest.fixture
def regression_data():
    """Generate regression data."""
    np.random.seed(42)
    x = np.random.randn(200)
    y = 2 * x + 1 + np.random.randn(200) * 0.5
    return x, y


@pytest.fixture
def classification_data():
    """Generate classification data."""
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    return X, y


@pytest.fixture
def trained_classifier(classification_data):
    """Train a simple classifier."""
    X, y = classification_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]
    
    return y_test, y_pred, y_score


@pytest.fixture
def time_series_data():
    """Generate time series data."""
    np.random.seed(42)
    return np.random.randn(100).cumsum()


@pytest.fixture
def feature_importance_data():
    """Generate feature importance data."""
    features = [f'feature_{i}' for i in range(10)]
    importances = np.random.rand(10)
    return features, importances


# Tests for Statistical Plots

class TestStatisticalPlots:
    """Test statistical plot generation."""
    
    def test_initialization(self):
        """Test StatisticalPlots initialization."""
        plots = StatisticalPlots(theme="default")
        assert plots.theme.name == "default"
    
    def test_plot_regression(self, regression_data):
        """Test regression plot."""
        x, y = regression_data
        plots = StatisticalPlots()
        
        fig = plots.plot_regression(
            x, y,
            title='Regression Analysis',
            xlabel='X',
            ylabel='Y'
        )
        
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_regression_with_series(self, regression_data):
        """Test regression plot with pandas Series."""
        x, y = regression_data
        plots = StatisticalPlots()
        
        x_series = pd.Series(x)
        y_series = pd.Series(y)
        
        fig = plots.plot_regression(x_series, y_series)
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_residuals(self):
        """Test residual plots."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.3
        
        plots = StatisticalPlots()
        fig = plots.plot_residuals(
            y_true,
            y_pred,
            title='Residual Analysis'
        )
        
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_roc_curve(self, trained_classifier):
        """Test ROC curve plot."""
        y_true, y_pred, y_score = trained_classifier
        
        plots = StatisticalPlots()
        fig = plots.plot_roc_curve(
            y_true,
            y_score,
            title='ROC Curve'
        )
        
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_roc_curve_with_list(self):
        """Test ROC curve with list inputs."""
        y_true = [0, 0, 1, 1, 0, 1, 1, 0, 1, 0]
        y_score = [0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.85, 0.15, 0.75, 0.25]
        
        plots = StatisticalPlots()
        fig = plots.plot_roc_curve(y_true, y_score)
        
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_confusion_matrix(self, trained_classifier):
        """Test confusion matrix plot."""
        y_true, y_pred, y_score = trained_classifier
        
        plots = StatisticalPlots()
        fig = plots.plot_confusion_matrix(
            y_true,
            y_pred,
            labels=['Class 0', 'Class 1'],
            title='Confusion Matrix'
        )
        
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_confusion_matrix_normalized(self, trained_classifier):
        """Test normalized confusion matrix."""
        y_true, y_pred, y_score = trained_classifier
        
        plots = StatisticalPlots()
        fig = plots.plot_confusion_matrix(
            y_true,
            y_pred,
            normalize=True,
            title='Normalized Confusion Matrix'
        )
        
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_feature_importance(self, feature_importance_data):
        """Test feature importance plot."""
        features, importances = feature_importance_data
        
        plots = StatisticalPlots()
        fig = plots.plot_feature_importance(
            features,
            importances,
            title='Feature Importance'
        )
        
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_feature_importance_top_n(self, feature_importance_data):
        """Test feature importance with top N."""
        features, importances = feature_importance_data
        
        plots = StatisticalPlots()
        fig = plots.plot_feature_importance(
            features,
            importances,
            top_n=5,
            title='Top 5 Features'
        )
        
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_learning_curve(self):
        """Test learning curve plot."""
        train_sizes = np.array([50, 100, 200, 300, 400])
        train_scores = np.array([0.7, 0.75, 0.8, 0.82, 0.83])
        val_scores = np.array([0.65, 0.72, 0.76, 0.78, 0.79])
        
        plots = StatisticalPlots()
        fig = plots.plot_learning_curve(
            train_sizes,
            train_scores,
            val_scores,
            title='Learning Curve'
        )
        
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_learning_curve_with_std(self):
        """Test learning curve with standard deviation."""
        train_sizes = np.array([50, 100, 200, 300, 400])
        # 2D array with multiple runs
        train_scores = np.array([
            [0.7, 0.71, 0.69],
            [0.75, 0.76, 0.74],
            [0.8, 0.81, 0.79],
            [0.82, 0.83, 0.81],
            [0.83, 0.84, 0.82]
        ])
        val_scores = np.array([
            [0.65, 0.66, 0.64],
            [0.72, 0.73, 0.71],
            [0.76, 0.77, 0.75],
            [0.78, 0.79, 0.77],
            [0.79, 0.80, 0.78]
        ])
        
        plots = StatisticalPlots()
        fig = plots.plot_learning_curve(
            train_sizes,
            train_scores,
            val_scores,
            title='Learning Curve with Confidence'
        )
        
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_validation_curve(self):
        """Test validation curve plot."""
        param_range = np.array([0.001, 0.01, 0.1, 1.0, 10.0])
        train_scores = np.array([0.6, 0.75, 0.85, 0.82, 0.78])
        val_scores = np.array([0.58, 0.72, 0.80, 0.81, 0.75])
        
        plots = StatisticalPlots()
        fig = plots.plot_validation_curve(
            param_range,
            train_scores,
            val_scores,
            param_name='C (Regularization)',
            title='Validation Curve'
        )
        
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_validation_curve_log_scale(self):
        """Test validation curve with log scale."""
        param_range = np.logspace(-3, 2, 6)
        train_scores = np.array([0.6, 0.7, 0.8, 0.85, 0.82, 0.78])
        val_scores = np.array([0.58, 0.68, 0.77, 0.82, 0.81, 0.75])
        
        plots = StatisticalPlots()
        fig = plots.plot_validation_curve(
            param_range,
            train_scores,
            val_scores,
            param_name='Alpha',
            xscale='log'
        )
        
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_lift_curve(self):
        """Test lift curve plot."""
        np.random.seed(42)
        y_true = np.random.choice([0, 1], 1000, p=[0.8, 0.2])
        y_score = np.random.rand(1000)
        
        plots = StatisticalPlots()
        fig = plots.plot_lift_curve(
            y_true,
            y_score,
            title='Lift Curve'
        )
        
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    @pytest.mark.skipif(not STATSMODELS_AVAILABLE, reason="statsmodels not installed")
    def test_plot_acf_pacf(self, time_series_data):
        """Test ACF/PACF plots."""
        plots = StatisticalPlots()
        fig = plots.plot_acf_pacf(
            time_series_data,
            lags=20,
            title='Autocorrelation Analysis'
        )
        
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    @pytest.mark.skipif(not STATSMODELS_AVAILABLE, reason="statsmodels not installed")
    def test_plot_acf_pacf_with_series(self, time_series_data):
        """Test ACF/PACF with pandas Series."""
        plots = StatisticalPlots()
        series = pd.Series(time_series_data)
        
        fig = plots.plot_acf_pacf(series, lags=30)
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)


# Tests for edge cases

class TestStatisticalPlotsEdgeCases:
    """Test edge cases for statistical plots."""
    
    def test_perfect_predictions(self):
        """Test plots with perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        
        plots = StatisticalPlots()
        
        # Confusion matrix should work
        fig = plots.plot_confusion_matrix(y_true, y_pred)
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_single_class_predictions(self):
        """Test handling of single class predictions."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0])
        
        plots = StatisticalPlots()
        
        # Should handle gracefully
        try:
            fig = plots.plot_confusion_matrix(y_true, y_pred)
            plt.close(fig)
        except Exception:
            # Some metrics may not be defined for single class
            pass
    
    def test_empty_feature_importance(self):
        """Test feature importance with empty lists."""
        plots = StatisticalPlots()
        
        try:
            fig = plots.plot_feature_importance([], [])
            plt.close(fig)
        except Exception:
            # Expected to fail with empty data
            pass
    
    def test_single_point_regression(self):
        """Test regression with minimal data."""
        plots = StatisticalPlots()
        
        # Single point should raise or handle gracefully
        try:
            fig = plots.plot_regression([1], [2])
            plt.close(fig)
        except Exception:
            # Expected for insufficient data
            pass
    
    def test_constant_values(self):
        """Test handling of constant values."""
        y_true = np.array([5.0] * 50)
        y_pred = np.array([5.0] * 50)
        
        plots = StatisticalPlots()
        fig = plots.plot_residuals(y_true, y_pred)
        
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_missing_values_in_regression(self):
        """Test regression plot with NaN values."""
        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([2, 4, 6, np.nan, 10])
        
        plots = StatisticalPlots()
        
        # Should handle or raise appropriately
        try:
            fig = plots.plot_regression(x, y)
            plt.close(fig)
        except Exception:
            # NaN values may cause issues
            pass


# Integration tests

class TestStatisticalPlotsIntegration:
    """Integration tests for statistical plots."""
    
    def test_full_model_diagnostics_workflow(self, trained_classifier):
        """Test complete model diagnostics workflow."""
        y_true, y_pred, y_score = trained_classifier
        plots = StatisticalPlots()
        
        # ROC Curve
        fig1 = plots.plot_roc_curve(y_true, y_score, title='Model ROC')
        assert isinstance(fig1, mpl_figure.Figure)
        
        # Confusion Matrix
        fig2 = plots.plot_confusion_matrix(y_true, y_pred, title='Model Confusion Matrix')
        assert isinstance(fig2, mpl_figure.Figure)
        
        # Lift Curve
        fig3 = plots.plot_lift_curve(y_true, y_score, title='Model Lift')
        assert isinstance(fig3, mpl_figure.Figure)
        
        plt.close('all')
    
    def test_different_themes(self, regression_data):
        """Test statistical plots with different themes."""
        x, y = regression_data
        
        for theme in ['default', 'dark', 'colorblind']:
            plots = StatisticalPlots(theme=theme)
            fig = plots.plot_regression(x, y)
            assert isinstance(fig, mpl_figure.Figure)
            plt.close(fig)
    
    def test_multiple_plots_same_figure(self, trained_classifier):
        """Test creating multiple diagnostic plots."""
        y_true, y_pred, y_score = trained_classifier
        plots = StatisticalPlots()
        
        figures = []
        
        # Create multiple plots
        figures.append(plots.plot_roc_curve(y_true, y_score))
        figures.append(plots.plot_confusion_matrix(y_true, y_pred))
        figures.append(plots.plot_lift_curve(y_true, y_score))
        
        # Verify all created
        assert len(figures) == 3
        assert all(isinstance(f, mpl_figure.Figure) for f in figures)
        
        plt.close('all')
    
    def test_learning_curves_comprehensive(self):
        """Test comprehensive learning curve analysis."""
        plots = StatisticalPlots()
        
        # Simulate cross-validation results
        train_sizes = np.array([100, 200, 400, 800, 1600])
        train_scores = np.random.rand(5, 5) * 0.2 + 0.7  # 5 sizes, 5 CV folds
        val_scores = np.random.rand(5, 5) * 0.2 + 0.65
        
        fig = plots.plot_learning_curve(
            train_sizes,
            train_scores,
            val_scores,
            title='Cross-Validated Learning Curve'
        )
        
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
