"""
Tests for Advanced Analytics Module

Tests correlation analysis, feature importance, model performance metrics,
and statistical significance testing.
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from src.analytics.advanced_analytics import (
    CorrelationAnalyzer,
    FeatureImportanceAnalyzer,
    ModelPerformanceAnalyzer,
    StatisticalTestAnalyzer,
    CorrelationResult,
    FeatureImportanceResult,
    ModelMetrics,
    StatisticalTestResult,
    AnalyticsReport,
)


@pytest.fixture
def sample_data():
    """Generate sample classification dataset"""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    return X, y


@pytest.fixture
def trained_model(sample_data):
    """Train a simple RandomForest model"""
    X, y = sample_data
    model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
    model.fit(X[:800], y[:800])
    return model


@pytest.fixture
def feature_names():
    """Generate feature names"""
    return [f'feature_{i}' for i in range(20)]


class TestCorrelationAnalyzer:
    """Test correlation analysis"""
    
    def test_pearson_correlation(self, sample_data, feature_names):
        """Test Pearson correlation"""
        X, _ = sample_data
        analyzer = CorrelationAnalyzer(threshold=0.7)
        result = analyzer.analyze(X, feature_names, method='pearson')
        
        assert isinstance(result, CorrelationResult)
        assert result.method == 'pearson'
        assert result.correlation_matrix.shape == (20, 20)
        assert len(result.feature_names) == 20
        
        # Diagonal should be 1.0
        np.testing.assert_array_almost_equal(
            np.diag(result.correlation_matrix),
            np.ones(20)
        )
    
    def test_spearman_correlation(self, sample_data, feature_names):
        """Test Spearman correlation"""
        X, _ = sample_data
        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze(X, feature_names, method='spearman')
        
        assert result.method == 'spearman'
        assert result.correlation_matrix.shape == (20, 20)
    
    def test_highly_correlated_pairs(self, sample_data, feature_names):
        """Test identification of highly correlated pairs"""
        X, _ = sample_data
        
        # Create highly correlated features
        X_modified = X.copy()
        X_modified[:, 1] = X_modified[:, 0] * 0.95 + np.random.randn(1000) * 0.1
        
        analyzer = CorrelationAnalyzer(threshold=0.7)
        result = analyzer.analyze(X_modified, feature_names, method='pearson')
        pairs = result.get_highly_correlated_pairs()
        
        assert len(pairs) > 0
        # Check that feature_0 and feature_1 are highly correlated
        pair_names = [(p[0], p[1]) for p in pairs]
        assert ('feature_0', 'feature_1') in pair_names or ('feature_1', 'feature_0') in pair_names
    
    def test_compare_feature_groups(self, sample_data, feature_names):
        """Test cross-correlation between feature groups"""
        X, _ = sample_data
        analyzer = CorrelationAnalyzer()
        
        group1 = [0, 1, 2, 3, 4]
        group2 = [5, 6, 7, 8, 9]
        
        cross_corr, top_corr = analyzer.compare_feature_groups(
            X, feature_names, group1, group2, method='pearson'
        )
        
        assert cross_corr.shape == (5, 5)
        assert len(top_corr) <= 10


class TestFeatureImportanceAnalyzer:
    """Test feature importance analysis"""
    
    def test_permutation_importance(self, sample_data, trained_model, feature_names):
        """Test permutation importance"""
        X, y = sample_data
        X_test, y_test = X[800:], y[800:]
        
        analyzer = FeatureImportanceAnalyzer(n_repeats=5)
        result = analyzer.permutation_importance(
            trained_model, X_test, y_test, feature_names
        )
        
        assert isinstance(result, FeatureImportanceResult)
        assert result.method == 'permutation'
        assert len(result.importances) == 20
        assert result.importances_std is not None
        assert len(result.importances_std) == 20
    
    def test_tree_based_importance(self, trained_model, feature_names):
        """Test tree-based importance"""
        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.tree_based_importance(trained_model, feature_names)
        
        assert result.method == 'tree_based'
        assert len(result.importances) == 20
        # Importances should sum to approximately 1.0
        assert abs(result.importances.sum() - 1.0) < 0.01
    
    def test_mutual_information_importance(self, sample_data, feature_names):
        """Test mutual information importance"""
        X, y = sample_data
        X_train, y_train = X[:800], y[:800]
        
        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.mutual_information_importance(X_train, y_train, feature_names)
        
        assert result.method == 'mutual_info'
        assert len(result.importances) == 20
        assert all(result.importances >= 0)  # MI is non-negative
    
    def test_get_top_features(self, trained_model, feature_names):
        """Test getting top N features"""
        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.tree_based_importance(trained_model, feature_names)
        
        top_5 = result.get_top_features(5)
        assert len(top_5) == 5
        # Should be sorted by importance (descending)
        importances = [imp for _, imp in top_5]
        assert importances == sorted(importances, reverse=True)
    
    def test_get_bottom_features(self, trained_model, feature_names):
        """Test getting bottom N features"""
        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.tree_based_importance(trained_model, feature_names)
        
        bottom_5 = result.get_bottom_features(5)
        assert len(bottom_5) == 5
        # Should be sorted by importance (ascending)
        importances = [imp for _, imp in bottom_5]
        assert importances == sorted(importances)
    
    def test_analyze_all(self, sample_data, trained_model, feature_names):
        """Test analyze_all method"""
        X, y = sample_data
        X_train, y_train = X[:800], y[:800]
        X_test, y_test = X[800:], y[800:]
        
        analyzer = FeatureImportanceAnalyzer(n_repeats=3)
        results = analyzer.analyze_all(
            trained_model, X_train, y_train, X_test, y_test, feature_names
        )
        
        assert len(results) >= 2  # At least tree_based and permutation
        methods = [r.method for r in results]
        assert 'tree_based' in methods


class TestModelPerformanceAnalyzer:
    """Test model performance metrics"""
    
    def test_analyze_basic_metrics(self):
        """Test basic performance metrics"""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 0])
        
        analyzer = ModelPerformanceAnalyzer()
        metrics = analyzer.analyze(y_true, y_pred)
        
        assert isinstance(metrics, ModelMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1 <= 1
        assert metrics.confusion_matrix.shape == (2, 2)
    
    def test_analyze_with_probabilities(self):
        """Test metrics with probability predictions"""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 0])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.7, 0.2])
        
        analyzer = ModelPerformanceAnalyzer()
        metrics = analyzer.analyze(y_true, y_pred, y_proba)
        
        assert 0 <= metrics.roc_auc <= 1
        assert 0 <= metrics.average_precision <= 1
        assert len(metrics.fpr) > 0
        assert len(metrics.tpr) > 0
        assert len(metrics.precision_curve) > 0
        assert len(metrics.recall_curve) > 0
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions"""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = y_true.copy()
        
        analyzer = ModelPerformanceAnalyzer()
        metrics = analyzer.analyze(y_true, y_pred)
        
        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
    
    def test_to_dict(self):
        """Test metrics to_dict conversion"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        
        analyzer = ModelPerformanceAnalyzer()
        metrics = analyzer.analyze(y_true, y_pred)
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert 'accuracy' in metrics_dict
        assert 'precision' in metrics_dict
        assert 'f1' in metrics_dict
        assert 'confusion_matrix' in metrics_dict
    
    def test_compare_models(self):
        """Test model comparison"""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        
        models_dict = {
            'model1': (np.array([0, 0, 1, 1, 0, 0, 1, 0]), None),
            'model2': (np.array([0, 0, 1, 1, 0, 1, 1, 0]), None),
        }
        
        analyzer = ModelPerformanceAnalyzer()
        results = analyzer.compare_models(models_dict, y_true)
        
        assert len(results) == 2
        assert 'model1' in results
        assert 'model2' in results
        assert results['model2'].accuracy > results['model1'].accuracy


class TestStatisticalTestAnalyzer:
    """Test statistical significance tests"""
    
    def test_chi_square_test(self):
        """Test chi-square test of independence"""
        # Create contingency table
        contingency = np.array([[10, 20], [30, 40]])
        
        analyzer = StatisticalTestAnalyzer()
        result = analyzer.chi_square_test(contingency)
        
        assert isinstance(result, StatisticalTestResult)
        assert result.test_name == 'chi_square'
        assert result.statistic > 0
        assert 0 <= result.p_value <= 1
        assert result.degrees_of_freedom is not None
    
    def test_t_test(self):
        """Test independent samples t-test"""
        group1 = np.random.normal(0, 1, 100)
        group2 = np.random.normal(0.5, 1, 100)
        
        analyzer = StatisticalTestAnalyzer()
        result = analyzer.t_test(group1, group2)
        
        assert result.test_name == 't_test'
        assert 0 <= result.p_value <= 1
        assert result.degrees_of_freedom == 198
    
    def test_anova_test(self):
        """Test one-way ANOVA"""
        group1 = np.random.normal(0, 1, 50)
        group2 = np.random.normal(0.5, 1, 50)
        group3 = np.random.normal(1.0, 1, 50)
        
        analyzer = StatisticalTestAnalyzer()
        result = analyzer.anova_test(group1, group2, group3)
        
        assert result.test_name == 'anova'
        assert 0 <= result.p_value <= 1
        assert result.degrees_of_freedom == 147
    
    def test_significance_flag(self):
        """Test is_significant flag"""
        # Create clearly different groups
        group1 = np.ones(100)
        group2 = np.ones(100) * 5
        
        analyzer = StatisticalTestAnalyzer(alpha=0.05)
        result = analyzer.t_test(group1, group2)
        
        assert result.is_significant == True
        assert result.p_value < 0.05
    
    def test_test_fraud_vs_normal(self):
        """Test fraud vs normal comparison"""
        feature_values = np.concatenate([
            np.random.normal(0, 1, 800),  # Normal
            np.random.normal(2, 1, 200),  # Fraud (higher mean)
        ])
        is_fraud = np.array([False] * 800 + [True] * 200)
        
        analyzer = StatisticalTestAnalyzer()
        result = analyzer.test_fraud_vs_normal(feature_values, is_fraud)
        
        assert result.test_name == 't_test'
        assert result.is_significant == True  # Should be significantly different


class TestAnalyticsReport:
    """Test analytics report"""
    
    def test_report_creation(self):
        """Test creating basic report"""
        dataset_info = {
            'total_transactions': 1000,
            'fraud_transactions': 100,
            'fraud_rate': 0.1,
        }
        
        report = AnalyticsReport(dataset_info=dataset_info)
        
        assert report.dataset_info == dataset_info
        assert report.correlation_results is None
        assert report.model_metrics is None
    
    def test_report_to_dict(self):
        """Test converting report to dictionary"""
        dataset_info = {'total': 1000}
        report = AnalyticsReport(dataset_info=dataset_info)
        
        report_dict = report.to_dict()
        
        assert isinstance(report_dict, dict)
        assert 'dataset_info' in report_dict
        assert report_dict['dataset_info'] == dataset_info
