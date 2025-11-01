"""
Comprehensive tests for all analytics modules.

Tests cover:
- StatisticalAnalyzer
- CorrelationAnalyzer
- DistributionFitter
- TrendAnalyzer
- DataProfiler
- StatisticalTests
- AnalysisReport
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.analytics.statistical_analyzer import StatisticalAnalyzer
from src.analytics.correlation_analyzer import CorrelationAnalyzer
from src.analytics.distribution_fitter import DistributionFitter
from src.analytics.trend_analyzer import TrendAnalyzer
from src.analytics.data_profiler import DataProfiler
from src.analytics.statistical_tests import StatisticalTests
from src.analytics.analysis_report import AnalysisReport


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_numeric_data():
    """Generate sample numeric data."""
    np.random.seed(42)
    return pd.Series(np.random.normal(100, 15, 1000))


@pytest.fixture
def sample_dataframe():
    """Generate sample DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        'amount': np.random.normal(100, 15, 1000),
        'count': np.random.poisson(10, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'is_fraud': np.random.choice([True, False], 1000, p=[0.1, 0.9]),
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='h'),
        'score': np.random.uniform(0, 100, 1000),
    })


@pytest.fixture
def timeseries_data():
    """Generate time-series data with trend and seasonality."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=365, freq='D')
    trend = np.linspace(100, 150, 365)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 7)  # Weekly seasonality
    noise = np.random.normal(0, 5, 365)
    values = trend + seasonal + noise
    
    return pd.DataFrame({
        'timestamp': dates,
        'value': values
    })


# ============================================================================
# StatisticalAnalyzer Tests
# ============================================================================

class TestStatisticalAnalyzer:
    
    def test_describe_dataset(self, sample_dataframe):
        """Test comprehensive dataset description."""
        analyzer = StatisticalAnalyzer()
        result = analyzer.describe_dataset(sample_dataframe)
        
        assert "dataset_info" in result
        assert result["dataset_info"]["row_count"] == 1000
        assert result["dataset_info"]["column_count"] == 6
        assert len(result["numeric_fields"]) == 3  # amount, count, score
        assert len(result["categorical_fields"]) == 1  # category
    
    def test_compute_statistics_normal_data(self, sample_numeric_data):
        """Test statistics computation on normal data."""
        analyzer = StatisticalAnalyzer()
        stats = analyzer.compute_statistics(sample_numeric_data)
        
        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "skewness" in stats
        assert "kurtosis" in stats
        assert stats["count"] == 1000
        assert 95 < stats["mean"] < 105  # Should be around 100
    
    def test_analyze_distribution(self, sample_dataframe):
        """Test distribution analysis."""
        analyzer = StatisticalAnalyzer()
        result = analyzer.analyze_distribution(sample_dataframe, 'amount')
        
        assert "histogram" in result
        assert "statistics" in result
        assert "kde" in result
        assert len(result["histogram"]["counts"]) == 30  # Default bins
    
    def test_detect_outliers_iqr(self, sample_numeric_data):
        """Test IQR outlier detection."""
        analyzer = StatisticalAnalyzer()
        
        # Add some outliers
        data_with_outliers = sample_numeric_data.copy()
        data_with_outliers.iloc[0] = 500  # Clear outlier
        
        result = analyzer.detect_outliers(
            pd.DataFrame({'value': data_with_outliers}),
            'value',
            method='iqr'
        )
        
        assert result["outlier_count"] > 0
        assert 0 in result["outlier_indices"]
    
    def test_detect_outliers_zscore(self, sample_numeric_data):
        """Test Z-score outlier detection."""
        analyzer = StatisticalAnalyzer()
        
        # Add outliers
        data_with_outliers = sample_numeric_data.copy()
        data_with_outliers.iloc[0] = 300
        
        result = analyzer.detect_outliers(
            pd.DataFrame({'value': data_with_outliers}),
            'value',
            method='zscore',
            threshold=3.0
        )
        
        assert result["outlier_count"] > 0
    
    def test_detect_outliers_isolation_forest(self, sample_numeric_data):
        """Test Isolation Forest outlier detection."""
        analyzer = StatisticalAnalyzer()
        
        result = analyzer.detect_outliers(
            pd.DataFrame({'value': sample_numeric_data}),
            'value',
            method='isolation_forest',
            contamination=0.1
        )
        
        assert "outlier_count" in result
        assert result["outlier_count"] > 0
    
    def test_categorical_analysis(self, sample_dataframe):
        """Test categorical field analysis."""
        analyzer = StatisticalAnalyzer()
        result = analyzer.categorical_analysis(sample_dataframe['category'])
        
        assert "unique_count" in result
        assert "mode" in result
        assert "entropy" in result
        assert result["unique_count"] == 3  # A, B, C
    
    def test_missing_value_analysis(self, sample_dataframe):
        """Test missing value analysis."""
        # Add missing values
        df_with_missing = sample_dataframe.copy()
        df_with_missing.loc[0:50, 'amount'] = np.nan
        
        analyzer = StatisticalAnalyzer()
        result = analyzer.missing_value_analysis(df_with_missing)
        
        assert "total_missing" in result
        assert result["total_missing"] > 0
        assert "amount" in result["fields_with_missing"]
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        analyzer = StatisticalAnalyzer()
        empty_series = pd.Series([], dtype=float)
        
        stats = analyzer.compute_statistics(empty_series)
        assert "error" in stats


# ============================================================================
# CorrelationAnalyzer Tests
# ============================================================================

class TestCorrelationAnalyzer:
    
    def test_correlation_matrix_pearson(self, sample_dataframe):
        """Test Pearson correlation matrix."""
        analyzer = CorrelationAnalyzer()
        corr_matrix = analyzer.correlation_matrix(sample_dataframe, method='pearson')
        
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape[0] == corr_matrix.shape[1]
        assert np.allclose(np.diag(corr_matrix), 1.0)  # Diagonal should be 1
    
    def test_correlation_matrix_spearman(self, sample_dataframe):
        """Test Spearman correlation matrix."""
        analyzer = CorrelationAnalyzer()
        corr_matrix = analyzer.correlation_matrix(sample_dataframe, method='spearman')
        
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape[0] == corr_matrix.shape[1]
    
    def test_correlation_with_pvalues(self, sample_dataframe):
        """Test correlation with p-values."""
        analyzer = CorrelationAnalyzer()
        corr_matrix, pval_matrix = analyzer.correlation_with_pvalues(sample_dataframe)
        
        assert isinstance(corr_matrix, pd.DataFrame)
        assert isinstance(pval_matrix, pd.DataFrame)
        assert corr_matrix.shape == pval_matrix.shape
    
    def test_find_strong_correlations(self):
        """Test finding strong correlations."""
        # Create data with known correlation
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        y = 0.9 * x + np.random.normal(0, 0.1, 100)  # Strong correlation
        
        df = pd.DataFrame({'x': x, 'y': y, 'z': np.random.normal(0, 1, 100)})
        
        analyzer = CorrelationAnalyzer()
        strong_corr = analyzer.find_strong_correlations(df, threshold=0.7)
        
        assert len(strong_corr) > 0
        assert strong_corr[0]['abs_correlation'] > 0.7
    
    def test_partial_correlation(self, sample_dataframe):
        """Test partial correlation."""
        analyzer = CorrelationAnalyzer()
        result = analyzer.partial_correlation(
            sample_dataframe,
            'amount',
            'score',
            control_vars=['count']
        )
        
        assert "partial_correlation" in result
        assert "zero_order_correlation" in result
        assert "p_value" in result
    
    def test_compare_correlation_methods(self, sample_dataframe):
        """Test comparison of correlation methods."""
        analyzer = CorrelationAnalyzer()
        result = analyzer.compare_correlation_methods(
            sample_dataframe,
            'amount',
            'score'
        )
        
        assert "pearson" in result
        assert "spearman" in result
        assert "kendall" in result
        assert "interpretation" in result


# ============================================================================
# DistributionFitter Tests
# ============================================================================

class TestDistributionFitter:
    
    def test_fit_normal_distribution(self, sample_numeric_data):
        """Test fitting normal distribution."""
        fitter = DistributionFitter()
        result = fitter.fit_distribution(sample_numeric_data, distribution='normal')
        
        assert "parameters" in result
        assert "goodness_of_fit" in result
        assert "mean" in result["parameters"]
        assert "std" in result["parameters"]
    
    def test_fit_lognormal_distribution(self):
        """Test fitting lognormal distribution."""
        # Generate lognormal data
        np.random.seed(42)
        data = np.random.lognormal(mean=3, sigma=0.5, size=1000)
        
        fitter = DistributionFitter()
        result = fitter.fit_distribution(data, distribution='lognormal')
        
        assert "parameters" in result
        assert "goodness_of_fit" in result
    
    def test_fit_exponential_distribution(self):
        """Test fitting exponential distribution."""
        np.random.seed(42)
        data = np.random.exponential(scale=10, size=1000)
        
        fitter = DistributionFitter()
        result = fitter.fit_distribution(data, distribution='exponential')
        
        assert "parameters" in result
        assert "ks_pvalue" in result["goodness_of_fit"]
    
    def test_fit_best_distribution(self, sample_numeric_data):
        """Test finding best distribution fit."""
        fitter = DistributionFitter()
        result = fitter.fit_best_distribution(
            sample_numeric_data,
            distributions=['normal', 'lognormal', 'exponential']
        )
        
        assert "best_distribution" in result
        assert "all_fits" in result
        assert len(result["all_fits"]) == 3
    
    def test_qq_plot_data(self, sample_numeric_data):
        """Test Q-Q plot data generation."""
        fitter = DistributionFitter()
        result = fitter.qq_plot_data(sample_numeric_data, distribution='normal')
        
        assert "sample_quantiles" in result
        assert "theoretical_quantiles" in result
        assert "r_squared" in result
        assert len(result["sample_quantiles"]) == len(result["theoretical_quantiles"])
    
    def test_probability_plot_data(self, sample_numeric_data):
        """Test P-P plot data generation."""
        fitter = DistributionFitter()
        result = fitter.probability_plot_data(sample_numeric_data, distribution='normal')
        
        assert "empirical_cdf" in result
        assert "theoretical_cdf" in result
        assert "max_deviation" in result


# ============================================================================
# TrendAnalyzer Tests
# ============================================================================

class TestTrendAnalyzer:
    
    def test_decompose_timeseries(self, timeseries_data):
        """Test time-series decomposition."""
        analyzer = TrendAnalyzer()
        result = analyzer.decompose_timeseries(
            timeseries_data,
            'timestamp',
            'value',
            period=7
        )
        
        assert "trend" in result
        assert "seasonal" in result
        assert "residual" in result
        assert len(result["observed"]) == 365
    
    def test_detect_seasonality(self, timeseries_data):
        """Test seasonality detection."""
        analyzer = TrendAnalyzer()
        result = analyzer.detect_seasonality(
            timeseries_data,
            'timestamp',
            'value'
        )
        
        assert "has_seasonality" in result
        assert "potential_periods" in result
    
    def test_analyze_trend_linear(self, timeseries_data):
        """Test linear trend analysis."""
        analyzer = TrendAnalyzer()
        result = analyzer.analyze_trend(
            timeseries_data,
            'timestamp',
            'value',
            trend_type='linear'
        )
        
        assert "slope" in result
        assert "r_squared" in result
        assert "direction" in result
        assert result["direction"] == "increasing"  # Data has positive trend
    
    def test_analyze_trend_polynomial(self, timeseries_data):
        """Test polynomial trend analysis."""
        analyzer = TrendAnalyzer()
        result = analyzer.analyze_trend(
            timeseries_data,
            'timestamp',
            'value',
            trend_type='polynomial'
        )
        
        assert "coefficients" in result
        assert len(result["coefficients"]) == 3  # Degree 2
    
    def test_detect_change_points(self, timeseries_data):
        """Test change point detection."""
        analyzer = TrendAnalyzer()
        result = analyzer.detect_change_points(
            timeseries_data,
            'timestamp',
            'value',
            window_size=10
        )
        
        assert "n_change_points" in result
        assert "change_points" in result
    
    def test_stationarity_test(self, timeseries_data):
        """Test stationarity test."""
        analyzer = TrendAnalyzer()
        result = analyzer.stationarity_test(
            timeseries_data,
            'timestamp',
            'value'
        )
        
        assert "adf_statistic" in result
        assert "p_value" in result
        assert "is_stationary" in result


# ============================================================================
# DataProfiler Tests
# ============================================================================

class TestDataProfiler:
    
    def test_profile_dataset(self, sample_dataframe):
        """Test complete dataset profiling."""
        profiler = DataProfiler()
        profile = profiler.profile_dataset(sample_dataframe)
        
        assert "overview" in profile
        assert "field_summary" in profile
        assert "quality_score" in profile
        assert profile["overview"]["n_rows"] == 1000
    
    def test_field_summary(self, sample_dataframe):
        """Test field-by-field summary."""
        profiler = DataProfiler()
        summary = profiler.field_summary(sample_dataframe)
        
        assert len(summary) == 6  # All fields
        assert "amount" in summary
        assert summary["amount"]["inferred_type"] == "numeric"
        assert "mean" in summary["amount"]
    
    def test_cardinality_analysis(self, sample_dataframe):
        """Test cardinality analysis."""
        profiler = DataProfiler()
        result = profiler.cardinality_analysis(sample_dataframe)
        
        assert "fields" in result
        assert "by_category" in result
        assert "category" in result["fields"]
        assert result["fields"]["category"]["unique_count"] == 3
    
    def test_completeness_report(self, sample_dataframe):
        """Test completeness assessment."""
        # Add missing values
        df_with_missing = sample_dataframe.copy()
        df_with_missing.loc[0:50, 'amount'] = np.nan
        
        profiler = DataProfiler()
        result = profiler.completeness_report(df_with_missing)
        
        assert "overall" in result
        assert "rows" in result
        assert "fields" in result
        assert result["overall"]["completeness_pct"] < 100
    
    def test_data_quality_score(self, sample_dataframe):
        """Test quality score computation."""
        profiler = DataProfiler()
        result = profiler.data_quality_score(sample_dataframe)
        
        assert "overall_score" in result
        assert "grade" in result
        assert "components" in result
        assert 0 <= result["overall_score"] <= 100
    
    def test_anomaly_summary(self, sample_dataframe):
        """Test anomaly detection."""
        # Add constant field
        df_with_anomaly = sample_dataframe.copy()
        df_with_anomaly['constant'] = 1
        
        profiler = DataProfiler()
        result = profiler.anomaly_summary(df_with_anomaly)
        
        assert "constant_fields" in result
        assert "constant" in result["constant_fields"]


# ============================================================================
# StatisticalTests Tests
# ============================================================================

class TestStatisticalTests:
    
    def test_normality_test_shapiro(self, sample_numeric_data):
        """Test Shapiro-Wilk normality test."""
        tester = StatisticalTests()
        result = tester.normality_test(sample_numeric_data, method='shapiro')
        
        assert "statistic" in result
        assert "p_value" in result
        assert "is_normal" in result
    
    def test_normality_test_anderson(self, sample_numeric_data):
        """Test Anderson-Darling normality test."""
        tester = StatisticalTests()
        result = tester.normality_test(sample_numeric_data, method='anderson')
        
        assert "statistic" in result
        assert "critical_value" in result
        assert "is_normal" in result
    
    def test_t_test_one_sample(self, sample_numeric_data):
        """Test one-sample t-test."""
        tester = StatisticalTests()
        result = tester.t_test_one_sample(sample_numeric_data, population_mean=100)
        
        assert "t_statistic" in result
        assert "p_value" in result
        assert "significant" in result
    
    def test_t_test_two_sample(self):
        """Test two-sample t-test."""
        np.random.seed(42)
        group1 = np.random.normal(100, 10, 100)
        group2 = np.random.normal(105, 10, 100)
        
        tester = StatisticalTests()
        result = tester.t_test_two_sample(group1, group2)
        
        assert "t_statistic" in result
        assert "p_value" in result
        assert "cohens_d" in result
        assert "effect_size" in result
    
    def test_t_test_paired(self):
        """Test paired t-test."""
        np.random.seed(42)
        before = np.random.normal(100, 10, 100)
        after = before + np.random.normal(5, 3, 100)  # After should be higher
        
        tester = StatisticalTests()
        result = tester.t_test_paired(before, after)
        
        assert "t_statistic" in result
        assert "p_value" in result
        assert "mean_difference" in result
    
    def test_chi_square_independence(self, sample_dataframe):
        """Test chi-square independence test."""
        tester = StatisticalTests()
        result = tester.chi_square_independence(
            sample_dataframe,
            'category',
            'is_fraud'
        )
        
        assert "chi2_statistic" in result
        assert "p_value" in result
        assert "cramers_v" in result
    
    def test_anova_one_way(self):
        """Test one-way ANOVA."""
        np.random.seed(42)
        group1 = np.random.normal(100, 10, 50)
        group2 = np.random.normal(105, 10, 50)
        group3 = np.random.normal(110, 10, 50)
        
        tester = StatisticalTests()
        result = tester.anova_one_way(group1, group2, group3)
        
        assert "f_statistic" in result
        assert "p_value" in result
        assert "eta_squared" in result
        assert result["n_groups"] == 3
    
    def test_mann_whitney_u(self):
        """Test Mann-Whitney U test."""
        np.random.seed(42)
        group1 = np.random.normal(100, 10, 50)
        group2 = np.random.normal(105, 10, 50)
        
        tester = StatisticalTests()
        result = tester.mann_whitney_u(group1, group2)
        
        assert "u_statistic" in result
        assert "p_value" in result
    
    def test_kruskal_wallis(self):
        """Test Kruskal-Wallis H test."""
        np.random.seed(42)
        group1 = np.random.normal(100, 10, 50)
        group2 = np.random.normal(105, 10, 50)
        group3 = np.random.normal(110, 10, 50)
        
        tester = StatisticalTests()
        result = tester.kruskal_wallis(group1, group2, group3)
        
        assert "h_statistic" in result
        assert "p_value" in result
    
    def test_wilcoxon_signed_rank(self):
        """Test Wilcoxon signed-rank test."""
        np.random.seed(42)
        before = np.random.normal(100, 10, 50)
        after = before + np.random.normal(5, 3, 50)
        
        tester = StatisticalTests()
        result = tester.wilcoxon_signed_rank(before, after)
        
        assert "statistic" in result
        assert "p_value" in result


# ============================================================================
# AnalysisReport Tests
# ============================================================================

class TestAnalysisReport:
    
    def test_generate_statistical_report(self, sample_dataframe):
        """Test statistical report generation."""
        # Create analysis results
        analyzer = StatisticalAnalyzer()
        results = analyzer.describe_dataset(sample_dataframe)
        
        reporter = AnalysisReport()
        report = reporter.generate_statistical_report(results)
        
        assert isinstance(report, str)
        assert "DATASET OVERVIEW" in report
        assert "Statistical Analysis Report" in report
    
    def test_summary_statistics(self, sample_dataframe):
        """Test summary statistics extraction."""
        analyzer = StatisticalAnalyzer()
        results = analyzer.describe_dataset(sample_dataframe)
        
        reporter = AnalysisReport()
        summary = reporter.summary_statistics(results)
        
        assert "timestamp" in summary
        assert "dataset" in summary
        assert summary["dataset"]["rows"] == 1000
    
    def test_create_comparison_report(self, sample_dataframe):
        """Test comparison report generation."""
        analyzer = StatisticalAnalyzer()
        before = analyzer.describe_dataset(sample_dataframe)
        
        # Create "after" with more rows
        after_df = pd.concat([sample_dataframe, sample_dataframe.head(100)])
        after = analyzer.describe_dataset(after_df)
        
        reporter = AnalysisReport()
        report = reporter.create_comparison_report(before, after)
        
        assert isinstance(report, str)
        assert "Comparison Report" in report
        assert "DATASET SIZE" in report
