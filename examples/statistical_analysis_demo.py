"""
Statistical Analysis Demo

Demonstrates comprehensive usage of all analytics modules:
- Statistical analysis
- Correlation analysis
- Distribution fitting
- Trend analysis
- Data profiling
- Statistical tests
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def generate_sample_data(n=10000):
    """Generate realistic sample financial data."""
    np.random.seed(42)
    
    # Transaction amounts (lognormal distribution)
    amounts = np.random.lognormal(mean=4.5, sigma=1.2, size=n)
    
    # Transaction counts (Poisson distribution)
    counts = np.random.poisson(lam=15, size=n)
    
    # Merchant categories
    categories = np.random.choice(
        ['retail', 'groceries', 'dining', 'travel', 'entertainment', 'utilities'],
        size=n,
        p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
    )
    
    # Fraud flags (10% fraud rate)
    is_fraud = np.random.choice([True, False], size=n, p=[0.1, 0.9])
    
    # Timestamps (last 30 days)
    start_date = datetime.now() - timedelta(days=30)
    timestamps = [start_date + timedelta(hours=i*0.072) for i in range(n)]
    
    # Risk scores (correlated with fraud)
    risk_scores = np.where(
        is_fraud,
        np.random.beta(8, 2, n) * 100,  # High risk for fraud
        np.random.beta(2, 8, n) * 100   # Low risk for legitimate
    )
    
    # Customer age
    age = np.random.normal(42, 15, n).clip(18, 90)
    
    return pd.DataFrame({
        'transaction_id': range(n),
        'timestamp': timestamps,
        'amount': amounts,
        'transaction_count': counts,
        'merchant_category': categories,
        'is_fraud': is_fraud,
        'risk_score': risk_scores,
        'customer_age': age,
    })


def demo_statistical_analysis():
    """Demonstrate Statistical Analyzer."""
    print("\n" + "="*80)
    print("DEMO: Statistical Analysis")
    print("="*80)
    
    df = generate_sample_data(5000)
    analyzer = StatisticalAnalyzer()
    
    # 1. Comprehensive dataset description
    print("\n1. Dataset Description")
    print("-" * 80)
    results = analyzer.describe_dataset(df, include_outliers=True, include_distribution=True)
    print(f"Rows: {results['dataset_info']['row_count']:,}")
    print(f"Columns: {results['dataset_info']['column_count']}")
    print(f"Numeric fields: {len(results['numeric_fields'])}")
    
    # 2. Detailed statistics for amount field
    print("\n2. Amount Field Statistics")
    print("-" * 80)
    amount_stats = results['numeric_fields']['amount']
    print(f"Mean: ${amount_stats['mean']:.2f}")
    print(f"Median: ${amount_stats['median']:.2f}")
    print(f"Std Dev: ${amount_stats['std']:.2f}")
    print(f"Skewness: {amount_stats['skewness']:.3f} ({amount_stats['skewness_interpretation']})")
    print(f"Kurtosis: {amount_stats['kurtosis']:.3f} ({amount_stats['kurtosis_interpretation']})")
    
    # 3. Outlier detection
    print("\n3. Outlier Detection (IQR method)")
    print("-" * 80)
    outliers = analyzer.detect_outliers(df, 'amount', method='iqr')
    print(f"Outliers found: {outliers['outlier_count']:,} ({outliers['outlier_percentage']:.2f}%)")
    print(f"Sample outlier values: {[f'${x:.2f}' for x in outliers['outlier_values_sample'][:5]]}")
    
    # 4. Distribution analysis
    print("\n4. Distribution Analysis")
    print("-" * 80)
    dist_analysis = analyzer.analyze_distribution(df, 'amount', bins=20)
    print(f"Field: {dist_analysis['field']}")
    print(f"Histogram bins: {dist_analysis['histogram']['bins']}")
    print(f"Min value: ${dist_analysis['statistics']['min']:.2f}")
    print(f"Max value: ${dist_analysis['statistics']['max']:.2f}")


def demo_correlation_analysis():
    """Demonstrate Correlation Analyzer."""
    print("\n" + "="*80)
    print("DEMO: Correlation Analysis")
    print("="*80)
    
    df = generate_sample_data(5000)
    analyzer = CorrelationAnalyzer()
    
    # 1. Correlation matrix
    print("\n1. Pearson Correlation Matrix")
    print("-" * 80)
    corr_matrix = analyzer.correlation_matrix(df, method='pearson')
    print(corr_matrix[['amount', 'risk_score', 'customer_age']].head())
    
    # 2. Find strong correlations
    print("\n2. Strong Correlations (|r| > 0.3)")
    print("-" * 80)
    strong_corr = analyzer.find_strong_correlations(df, threshold=0.3, include_pvalues=True)
    for corr in strong_corr[:5]:
        print(f"{corr['field1']} <-> {corr['field2']}: "
              f"r={corr['correlation']:.3f} ({corr['strength']}, {corr['direction']}), "
              f"p={corr['p_value']:.4f}")
    
    # 3. Compare correlation methods
    print("\n3. Correlation Method Comparison (amount vs risk_score)")
    print("-" * 80)
    comparison = analyzer.compare_correlation_methods(df, 'amount', 'risk_score')
    print(f"Pearson: r={comparison['pearson']['correlation']:.3f}, p={comparison['pearson']['p_value']:.4f}")
    print(f"Spearman: r={comparison['spearman']['correlation']:.3f}, p={comparison['spearman']['p_value']:.4f}")
    print(f"Kendall: τ={comparison['kendall']['correlation']:.3f}, p={comparison['kendall']['p_value']:.4f}")
    print(f"Interpretation: {comparison['interpretation']}")


def demo_distribution_fitting():
    """Demonstrate Distribution Fitter."""
    print("\n" + "="*80)
    print("DEMO: Distribution Fitting")
    print("="*80)
    
    df = generate_sample_data(5000)
    fitter = DistributionFitter()
    
    # 1. Fit single distribution
    print("\n1. Fit Normal Distribution to Customer Age")
    print("-" * 80)
    result = fitter.fit_distribution(df['customer_age'], distribution='normal')
    print(f"Distribution: {result['distribution']}")
    print(f"Parameters: μ={result['parameters']['mean']:.2f}, σ={result['parameters']['std']:.2f}")
    print(f"KS test: statistic={result['goodness_of_fit']['ks_statistic']:.4f}, "
          f"p-value={result['goodness_of_fit']['ks_pvalue']:.4f}")
    print(f"Interpretation: {result['interpretation']}")
    
    # 2. Find best distribution for transaction amounts
    print("\n2. Best Distribution Fit for Transaction Amounts")
    print("-" * 80)
    best_fit = fitter.fit_best_distribution(
        df['amount'],
        distributions=['normal', 'lognormal', 'exponential', 'gamma'],
        criterion='bic'
    )
    print(f"Best distribution: {best_fit['best_distribution']['distribution']}")
    print(f"Selection criterion: {best_fit['criterion']}")
    print("\nAll distributions tested:")
    for fit in best_fit['all_fits']:
        marker = "★" if fit['selected'] else " "
        print(f"{marker} {fit['distribution']:12s} - BIC: {fit['bic']:.2f}, "
              f"KS p-value: {fit['ks_pvalue']:.4f}")


def demo_trend_analysis():
    """Demonstrate Trend Analyzer."""
    print("\n" + "="*80)
    print("DEMO: Trend Analysis")
    print("="*80)
    
    df = generate_sample_data(720)  # 30 days of hourly data
    analyzer = TrendAnalyzer()
    
    # 1. Detect seasonality
    print("\n1. Seasonality Detection")
    print("-" * 80)
    seasonality = analyzer.detect_seasonality(df, 'timestamp', 'amount', max_period=168)  # 1 week
    print(f"Has seasonality: {seasonality['has_seasonality']}")
    if seasonality['has_seasonality']:
        print(f"Dominant period: {seasonality['dominant_period']} observations")
        print("Potential periods:")
        for period in seasonality['potential_periods'][:3]:
            print(f"  Period {period['period']}: ACF={period['acf']:.3f} ({period['strength']})")
    
    # 2. Linear trend analysis
    print("\n2. Linear Trend Analysis")
    print("-" * 80)
    trend = analyzer.analyze_trend(df, 'timestamp', 'amount', trend_type='linear')
    print(f"Slope: {trend['slope']:.4f}")
    print(f"R-squared: {trend['r_squared']:.4f}")
    print(f"Direction: {trend['direction']}")
    print(f"Significance: {trend['significance']} (p={trend['p_value']:.4f})")
    
    # 3. Stationarity test
    print("\n3. Stationarity Test (Augmented Dickey-Fuller)")
    print("-" * 80)
    stationarity = analyzer.stationarity_test(df, 'timestamp', 'amount')
    print(f"ADF statistic: {stationarity['adf_statistic']:.4f}")
    print(f"p-value: {stationarity['p_value']:.4f}")
    print(f"Result: {stationarity['interpretation']}")


def demo_data_profiling():
    """Demonstrate Data Profiler."""
    print("\n" + "="*80)
    print("DEMO: Data Profiling")
    print("="*80)
    
    df = generate_sample_data(5000)
    # Add some missing values
    df.loc[0:100, 'amount'] = np.nan
    df.loc[50:150, 'risk_score'] = np.nan
    
    profiler = DataProfiler()
    
    # 1. Complete dataset profile
    print("\n1. Complete Dataset Profile")
    print("-" * 80)
    profile = profiler.profile_dataset(df)
    print(f"Rows: {profile['overview']['n_rows']:,}")
    print(f"Columns: {profile['overview']['n_columns']}")
    print(f"Memory: {profile['overview']['memory_usage_mb']:.2f} MB")
    
    # 2. Data quality score
    print("\n2. Data Quality Assessment")
    print("-" * 80)
    quality = profile['quality_score']
    print(f"Overall Score: {quality['overall_score']:.1f}/100 ({quality['grade']})")
    print(f"  Completeness: {quality['components']['completeness']:.1f}/100")
    print(f"  Validity: {quality['components']['validity']:.1f}/100")
    print(f"  Consistency: {quality['components']['consistency']:.1f}/100")
    print(f"  Uniqueness: {quality['components']['uniqueness']:.1f}/100")
    print(f"\n{quality['interpretation']}")
    
    # 3. Completeness report
    print("\n3. Completeness Report")
    print("-" * 80)
    completeness = profiler.completeness_report(df)
    print(f"Overall completeness: {completeness['overall']['completeness_pct']:.2f}%")
    print(f"Complete rows: {completeness['rows']['complete']:,} ({completeness['rows']['completeness_pct']:.1f}%)")
    print("\nFields with missing values:")
    for field, info in completeness['fields'].items():
        if info['missing'] > 0:
            print(f"  {field}: {info['missing']:,} missing ({info['completeness_pct']:.1f}% complete)")


def demo_statistical_tests():
    """Demonstrate Statistical Tests."""
    print("\n" + "="*80)
    print("DEMO: Statistical Tests")
    print("="*80)
    
    df = generate_sample_data(5000)
    tester = StatisticalTests(alpha=0.05)
    
    # 1. Normality test
    print("\n1. Normality Test (Shapiro-Wilk)")
    print("-" * 80)
    normality = tester.normality_test(df['customer_age'].dropna(), method='shapiro')
    print(f"Test: {normality['test']}")
    print(f"Statistic: {normality['statistic']:.4f}")
    print(f"p-value: {normality['p_value']:.4f}")
    print(f"Result: {normality['interpretation']}")
    
    # 2. Two-sample t-test (fraud vs non-fraud amounts)
    print("\n2. Two-Sample t-Test (Fraud vs Non-Fraud Amounts)")
    print("-" * 80)
    fraud_amounts = df[df['is_fraud']]['amount'].dropna()
    normal_amounts = df[~df['is_fraud']]['amount'].dropna()
    
    ttest = tester.t_test_two_sample(fraud_amounts, normal_amounts)
    print(f"Fraud mean: ${ttest['group1_mean']:.2f}")
    print(f"Normal mean: ${ttest['group2_mean']:.2f}")
    print(f"Difference: ${ttest['difference']:.2f}")
    print(f"t-statistic: {ttest['t_statistic']:.4f}")
    print(f"p-value: {ttest['p_value']:.6f}")
    print(f"Cohen's d: {ttest['cohens_d']:.3f} ({ttest['effect_size']} effect)")
    print(f"Result: {ttest['interpretation']}")
    
    # 3. Chi-square test (category vs fraud)
    print("\n3. Chi-Square Test (Merchant Category vs Fraud)")
    print("-" * 80)
    chi_square = tester.chi_square_independence(df, 'merchant_category', 'is_fraud')
    print(f"χ² statistic: {chi_square['chi2_statistic']:.4f}")
    print(f"p-value: {chi_square['p_value']:.6f}")
    print(f"Cramér's V: {chi_square['cramers_v']:.3f} ({chi_square['effect_size']} effect)")
    print(f"Result: {chi_square['interpretation']}")
    
    # 4. ANOVA (amounts across categories)
    print("\n4. One-Way ANOVA (Amounts Across Merchant Categories)")
    print("-" * 80)
    groups = [df[df['merchant_category'] == cat]['amount'].dropna().values 
              for cat in df['merchant_category'].unique()]
    
    anova = tester.anova_one_way(*groups)
    print(f"F-statistic: {anova['f_statistic']:.4f}")
    print(f"p-value: {anova['p_value']:.6f}")
    print(f"η² (eta-squared): {anova['eta_squared']:.3f} ({anova['effect_size']} effect)")
    print(f"Result: {anova['interpretation']}")


def demo_analysis_report():
    """Demonstrate Analysis Report."""
    print("\n" + "="*80)
    print("DEMO: Analysis Report")
    print("="*80)
    
    df = generate_sample_data(5000)
    
    # Generate comprehensive analysis
    stat_analyzer = StatisticalAnalyzer()
    corr_analyzer = CorrelationAnalyzer()
    profiler = DataProfiler()
    
    analysis_results = stat_analyzer.describe_dataset(df)
    analysis_results['strong_correlations'] = corr_analyzer.find_strong_correlations(df, threshold=0.3)
    analysis_results['quality_score'] = profiler.data_quality_score(df)
    
    # Generate report
    reporter = AnalysisReport()
    
    print("\n1. Comprehensive Text Report")
    print("-" * 80)
    report = reporter.generate_statistical_report(analysis_results, title="Transaction Data Analysis")
    print(report)
    
    print("\n2. Summary Statistics")
    print("-" * 80)
    summary = reporter.summary_statistics(analysis_results)
    print(f"Dataset: {summary['dataset']['rows']:,} rows, {summary['dataset']['columns']} columns")
    print(f"Quality: {summary['quality']['overall_score']:.1f}/100 ({summary['quality']['grade']})")
    print(f"Strong correlations found: {summary['insights']['strong_correlations_found']}")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print(" "*25 + "STATISTICAL ANALYSIS DEMO")
    print(" "*20 + "Comprehensive Analytics Toolkit")
    print("="*80)
    
    demos = [
        ("Statistical Analysis", demo_statistical_analysis),
        ("Correlation Analysis", demo_correlation_analysis),
        ("Distribution Fitting", demo_distribution_fitting),
        ("Trend Analysis", demo_trend_analysis),
        ("Data Profiling", demo_data_profiling),
        ("Statistical Tests", demo_statistical_tests),
        ("Analysis Reports", demo_analysis_report),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n❌ Error in {name} demo: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print(" "*30 + "DEMO COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
