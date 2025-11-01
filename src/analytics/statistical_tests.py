"""
Statistical Tests for hypothesis testing and significance testing.

Provides methods for:
- Normality tests (Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov)
- t-tests (one-sample, two-sample independent, paired)
- Chi-square tests (independence, goodness-of-fit)
- ANOVA (one-way, two-way)
- Non-parametric tests (Mann-Whitney U, Kruskal-Wallis, Wilcoxon)
- Proportion tests
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, kruskal
import logging

logger = logging.getLogger(__name__)


class StatisticalTests:
    """
    Comprehensive statistical hypothesis testing.
    
    Provides parametric and non-parametric tests for comparing
    groups and testing assumptions.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize Statistical Tests.
        
        Args:
            alpha: Significance level (default: 0.05)
        """
        self.alpha = alpha
    
    def normality_test(
        self,
        data: Union[pd.Series, np.ndarray],
        method: str = 'shapiro'
    ) -> Dict[str, Any]:
        """
        Test if data follows a normal distribution.
        
        Args:
            data: Input data
            method: Test method ('shapiro', 'anderson', 'ks')
            
        Returns:
            Dictionary with test results
        """
        if isinstance(data, pd.Series):
            data = data.dropna().values
        else:
            data = data[~np.isnan(data)]
        
        if len(data) < 3:
            return {"error": "Insufficient data (need at least 3 points)"}
        
        if method == 'shapiro':
            # Shapiro-Wilk test
            if len(data) > 5000:
                logger.warning("Shapiro-Wilk test not recommended for n > 5000. Consider using Anderson-Darling.")
            
            statistic, p_value = stats.shapiro(data)
            
            return {
                "test": "Shapiro-Wilk",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "is_normal": p_value >= self.alpha,
                "alpha": self.alpha,
                "interpretation": f"Data {'appears to be' if p_value >= self.alpha else 'does not appear to be'} normally distributed (p={p_value:.4f})",
                "n_observations": len(data),
            }
        
        elif method == 'anderson':
            # Anderson-Darling test
            result = stats.anderson(data, dist='norm')
            
            # Get critical value for alpha
            critical_values = dict(zip([0.15, 0.10, 0.05, 0.025, 0.01], result.critical_values))
            critical_value = critical_values.get(self.alpha, result.critical_values[2])  # Default to 5%
            
            is_normal = result.statistic < critical_value
            
            return {
                "test": "Anderson-Darling",
                "statistic": float(result.statistic),
                "critical_value": float(critical_value),
                "is_normal": is_normal,
                "alpha": self.alpha,
                "all_critical_values": {
                    "15%": float(result.critical_values[0]),
                    "10%": float(result.critical_values[1]),
                    "5%": float(result.critical_values[2]),
                    "2.5%": float(result.critical_values[3]),
                    "1%": float(result.critical_values[4]),
                },
                "interpretation": f"Data {'appears to be' if is_normal else 'does not appear to be'} normally distributed",
                "n_observations": len(data),
            }
        
        elif method == 'ks':
            # Kolmogorov-Smirnov test
            # Fit normal distribution
            mu, sigma = data.mean(), data.std()
            statistic, p_value = stats.kstest(data, lambda x: stats.norm.cdf(x, mu, sigma))
            
            return {
                "test": "Kolmogorov-Smirnov",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "is_normal": p_value >= self.alpha,
                "alpha": self.alpha,
                "fitted_mean": float(mu),
                "fitted_std": float(sigma),
                "interpretation": f"Data {'appears to be' if p_value >= self.alpha else 'does not appear to be'} normally distributed (p={p_value:.4f})",
                "n_observations": len(data),
            }
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'shapiro', 'anderson', or 'ks'")
    
    def t_test_one_sample(
        self,
        data: Union[pd.Series, np.ndarray],
        population_mean: float
    ) -> Dict[str, Any]:
        """
        One-sample t-test: Test if sample mean differs from population mean.
        
        Args:
            data: Sample data
            population_mean: Hypothesized population mean
            
        Returns:
            Dictionary with test results
        """
        if isinstance(data, pd.Series):
            data = data.dropna().values
        else:
            data = data[~np.isnan(data)]
        
        if len(data) < 2:
            return {"error": "Insufficient data"}
        
        statistic, p_value = stats.ttest_1samp(data, population_mean)
        
        sample_mean = data.mean()
        diff = sample_mean - population_mean
        
        return {
            "test": "One-sample t-test",
            "sample_mean": float(sample_mean),
            "population_mean": float(population_mean),
            "difference": float(diff),
            "t_statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
            "alpha": self.alpha,
            "degrees_of_freedom": len(data) - 1,
            "n_observations": len(data),
            "interpretation": f"Sample mean {'significantly differs' if p_value < self.alpha else 'does not significantly differ'} from {population_mean} (p={p_value:.4f})",
        }
    
    def t_test_two_sample(
        self,
        group1: Union[pd.Series, np.ndarray],
        group2: Union[pd.Series, np.ndarray],
        equal_var: bool = True
    ) -> Dict[str, Any]:
        """
        Two-sample independent t-test: Compare means of two groups.
        
        Args:
            group1: First group data
            group2: Second group data
            equal_var: Assume equal variances (True = Student's t-test, False = Welch's t-test)
            
        Returns:
            Dictionary with test results
        """
        if isinstance(group1, pd.Series):
            group1 = group1.dropna().values
        else:
            group1 = group1[~np.isnan(group1)]
        
        if isinstance(group2, pd.Series):
            group2 = group2.dropna().values
        else:
            group2 = group2[~np.isnan(group2)]
        
        if len(group1) < 2 or len(group2) < 2:
            return {"error": "Insufficient data in one or both groups"}
        
        statistic, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
        
        mean1 = group1.mean()
        mean2 = group2.mean()
        diff = mean1 - mean2
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group1) - 1) * group1.std() ** 2 + (len(group2) - 1) * group2.std() ** 2) / (len(group1) + len(group2) - 2))
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        
        return {
            "test": "Welch's t-test" if not equal_var else "Student's t-test",
            "group1_mean": float(mean1),
            "group2_mean": float(mean2),
            "difference": float(diff),
            "t_statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
            "alpha": self.alpha,
            "group1_n": len(group1),
            "group2_n": len(group2),
            "cohens_d": float(cohens_d),
            "effect_size": self._interpret_cohens_d(cohens_d),
            "interpretation": f"Groups {'significantly differ' if p_value < self.alpha else 'do not significantly differ'} (p={p_value:.4f}, d={cohens_d:.2f})",
        }
    
    def t_test_paired(
        self,
        before: Union[pd.Series, np.ndarray],
        after: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Paired t-test: Compare means of paired samples (e.g., before/after).
        
        Args:
            before: Before measurements
            after: After measurements
            
        Returns:
            Dictionary with test results
        """
        if isinstance(before, pd.Series):
            before = before.dropna().values
        else:
            before = before[~np.isnan(before)]
        
        if isinstance(after, pd.Series):
            after = after.dropna().values
        else:
            after = after[~np.isnan(after)]
        
        if len(before) != len(after):
            return {"error": "Groups must have same length for paired test"}
        
        if len(before) < 2:
            return {"error": "Insufficient data"}
        
        statistic, p_value = stats.ttest_rel(before, after)
        
        mean_before = before.mean()
        mean_after = after.mean()
        mean_diff = (after - before).mean()
        
        return {
            "test": "Paired t-test",
            "before_mean": float(mean_before),
            "after_mean": float(mean_after),
            "mean_difference": float(mean_diff),
            "t_statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
            "alpha": self.alpha,
            "n_pairs": len(before),
            "interpretation": f"Paired measurements {'significantly differ' if p_value < self.alpha else 'do not significantly differ'} (p={p_value:.4f})",
        }
    
    def chi_square_independence(
        self,
        df: pd.DataFrame,
        var1: str,
        var2: str
    ) -> Dict[str, Any]:
        """
        Chi-square test of independence: Test if two categorical variables are independent.
        
        Args:
            df: Input DataFrame
            var1: First categorical variable
            var2: Second categorical variable
            
        Returns:
            Dictionary with test results
        """
        # Create contingency table
        contingency_table = pd.crosstab(df[var1], df[var2])
        
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Cramér's V (effect size)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape[0], contingency_table.shape[1]) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        return {
            "test": "Chi-square test of independence",
            "variable1": var1,
            "variable2": var2,
            "chi2_statistic": float(chi2),
            "p_value": float(p_value),
            "degrees_of_freedom": int(dof),
            "significant": p_value < self.alpha,
            "alpha": self.alpha,
            "cramers_v": float(cramers_v),
            "effect_size": self._interpret_cramers_v(cramers_v),
            "contingency_table": contingency_table.to_dict(),
            "interpretation": f"Variables {'are dependent' if p_value < self.alpha else 'are independent'} (p={p_value:.4f}, V={cramers_v:.2f})",
        }
    
    def chi_square_goodness_of_fit(
        self,
        observed: Union[pd.Series, np.ndarray],
        expected: Union[pd.Series, np.ndarray, None] = None
    ) -> Dict[str, Any]:
        """
        Chi-square goodness-of-fit test: Test if observed frequencies match expected.
        
        Args:
            observed: Observed frequencies
            expected: Expected frequencies (None = uniform distribution)
            
        Returns:
            Dictionary with test results
        """
        if isinstance(observed, pd.Series):
            observed = observed.values
        
        if expected is None:
            # Uniform distribution
            expected = np.full(len(observed), observed.sum() / len(observed))
        elif isinstance(expected, pd.Series):
            expected = expected.values
        
        chi2, p_value = stats.chisquare(observed, expected)
        
        return {
            "test": "Chi-square goodness-of-fit",
            "chi2_statistic": float(chi2),
            "p_value": float(p_value),
            "degrees_of_freedom": len(observed) - 1,
            "significant": p_value < self.alpha,
            "alpha": self.alpha,
            "interpretation": f"Observed frequencies {'do not match' if p_value < self.alpha else 'match'} expected distribution (p={p_value:.4f})",
        }
    
    def anova_one_way(
        self,
        *groups: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """
        One-way ANOVA: Compare means across multiple groups.
        
        Args:
            *groups: Multiple groups to compare
            
        Returns:
            Dictionary with test results
        """
        if len(groups) < 2:
            return {"error": "Need at least 2 groups"}
        
        # Clean groups
        clean_groups = []
        for g in groups:
            if isinstance(g, pd.Series):
                clean_groups.append(g.dropna().values)
            else:
                clean_groups.append(g[~np.isnan(g)])
        
        # Check for sufficient data
        if any(len(g) < 2 for g in clean_groups):
            return {"error": "All groups must have at least 2 observations"}
        
        f_stat, p_value = f_oneway(*clean_groups)
        
        # Group statistics
        group_stats = []
        for i, g in enumerate(clean_groups):
            group_stats.append({
                "group": i + 1,
                "n": len(g),
                "mean": float(g.mean()),
                "std": float(g.std()),
            })
        
        # Effect size (eta-squared)
        grand_mean = np.concatenate(clean_groups).mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in clean_groups)
        ss_total = sum(np.sum((g - grand_mean) ** 2) for g in clean_groups)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        return {
            "test": "One-way ANOVA",
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
            "alpha": self.alpha,
            "n_groups": len(groups),
            "group_statistics": group_stats,
            "eta_squared": float(eta_squared),
            "effect_size": self._interpret_eta_squared(eta_squared),
            "interpretation": f"Groups {'have significantly different means' if p_value < self.alpha else 'do not have significantly different means'} (p={p_value:.4f}, η²={eta_squared:.3f})",
        }
    
    def mann_whitney_u(
        self,
        group1: Union[pd.Series, np.ndarray],
        group2: Union[pd.Series, np.ndarray],
        alternative: str = 'two-sided'
    ) -> Dict[str, Any]:
        """
        Mann-Whitney U test: Non-parametric alternative to two-sample t-test.
        
        Args:
            group1: First group data
            group2: Second group data
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            Dictionary with test results
        """
        if isinstance(group1, pd.Series):
            group1 = group1.dropna().values
        else:
            group1 = group1[~np.isnan(group1)]
        
        if isinstance(group2, pd.Series):
            group2 = group2.dropna().values
        else:
            group2 = group2[~np.isnan(group2)]
        
        if len(group1) < 1 or len(group2) < 1:
            return {"error": "Insufficient data"}
        
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)
        
        return {
            "test": "Mann-Whitney U",
            "u_statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
            "alpha": self.alpha,
            "alternative": alternative,
            "group1_median": float(np.median(group1)),
            "group2_median": float(np.median(group2)),
            "group1_n": len(group1),
            "group2_n": len(group2),
            "interpretation": f"Groups {'significantly differ' if p_value < self.alpha else 'do not significantly differ'} (p={p_value:.4f})",
        }
    
    def kruskal_wallis(
        self,
        *groups: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Kruskal-Wallis H test: Non-parametric alternative to one-way ANOVA.
        
        Args:
            *groups: Multiple groups to compare
            
        Returns:
            Dictionary with test results
        """
        if len(groups) < 2:
            return {"error": "Need at least 2 groups"}
        
        # Clean groups
        clean_groups = []
        for g in groups:
            if isinstance(g, pd.Series):
                clean_groups.append(g.dropna().values)
            else:
                clean_groups.append(g[~np.isnan(g)])
        
        if any(len(g) < 1 for g in clean_groups):
            return {"error": "All groups must have at least 1 observation"}
        
        h_stat, p_value = kruskal(*clean_groups)
        
        # Group medians
        group_stats = []
        for i, g in enumerate(clean_groups):
            group_stats.append({
                "group": i + 1,
                "n": len(g),
                "median": float(np.median(g)),
            })
        
        return {
            "test": "Kruskal-Wallis H",
            "h_statistic": float(h_stat),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
            "alpha": self.alpha,
            "n_groups": len(groups),
            "group_statistics": group_stats,
            "interpretation": f"Groups {'have significantly different distributions' if p_value < self.alpha else 'do not have significantly different distributions'} (p={p_value:.4f})",
        }
    
    def wilcoxon_signed_rank(
        self,
        before: Union[pd.Series, np.ndarray],
        after: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Wilcoxon signed-rank test: Non-parametric alternative to paired t-test.
        
        Args:
            before: Before measurements
            after: After measurements
            
        Returns:
            Dictionary with test results
        """
        if isinstance(before, pd.Series):
            before = before.dropna().values
        else:
            before = before[~np.isnan(before)]
        
        if isinstance(after, pd.Series):
            after = after.dropna().values
        else:
            after = after[~np.isnan(after)]
        
        if len(before) != len(after):
            return {"error": "Groups must have same length"}
        
        if len(before) < 1:
            return {"error": "Insufficient data"}
        
        statistic, p_value = stats.wilcoxon(before, after)
        
        return {
            "test": "Wilcoxon signed-rank",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
            "alpha": self.alpha,
            "before_median": float(np.median(before)),
            "after_median": float(np.median(after)),
            "n_pairs": len(before),
            "interpretation": f"Paired measurements {'significantly differ' if p_value < self.alpha else 'do not significantly differ'} (p={p_value:.4f})",
        }
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_cramers_v(self, v: float) -> str:
        """Interpret Cramér's V effect size."""
        if v < 0.1:
            return "negligible"
        elif v < 0.3:
            return "small"
        elif v < 0.5:
            return "medium"
        else:
            return "large"
    
    def _interpret_eta_squared(self, eta2: float) -> str:
        """Interpret eta-squared effect size."""
        if eta2 < 0.01:
            return "negligible"
        elif eta2 < 0.06:
            return "small"
        elif eta2 < 0.14:
            return "medium"
        else:
            return "large"
