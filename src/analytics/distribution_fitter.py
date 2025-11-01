"""
Distribution Fitter for fitting probability distributions and goodness-of-fit testing.

Provides methods for:
- Fitting common distributions (normal, lognormal, exponential, gamma, beta, etc.)
- Goodness-of-fit tests (Kolmogorov-Smirnov, Anderson-Darling, Chi-square)
- Best distribution selection
- Q-Q plot data generation
- Distribution parameter estimation
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, lognorm, expon, gamma, beta, weibull_min, uniform, poisson
import logging

logger = logging.getLogger(__name__)


class DistributionFitter:
    """
    Fit probability distributions to data and perform goodness-of-fit tests.
    
    Supports multiple distribution families and provides statistical
    tests to evaluate fit quality.
    """
    
    # Available distributions
    DISTRIBUTIONS = {
        'normal': norm,
        'lognormal': lognorm,
        'exponential': expon,
        'gamma': gamma,
        'beta': beta,
        'weibull': weibull_min,
        'uniform': uniform,
    }
    
    def __init__(self):
        """Initialize the Distribution Fitter."""
        pass
    
    def fit_distribution(
        self,
        data: Union[pd.Series, np.ndarray],
        distribution: str = 'normal',
        **fit_kwargs
    ) -> Dict[str, Any]:
        """
        Fit a specific distribution to data.
        
        Args:
            data: Input data
            distribution: Distribution name ('normal', 'lognormal', 'exponential', etc.)
            **fit_kwargs: Additional arguments for fitting
            
        Returns:
            Dictionary with fitted parameters and goodness-of-fit statistics
        """
        if isinstance(data, pd.Series):
            data = data.dropna().values
        else:
            data = data[~np.isnan(data)]
        
        if len(data) < 10:
            return {"error": "Insufficient data (need at least 10 points)"}
        
        if distribution not in self.DISTRIBUTIONS:
            raise ValueError(f"Unknown distribution: {distribution}. Available: {list(self.DISTRIBUTIONS.keys())}")
        
        dist_class = self.DISTRIBUTIONS[distribution]
        
        try:
            # Fit distribution
            params = dist_class.fit(data, **fit_kwargs)
            
            # Goodness-of-fit tests
            ks_stat, ks_pval = stats.kstest(data, dist_class(*params).cdf)
            
            # Anderson-Darling test (only for specific distributions)
            if distribution == 'normal':
                ad_result = stats.anderson(data, dist='norm')
                ad_stat = ad_result.statistic
                ad_critical = ad_result.critical_values[2]  # 5% significance level
            else:
                ad_stat = None
                ad_critical = None
            
            # Chi-square test
            chi2_stat, chi2_pval = self._chi_square_test(data, dist_class, params)
            
            # AIC and BIC
            log_likelihood = np.sum(dist_class.logpdf(data, *params))
            n = len(data)
            k = len(params)
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood
            
            return {
                "distribution": distribution,
                "parameters": self._format_parameters(distribution, params),
                "parameter_values": params,
                "n_observations": n,
                "goodness_of_fit": {
                    "ks_statistic": float(ks_stat),
                    "ks_pvalue": float(ks_pval),
                    "ks_reject_null": ks_pval < 0.05,
                    "ad_statistic": float(ad_stat) if ad_stat is not None else None,
                    "ad_critical_5pct": float(ad_critical) if ad_critical is not None else None,
                    "ad_reject_null": (ad_stat > ad_critical) if ad_stat is not None else None,
                    "chi2_statistic": float(chi2_stat) if chi2_stat is not None else None,
                    "chi2_pvalue": float(chi2_pval) if chi2_pval is not None else None,
                    "chi2_reject_null": (chi2_pval < 0.05) if chi2_pval is not None else None,
                },
                "information_criteria": {
                    "log_likelihood": float(log_likelihood),
                    "aic": float(aic),
                    "bic": float(bic),
                },
                "interpretation": self._interpret_fit(ks_pval, ad_stat, ad_critical)
            }
            
        except Exception as e:
            logger.error(f"Distribution fitting failed: {e}")
            return {"error": str(e)}
    
    def fit_best_distribution(
        self,
        data: Union[pd.Series, np.ndarray],
        distributions: Optional[List[str]] = None,
        criterion: str = 'bic'
    ) -> Dict[str, Any]:
        """
        Try multiple distributions and select the best fit.
        
        Args:
            data: Input data
            distributions: List of distributions to try (None = all)
            criterion: Selection criterion ('aic', 'bic', 'ks')
            
        Returns:
            Dictionary with best distribution and comparison results
        """
        if distributions is None:
            distributions = list(self.DISTRIBUTIONS.keys())
        
        results = []
        
        for dist_name in distributions:
            logger.info(f"Fitting {dist_name} distribution...")
            result = self.fit_distribution(data, distribution=dist_name)
            
            if "error" not in result:
                results.append(result)
        
        if len(results) == 0:
            return {"error": "No distributions could be fitted"}
        
        # Select best based on criterion
        if criterion == 'aic':
            best = min(results, key=lambda x: x["information_criteria"]["aic"])
        elif criterion == 'bic':
            best = min(results, key=lambda x: x["information_criteria"]["bic"])
        elif criterion == 'ks':
            # Lower KS statistic is better
            best = min(results, key=lambda x: x["goodness_of_fit"]["ks_statistic"])
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        # Create comparison table
        comparison = []
        for result in results:
            comparison.append({
                "distribution": result["distribution"],
                "ks_statistic": result["goodness_of_fit"]["ks_statistic"],
                "ks_pvalue": result["goodness_of_fit"]["ks_pvalue"],
                "aic": result["information_criteria"]["aic"],
                "bic": result["information_criteria"]["bic"],
                "selected": (result["distribution"] == best["distribution"])
            })
        
        # Sort by criterion
        comparison.sort(key=lambda x: x[criterion] if criterion in ['aic', 'bic'] else x['ks_statistic'])
        
        return {
            "best_distribution": best,
            "criterion": criterion,
            "all_fits": comparison,
            "n_distributions_tested": len(results),
        }
    
    def qq_plot_data(
        self,
        data: Union[pd.Series, np.ndarray],
        distribution: str = 'normal',
        params: Optional[Tuple] = None
    ) -> Dict[str, Any]:
        """
        Generate data for Q-Q (quantile-quantile) plot.
        
        Args:
            data: Input data
            distribution: Distribution to compare against
            params: Distribution parameters (None = fit automatically)
            
        Returns:
            Dictionary with Q-Q plot data
        """
        if isinstance(data, pd.Series):
            data = data.dropna().values
        else:
            data = data[~np.isnan(data)]
        
        # Fit distribution if params not provided
        if params is None:
            fit_result = self.fit_distribution(data, distribution=distribution)
            if "error" in fit_result:
                return fit_result
            params = fit_result["parameter_values"]
        
        dist_class = self.DISTRIBUTIONS[distribution]
        
        # Sort data
        sorted_data = np.sort(data)
        
        # Theoretical quantiles
        n = len(data)
        theoretical_quantiles = dist_class.ppf(
            np.linspace(0.01, 0.99, n),
            *params
        )
        
        # Calculate R-squared for Q-Q plot
        correlation = np.corrcoef(theoretical_quantiles, sorted_data)[0, 1]
        r_squared = correlation ** 2
        
        return {
            "distribution": distribution,
            "sample_quantiles": sorted_data.tolist(),
            "theoretical_quantiles": theoretical_quantiles.tolist(),
            "n_points": n,
            "r_squared": float(r_squared),
            "interpretation": "Good fit" if r_squared > 0.95 else "Poor fit" if r_squared < 0.80 else "Moderate fit"
        }
    
    def probability_plot_data(
        self,
        data: Union[pd.Series, np.ndarray],
        distribution: str = 'normal'
    ) -> Dict[str, Any]:
        """
        Generate data for probability plot (P-P plot).
        
        Args:
            data: Input data
            distribution: Distribution to compare against
            
        Returns:
            Dictionary with P-P plot data
        """
        if isinstance(data, pd.Series):
            data = data.dropna().values
        else:
            data = data[~np.isnan(data)]
        
        # Fit distribution
        fit_result = self.fit_distribution(data, distribution=distribution)
        if "error" in fit_result:
            return fit_result
        
        params = fit_result["parameter_values"]
        dist_class = self.DISTRIBUTIONS[distribution]
        
        # Empirical CDF
        sorted_data = np.sort(data)
        n = len(data)
        empirical_cdf = np.arange(1, n + 1) / n
        
        # Theoretical CDF
        theoretical_cdf = dist_class.cdf(sorted_data, *params)
        
        # Calculate max deviation
        max_deviation = np.max(np.abs(empirical_cdf - theoretical_cdf))
        
        return {
            "distribution": distribution,
            "empirical_cdf": empirical_cdf.tolist(),
            "theoretical_cdf": theoretical_cdf.tolist(),
            "max_deviation": float(max_deviation),
            "interpretation": "Good fit" if max_deviation < 0.05 else "Poor fit" if max_deviation > 0.15 else "Moderate fit"
        }
    
    def _chi_square_test(
        self,
        data: np.ndarray,
        dist_class,
        params: Tuple,
        bins: int = 10
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Perform chi-square goodness-of-fit test.
        
        Args:
            data: Input data
            dist_class: Scipy distribution class
            params: Distribution parameters
            bins: Number of bins for histogram
            
        Returns:
            Tuple of (chi2_statistic, p_value)
        """
        try:
            # Create histogram
            observed, bin_edges = np.histogram(data, bins=bins)
            
            # Expected frequencies
            cdf_vals = dist_class.cdf(bin_edges, *params)
            expected = np.diff(cdf_vals) * len(data)
            
            # Remove bins with very low expected frequency
            mask = expected >= 5
            observed = observed[mask]
            expected = expected[mask]
            
            if len(observed) < 2:
                return None, None
            
            # Chi-square test
            chi2_stat = np.sum((observed - expected) ** 2 / expected)
            df = len(observed) - 1 - len(params)
            
            if df > 0:
                p_value = 1 - stats.chi2.cdf(chi2_stat, df)
                return chi2_stat, p_value
            else:
                return None, None
                
        except Exception as e:
            logger.warning(f"Chi-square test failed: {e}")
            return None, None
    
    def _format_parameters(self, distribution: str, params: Tuple) -> Dict[str, float]:
        """
        Format distribution parameters with meaningful names.
        
        Args:
            distribution: Distribution name
            params: Parameter tuple from scipy
            
        Returns:
            Dictionary with named parameters
        """
        if distribution == 'normal':
            return {"mean": float(params[0]), "std": float(params[1])}
        elif distribution == 'lognormal':
            return {"shape": float(params[0]), "loc": float(params[1]), "scale": float(params[2])}
        elif distribution == 'exponential':
            return {"loc": float(params[0]), "scale": float(params[1])}
        elif distribution == 'gamma':
            return {"shape": float(params[0]), "loc": float(params[1]), "scale": float(params[2])}
        elif distribution == 'beta':
            return {"a": float(params[0]), "b": float(params[1]), "loc": float(params[2]), "scale": float(params[3])}
        elif distribution == 'weibull':
            return {"shape": float(params[0]), "loc": float(params[1]), "scale": float(params[2])}
        elif distribution == 'uniform':
            return {"loc": float(params[0]), "scale": float(params[1])}
        else:
            return {f"param_{i}": float(p) for i, p in enumerate(params)}
    
    def _interpret_fit(
        self,
        ks_pval: float,
        ad_stat: Optional[float],
        ad_critical: Optional[float]
    ) -> str:
        """
        Interpret goodness-of-fit test results.
        
        Args:
            ks_pval: Kolmogorov-Smirnov p-value
            ad_stat: Anderson-Darling statistic
            ad_critical: Anderson-Darling critical value
            
        Returns:
            Interpretation string
        """
        interpretations = []
        
        # KS test
        if ks_pval >= 0.05:
            interpretations.append("KS test: Cannot reject null hypothesis (good fit)")
        else:
            interpretations.append("KS test: Reject null hypothesis (poor fit)")
        
        # AD test
        if ad_stat is not None and ad_critical is not None:
            if ad_stat <= ad_critical:
                interpretations.append("AD test: Cannot reject null hypothesis (good fit)")
            else:
                interpretations.append("AD test: Reject null hypothesis (poor fit)")
        
        return " | ".join(interpretations)
