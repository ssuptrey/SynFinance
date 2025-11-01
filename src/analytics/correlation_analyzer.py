"""
Correlation Analyzer for computing and analyzing correlation matrices.

Provides methods for:
- Pearson correlation (linear relationships)
- Spearman correlation (monotonic relationships)
- Kendall correlation (ordinal data)
- Partial correlation
- Statistical significance testing
- Strong correlation detection
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import logging

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Comprehensive correlation analysis for datasets.
    
    Supports multiple correlation coefficients and provides
    statistical significance testing.
    """
    
    def __init__(self):
        """Initialize the Correlation Analyzer."""
        pass
    
    def correlation_matrix(
        self,
        df: pd.DataFrame,
        method: str = "pearson",
        fields: Optional[List[str]] = None,
        min_periods: int = 30
    ) -> pd.DataFrame:
        """
        Compute correlation matrix for numeric fields.
        
        Args:
            df: Input DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
            fields: List of fields to include (None = all numeric fields)
            min_periods: Minimum number of observations required
            
        Returns:
            DataFrame with correlation matrix
        """
        # Select numeric fields
        if fields is None:
            numeric_df = df.select_dtypes(include='number')
        else:
            numeric_df = df[fields].select_dtypes(include='number')
        
        if len(numeric_df.columns) == 0:
            raise ValueError("No numeric fields found")
        
        logger.info(f"Computing {method} correlation matrix for {len(numeric_df.columns)} fields")
        
        # Compute correlation
        corr_matrix = numeric_df.corr(method=method, min_periods=min_periods)
        
        return corr_matrix
    
    def correlation_with_pvalues(
        self,
        df: pd.DataFrame,
        method: str = "pearson",
        fields: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute correlation matrix with p-values for significance testing.
        
        Args:
            df: Input DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
            fields: List of fields to include (None = all numeric fields)
            
        Returns:
            Tuple of (correlation_matrix, pvalue_matrix)
        """
        # Select numeric fields
        if fields is None:
            numeric_df = df.select_dtypes(include='number')
        else:
            numeric_df = df[fields].select_dtypes(include='number')
        
        if len(numeric_df.columns) == 0:
            raise ValueError("No numeric fields found")
        
        cols = numeric_df.columns.tolist()
        n = len(cols)
        
        # Initialize matrices
        corr_matrix = np.zeros((n, n))
        pval_matrix = np.zeros((n, n))
        
        # Choose correlation function
        if method == "pearson":
            corr_func = pearsonr
        elif method == "spearman":
            corr_func = spearmanr
        elif method == "kendall":
            corr_func = kendalltau
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Compute pairwise correlations
        for i in range(n):
            for j in range(n):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    pval_matrix[i, j] = 0.0
                elif i < j:
                    # Remove NaNs
                    mask = ~(numeric_df.iloc[:, i].isna() | numeric_df.iloc[:, j].isna())
                    x = numeric_df.iloc[:, i][mask]
                    y = numeric_df.iloc[:, j][mask]
                    
                    if len(x) > 2:
                        try:
                            corr, pval = corr_func(x, y)
                            corr_matrix[i, j] = corr
                            corr_matrix[j, i] = corr
                            pval_matrix[i, j] = pval
                            pval_matrix[j, i] = pval
                        except Exception as e:
                            logger.warning(f"Correlation calculation failed for {cols[i]} vs {cols[j]}: {e}")
                            corr_matrix[i, j] = np.nan
                            corr_matrix[j, i] = np.nan
                            pval_matrix[i, j] = np.nan
                            pval_matrix[j, i] = np.nan
                    else:
                        corr_matrix[i, j] = np.nan
                        corr_matrix[j, i] = np.nan
                        pval_matrix[i, j] = np.nan
                        pval_matrix[j, i] = np.nan
        
        # Convert to DataFrames
        corr_df = pd.DataFrame(corr_matrix, index=cols, columns=cols)
        pval_df = pd.DataFrame(pval_matrix, index=cols, columns=cols)
        
        return corr_df, pval_df
    
    def find_strong_correlations(
        self,
        df: pd.DataFrame,
        threshold: float = 0.7,
        method: str = "pearson",
        include_pvalues: bool = True,
        max_pvalue: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Find strong correlations above threshold.
        
        Args:
            df: Input DataFrame
            threshold: Absolute correlation threshold (default: 0.7)
            method: Correlation method
            include_pvalues: Include significance testing
            max_pvalue: Maximum p-value to consider significant
            
        Returns:
            List of strong correlation pairs with statistics
        """
        if include_pvalues:
            corr_matrix, pval_matrix = self.correlation_with_pvalues(df, method=method)
        else:
            corr_matrix = self.correlation_matrix(df, method=method)
            pval_matrix = None
        
        strong_correlations = []
        
        # Iterate through upper triangle (avoid duplicates)
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold and not np.isnan(corr_value):
                    field1 = corr_matrix.index[i]
                    field2 = corr_matrix.columns[j]
                    
                    result = {
                        "field1": field1,
                        "field2": field2,
                        "correlation": float(corr_value),
                        "abs_correlation": abs(float(corr_value)),
                        "direction": "positive" if corr_value > 0 else "negative",
                        "method": method,
                    }
                    
                    # Interpret strength
                    abs_corr = abs(corr_value)
                    if abs_corr >= 0.9:
                        strength = "very strong"
                    elif abs_corr >= 0.7:
                        strength = "strong"
                    elif abs_corr >= 0.5:
                        strength = "moderate"
                    elif abs_corr >= 0.3:
                        strength = "weak"
                    else:
                        strength = "very weak"
                    
                    result["strength"] = strength
                    
                    # Add p-value if available
                    if pval_matrix is not None:
                        pval = pval_matrix.iloc[i, j]
                        result["p_value"] = float(pval)
                        result["significant"] = pval < max_pvalue
                    
                    strong_correlations.append(result)
        
        # Sort by absolute correlation
        strong_correlations.sort(key=lambda x: x["abs_correlation"], reverse=True)
        
        logger.info(f"Found {len(strong_correlations)} strong correlations (|r| >= {threshold})")
        
        return strong_correlations
    
    def partial_correlation(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        control_vars: List[str]
    ) -> Dict[str, Any]:
        """
        Compute partial correlation controlling for other variables.
        
        Partial correlation measures the relationship between x and y
        while controlling for the effects of control_vars.
        
        Args:
            df: Input DataFrame
            x: First variable
            y: Second variable
            control_vars: Variables to control for
            
        Returns:
            Dictionary with partial correlation and statistics
        """
        # Select relevant columns
        all_vars = [x, y] + control_vars
        subset = df[all_vars].dropna()
        
        if len(subset) < 10:
            return {"error": "Insufficient data after removing NaNs"}
        
        # Compute correlation matrix
        corr_matrix = subset.corr().values
        
        # Indices
        x_idx = all_vars.index(x)
        y_idx = all_vars.index(y)
        
        # Compute partial correlation using matrix inversion
        # partial_corr(x, y | Z) = -r_xy / sqrt(r_xx * r_yy)
        # where r_xy is from inverse correlation matrix
        
        try:
            inv_corr = np.linalg.inv(corr_matrix)
            partial_corr = -inv_corr[x_idx, y_idx] / np.sqrt(
                inv_corr[x_idx, x_idx] * inv_corr[y_idx, y_idx]
            )
        except np.linalg.LinAlgError:
            return {"error": "Matrix inversion failed (singular matrix)"}
        
        # Compute significance using Fisher's z-transformation
        n = len(subset)
        k = len(control_vars)
        z = 0.5 * np.log((1 + partial_corr) / (1 - partial_corr))
        se = 1 / np.sqrt(n - k - 3)
        z_score = z / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return {
            "x": x,
            "y": y,
            "control_vars": control_vars,
            "partial_correlation": float(partial_corr),
            "n_observations": n,
            "z_score": float(z_score),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            
            # Also include zero-order correlation for comparison
            "zero_order_correlation": float(corr_matrix[x_idx, y_idx]),
        }
    
    def correlation_heatmap_data(
        self,
        df: pd.DataFrame,
        method: str = "pearson",
        fields: Optional[List[str]] = None,
        annotate_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Prepare data for correlation heatmap visualization.
        
        Args:
            df: Input DataFrame
            method: Correlation method
            fields: Fields to include
            annotate_threshold: Threshold for annotating cells
            
        Returns:
            Dictionary with heatmap data and metadata
        """
        corr_matrix = self.correlation_matrix(df, method=method, fields=fields)
        
        # Find cells to annotate (strong correlations)
        annotations = []
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                if i != j:  # Exclude diagonal
                    value = corr_matrix.iloc[i, j]
                    if abs(value) >= annotate_threshold and not np.isnan(value):
                        annotations.append({
                            "row": corr_matrix.index[i],
                            "col": corr_matrix.columns[j],
                            "value": float(value),
                        })
        
        return {
            "matrix": corr_matrix.values.tolist(),
            "fields": corr_matrix.columns.tolist(),
            "method": method,
            "annotations": annotations,
            "vmin": -1.0,
            "vmax": 1.0,
        }
    
    def compare_correlation_methods(
        self,
        df: pd.DataFrame,
        field1: str,
        field2: str
    ) -> Dict[str, Any]:
        """
        Compare different correlation methods for the same pair of variables.
        
        Args:
            df: Input DataFrame
            field1: First field
            field2: Second field
            
        Returns:
            Dictionary with correlations from all methods
        """
        # Clean data
        subset = df[[field1, field2]].dropna()
        
        if len(subset) < 3:
            return {"error": "Insufficient data"}
        
        x = subset[field1]
        y = subset[field2]
        
        # Pearson
        pearson_corr, pearson_pval = pearsonr(x, y)
        
        # Spearman
        spearman_corr, spearman_pval = spearmanr(x, y)
        
        # Kendall
        kendall_corr, kendall_pval = kendalltau(x, y)
        
        return {
            "field1": field1,
            "field2": field2,
            "n_observations": len(subset),
            "pearson": {
                "correlation": float(pearson_corr),
                "p_value": float(pearson_pval),
                "significant": pearson_pval < 0.05,
            },
            "spearman": {
                "correlation": float(spearman_corr),
                "p_value": float(spearman_pval),
                "significant": spearman_pval < 0.05,
            },
            "kendall": {
                "correlation": float(kendall_corr),
                "p_value": float(kendall_pval),
                "significant": kendall_pval < 0.05,
            },
            "interpretation": self._interpret_correlation_comparison(
                pearson_corr, spearman_corr, kendall_corr
            )
        }
    
    def _interpret_correlation_comparison(
        self,
        pearson: float,
        spearman: float,
        kendall: float
    ) -> str:
        """
        Interpret differences between correlation methods.
        
        Args:
            pearson: Pearson correlation
            spearman: Spearman correlation
            kendall: Kendall correlation
            
        Returns:
            Interpretation string
        """
        # If all similar, relationship is likely linear
        if abs(pearson - spearman) < 0.1 and abs(pearson - kendall) < 0.1:
            return "All methods agree: relationship is approximately linear"
        
        # If Spearman/Kendall much higher than Pearson
        if (spearman - pearson) > 0.2 or (kendall - pearson) > 0.2:
            return "Monotonic but not linear relationship (Spearman/Kendall > Pearson)"
        
        # If Pearson much higher
        if (pearson - spearman) > 0.2 or (pearson - kendall) > 0.2:
            return "Linear but not strictly monotonic (Pearson > Spearman/Kendall)"
        
        return "Methods show moderate agreement"
