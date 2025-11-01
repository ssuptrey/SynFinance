"""
Statistical Analyzer for comprehensive descriptive statistics and distribution analysis.

Provides methods for:
- Descriptive statistics (mean, median, mode, std, variance, quartiles, skewness, kurtosis)
- Distribution analysis (histograms, KDE, shape metrics)
- Outlier detection (IQR, Z-score, Isolation Forest)
- Missing value analysis
- Categorical field analysis
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for datasets.
    
    Provides descriptive statistics, distribution analysis, outlier detection,
    and data quality metrics.
    """
    
    def __init__(self):
        """Initialize the Statistical Analyzer."""
        self.numeric_dtypes = [np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]
    
    def describe_dataset(
        self,
        df: pd.DataFrame,
        include_outliers: bool = True,
        include_distribution: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive descriptive statistics for entire dataset.
        
        Args:
            df: Input DataFrame
            include_outliers: Include outlier detection results
            include_distribution: Include distribution shape metrics
            
        Returns:
            Dictionary with comprehensive statistics for all fields
        """
        logger.info(f"Analyzing dataset with {len(df)} rows and {len(df.columns)} columns")
        
        results = {
            "dataset_info": {
                "row_count": len(df),
                "column_count": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                "numeric_columns": df.select_dtypes(include='number').columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
                "datetime_columns": df.select_dtypes(include='datetime').columns.tolist(),
            },
            "numeric_fields": {},
            "categorical_fields": {},
            "datetime_fields": {},
        }
        
        # Analyze numeric fields
        for col in results["dataset_info"]["numeric_columns"]:
            results["numeric_fields"][col] = self.compute_statistics(
                df[col],
                include_outliers=include_outliers,
                include_distribution=include_distribution
            )
        
        # Analyze categorical fields
        for col in results["dataset_info"]["categorical_columns"]:
            results["categorical_fields"][col] = self.categorical_analysis(df[col])
        
        # Analyze datetime fields
        for col in results["dataset_info"]["datetime_columns"]:
            results["datetime_fields"][col] = self._datetime_analysis(df[col])
        
        return results
    
    def compute_statistics(
        self,
        series: pd.Series,
        include_outliers: bool = True,
        include_distribution: bool = True
    ) -> Dict[str, Any]:
        """
        Compute comprehensive statistics for a numeric series.
        
        Args:
            series: Pandas Series with numeric data
            include_outliers: Include outlier detection
            include_distribution: Include distribution shape metrics
            
        Returns:
            Dictionary with all statistics
        """
        # Remove NaN values for calculation
        clean_data = series.dropna()
        
        if len(clean_data) == 0:
            return {"error": "No valid data"}
        
        stats_dict = {
            # Basic statistics
            "count": len(series),
            "missing": series.isna().sum(),
            "missing_pct": (series.isna().sum() / len(series)) * 100,
            
            # Central tendency
            "mean": float(clean_data.mean()),
            "median": float(clean_data.median()),
            "mode": float(clean_data.mode()[0]) if len(clean_data.mode()) > 0 else None,
            
            # Spread
            "std": float(clean_data.std()),
            "variance": float(clean_data.var()),
            "min": float(clean_data.min()),
            "max": float(clean_data.max()),
            "range": float(clean_data.max() - clean_data.min()),
            
            # Quartiles
            "q1": float(clean_data.quantile(0.25)),
            "q2": float(clean_data.quantile(0.50)),
            "q3": float(clean_data.quantile(0.75)),
            "iqr": float(clean_data.quantile(0.75) - clean_data.quantile(0.25)),
            
            # Additional percentiles
            "p5": float(clean_data.quantile(0.05)),
            "p95": float(clean_data.quantile(0.95)),
            "p99": float(clean_data.quantile(0.99)),
        }
        
        # Distribution shape metrics
        if include_distribution:
            stats_dict.update({
                "skewness": float(stats.skew(clean_data)),
                "kurtosis": float(stats.kurtosis(clean_data)),
                "cv": float(clean_data.std() / clean_data.mean()) if clean_data.mean() != 0 else None,  # Coefficient of variation
            })
            
            # Interpret skewness
            skew = stats_dict["skewness"]
            if abs(skew) < 0.5:
                skew_interp = "approximately symmetric"
            elif skew > 0.5:
                skew_interp = "right-skewed (positive)"
            else:
                skew_interp = "left-skewed (negative)"
            stats_dict["skewness_interpretation"] = skew_interp
            
            # Interpret kurtosis
            kurt = stats_dict["kurtosis"]
            if abs(kurt) < 0.5:
                kurt_interp = "approximately normal (mesokurtic)"
            elif kurt > 0.5:
                kurt_interp = "heavy-tailed (leptokurtic)"
            else:
                kurt_interp = "light-tailed (platykurtic)"
            stats_dict["kurtosis_interpretation"] = kurt_interp
        
        # Outlier detection
        if include_outliers:
            outliers_iqr = self._detect_outliers_iqr(clean_data)
            outliers_zscore = self._detect_outliers_zscore(clean_data)
            
            stats_dict["outliers"] = {
                "iqr_method": {
                    "count": len(outliers_iqr),
                    "percentage": (len(outliers_iqr) / len(clean_data)) * 100,
                    "indices": outliers_iqr.tolist() if len(outliers_iqr) < 100 else outliers_iqr[:100].tolist(),
                },
                "zscore_method": {
                    "count": len(outliers_zscore),
                    "percentage": (len(outliers_zscore) / len(clean_data)) * 100,
                    "indices": outliers_zscore.tolist() if len(outliers_zscore) < 100 else outliers_zscore[:100].tolist(),
                }
            }
        
        return stats_dict
    
    def analyze_distribution(
        self,
        df: pd.DataFrame,
        field: str,
        bins: int = 30,
        kde: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze distribution of a numeric field.
        
        Args:
            df: Input DataFrame
            field: Field name to analyze
            bins: Number of histogram bins
            kde: Whether to compute Kernel Density Estimate
            
        Returns:
            Dictionary with distribution analysis
        """
        if field not in df.columns:
            raise ValueError(f"Field '{field}' not found in DataFrame")
        
        clean_data = df[field].dropna()
        
        if len(clean_data) == 0:
            return {"error": "No valid data"}
        
        # Histogram
        hist, bin_edges = np.histogram(clean_data, bins=bins)
        
        result = {
            "field": field,
            "histogram": {
                "counts": hist.tolist(),
                "bin_edges": bin_edges.tolist(),
                "bins": bins,
            },
            "statistics": self.compute_statistics(clean_data, include_distribution=True),
        }
        
        # KDE (Kernel Density Estimate)
        if kde and len(clean_data) > 1:
            try:
                kde_obj = stats.gaussian_kde(clean_data)
                x_range = np.linspace(clean_data.min(), clean_data.max(), 100)
                kde_values = kde_obj(x_range)
                
                result["kde"] = {
                    "x": x_range.tolist(),
                    "density": kde_values.tolist(),
                }
            except Exception as e:
                logger.warning(f"KDE calculation failed: {e}")
                result["kde"] = None
        
        return result
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        field: str,
        method: str = "iqr",
        threshold: float = 3.0,
        contamination: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect outliers using specified method.
        
        Args:
            df: Input DataFrame
            field: Field name to analyze
            method: Method to use ('iqr', 'zscore', 'isolation_forest')
            threshold: Threshold for zscore method (default: 3.0)
            contamination: Expected proportion of outliers for isolation forest (default: 0.1)
            
        Returns:
            Dictionary with outlier detection results
        """
        if field not in df.columns:
            raise ValueError(f"Field '{field}' not found in DataFrame")
        
        clean_data = df[field].dropna()
        
        if len(clean_data) == 0:
            return {"error": "No valid data"}
        
        if method == "iqr":
            outlier_indices = self._detect_outliers_iqr(clean_data)
            outlier_values = clean_data.iloc[outlier_indices]
        elif method == "zscore":
            outlier_indices = self._detect_outliers_zscore(clean_data, threshold=threshold)
            outlier_values = clean_data.iloc[outlier_indices]
        elif method == "isolation_forest":
            outlier_indices, outlier_values = self._detect_outliers_isolation_forest(
                clean_data, contamination=contamination
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            "method": method,
            "total_count": len(clean_data),
            "outlier_count": len(outlier_indices),
            "outlier_percentage": (len(outlier_indices) / len(clean_data)) * 100,
            "outlier_indices": outlier_indices.tolist() if len(outlier_indices) < 1000 else outlier_indices[:1000].tolist(),
            "outlier_values_sample": outlier_values.head(100).tolist() if isinstance(outlier_values, pd.Series) else outlier_values[:100].tolist(),
            "outlier_statistics": {
                "min": float(outlier_values.min()) if len(outlier_values) > 0 else None,
                "max": float(outlier_values.max()) if len(outlier_values) > 0 else None,
                "mean": float(outlier_values.mean()) if len(outlier_values) > 0 else None,
            }
        }
    
    def categorical_analysis(self, series: pd.Series) -> Dict[str, Any]:
        """
        Analyze categorical field.
        
        Args:
            series: Categorical series
            
        Returns:
            Dictionary with categorical analysis
        """
        value_counts = series.value_counts()
        
        return {
            "count": len(series),
            "missing": series.isna().sum(),
            "missing_pct": (series.isna().sum() / len(series)) * 100,
            "unique_count": series.nunique(),
            "cardinality": series.nunique() / len(series),  # Ratio of unique to total
            "mode": series.mode()[0] if len(series.mode()) > 0 else None,
            "mode_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            "mode_percentage": (value_counts.iloc[0] / len(series)) * 100 if len(value_counts) > 0 else 0,
            "top_10_values": value_counts.head(10).to_dict(),
            "entropy": stats.entropy(value_counts),  # Shannon entropy
        }
    
    def missing_value_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive missing value analysis.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with missing value analysis
        """
        missing_counts = df.isna().sum()
        total_cells = len(df) * len(df.columns)
        total_missing = missing_counts.sum()
        
        # Fields with missing values
        fields_with_missing = missing_counts[missing_counts > 0].sort_values(ascending=False)
        
        # Missing patterns (row-wise)
        rows_with_missing = df.isna().any(axis=1).sum()
        complete_rows = len(df) - rows_with_missing
        
        return {
            "total_cells": total_cells,
            "total_missing": int(total_missing),
            "missing_percentage": (total_missing / total_cells) * 100,
            "complete_rows": complete_rows,
            "incomplete_rows": rows_with_missing,
            "complete_rows_pct": (complete_rows / len(df)) * 100,
            "fields_with_missing": {
                col: {
                    "count": int(count),
                    "percentage": (count / len(df)) * 100
                }
                for col, count in fields_with_missing.items()
            }
        }
    
    def _detect_outliers_iqr(self, series: pd.Series, multiplier: float = 1.5) -> np.ndarray:
        """
        Detect outliers using IQR method.
        
        Args:
            series: Input series
            multiplier: IQR multiplier (default: 1.5)
            
        Returns:
            Array of outlier indices
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        outliers = (series < lower_bound) | (series > upper_bound)
        return np.where(outliers)[0]
    
    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> np.ndarray:
        """
        Detect outliers using Z-score method.
        
        Args:
            series: Input series
            threshold: Z-score threshold (default: 3.0)
            
        Returns:
            Array of outlier indices
        """
        z_scores = np.abs(stats.zscore(series))
        return np.where(z_scores > threshold)[0]
    
    def _detect_outliers_isolation_forest(
        self,
        series: pd.Series,
        contamination: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect outliers using Isolation Forest.
        
        Args:
            series: Input series
            contamination: Expected proportion of outliers
            
        Returns:
            Tuple of (outlier_indices, outlier_values)
        """
        # Reshape for sklearn
        X = series.values.reshape(-1, 1)
        
        # Fit Isolation Forest
        clf = IsolationForest(contamination=contamination, random_state=42)
        predictions = clf.fit_predict(X)
        
        # -1 indicates outlier
        outlier_mask = predictions == -1
        outlier_indices = np.where(outlier_mask)[0]
        outlier_values = series.iloc[outlier_indices].values
        
        return outlier_indices, outlier_values
    
    def _datetime_analysis(self, series: pd.Series) -> Dict[str, Any]:
        """
        Analyze datetime field.
        
        Args:
            series: Datetime series
            
        Returns:
            Dictionary with datetime analysis
        """
        clean_data = series.dropna()
        
        if len(clean_data) == 0:
            return {"error": "No valid data"}
        
        return {
            "count": len(series),
            "missing": series.isna().sum(),
            "missing_pct": (series.isna().sum() / len(series)) * 100,
            "min_date": str(clean_data.min()),
            "max_date": str(clean_data.max()),
            "date_range_days": (clean_data.max() - clean_data.min()).days,
            "unique_dates": clean_data.nunique(),
        }
