"""
Trend Analyzer for time-series analysis and trend detection.

Provides methods for:
- Time-series decomposition (trend, seasonal, residual)
- Seasonality detection
- Trend identification (linear, polynomial, exponential)
- Change point detection
- Simple forecasting
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats, signal
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
import logging

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """
    Analyze trends and patterns in time-series data.
    
    Provides decomposition, seasonality detection, and trend fitting.
    """
    
    def __init__(self):
        """Initialize the Trend Analyzer."""
        pass
    
    def decompose_timeseries(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        value_col: str,
        period: Optional[int] = None,
        model: str = 'additive'
    ) -> Dict[str, Any]:
        """
        Decompose time-series into trend, seasonal, and residual components.
        
        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column
            value_col: Name of value column
            period: Seasonal period (None = auto-detect)
            model: 'additive' or 'multiplicative'
            
        Returns:
            Dictionary with decomposition components
        """
        # Sort by timestamp
        df_sorted = df.sort_values(timestamp_col)
        
        # Create time series
        ts = pd.Series(
            df_sorted[value_col].values,
            index=pd.to_datetime(df_sorted[timestamp_col])
        )
        
        # Remove NaNs
        ts = ts.dropna()
        
        if len(ts) < 4:
            return {"error": "Insufficient data for decomposition"}
        
        # Auto-detect period if not provided
        if period is None:
            period = self._detect_period(ts)
            logger.info(f"Auto-detected period: {period}")
        
        if period is None or period >= len(ts) // 2:
            return {"error": "Cannot determine valid seasonal period"}
        
        try:
            # Perform decomposition
            decomposition = seasonal_decompose(ts, model=model, period=period, extrapolate_trend='freq')
            
            return {
                "timestamp": ts.index.astype(str).tolist(),
                "observed": ts.values.tolist(),
                "trend": decomposition.trend.values.tolist(),
                "seasonal": decomposition.seasonal.values.tolist(),
                "residual": decomposition.resid.values.tolist(),
                "period": period,
                "model": model,
                "trend_strength": self._compute_trend_strength(decomposition),
                "seasonal_strength": self._compute_seasonal_strength(decomposition),
            }
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            return {"error": str(e)}
    
    def detect_seasonality(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        value_col: str,
        max_period: int = 365
    ) -> Dict[str, Any]:
        """
        Detect seasonal patterns in time-series data.
        
        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column
            value_col: Name of value column
            max_period: Maximum period to test
            
        Returns:
            Dictionary with seasonality detection results
        """
        # Sort by timestamp
        df_sorted = df.sort_values(timestamp_col)
        ts = df_sorted[value_col].dropna().values
        
        if len(ts) < 10:
            return {"error": "Insufficient data"}
        
        # Limit max_period to half the time series length
        max_period = min(max_period, len(ts) // 2)
        
        # Auto-correlation function
        acf_values = acf(ts, nlags=min(max_period, len(ts) - 1), fft=True)
        
        # Find peaks in ACF (potential seasonal periods)
        peaks, properties = signal.find_peaks(acf_values[1:], height=0.3, distance=2)
        peaks = peaks + 1  # Adjust for removed lag-0
        
        potential_periods = []
        for peak in peaks:
            potential_periods.append({
                "period": int(peak),
                "acf": float(acf_values[peak]),
                "strength": "strong" if acf_values[peak] > 0.7 else "moderate" if acf_values[peak] > 0.5 else "weak"
            })
        
        # Sort by ACF value
        potential_periods.sort(key=lambda x: x["acf"], reverse=True)
        
        return {
            "has_seasonality": len(potential_periods) > 0,
            "dominant_period": potential_periods[0]["period"] if len(potential_periods) > 0 else None,
            "potential_periods": potential_periods[:5],  # Top 5
            "acf_values": acf_values.tolist(),
        }
    
    def analyze_trend(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        value_col: str,
        trend_type: str = 'linear'
    ) -> Dict[str, Any]:
        """
        Fit trend line to time-series data.
        
        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column
            value_col: Name of value column
            trend_type: 'linear', 'polynomial', or 'exponential'
            
        Returns:
            Dictionary with trend analysis results
        """
        # Sort by timestamp
        df_sorted = df.sort_values(timestamp_col).dropna(subset=[value_col])
        
        if len(df_sorted) < 3:
            return {"error": "Insufficient data"}
        
        # Convert timestamp to numeric (days since first observation)
        timestamps = pd.to_datetime(df_sorted[timestamp_col])
        x = (timestamps - timestamps.min()).dt.total_seconds() / 86400  # Days
        y = df_sorted[value_col].values
        
        if trend_type == 'linear':
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Fitted values
            y_pred = slope * x + intercept
            
            return {
                "trend_type": "linear",
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "std_error": float(std_err),
                "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat",
                "significance": "significant" if p_value < 0.05 else "not significant",
                "x": x.tolist(),
                "y_observed": y.tolist(),
                "y_fitted": y_pred.tolist(),
            }
        
        elif trend_type == 'polynomial':
            # Polynomial regression (degree 2)
            coeffs = np.polyfit(x, y, deg=2)
            y_pred = np.polyval(coeffs, x)
            
            # R-squared
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return {
                "trend_type": "polynomial",
                "coefficients": coeffs.tolist(),
                "r_squared": float(r_squared),
                "x": x.tolist(),
                "y_observed": y.tolist(),
                "y_fitted": y_pred.tolist(),
            }
        
        elif trend_type == 'exponential':
            # Exponential: y = a * exp(b * x)
            # Take log: log(y) = log(a) + b * x
            
            # Remove non-positive values
            mask = y > 0
            x_pos = x[mask]
            y_pos = y[mask]
            
            if len(y_pos) < 3:
                return {"error": "Insufficient positive values for exponential fit"}
            
            log_y = np.log(y_pos)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_pos, log_y)
            
            # Parameters
            a = np.exp(intercept)
            b = slope
            
            # Fitted values
            y_pred = a * np.exp(b * x_pos)
            
            return {
                "trend_type": "exponential",
                "a": float(a),
                "b": float(b),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "growth_rate": float(b),
                "direction": "exponential growth" if b > 0 else "exponential decay",
                "x": x_pos.tolist(),
                "y_observed": y_pos.tolist(),
                "y_fitted": y_pred.tolist(),
            }
        
        else:
            raise ValueError(f"Unknown trend type: {trend_type}")
    
    def detect_change_points(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        value_col: str,
        window_size: int = 10
    ) -> Dict[str, Any]:
        """
        Detect significant change points in time-series.
        
        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column
            value_col: Name of value column
            window_size: Window size for computing statistics
            
        Returns:
            Dictionary with change point detection results
        """
        # Sort by timestamp
        df_sorted = df.sort_values(timestamp_col).dropna(subset=[value_col])
        
        if len(df_sorted) < window_size * 2:
            return {"error": "Insufficient data"}
        
        timestamps = pd.to_datetime(df_sorted[timestamp_col])
        values = df_sorted[value_col].values
        
        change_points = []
        
        # Sliding window approach
        for i in range(window_size, len(values) - window_size):
            window_before = values[i - window_size:i]
            window_after = values[i:i + window_size]
            
            # T-test for change in mean
            t_stat, p_val = stats.ttest_ind(window_before, window_after)
            
            # Significant change detected
            if p_val < 0.01:  # Strict threshold
                change_points.append({
                    "index": int(i),
                    "timestamp": str(timestamps.iloc[i]),
                    "value": float(values[i]),
                    "mean_before": float(np.mean(window_before)),
                    "mean_after": float(np.mean(window_after)),
                    "t_statistic": float(t_stat),
                    "p_value": float(p_val),
                    "change_magnitude": float(np.mean(window_after) - np.mean(window_before)),
                })
        
        return {
            "n_change_points": len(change_points),
            "change_points": change_points,
            "window_size": window_size,
        }
    
    def stationarity_test(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        value_col: str
    ) -> Dict[str, Any]:
        """
        Test for stationarity using Augmented Dickey-Fuller test.
        
        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column
            value_col: Name of value column
            
        Returns:
            Dictionary with stationarity test results
        """
        # Sort and get values
        df_sorted = df.sort_values(timestamp_col).dropna(subset=[value_col])
        values = df_sorted[value_col].values
        
        if len(values) < 10:
            return {"error": "Insufficient data"}
        
        # ADF test
        adf_result = adfuller(values, autolag='AIC')
        
        return {
            "test": "Augmented Dickey-Fuller",
            "adf_statistic": float(adf_result[0]),
            "p_value": float(adf_result[1]),
            "n_lags": int(adf_result[2]),
            "n_observations": int(adf_result[3]),
            "critical_values": {
                "1%": float(adf_result[4]['1%']),
                "5%": float(adf_result[4]['5%']),
                "10%": float(adf_result[4]['10%']),
            },
            "is_stationary": adf_result[1] < 0.05,
            "interpretation": "Stationary" if adf_result[1] < 0.05 else "Non-stationary"
        }
    
    def _detect_period(self, ts: pd.Series) -> Optional[int]:
        """
        Auto-detect seasonal period using ACF.
        
        Args:
            ts: Time series
            
        Returns:
            Detected period or None
        """
        try:
            # Compute ACF
            max_lag = min(len(ts) // 2, 365)
            acf_values = acf(ts.values, nlags=max_lag, fft=True)
            
            # Find first significant peak after lag 1
            for lag in range(2, len(acf_values)):
                if acf_values[lag] > 0.6:  # Strong correlation
                    return lag
            
            # Default to 7 (weekly) if nothing found
            return 7
        except Exception as e:
            logger.warning(f"Period detection failed: {e}")
            return None
    
    def _compute_trend_strength(self, decomposition) -> float:
        """Compute strength of trend component."""
        var_resid = np.nanvar(decomposition.resid)
        var_detrend = np.nanvar(decomposition.resid + decomposition.seasonal)
        
        if var_detrend == 0:
            return 0.0
        
        return max(0, 1 - (var_resid / var_detrend))
    
    def _compute_seasonal_strength(self, decomposition) -> float:
        """Compute strength of seasonal component."""
        var_resid = np.nanvar(decomposition.resid)
        var_deseas = np.nanvar(decomposition.resid + decomposition.trend)
        
        if var_deseas == 0:
            return 0.0
        
        return max(0, 1 - (var_resid / var_deseas))
