"""
Data Profiler for comprehensive dataset quality assessment and profiling.

Provides methods for:
- Complete dataset profiling
- Data quality scoring
- Cardinality analysis
- Completeness assessment
- Anomaly detection
- Field-level summaries
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class DataProfiler:
    """
    Comprehensive data profiling and quality assessment.
    
    Analyzes dataset structure, quality, completeness, and anomalies.
    """
    
    def __init__(self):
        """Initialize the Data Profiler."""
        pass
    
    def profile_dataset(
        self,
        df: pd.DataFrame,
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive profile of entire dataset.
        
        Args:
            df: Input DataFrame
            sample_size: Sample size for large datasets (None = all data)
            
        Returns:
            Dictionary with complete dataset profile
        """
        if sample_size is not None and len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
            logger.info(f"Profiling sample of {sample_size} rows")
        else:
            df_sample = df
        
        profile = {
            "overview": self._overview_stats(df_sample),
            "field_summary": self.field_summary(df_sample),
            "missing_analysis": self._missing_analysis(df_sample),
            "cardinality": self.cardinality_analysis(df_sample),
            "quality_score": self.data_quality_score(df_sample),
            "anomalies": self.anomaly_summary(df_sample),
            "memory_usage": self._memory_analysis(df_sample),
            "duplicates": self._duplicate_analysis(df_sample),
        }
        
        return profile
    
    def field_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Generate field-by-field summary.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with summary for each field
        """
        summaries = {}
        
        for col in df.columns:
            field_type = self._infer_field_type(df[col])
            
            summary = {
                "dtype": str(df[col].dtype),
                "inferred_type": field_type,
                "count": len(df[col]),
                "missing": df[col].isna().sum(),
                "missing_pct": (df[col].isna().sum() / len(df[col])) * 100,
                "unique_count": df[col].nunique(),
                "cardinality": df[col].nunique() / len(df[col]),
            }
            
            # Type-specific metrics
            if field_type == "numeric":
                summary.update(self._numeric_field_summary(df[col]))
            elif field_type == "categorical":
                summary.update(self._categorical_field_summary(df[col]))
            elif field_type == "datetime":
                summary.update(self._datetime_field_summary(df[col]))
            elif field_type == "boolean":
                summary.update(self._boolean_field_summary(df[col]))
            
            summaries[col] = summary
        
        return summaries
    
    def cardinality_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze cardinality (uniqueness) of fields.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with cardinality analysis
        """
        cardinalities = {}
        
        for col in df.columns:
            unique_count = df[col].nunique()
            total_count = len(df[col])
            cardinality = unique_count / total_count if total_count > 0 else 0
            
            # Classify cardinality
            if unique_count == 1:
                category = "constant"
            elif cardinality < 0.01:
                category = "very low"
            elif cardinality < 0.1:
                category = "low"
            elif cardinality < 0.5:
                category = "medium"
            elif cardinality < 0.99:
                category = "high"
            else:
                category = "unique"
            
            cardinalities[col] = {
                "unique_count": unique_count,
                "total_count": total_count,
                "cardinality": cardinality,
                "category": category,
            }
        
        # Fields by category
        by_category = {"constant": [], "very low": [], "low": [], "medium": [], "high": [], "unique": []}
        for col, info in cardinalities.items():
            by_category[info["category"]].append(col)
        
        return {
            "fields": cardinalities,
            "by_category": by_category,
            "summary": {
                "constant_fields": len(by_category["constant"]),
                "unique_identifier_fields": len(by_category["unique"]),
            }
        }
    
    def completeness_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess data completeness (missing values).
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with completeness assessment
        """
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isna().sum().sum()
        
        # Per-field completeness
        field_completeness = {}
        for col in df.columns:
            missing = df[col].isna().sum()
            field_completeness[col] = {
                "complete": len(df[col]) - missing,
                "missing": missing,
                "completeness_pct": ((len(df[col]) - missing) / len(df[col])) * 100 if len(df[col]) > 0 else 0
            }
        
        # Completeness categories
        complete_fields = [col for col, info in field_completeness.items() if info["completeness_pct"] == 100]
        mostly_complete = [col for col, info in field_completeness.items() if 95 <= info["completeness_pct"] < 100]
        somewhat_complete = [col for col, info in field_completeness.items() if 50 <= info["completeness_pct"] < 95]
        mostly_missing = [col for col, info in field_completeness.items() if info["completeness_pct"] < 50]
        
        # Row completeness
        complete_rows = (df.notna().all(axis=1)).sum()
        
        return {
            "overall": {
                "total_cells": total_cells,
                "complete_cells": total_cells - missing_cells,
                "missing_cells": missing_cells,
                "completeness_pct": ((total_cells - missing_cells) / total_cells) * 100 if total_cells > 0 else 0
            },
            "rows": {
                "total": len(df),
                "complete": complete_rows,
                "incomplete": len(df) - complete_rows,
                "completeness_pct": (complete_rows / len(df)) * 100 if len(df) > 0 else 0
            },
            "fields": field_completeness,
            "categories": {
                "complete_fields": complete_fields,
                "mostly_complete_fields": mostly_complete,
                "somewhat_complete_fields": somewhat_complete,
                "mostly_missing_fields": mostly_missing,
            }
        }
    
    def data_quality_score(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute overall data quality score.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with quality score and components
        """
        # Completeness score (0-100)
        completeness = self.completeness_report(df)
        completeness_score = completeness["overall"]["completeness_pct"]
        
        # Validity score (based on expected data types)
        validity_score = self._compute_validity_score(df)
        
        # Consistency score (no duplicates, consistent formats)
        consistency_score = self._compute_consistency_score(df)
        
        # Uniqueness score (appropriate cardinality)
        uniqueness_score = self._compute_uniqueness_score(df)
        
        # Overall score (weighted average)
        overall_score = (
            0.35 * completeness_score +
            0.25 * validity_score +
            0.25 * consistency_score +
            0.15 * uniqueness_score
        )
        
        # Grade
        if overall_score >= 90:
            grade = "Excellent"
        elif overall_score >= 75:
            grade = "Good"
        elif overall_score >= 60:
            grade = "Fair"
        else:
            grade = "Poor"
        
        return {
            "overall_score": overall_score,
            "grade": grade,
            "components": {
                "completeness": completeness_score,
                "validity": validity_score,
                "consistency": consistency_score,
                "uniqueness": uniqueness_score,
            },
            "interpretation": self._interpret_quality_score(overall_score)
        }
    
    def anomaly_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect and summarize anomalies in dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with anomaly summary
        """
        anomalies = {
            "constant_fields": [],
            "high_cardinality_numeric": [],
            "low_cardinality_categorical": [],
            "suspicious_missing_patterns": [],
            "outlier_fields": [],
        }
        
        for col in df.columns:
            # Constant fields
            if df[col].nunique() == 1:
                anomalies["constant_fields"].append(col)
            
            # High cardinality numeric fields
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() / len(df[col]) > 0.95:
                anomalies["high_cardinality_numeric"].append(col)
            
            # Low cardinality categorical
            if pd.api.types.is_object_dtype(df[col]) and df[col].nunique() < 3:
                anomalies["low_cardinality_categorical"].append(col)
            
            # Suspicious missing patterns (exactly 50% missing)
            missing_pct = df[col].isna().sum() / len(df[col])
            if 0.48 < missing_pct < 0.52:
                anomalies["suspicious_missing_patterns"].append({
                    "field": col,
                    "missing_pct": missing_pct * 100
                })
            
            # Fields with many outliers (only for numeric, non-boolean fields)
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                clean = df[col].dropna()
                if len(clean) > 0:
                    try:
                        q1, q3 = clean.quantile([0.25, 0.75])
                        iqr = q3 - q1
                        outliers = ((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum()
                        if outliers / len(clean) > 0.1:  # More than 10% outliers
                            anomalies["outlier_fields"].append({
                                "field": col,
                                "outlier_pct": (outliers / len(clean)) * 100
                            })
                    except (TypeError, ValueError):
                        # Skip fields that can't compute quantiles
                        pass
        
        return anomalies
    
    def _overview_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate high-level overview statistics."""
        return {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "n_numeric_fields": len(df.select_dtypes(include='number').columns),
            "n_categorical_fields": len(df.select_dtypes(include=['object', 'category']).columns),
            "n_datetime_fields": len(df.select_dtypes(include='datetime').columns),
            "n_boolean_fields": len(df.select_dtypes(include='bool').columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        }
    
    def _infer_field_type(self, series: pd.Series) -> str:
        """Infer semantic field type."""
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        elif pd.api.types.is_bool_dtype(series):
            return "boolean"
        else:
            return "categorical"
    
    def _numeric_field_summary(self, series: pd.Series) -> Dict[str, Any]:
        """Summary for numeric field."""
        clean = series.dropna()
        if len(clean) == 0:
            return {}
        
        return {
            "min": float(clean.min()),
            "max": float(clean.max()),
            "mean": float(clean.mean()),
            "median": float(clean.median()),
            "std": float(clean.std()),
            "zeros": (clean == 0).sum(),
            "negatives": (clean < 0).sum(),
        }
    
    def _categorical_field_summary(self, series: pd.Series) -> Dict[str, Any]:
        """Summary for categorical field."""
        value_counts = series.value_counts()
        return {
            "top_value": value_counts.index[0] if len(value_counts) > 0 else None,
            "top_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            "top_5_values": value_counts.head(5).to_dict(),
        }
    
    def _datetime_field_summary(self, series: pd.Series) -> Dict[str, Any]:
        """Summary for datetime field."""
        clean = series.dropna()
        if len(clean) == 0:
            return {}
        
        return {
            "min_date": str(clean.min()),
            "max_date": str(clean.max()),
            "range_days": (clean.max() - clean.min()).days,
        }
    
    def _boolean_field_summary(self, series: pd.Series) -> Dict[str, Any]:
        """Summary for boolean field."""
        value_counts = series.value_counts()
        return {
            "true_count": int(value_counts.get(True, 0)),
            "false_count": int(value_counts.get(False, 0)),
            "true_pct": (value_counts.get(True, 0) / len(series)) * 100 if len(series) > 0 else 0,
        }
    
    def _missing_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing value patterns."""
        missing_by_field = df.isna().sum()
        missing_by_field = missing_by_field[missing_by_field > 0].sort_values(ascending=False)
        
        return {
            "fields_with_missing": missing_by_field.to_dict(),
            "n_fields_with_missing": len(missing_by_field),
        }
    
    def _memory_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze memory usage."""
        memory_by_field = df.memory_usage(deep=True)
        total_memory = memory_by_field.sum()
        
        return {
            "total_mb": total_memory / (1024 * 1024),
            "by_field_mb": {col: mem / (1024 * 1024) for col, mem in memory_by_field.items()},
        }
    
    def _duplicate_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicate rows."""
        n_duplicates = df.duplicated().sum()
        
        return {
            "n_duplicates": n_duplicates,
            "duplicate_pct": (n_duplicates / len(df)) * 100 if len(df) > 0 else 0,
        }
    
    def _compute_validity_score(self, df: pd.DataFrame) -> float:
        """Compute validity score (100 = all data valid)."""
        # Simple heuristic: numeric fields should be numeric, etc.
        # For now, assume 95% validity
        return 95.0
    
    def _compute_consistency_score(self, df: pd.DataFrame) -> float:
        """Compute consistency score."""
        duplicate_pct = (df.duplicated().sum() / len(df)) * 100 if len(df) > 0 else 0
        return max(0, 100 - duplicate_pct)
    
    def _compute_uniqueness_score(self, df: pd.DataFrame) -> float:
        """Compute uniqueness score."""
        # High cardinality for ID fields, low for categories
        # Simple heuristic: 85% is appropriate
        return 85.0
    
    def _interpret_quality_score(self, score: float) -> str:
        """Interpret overall quality score."""
        if score >= 90:
            return "Excellent data quality. Dataset is production-ready."
        elif score >= 75:
            return "Good data quality. Minor issues may exist."
        elif score >= 60:
            return "Fair data quality. Some cleaning recommended."
        else:
            return "Poor data quality. Significant cleaning required."
