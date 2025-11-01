"""
Analysis Report Generator for creating comprehensive statistical reports.

Provides methods for:
- Generating text reports
- Exporting to JSON
- Exporting to CSV
- Creating summary statistics
"""

from typing import Dict, List, Optional, Any
import json
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AnalysisReport:
    """
    Generate comprehensive analysis reports in multiple formats.
    
    Consolidates results from all analytics modules into
    readable reports.
    """
    
    def __init__(self):
        """Initialize the Analysis Report generator."""
        pass
    
    def generate_statistical_report(
        self,
        analysis_results: Dict[str, Any],
        title: str = "Statistical Analysis Report"
    ) -> str:
        """
        Generate comprehensive text report from analysis results.
        
        Args:
            analysis_results: Dictionary with all analysis results
            title: Report title
            
        Returns:
            Formatted text report
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"{title:^80}")
        lines.append(f"{'Generated: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^80}")
        lines.append("=" * 80)
        lines.append("")
        
        # Dataset overview
        if "dataset_info" in analysis_results:
            lines.append("DATASET OVERVIEW")
            lines.append("-" * 80)
            info = analysis_results["dataset_info"]
            lines.append(f"  Rows: {info.get('row_count', 'N/A'):,}")
            lines.append(f"  Columns: {info.get('column_count', 'N/A'):,}")
            lines.append(f"  Memory Usage: {info.get('memory_usage_mb', 0):.2f} MB")
            lines.append(f"  Numeric Fields: {len(info.get('numeric_columns', []))}")
            lines.append(f"  Categorical Fields: {len(info.get('categorical_columns', []))}")
            lines.append(f"  Datetime Fields: {len(info.get('datetime_columns', []))}")
            lines.append("")
        
        # Numeric fields summary
        if "numeric_fields" in analysis_results:
            lines.append("NUMERIC FIELDS SUMMARY")
            lines.append("-" * 80)
            for field, stats in list(analysis_results["numeric_fields"].items())[:10]:  # First 10
                if "error" not in stats:
                    lines.append(f"\n  {field}:")
                    lines.append(f"    Count: {stats.get('count', 'N/A'):,}")
                    lines.append(f"    Mean: {stats.get('mean', 0):.2f}")
                    lines.append(f"    Median: {stats.get('median', 0):.2f}")
                    lines.append(f"    Std Dev: {stats.get('std', 0):.2f}")
                    lines.append(f"    Range: [{stats.get('min', 0):.2f}, {stats.get('max', 0):.2f}]")
                    if 'skewness' in stats:
                        lines.append(f"    Skewness: {stats['skewness']:.2f} ({stats.get('skewness_interpretation', '')})")
            lines.append("")
        
        # Categorical fields summary
        if "categorical_fields" in analysis_results:
            lines.append("CATEGORICAL FIELDS SUMMARY")
            lines.append("-" * 80)
            for field, stats in list(analysis_results["categorical_fields"].items())[:10]:
                if "error" not in stats:
                    lines.append(f"\n  {field}:")
                    lines.append(f"    Unique Values: {stats.get('unique_count', 'N/A')}")
                    lines.append(f"    Mode: {stats.get('mode', 'N/A')} ({stats.get('mode_percentage', 0):.1f}%)")
                    lines.append(f"    Cardinality: {stats.get('cardinality', 0):.3f}")
            lines.append("")
        
        # Quality score
        if "quality_score" in analysis_results:
            lines.append("DATA QUALITY ASSESSMENT")
            lines.append("-" * 80)
            qs = analysis_results["quality_score"]
            lines.append(f"  Overall Score: {qs.get('overall_score', 0):.1f}/100 ({qs.get('grade', 'N/A')})")
            lines.append(f"  Completeness: {qs.get('components', {}).get('completeness', 0):.1f}/100")
            lines.append(f"  Validity: {qs.get('components', {}).get('validity', 0):.1f}/100")
            lines.append(f"  Consistency: {qs.get('components', {}).get('consistency', 0):.1f}/100")
            lines.append(f"  Uniqueness: {qs.get('components', {}).get('uniqueness', 0):.1f}/100")
            lines.append(f"\n  {qs.get('interpretation', '')}")
            lines.append("")
        
        # Correlation analysis
        if "strong_correlations" in analysis_results:
            lines.append("STRONG CORRELATIONS")
            lines.append("-" * 80)
            correlations = analysis_results["strong_correlations"][:10]  # Top 10
            if correlations:
                for corr in correlations:
                    lines.append(f"  {corr['field1']} <-> {corr['field2']}: "
                               f"r={corr['correlation']:.3f} ({corr['strength']}, {corr['direction']})")
            else:
                lines.append("  No strong correlations found.")
            lines.append("")
        
        # Outliers
        if "outliers" in analysis_results:
            lines.append("OUTLIER DETECTION")
            lines.append("-" * 80)
            outliers = analysis_results["outliers"]
            lines.append(f"  Method: {outliers.get('method', 'N/A')}")
            lines.append(f"  Outliers Found: {outliers.get('outlier_count', 0):,} "
                       f"({outliers.get('outlier_percentage', 0):.2f}%)")
            lines.append("")
        
        lines.append("=" * 80)
        lines.append(f"{'End of Report':^80}")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def export_to_json(
        self,
        analysis_results: Dict[str, Any],
        filepath: str
    ) -> None:
        """
        Export analysis results to JSON file.
        
        Args:
            analysis_results: Dictionary with all analysis results
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        logger.info(f"Exported analysis to JSON: {filepath}")
    
    def export_to_csv(
        self,
        analysis_results: Dict[str, Any],
        output_dir: str
    ) -> List[str]:
        """
        Export analysis results to CSV files.
        
        Args:
            analysis_results: Dictionary with all analysis results
            output_dir: Output directory path
            
        Returns:
            List of created file paths
        """
        created_files = []
        
        # Export numeric fields summary
        if "numeric_fields" in analysis_results:
            numeric_df = pd.DataFrame(analysis_results["numeric_fields"]).T
            filepath = f"{output_dir}/numeric_fields_summary.csv"
            numeric_df.to_csv(filepath)
            created_files.append(filepath)
            logger.info(f"Exported numeric fields summary: {filepath}")
        
        # Export categorical fields summary
        if "categorical_fields" in analysis_results:
            categorical_df = pd.DataFrame(analysis_results["categorical_fields"]).T
            filepath = f"{output_dir}/categorical_fields_summary.csv"
            categorical_df.to_csv(filepath)
            created_files.append(filepath)
            logger.info(f"Exported categorical fields summary: {filepath}")
        
        # Export strong correlations
        if "strong_correlations" in analysis_results:
            corr_df = pd.DataFrame(analysis_results["strong_correlations"])
            filepath = f"{output_dir}/strong_correlations.csv"
            corr_df.to_csv(filepath, index=False)
            created_files.append(filepath)
            logger.info(f"Exported strong correlations: {filepath}")
        
        return created_files
    
    def summary_statistics(
        self,
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract high-level summary statistics from analysis results.
        
        Args:
            analysis_results: Dictionary with all analysis results
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "dataset": {},
            "quality": {},
            "insights": {},
        }
        
        # Dataset summary
        if "dataset_info" in analysis_results:
            info = analysis_results["dataset_info"]
            summary["dataset"] = {
                "rows": info.get("row_count", 0),
                "columns": info.get("column_count", 0),
                "numeric_fields": len(info.get("numeric_columns", [])),
                "categorical_fields": len(info.get("categorical_columns", [])),
                "memory_mb": info.get("memory_usage_mb", 0),
            }
        
        # Quality summary
        if "quality_score" in analysis_results:
            qs = analysis_results["quality_score"]
            summary["quality"] = {
                "overall_score": qs.get("overall_score", 0),
                "grade": qs.get("grade", "N/A"),
                "completeness": qs.get("components", {}).get("completeness", 0),
                "validity": qs.get("components", {}).get("validity", 0),
            }
        
        # Insights
        summary["insights"] = {
            "strong_correlations_found": len(analysis_results.get("strong_correlations", [])),
            "fields_with_outliers": 1 if "outliers" in analysis_results else 0,
            "anomalies_detected": len(analysis_results.get("anomalies", {}).get("constant_fields", [])),
        }
        
        return summary
    
    def create_comparison_report(
        self,
        before_results: Dict[str, Any],
        after_results: Dict[str, Any],
        title: str = "Comparison Report"
    ) -> str:
        """
        Generate comparison report between two analysis results.
        
        Args:
            before_results: Analysis results from "before" dataset
            after_results: Analysis results from "after" dataset
            title: Report title
            
        Returns:
            Formatted comparison report
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"{title:^80}")
        lines.append("=" * 80)
        lines.append("")
        
        # Dataset size comparison
        before_rows = before_results.get("dataset_info", {}).get("row_count", 0)
        after_rows = after_results.get("dataset_info", {}).get("row_count", 0)
        
        lines.append("DATASET SIZE")
        lines.append("-" * 80)
        lines.append(f"  Before: {before_rows:,} rows")
        lines.append(f"  After: {after_rows:,} rows")
        lines.append(f"  Change: {after_rows - before_rows:+,} rows ({((after_rows - before_rows) / before_rows * 100):.1f}%)")
        lines.append("")
        
        # Quality score comparison
        before_score = before_results.get("quality_score", {}).get("overall_score", 0)
        after_score = after_results.get("quality_score", {}).get("overall_score", 0)
        
        lines.append("DATA QUALITY")
        lines.append("-" * 80)
        lines.append(f"  Before: {before_score:.1f}/100")
        lines.append(f"  After: {after_score:.1f}/100")
        lines.append(f"  Change: {after_score - before_score:+.1f}")
        lines.append("")
        
        lines.append("=" * 80)
        return "\n".join(lines)
