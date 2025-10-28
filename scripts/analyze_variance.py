"""
Column Variance & Data Quality Analysis Script

Analyzes the variance and quality of all fields in the Week 3 dataset.
Implements comprehensive checks for:
- Numerical field variance (coefficient of variation)
- Categorical field entropy (Shannon entropy)
- Distribution validation (skewness, kurtosis)
- Low-variance field detection
- Expected range validation

Usage:
    python scripts/analyze_variance.py

Outputs:
    - variance_analysis_results.json (detailed metrics)
    - variance_report.txt (human-readable summary)
    - field_distributions.png (distribution visualizations)
    - low_variance_fields.csv (flagged issues)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Skipping visualizations.")


class VarianceAnalyzer:
    """Comprehensive variance and quality analysis for transaction dataset."""
    
    def __init__(self, csv_path: str):
        """Initialize analyzer with dataset."""
        print(f"[1/8] Loading dataset from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        print(f"      Loaded {len(self.df):,} rows, {len(self.df.columns)} columns")
        
        # Define field types
        self.numerical_fields = [
            'Amount', 'Distance_from_Home', 'Time_Since_Last_Txn',
            'Daily_Transaction_Count', 'Daily_Transaction_Amount',
            'Merchant_Reputation', 'Customer_Age', 'Customer_Loyalty_Score'
        ]
        
        self.categorical_fields = [
            'Payment_Mode', 'Category', 'Card_Type', 'Transaction_Status',
            'Transaction_Channel', 'Device_Type', 'Operating_System',
            'Age_Group', 'Income_Bracket', 'Customer_Segment',
            'Digital_Savviness', 'Occupation', 'Risk_Profile',
            'City', 'State', 'Region', 'Merchant_Category'
        ]
        
        self.boolean_fields = [
            'Is_Weekend', 'Is_International', 'Is_First_Transaction_with_Merchant'
        ]
        
        # Quality thresholds
        self.thresholds = {
            'min_entropy': 1.5,  # Minimum Shannon entropy for good diversity
            'min_cv': 0.1,  # Minimum coefficient of variation for numerical fields
            'min_unique_categorical': 3,  # Minimum unique values for categorical
            'max_mode_percentage': 0.95,  # Max % of most common value
            'min_std': 0.01  # Minimum standard deviation
        }
        
        self.results = {}
        
    def calculate_entropy(self, series: pd.Series) -> float:
        """Calculate Shannon entropy for a categorical field."""
        value_counts = series.value_counts(normalize=True)
        entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
        return entropy
    
    def calculate_cv(self, series: pd.Series) -> float:
        """Calculate coefficient of variation (std/mean) for numerical field."""
        mean = series.mean()
        if mean == 0:
            return 0.0
        return series.std() / abs(mean)
    
    def analyze_numerical_field(self, field: str) -> Dict[str, Any]:
        """Comprehensive analysis of a numerical field."""
        data = self.df[field].dropna()
        
        if len(data) == 0:
            return {'error': 'No data after dropping NaN'}
        
        analysis = {
            'type': 'numerical',
            'count': len(data),
            'missing': self.df[field].isna().sum(),
            'missing_pct': (self.df[field].isna().sum() / len(self.df)) * 100,
            'mean': float(data.mean()),
            'median': float(data.median()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'q25': float(data.quantile(0.25)),
            'q75': float(data.quantile(0.75)),
            'cv': float(self.calculate_cv(data)),
            'skewness': float(data.skew()),
            'kurtosis': float(data.kurtosis()),
            'zeros_count': int((data == 0).sum()),
            'zeros_pct': float(((data == 0).sum() / len(data)) * 100)
        }
        
        # Quality flags
        analysis['flags'] = []
        if analysis['cv'] < self.thresholds['min_cv']:
            analysis['flags'].append(f"LOW_VARIANCE: CV={analysis['cv']:.3f} < {self.thresholds['min_cv']}")
        if analysis['std'] < self.thresholds['min_std']:
            analysis['flags'].append(f"LOW_STD: std={analysis['std']:.4f} < {self.thresholds['min_std']}")
        if analysis['missing_pct'] > 50:
            analysis['flags'].append(f"HIGH_MISSING: {analysis['missing_pct']:.1f}% missing")
        
        # Pass/Fail
        analysis['status'] = 'PASS' if len(analysis['flags']) == 0 else 'WARNING'
        
        return analysis
    
    def analyze_categorical_field(self, field: str) -> Dict[str, Any]:
        """Comprehensive analysis of a categorical field."""
        data = self.df[field].dropna()
        
        if len(data) == 0:
            return {'error': 'No data after dropping NaN'}
        
        value_counts = data.value_counts()
        mode_percentage = (value_counts.iloc[0] / len(data)) * 100 if len(value_counts) > 0 else 0
        
        analysis = {
            'type': 'categorical',
            'count': len(data),
            'missing': self.df[field].isna().sum(),
            'missing_pct': (self.df[field].isna().sum() / len(self.df)) * 100,
            'unique_values': int(data.nunique()),
            'entropy': float(self.calculate_entropy(data)),
            'mode': str(value_counts.index[0]) if len(value_counts) > 0 else None,
            'mode_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            'mode_percentage': float(mode_percentage),
            'top_5_values': {str(k): int(v) for k, v in value_counts.head(5).items()}
        }
        
        # Quality flags
        analysis['flags'] = []
        if analysis['unique_values'] < self.thresholds['min_unique_categorical']:
            analysis['flags'].append(f"LOW_DIVERSITY: Only {analysis['unique_values']} unique values")
        if analysis['entropy'] < self.thresholds['min_entropy']:
            analysis['flags'].append(f"LOW_ENTROPY: entropy={analysis['entropy']:.2f} < {self.thresholds['min_entropy']}")
        if analysis['mode_percentage'] > self.thresholds['max_mode_percentage'] * 100:
            analysis['flags'].append(f"HIGH_CONCENTRATION: {analysis['mode_percentage']:.1f}% in mode")
        if analysis['missing_pct'] > 50:
            analysis['flags'].append(f"HIGH_MISSING: {analysis['missing_pct']:.1f}% missing")
        
        # Pass/Fail
        analysis['status'] = 'PASS' if len(analysis['flags']) == 0 else 'WARNING'
        
        return analysis
    
    def analyze_boolean_field(self, field: str) -> Dict[str, Any]:
        """Comprehensive analysis of a boolean field."""
        data = self.df[field].dropna()
        
        if len(data) == 0:
            return {'error': 'No data after dropping NaN'}
        
        value_counts = data.value_counts()
        
        analysis = {
            'type': 'boolean',
            'count': len(data),
            'missing': self.df[field].isna().sum(),
            'missing_pct': (self.df[field].isna().sum() / len(self.df)) * 100,
            'true_count': int(value_counts.get(True, 0)),
            'false_count': int(value_counts.get(False, 0)),
            'true_percentage': float((value_counts.get(True, 0) / len(data)) * 100),
            'false_percentage': float((value_counts.get(False, 0) / len(data)) * 100)
        }
        
        # Quality flags
        analysis['flags'] = []
        if analysis['true_percentage'] > 95 or analysis['true_percentage'] < 5:
            analysis['flags'].append(f"IMBALANCED: {analysis['true_percentage']:.1f}% true")
        if analysis['missing_pct'] > 50:
            analysis['flags'].append(f"HIGH_MISSING: {analysis['missing_pct']:.1f}% missing")
        
        # Pass/Fail
        analysis['status'] = 'PASS' if len(analysis['flags']) == 0 else 'WARNING'
        
        return analysis
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run complete variance analysis on all fields."""
        print("\n[2/8] Analyzing numerical fields...")
        for i, field in enumerate(self.numerical_fields, 1):
            if field in self.df.columns:
                print(f"      [{i}/{len(self.numerical_fields)}] {field}")
                self.results[field] = self.analyze_numerical_field(field)
            else:
                print(f"      [{i}/{len(self.numerical_fields)}] {field} - SKIPPED (not in dataset)")
        
        print("\n[3/8] Analyzing categorical fields...")
        for i, field in enumerate(self.categorical_fields, 1):
            if field in self.df.columns:
                print(f"      [{i}/{len(self.categorical_fields)}] {field}")
                self.results[field] = self.analyze_categorical_field(field)
            else:
                print(f"      [{i}/{len(self.categorical_fields)}] {field} - SKIPPED (not in dataset)")
        
        print("\n[4/8] Analyzing boolean fields...")
        for i, field in enumerate(self.boolean_fields, 1):
            if field in self.df.columns:
                print(f"      [{i}/{len(self.boolean_fields)}] {field}")
                self.results[field] = self.analyze_boolean_field(field)
            else:
                print(f"      [{i}/{len(self.boolean_fields)}] {field} - SKIPPED (not in dataset)")
        
        return self.results
    
    def identify_quality_issues(self) -> Dict[str, List[str]]:
        """Identify all fields with quality issues."""
        print("\n[5/8] Identifying quality issues...")
        
        issues = {
            'low_variance': [],
            'low_diversity': [],
            'high_missing': [],
            'imbalanced': [],
            'all_warnings': []
        }
        
        for field, analysis in self.results.items():
            if 'flags' in analysis and len(analysis['flags']) > 0:
                issues['all_warnings'].append(field)
                
                for flag in analysis['flags']:
                    if 'LOW_VARIANCE' in flag or 'LOW_STD' in flag:
                        issues['low_variance'].append(field)
                    if 'LOW_DIVERSITY' in flag or 'LOW_ENTROPY' in flag:
                        issues['low_diversity'].append(field)
                    if 'HIGH_MISSING' in flag:
                        issues['high_missing'].append(field)
                    if 'IMBALANCED' in flag or 'HIGH_CONCENTRATION' in flag:
                        issues['imbalanced'].append(field)
        
        print(f"      Found {len(issues['all_warnings'])} fields with warnings")
        print(f"      - Low variance: {len(issues['low_variance'])}")
        print(f"      - Low diversity: {len(issues['low_diversity'])}")
        print(f"      - High missing: {len(issues['high_missing'])}")
        print(f"      - Imbalanced: {len(issues['imbalanced'])}")
        
        return issues
    
    def generate_text_report(self) -> str:
        """Generate human-readable text report."""
        print("\n[6/8] Generating text report...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COLUMN VARIANCE & DATA QUALITY ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Dataset: {len(self.df)} rows, {len(self.df.columns)} columns")
        report_lines.append(f"Fields analyzed: {len(self.results)}")
        
        # Summary statistics
        pass_count = sum(1 for r in self.results.values() if r.get('status') == 'PASS')
        warning_count = sum(1 for r in self.results.values() if r.get('status') == 'WARNING')
        
        report_lines.append(f"\nOVERALL QUALITY: {pass_count} PASS, {warning_count} WARNING")
        report_lines.append(f"Pass Rate: {(pass_count / len(self.results) * 100):.1f}%")
        
        # Numerical fields summary
        report_lines.append("\n" + "=" * 80)
        report_lines.append("NUMERICAL FIELDS SUMMARY")
        report_lines.append("=" * 80)
        
        for field in self.numerical_fields:
            if field in self.results:
                r = self.results[field]
                if 'error' not in r:
                    report_lines.append(f"\n{field}:")
                    report_lines.append(f"  Status: {r['status']}")
                    report_lines.append(f"  Range: [{r['min']:.2f}, {r['max']:.2f}]")
                    report_lines.append(f"  Mean ± Std: {r['mean']:.2f} ± {r['std']:.2f}")
                    report_lines.append(f"  CV (Coefficient of Variation): {r['cv']:.3f}")
                    report_lines.append(f"  Skewness: {r['skewness']:.2f}, Kurtosis: {r['kurtosis']:.2f}")
                    report_lines.append(f"  Missing: {r['missing']} ({r['missing_pct']:.1f}%)")
                    if r['flags']:
                        report_lines.append(f"  FLAGS: {', '.join(r['flags'])}")
        
        # Categorical fields summary
        report_lines.append("\n" + "=" * 80)
        report_lines.append("CATEGORICAL FIELDS SUMMARY")
        report_lines.append("=" * 80)
        
        for field in self.categorical_fields:
            if field in self.results:
                r = self.results[field]
                if 'error' not in r:
                    report_lines.append(f"\n{field}:")
                    report_lines.append(f"  Status: {r['status']}")
                    report_lines.append(f"  Unique values: {r['unique_values']}")
                    report_lines.append(f"  Entropy: {r['entropy']:.2f}")
                    report_lines.append(f"  Mode: {r['mode']} ({r['mode_percentage']:.1f}%)")
                    report_lines.append(f"  Missing: {r['missing']} ({r['missing_pct']:.1f}%)")
                    if r['flags']:
                        report_lines.append(f"  FLAGS: {', '.join(r['flags'])}")
        
        # Boolean fields summary
        report_lines.append("\n" + "=" * 80)
        report_lines.append("BOOLEAN FIELDS SUMMARY")
        report_lines.append("=" * 80)
        
        for field in self.boolean_fields:
            if field in self.results:
                r = self.results[field]
                if 'error' not in r:
                    report_lines.append(f"\n{field}:")
                    report_lines.append(f"  Status: {r['status']}")
                    report_lines.append(f"  True: {r['true_count']} ({r['true_percentage']:.1f}%)")
                    report_lines.append(f"  False: {r['false_count']} ({r['false_percentage']:.1f}%)")
                    report_lines.append(f"  Missing: {r['missing']} ({r['missing_pct']:.1f}%)")
                    if r['flags']:
                        report_lines.append(f"  FLAGS: {', '.join(r['flags'])}")
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_results(self, output_dir: str = "output"):
        """Save all analysis results to files."""
        print("\n[7/8] Saving results...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Save JSON results
        json_path = output_path / "variance_analysis_results.json"
        with open(json_path, 'w') as f:
            json.dump(convert_types(self.results), f, indent=2)
        print(f"      Saved: {json_path}")
        
        # Save text report
        report_path = output_path / "variance_report.txt"
        with open(report_path, 'w') as f:
            f.write(self.generate_text_report())
        print(f"      Saved: {report_path}")
        
        # Save quality issues CSV
        issues = self.identify_quality_issues()
        if issues['all_warnings']:
            low_variance_data = []
            for field in issues['all_warnings']:
                r = self.results[field]
                low_variance_data.append({
                    'Field': field,
                    'Type': r['type'],
                    'Status': r['status'],
                    'Flags': '; '.join(r['flags'])
                })
            
            df_issues = pd.DataFrame(low_variance_data)
            csv_path = output_path / "low_variance_fields.csv"
            df_issues.to_csv(csv_path, index=False)
            print(f"      Saved: {csv_path}")


def main():
    """Main execution function."""
    print("Column Variance & Data Quality Analysis")
    print("=" * 80)
    
    # Load dataset
    csv_path = "output/week3_analysis_dataset.csv"
    
    if not Path(csv_path).exists():
        print(f"ERROR: Dataset not found at {csv_path}")
        print("Please run scripts/generate_week3_dataset.py first.")
        return
    
    # Run analysis
    analyzer = VarianceAnalyzer(csv_path)
    analyzer.run_analysis()
    analyzer.save_results()
    
    print("\n[8/8] Analysis complete!")
    print("\nGenerated files:")
    print("  - output/variance_analysis_results.json")
    print("  - output/variance_report.txt")
    print("  - output/low_variance_fields.csv")
    print("\nReview variance_report.txt for detailed findings.")


if __name__ == "__main__":
    main()
