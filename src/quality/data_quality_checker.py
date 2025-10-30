"""
Data Quality Checker for SynFinance

Comprehensive validation framework for transaction datasets including:
- Schema validation (types, required fields, ranges)
- Statistical validation (distributions, outliers, correlations)
- Business rule validation (fraud rate, anomaly rate, patterns)
- Temporal validation (date sequences, time patterns)
- Referential integrity (ID consistency)

Week 7 Day 3: Automated Quality Assurance Framework
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime
from enum import Enum

import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Types of quality violations"""
    SCHEMA = "schema"
    STATISTICAL = "statistical"
    BUSINESS_RULE = "business_rule"
    TEMPORAL = "temporal"
    REFERENTIAL = "referential"


class Severity(Enum):
    """Severity levels for violations"""
    CRITICAL = "critical"  # Blocks deployment
    ERROR = "error"        # Major issue, should fix
    WARNING = "warning"    # Minor issue, optional fix
    INFO = "info"          # Informational only


@dataclass
class QualityViolation:
    """
    Represents a single quality violation
    
    Attributes:
        violation_type: Type of violation
        severity: Severity level
        field: Field name (if applicable)
        message: Description of the violation
        actual_value: Actual value found
        expected_value: Expected value/range
        count: Number of occurrences
        percentage: Percentage of dataset affected
    """
    violation_type: ViolationType
    severity: Severity
    field: Optional[str]
    message: str
    actual_value: Any
    expected_value: Any
    count: int = 0
    percentage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'type': self.violation_type.value,
            'severity': self.severity.value,
            'field': self.field,
            'message': self.message,
            'actual': str(self.actual_value),
            'expected': str(self.expected_value),
            'count': self.count,
            'percentage': round(self.percentage, 4)
        }


@dataclass
class ValidationResult:
    """
    Result of a single validation check
    
    Attributes:
        check_name: Name of the validation check
        passed: Whether the check passed
        violations: List of violations found
        metrics: Additional metrics from the check
    """
    check_name: str
    passed: bool
    violations: List[QualityViolation] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'check': self.check_name,
            'passed': self.passed,
            'violations': [v.to_dict() for v in self.violations],
            'metrics': self.metrics
        }


@dataclass
class QualityReport:
    """
    Comprehensive quality report for a dataset
    
    Attributes:
        dataset_name: Name/identifier of the dataset
        timestamp: When the check was performed
        total_records: Number of records checked
        quality_score: Overall quality score (0-100)
        validation_results: List of all validation results
        summary: Summary statistics
        recommendations: Suggested actions
    """
    dataset_name: str
    timestamp: datetime
    total_records: int
    quality_score: float
    validation_results: List[ValidationResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def total_violations(self) -> int:
        """Total number of violations"""
        return sum(len(r.violations) for r in self.validation_results)
    
    @property
    def critical_violations(self) -> int:
        """Number of critical violations"""
        return sum(
            sum(1 for v in r.violations if v.severity == Severity.CRITICAL)
            for r in self.validation_results
        )
    
    @property
    def passed_checks(self) -> int:
        """Number of checks that passed"""
        return sum(1 for r in self.validation_results if r.passed)
    
    @property
    def failed_checks(self) -> int:
        """Number of checks that failed"""
        return sum(1 for r in self.validation_results if not r.passed)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'dataset': self.dataset_name,
            'timestamp': self.timestamp.isoformat(),
            'total_records': self.total_records,
            'quality_score': round(self.quality_score, 2),
            'checks': {
                'total': len(self.validation_results),
                'passed': self.passed_checks,
                'failed': self.failed_checks
            },
            'violations': {
                'total': self.total_violations,
                'critical': self.critical_violations
            },
            'validation_results': [r.to_dict() for r in self.validation_results],
            'summary': self.summary,
            'recommendations': self.recommendations
        }


class DataQualityChecker:
    """
    Comprehensive data quality validation framework
    
    Performs multi-level validation on transaction datasets:
    - Schema validation (field presence, types, ranges)
    - Statistical validation (distributions, outliers)
    - Business rule validation (fraud rates, patterns)
    - Temporal validation (date sequences, patterns)
    - Referential integrity (ID consistency)
    """
    
    # Expected schema for transaction data
    REQUIRED_FIELDS = {
        'transaction_id', 'customer_id', 'merchant_id', 'amount',
        'payment_mode', 'category', 'timestamp'
    }
    
    OPTIONAL_FIELDS = {
        'is_fraud', 'fraud_type', 'fraud_confidence', 'is_anomaly',
        'anomaly_score', 'city', 'state', 'distance', 'merchant_reputation'
    }
    
    NUMERIC_FIELDS = {
        'amount', 'distance', 'merchant_reputation', 'anomaly_score', 'fraud_confidence'
    }
    
    CATEGORICAL_FIELDS = {
        'payment_mode', 'category', 'city', 'state', 'fraud_type'
    }
    
    # Expected ranges for numeric fields
    FIELD_RANGES = {
        'amount': (0, 1000000),  # 0 to 10 lakh
        'distance': (0, 5000),   # 0 to 5000 km
        'merchant_reputation': (0, 1),  # 0 to 1
        'anomaly_score': (0, 1),  # 0 to 1
        'fraud_confidence': (0, 1),  # 0 to 1
    }
    
    # Business rule thresholds
    FRAUD_RATE_RANGE = (0.005, 0.025)  # 0.5% to 2.5%
    ANOMALY_RATE_RANGE = (0.05, 0.20)  # 5% to 20%
    MISSING_VALUE_THRESHOLD = 0.01  # Max 1% missing
    OUTLIER_THRESHOLD = 0.05  # Max 5% outliers
    CORRELATION_THRESHOLD = 0.90  # Max correlation (no duplicates)
    
    def __init__(self, dataset_name: str = "transaction_data"):
        """
        Initialize quality checker
        
        Args:
            dataset_name: Name/identifier for the dataset
        """
        self.dataset_name = dataset_name
        self.validation_results: List[ValidationResult] = []
        
    def check_quality(self, df: pd.DataFrame) -> QualityReport:
        """
        Perform comprehensive quality check on dataset
        
        Args:
            df: DataFrame to validate
            
        Returns:
            QualityReport with all results
        """
        logger.info(f"Starting quality check on {self.dataset_name} ({len(df)} records)")
        
        self.validation_results = []
        
        # Run all validation checks
        self._validate_schema(df)
        self._validate_statistical_properties(df)
        self._validate_business_rules(df)
        self._validate_temporal_patterns(df)
        self._validate_referential_integrity(df)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score()
        
        # Generate summary and recommendations
        summary = self._generate_summary(df)
        recommendations = self._generate_recommendations()
        
        report = QualityReport(
            dataset_name=self.dataset_name,
            timestamp=datetime.now(),
            total_records=len(df),
            quality_score=quality_score,
            validation_results=self.validation_results,
            summary=summary,
            recommendations=recommendations
        )
        
        logger.info(f"Quality check complete. Score: {quality_score:.2f}/100")
        return report
    
    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Validate dataset schema"""
        violations = []
        
        # Check required fields
        missing_fields = self.REQUIRED_FIELDS - set(df.columns)
        if missing_fields:
            violations.append(QualityViolation(
                violation_type=ViolationType.SCHEMA,
                severity=Severity.CRITICAL,
                field=None,
                message=f"Missing required fields: {missing_fields}",
                actual_value=list(df.columns),
                expected_value=list(self.REQUIRED_FIELDS),
                count=len(missing_fields),
                percentage=100.0
            ))
        
        # Check data types and ranges for numeric fields
        for field in self.NUMERIC_FIELDS:
            if field not in df.columns:
                continue
                
            # Check if numeric
            if not pd.api.types.is_numeric_dtype(df[field]):
                violations.append(QualityViolation(
                    violation_type=ViolationType.SCHEMA,
                    severity=Severity.ERROR,
                    field=field,
                    message=f"Field {field} should be numeric",
                    actual_value=str(df[field].dtype),
                    expected_value="numeric",
                    count=len(df),
                    percentage=100.0
                ))
                continue
            
            # Check range
            if field in self.FIELD_RANGES:
                min_val, max_val = self.FIELD_RANGES[field]
                out_of_range = ((df[field] < min_val) | (df[field] > max_val)).sum()
                
                if out_of_range > 0:
                    violations.append(QualityViolation(
                        violation_type=ViolationType.SCHEMA,
                        severity=Severity.ERROR,
                        field=field,
                        message=f"Field {field} has values outside valid range",
                        actual_value=f"[{df[field].min()}, {df[field].max()}]",
                        expected_value=f"[{min_val}, {max_val}]",
                        count=out_of_range,
                        percentage=(out_of_range / len(df)) * 100
                    ))
        
        # Check for missing values in required fields
        for field in self.REQUIRED_FIELDS:
            if field not in df.columns:
                continue
                
            missing_count = df[field].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            if missing_pct > self.MISSING_VALUE_THRESHOLD * 100:
                violations.append(QualityViolation(
                    violation_type=ViolationType.SCHEMA,
                    severity=Severity.ERROR if missing_pct > 5 else Severity.WARNING,
                    field=field,
                    message=f"Field {field} has excessive missing values",
                    actual_value=missing_pct,
                    expected_value=f"<{self.MISSING_VALUE_THRESHOLD * 100}%",
                    count=missing_count,
                    percentage=missing_pct
                ))
        
        self.validation_results.append(ValidationResult(
            check_name="Schema Validation",
            passed=len(violations) == 0,
            violations=violations,
            metrics={
                'total_fields': len(df.columns),
                'required_fields_present': len(self.REQUIRED_FIELDS & set(df.columns)),
                'numeric_fields_valid': sum(
                    1 for f in self.NUMERIC_FIELDS 
                    if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
                )
            }
        ))
    
    def _validate_statistical_properties(self, df: pd.DataFrame) -> None:
        """Validate statistical properties of the data"""
        violations = []
        
        # Check for outliers in numeric fields using IQR method
        for field in self.NUMERIC_FIELDS:
            if field not in df.columns or df[field].isna().all():
                continue
            
            Q1 = df[field].quantile(0.25)
            Q3 = df[field].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = ((df[field] < lower_bound) | (df[field] > upper_bound)).sum()
            outlier_pct = (outliers / len(df)) * 100
            
            if outlier_pct > self.OUTLIER_THRESHOLD * 100:
                violations.append(QualityViolation(
                    violation_type=ViolationType.STATISTICAL,
                    severity=Severity.WARNING,
                    field=field,
                    message=f"Field {field} has excessive outliers",
                    actual_value=outlier_pct,
                    expected_value=f"<{self.OUTLIER_THRESHOLD * 100}%",
                    count=outliers,
                    percentage=outlier_pct
                ))
        
        # Check for duplicate IDs
        if 'transaction_id' in df.columns:
            duplicates = df['transaction_id'].duplicated().sum()
            if duplicates > 0:
                violations.append(QualityViolation(
                    violation_type=ViolationType.STATISTICAL,
                    severity=Severity.CRITICAL,
                    field='transaction_id',
                    message="Duplicate transaction IDs found",
                    actual_value=duplicates,
                    expected_value=0,
                    count=duplicates,
                    percentage=(duplicates / len(df)) * 100
                ))
        
        # Check for high correlations (potential duplicates)
        numeric_df = df[list(self.NUMERIC_FIELDS & set(df.columns))].select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr().abs()
            
            # Find high correlations (excluding diagonal)
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > self.CORRELATION_THRESHOLD:
                        high_corr_pairs.append((
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j]
                        ))
            
            if high_corr_pairs:
                for col1, col2, corr_val in high_corr_pairs:
                    violations.append(QualityViolation(
                        violation_type=ViolationType.STATISTICAL,
                        severity=Severity.WARNING,
                        field=f"{col1} vs {col2}",
                        message=f"High correlation between {col1} and {col2}",
                        actual_value=corr_val,
                        expected_value=f"<{self.CORRELATION_THRESHOLD}",
                        count=1,
                        percentage=0.0
                    ))
        
        self.validation_results.append(ValidationResult(
            check_name="Statistical Validation",
            passed=len(violations) == 0,
            violations=violations,
            metrics={
                'outlier_fields_checked': len(self.NUMERIC_FIELDS & set(df.columns)),
                'correlation_pairs_checked': len(numeric_df.columns) * (len(numeric_df.columns) - 1) // 2
            }
        ))
    
    def _validate_business_rules(self, df: pd.DataFrame) -> None:
        """Validate business rules"""
        violations = []
        
        # Check fraud rate
        if 'is_fraud' in df.columns:
            fraud_rate = df['is_fraud'].mean()
            min_rate, max_rate = self.FRAUD_RATE_RANGE
            
            if not (min_rate <= fraud_rate <= max_rate):
                violations.append(QualityViolation(
                    violation_type=ViolationType.BUSINESS_RULE,
                    severity=Severity.ERROR,
                    field='is_fraud',
                    message="Fraud rate outside expected range",
                    actual_value=fraud_rate,
                    expected_value=f"[{min_rate}, {max_rate}]",
                    count=df['is_fraud'].sum(),
                    percentage=fraud_rate * 100
                ))
        
        # Check anomaly rate
        if 'is_anomaly' in df.columns:
            anomaly_rate = df['is_anomaly'].mean()
            min_rate, max_rate = self.ANOMALY_RATE_RANGE
            
            if not (min_rate <= anomaly_rate <= max_rate):
                violations.append(QualityViolation(
                    violation_type=ViolationType.BUSINESS_RULE,
                    severity=Severity.WARNING,
                    field='is_anomaly',
                    message="Anomaly rate outside expected range",
                    actual_value=anomaly_rate,
                    expected_value=f"[{min_rate}, {max_rate}]",
                    count=df['is_anomaly'].sum(),
                    percentage=anomaly_rate * 100
                ))
        
        # Check payment mode distribution
        if 'payment_mode' in df.columns:
            mode_counts = df['payment_mode'].value_counts()
            # Each mode should have at least 1% of transactions
            rare_modes = mode_counts[mode_counts / len(df) < 0.01]
            
            if len(rare_modes) > 0:
                violations.append(QualityViolation(
                    violation_type=ViolationType.BUSINESS_RULE,
                    severity=Severity.INFO,
                    field='payment_mode',
                    message=f"Some payment modes have very low representation: {list(rare_modes.index)}",
                    actual_value=list(rare_modes.index),
                    expected_value=">1% each",
                    count=len(rare_modes),
                    percentage=0.0
                ))
        
        # Check amount distribution (should be right-skewed)
        if 'amount' in df.columns:
            skewness = df['amount'].skew()
            if skewness < 0.5:  # Should be positively skewed
                violations.append(QualityViolation(
                    violation_type=ViolationType.BUSINESS_RULE,
                    severity=Severity.WARNING,
                    field='amount',
                    message="Amount distribution not right-skewed as expected",
                    actual_value=skewness,
                    expected_value=">0.5",
                    count=0,
                    percentage=0.0
                ))
        
        self.validation_results.append(ValidationResult(
            check_name="Business Rule Validation",
            passed=len(violations) == 0,
            violations=violations,
            metrics={
                'fraud_rate': df['is_fraud'].mean() if 'is_fraud' in df.columns else None,
                'anomaly_rate': df['is_anomaly'].mean() if 'is_anomaly' in df.columns else None,
                'unique_payment_modes': df['payment_mode'].nunique() if 'payment_mode' in df.columns else None
            }
        ))
    
    def _validate_temporal_patterns(self, df: pd.DataFrame) -> None:
        """Validate temporal patterns in the data"""
        violations = []
        
        if 'timestamp' not in df.columns:
            self.validation_results.append(ValidationResult(
                check_name="Temporal Validation",
                passed=True,
                violations=[],
                metrics={'skipped': 'timestamp column not present'}
            ))
            return
        
        # Convert to datetime if needed
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
            try:
                df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            except Exception as e:
                violations.append(QualityViolation(
                    violation_type=ViolationType.TEMPORAL,
                    severity=Severity.ERROR,
                    field='timestamp',
                    message=f"Could not parse timestamp column: {str(e)}",
                    actual_value=str(df_copy['timestamp'].dtype),
                    expected_value="datetime",
                    count=len(df),
                    percentage=100.0
                ))
                self.validation_results.append(ValidationResult(
                    check_name="Temporal Validation",
                    passed=False,
                    violations=violations,
                    metrics={}
                ))
                return
        
        # Check for future dates
        now = pd.Timestamp.now()
        future_dates = (df_copy['timestamp'] > now).sum()
        if future_dates > 0:
            violations.append(QualityViolation(
                violation_type=ViolationType.TEMPORAL,
                severity=Severity.ERROR,
                field='timestamp',
                message="Transactions with future timestamps found",
                actual_value=future_dates,
                expected_value=0,
                count=future_dates,
                percentage=(future_dates / len(df)) * 100
            ))
        
        # Check for reasonable date range (not older than 10 years)
        ten_years_ago = now - pd.Timedelta(days=3650)
        old_dates = (df_copy['timestamp'] < ten_years_ago).sum()
        if old_dates > 0:
            violations.append(QualityViolation(
                violation_type=ViolationType.TEMPORAL,
                severity=Severity.WARNING,
                field='timestamp',
                message="Transactions older than 10 years found",
                actual_value=old_dates,
                expected_value=0,
                count=old_dates,
                percentage=(old_dates / len(df)) * 100
            ))
        
        # Check for time gaps (no gaps > 30 days for production data)
        df_sorted = df_copy.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff()
        max_gap = time_diffs.max()
        if pd.notna(max_gap) and max_gap > pd.Timedelta(days=30):
            violations.append(QualityViolation(
                violation_type=ViolationType.TEMPORAL,
                severity=Severity.INFO,
                field='timestamp',
                message=f"Large time gap found: {max_gap}",
                actual_value=str(max_gap),
                expected_value="<30 days",
                count=1,
                percentage=0.0
            ))
        
        self.validation_results.append(ValidationResult(
            check_name="Temporal Validation",
            passed=len(violations) == 0,
            violations=violations,
            metrics={
                'date_range_days': (df_copy['timestamp'].max() - df_copy['timestamp'].min()).days,
                'max_gap_days': max_gap.days if pd.notna(max_gap) else None
            }
        ))
    
    def _validate_referential_integrity(self, df: pd.DataFrame) -> None:
        """Validate referential integrity"""
        violations = []
        
        # Check customer_id consistency
        if 'customer_id' in df.columns:
            null_customers = df['customer_id'].isna().sum()
            if null_customers > 0:
                violations.append(QualityViolation(
                    violation_type=ViolationType.REFERENTIAL,
                    severity=Severity.ERROR,
                    field='customer_id',
                    message="NULL customer IDs found",
                    actual_value=null_customers,
                    expected_value=0,
                    count=null_customers,
                    percentage=(null_customers / len(df)) * 100
                ))
        
        # Check merchant_id consistency
        if 'merchant_id' in df.columns:
            null_merchants = df['merchant_id'].isna().sum()
            if null_merchants > 0:
                violations.append(QualityViolation(
                    violation_type=ViolationType.REFERENTIAL,
                    severity=Severity.ERROR,
                    field='merchant_id',
                    message="NULL merchant IDs found",
                    actual_value=null_merchants,
                    expected_value=0,
                    count=null_merchants,
                    percentage=(null_merchants / len(df)) * 100
                ))
        
        # Check if fraud transactions have fraud_type
        if 'is_fraud' in df.columns and 'fraud_type' in df.columns:
            fraud_df = df[df['is_fraud'] == True]
            missing_fraud_type = fraud_df['fraud_type'].isna().sum()
            
            if missing_fraud_type > 0:
                violations.append(QualityViolation(
                    violation_type=ViolationType.REFERENTIAL,
                    severity=Severity.WARNING,
                    field='fraud_type',
                    message="Fraud transactions missing fraud_type",
                    actual_value=missing_fraud_type,
                    expected_value=0,
                    count=missing_fraud_type,
                    percentage=(missing_fraud_type / len(fraud_df)) * 100 if len(fraud_df) > 0 else 0
                ))
        
        self.validation_results.append(ValidationResult(
            check_name="Referential Integrity",
            passed=len(violations) == 0,
            violations=violations,
            metrics={
                'unique_customers': df['customer_id'].nunique() if 'customer_id' in df.columns else None,
                'unique_merchants': df['merchant_id'].nunique() if 'merchant_id' in df.columns else None
            }
        ))
    
    def _calculate_quality_score(self) -> float:
        """
        Calculate overall quality score (0-100)
        
        Scoring:
        - Critical violations: -20 points each
        - Error violations: -5 points each
        - Warning violations: -1 point each
        - Info violations: -0.1 points each
        """
        score = 100.0
        
        for result in self.validation_results:
            for violation in result.violations:
                if violation.severity == Severity.CRITICAL:
                    score -= 20
                elif violation.severity == Severity.ERROR:
                    score -= 5
                elif violation.severity == Severity.WARNING:
                    score -= 1
                elif violation.severity == Severity.INFO:
                    score -= 0.1
        
        return max(0.0, min(100.0, score))
    
    def _generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics"""
        return {
            'total_records': len(df),
            'total_fields': len(df.columns),
            'numeric_fields': len([f for f in df.columns if pd.api.types.is_numeric_dtype(df[f])]),
            'categorical_fields': len([f for f in df.columns if df[f].dtype == 'object']),
            'missing_values': df.isna().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on violations"""
        recommendations = []
        
        # Count violations by severity
        critical_count = sum(
            sum(1 for v in r.violations if v.severity == Severity.CRITICAL)
            for r in self.validation_results
        )
        error_count = sum(
            sum(1 for v in r.violations if v.severity == Severity.ERROR)
            for r in self.validation_results
        )
        
        if critical_count > 0:
            recommendations.append(
                f"CRITICAL: Fix {critical_count} critical violations before deployment"
            )
        
        if error_count > 0:
            recommendations.append(
                f"ERROR: Address {error_count} error-level issues to improve data quality"
            )
        
        # Specific recommendations based on violation types
        for result in self.validation_results:
            for violation in result.violations:
                if violation.violation_type == ViolationType.SCHEMA:
                    if violation.severity in [Severity.CRITICAL, Severity.ERROR]:
                        recommendations.append(
                            f"Fix schema issue in field '{violation.field}': {violation.message}"
                        )
                elif violation.violation_type == ViolationType.BUSINESS_RULE:
                    if 'fraud_rate' in violation.message.lower():
                        recommendations.append(
                            "Adjust fraud rate parameters in data generation configuration"
                        )
                    elif 'anomaly_rate' in violation.message.lower():
                        recommendations.append(
                            "Adjust anomaly rate parameters in data generation configuration"
                        )
        
        if not recommendations:
            recommendations.append("Dataset passes all quality checks. Ready for production use.")
        
        return list(set(recommendations))  # Remove duplicates
