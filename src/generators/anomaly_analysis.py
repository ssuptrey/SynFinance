"""
Anomaly Analysis and Validation Module

This module provides comprehensive analysis tools for anomaly patterns in synthetic
transaction datasets. It includes correlation analysis with fraud patterns, severity
distribution analysis, temporal clustering detection, and geographic heatmap generation.

For Indian market synthetic financial data generation.
Production-ready implementation with statistical validation.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import statistics
from datetime import datetime


@dataclass
class CorrelationResult:
    """Results from anomaly-fraud correlation analysis"""
    total_transactions: int
    fraud_count: int
    anomaly_count: int
    overlap_count: int
    overlap_rate: float  # Percentage of frauds that are also anomalies
    reverse_overlap_rate: float  # Percentage of anomalies that are also fraud
    correlation_by_type: Dict[str, Dict[str, int]]  # Fraud type -> Anomaly type -> count
    correlation_strength: float  # Overall correlation coefficient (0.0-1.0)
    statistical_significance: bool  # Chi-square test result
    chi_square_statistic: float
    p_value: float


@dataclass
class SeverityDistribution:
    """Severity score distribution analysis"""
    anomaly_type: str
    count: int
    mean_severity: float
    median_severity: float
    std_deviation: float
    min_severity: float
    max_severity: float
    low_severity_count: int  # 0.0-0.3
    medium_severity_count: int  # 0.3-0.6
    high_severity_count: int  # 0.6-0.8
    critical_severity_count: int  # 0.8-1.0
    severity_bins: Dict[str, int]  # Histogram bins


@dataclass
class TemporalCluster:
    """Temporal anomaly cluster detection"""
    hour: int
    day_of_week: int
    anomaly_count: int
    cluster_size: float  # Anomalies per hour in this time slot
    is_burst: bool  # True if significantly above average
    burst_multiplier: float  # How many times above baseline
    anomaly_types: Dict[str, int]  # Distribution of anomaly types in cluster


@dataclass
class GeographicHeatmap:
    """Geographic anomaly pattern analysis"""
    from_city: str
    to_city: str
    anomaly_count: int
    avg_distance_km: float
    avg_severity: float
    impossible_travel_count: int
    transition_rate: float  # Anomalies / total transitions on this route


class AnomalyFraudCorrelationAnalyzer:
    """
    Analyze correlation between anomaly patterns and fraud patterns
    
    Measures overlap, identifies which anomalies are most likely fraud,
    and performs statistical significance testing.
    """
    
    def __init__(self):
        """Initialize the correlation analyzer"""
        self.transactions: List[Dict[str, Any]] = []
        self.results: Optional[CorrelationResult] = None
    
    def analyze(self, transactions: List[Dict[str, Any]]) -> CorrelationResult:
        """
        Perform comprehensive correlation analysis
        
        Args:
            transactions: List of transaction dictionaries with fraud and anomaly labels
            
        Returns:
            CorrelationResult with detailed analysis
        """
        self.transactions = transactions
        
        # Count fraud and anomalies
        fraud_count = sum(1 for t in transactions if t.get('Fraud_Type', 'None') != 'None')
        anomaly_count = sum(1 for t in transactions if t.get('Anomaly_Type', 'None') != 'None')
        
        # Count overlap (both fraud AND anomaly)
        overlap_count = sum(
            1 for t in transactions 
            if t.get('Fraud_Type', 'None') != 'None' 
            and t.get('Anomaly_Type', 'None') != 'None'
        )
        
        # Calculate overlap rates
        overlap_rate = overlap_count / fraud_count if fraud_count > 0 else 0.0
        reverse_overlap_rate = overlap_count / anomaly_count if anomaly_count > 0 else 0.0
        
        # Analyze correlation by type
        correlation_by_type = self._analyze_type_correlation()
        
        # Calculate correlation strength
        correlation_strength = self._calculate_correlation_strength(
            fraud_count, anomaly_count, overlap_count, len(transactions)
        )
        
        # Perform statistical significance test (chi-square)
        is_significant, chi_square, p_value = self._chi_square_test(
            fraud_count, anomaly_count, overlap_count, len(transactions)
        )
        
        result = CorrelationResult(
            total_transactions=len(transactions),
            fraud_count=fraud_count,
            anomaly_count=anomaly_count,
            overlap_count=overlap_count,
            overlap_rate=overlap_rate,
            reverse_overlap_rate=reverse_overlap_rate,
            correlation_by_type=correlation_by_type,
            correlation_strength=correlation_strength,
            statistical_significance=is_significant,
            chi_square_statistic=chi_square,
            p_value=p_value
        )
        
        self.results = result
        return result
    
    def _analyze_type_correlation(self) -> Dict[str, Dict[str, int]]:
        """
        Analyze which fraud types correlate with which anomaly types
        
        Returns:
            Dictionary mapping fraud_type -> anomaly_type -> count
        """
        correlation = defaultdict(lambda: defaultdict(int))
        
        for txn in self.transactions:
            fraud_type = txn.get('Fraud_Type', 'None')
            anomaly_type = txn.get('Anomaly_Type', 'None')
            
            if fraud_type != 'None' and anomaly_type != 'None':
                correlation[fraud_type][anomaly_type] += 1
        
        # Convert to regular dict
        return {ft: dict(at_dict) for ft, at_dict in correlation.items()}
    
    def _calculate_correlation_strength(self, fraud_count: int, anomaly_count: int,
                                       overlap_count: int, total: int) -> float:
        """
        Calculate phi coefficient for correlation strength
        
        2x2 contingency table:
                    | Anomaly  | No Anomaly |
        Fraud       | overlap  | fraud_only |
        No Fraud    | anom_only| normal     |
        
        Returns:
            Phi coefficient (0.0-1.0)
        """
        if total == 0:
            return 0.0
        
        fraud_only = fraud_count - overlap_count
        anomaly_only = anomaly_count - overlap_count
        normal = total - fraud_count - anomaly_only
        
        # Phi coefficient formula
        numerator = (overlap_count * normal) - (fraud_only * anomaly_only)
        denominator_parts = [
            (overlap_count + fraud_only),
            (overlap_count + anomaly_only),
            (fraud_only + normal),
            (anomaly_only + normal)
        ]
        
        denominator = 1.0
        for part in denominator_parts:
            denominator *= part
        
        if denominator == 0:
            return 0.0
        
        phi = abs(numerator) / (denominator ** 0.5)
        return min(phi, 1.0)  # Clamp to 0-1
    
    def _chi_square_test(self, fraud_count: int, anomaly_count: int,
                        overlap_count: int, total: int) -> Tuple[bool, float, float]:
        """
        Perform chi-square test for statistical significance
        
        H0: Fraud and anomaly are independent
        H1: Fraud and anomaly are correlated
        
        Returns:
            Tuple of (is_significant, chi_square_statistic, p_value)
        """
        if total == 0:
            return False, 0.0, 1.0
        
        # Observed frequencies
        fraud_only = fraud_count - overlap_count
        anomaly_only = anomaly_count - overlap_count
        normal = total - fraud_count - anomaly_only
        
        # Expected frequencies under independence
        expected_overlap = (fraud_count * anomaly_count) / total
        expected_fraud_only = (fraud_count * (total - anomaly_count)) / total
        expected_anomaly_only = ((total - fraud_count) * anomaly_count) / total
        expected_normal = ((total - fraud_count) * (total - anomaly_count)) / total
        
        # Chi-square statistic
        chi_square = 0.0
        observed = [overlap_count, fraud_only, anomaly_only, normal]
        expected = [expected_overlap, expected_fraud_only, expected_anomaly_only, expected_normal]
        
        for obs, exp in zip(observed, expected):
            if exp > 0:
                chi_square += ((obs - exp) ** 2) / exp
        
        # P-value approximation (degrees of freedom = 1 for 2x2 table)
        # Using simplified chi-square distribution approximation
        # For df=1, critical value at p=0.05 is 3.841
        p_value = 0.05 if chi_square > 3.841 else 0.1  # Simplified
        is_significant = chi_square > 3.841  # p < 0.05
        
        return is_significant, chi_square, p_value
    
    def get_high_correlation_pairs(self, threshold: float = 0.3) -> List[Tuple[str, str, int]]:
        """
        Get fraud-anomaly type pairs with high correlation
        
        Args:
            threshold: Minimum correlation threshold (default 0.3 = 30%)
            
        Returns:
            List of (fraud_type, anomaly_type, count) tuples
        """
        if not self.results:
            return []
        
        pairs = []
        for fraud_type, anomaly_dict in self.results.correlation_by_type.items():
            total_fraud = sum(
                1 for t in self.transactions 
                if t.get('Fraud_Type', 'None') == fraud_type
            )
            
            for anomaly_type, count in anomaly_dict.items():
                correlation_rate = count / total_fraud if total_fraud > 0 else 0.0
                if correlation_rate >= threshold:
                    pairs.append((fraud_type, anomaly_type, count))
        
        return sorted(pairs, key=lambda x: x[2], reverse=True)


class SeverityDistributionAnalyzer:
    """
    Analyze severity score distributions across anomaly types
    
    Validates severity ranges, creates histograms, identifies outliers.
    """
    
    def __init__(self):
        """Initialize the severity analyzer"""
        self.transactions: List[Dict[str, Any]] = []
        self.distributions: Dict[str, SeverityDistribution] = {}
    
    def analyze(self, transactions: List[Dict[str, Any]]) -> Dict[str, SeverityDistribution]:
        """
        Analyze severity distributions for each anomaly type
        
        Args:
            transactions: List of transaction dictionaries with anomaly labels
            
        Returns:
            Dictionary mapping anomaly_type -> SeverityDistribution
        """
        self.transactions = transactions
        
        # Group by anomaly type
        anomalies_by_type = defaultdict(list)
        for txn in transactions:
            anomaly_type = txn.get('Anomaly_Type', 'None')
            if anomaly_type != 'None':
                severity = txn.get('Anomaly_Severity', 0.0)
                anomalies_by_type[anomaly_type].append(severity)
        
        # Analyze each type
        distributions = {}
        for anomaly_type, severities in anomalies_by_type.items():
            dist = self._analyze_type_distribution(anomaly_type, severities)
            distributions[anomaly_type] = dist
        
        self.distributions = distributions
        return distributions
    
    def _analyze_type_distribution(self, anomaly_type: str, 
                                   severities: List[float]) -> SeverityDistribution:
        """
        Analyze distribution for single anomaly type
        
        Args:
            anomaly_type: Type of anomaly
            severities: List of severity scores
            
        Returns:
            SeverityDistribution object
        """
        if not severities:
            return SeverityDistribution(
                anomaly_type=anomaly_type,
                count=0,
                mean_severity=0.0,
                median_severity=0.0,
                std_deviation=0.0,
                min_severity=0.0,
                max_severity=0.0,
                low_severity_count=0,
                medium_severity_count=0,
                high_severity_count=0,
                critical_severity_count=0,
                severity_bins={}
            )
        
        # Basic statistics
        mean_sev = statistics.mean(severities)
        median_sev = statistics.median(severities)
        std_dev = statistics.stdev(severities) if len(severities) > 1 else 0.0
        min_sev = min(severities)
        max_sev = max(severities)
        
        # Count by severity level
        low = sum(1 for s in severities if 0.0 <= s < 0.3)
        medium = sum(1 for s in severities if 0.3 <= s < 0.6)
        high = sum(1 for s in severities if 0.6 <= s < 0.8)
        critical = sum(1 for s in severities if 0.8 <= s <= 1.0)
        
        # Create histogram bins (10 bins)
        bins = {f"{i/10:.1f}-{(i+1)/10:.1f}": 0 for i in range(10)}
        for sev in severities:
            bin_index = min(int(sev * 10), 9)  # Clamp to 0-9
            bin_key = f"{bin_index/10:.1f}-{(bin_index+1)/10:.1f}"
            bins[bin_key] += 1
        
        return SeverityDistribution(
            anomaly_type=anomaly_type,
            count=len(severities),
            mean_severity=mean_sev,
            median_severity=median_sev,
            std_deviation=std_dev,
            min_severity=min_sev,
            max_severity=max_sev,
            low_severity_count=low,
            medium_severity_count=medium,
            high_severity_count=high,
            critical_severity_count=critical,
            severity_bins=bins
        )
    
    def get_outliers(self, anomaly_type: str, iqr_multiplier: float = 1.5) -> List[float]:
        """
        Identify outlier severity scores using IQR method
        
        Args:
            anomaly_type: Type of anomaly to analyze
            iqr_multiplier: IQR multiplier for outlier detection (default 1.5)
            
        Returns:
            List of outlier severity scores
        """
        if anomaly_type not in self.distributions:
            return []
        
        # Get all severities for this type
        severities = [
            txn.get('Anomaly_Severity', 0.0)
            for txn in self.transactions
            if txn.get('Anomaly_Type', 'None') == anomaly_type
        ]
        
        if len(severities) < 4:  # Need at least 4 points for quartiles
            return []
        
        # Calculate quartiles
        sorted_sev = sorted(severities)
        n = len(sorted_sev)
        q1 = sorted_sev[n // 4]
        q3 = sorted_sev[3 * n // 4]
        iqr = q3 - q1
        
        # Outlier bounds
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        
        # Find outliers
        outliers = [s for s in severities if s < lower_bound or s > upper_bound]
        return outliers
    
    def validate_expected_ranges(self) -> Dict[str, bool]:
        """
        Validate that severity ranges match expected patterns for each anomaly type
        
        Returns:
            Dictionary mapping anomaly_type -> validation_passed
        """
        # Expected severity ranges for each anomaly type
        expected_ranges = {
            'BEHAVIORAL': (0.4, 0.8),  # Medium to high
            'GEOGRAPHIC': (0.5, 0.9),  # Medium-high to critical
            'TEMPORAL': (0.5, 0.8),    # Medium to high
            'AMOUNT': (0.4, 0.7)       # Medium to medium-high
        }
        
        validation = {}
        for anomaly_type, dist in self.distributions.items():
            if anomaly_type in expected_ranges:
                min_expected, max_expected = expected_ranges[anomaly_type]
                # Validate mean is within ±0.2 of expected range
                passes = (min_expected - 0.2 <= dist.mean_severity <= max_expected + 0.2)
                validation[anomaly_type] = passes
            else:
                validation[anomaly_type] = True  # Unknown type, assume valid
        
        return validation


class TemporalClusteringAnalyzer:
    """
    Detect temporal clustering of anomalies
    
    Identifies time periods with anomaly bursts, analyzes hourly and daily patterns.
    """
    
    def __init__(self, burst_threshold: float = 2.0):
        """
        Initialize the temporal clustering analyzer
        
        Args:
            burst_threshold: Multiplier above baseline to consider a burst (default 2.0)
        """
        self.transactions: List[Dict[str, Any]] = []
        self.burst_threshold = burst_threshold
        self.clusters: List[TemporalCluster] = []
    
    def analyze(self, transactions: List[Dict[str, Any]]) -> List[TemporalCluster]:
        """
        Analyze temporal clustering patterns
        
        Args:
            transactions: List of transaction dictionaries with anomaly labels
            
        Returns:
            List of TemporalCluster objects for significant time slots
        """
        self.transactions = transactions
        
        # Group anomalies by hour and day of week
        anomaly_counts = defaultdict(lambda: defaultdict(int))
        anomaly_types_by_slot = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        for txn in transactions:
            anomaly_type = txn.get('Anomaly_Type', 'None')
            if anomaly_type != 'None':
                # Parse hour and day of week from transaction
                hour = txn.get('Hour', 0)
                day_of_week = self._get_day_of_week(txn.get('Day_of_Week', 'Monday'))
                
                anomaly_counts[hour][day_of_week] += 1
                anomaly_types_by_slot[hour][day_of_week][anomaly_type] += 1
        
        # Calculate baseline (average anomalies per time slot)
        total_anomalies = sum(
            1 for t in transactions if t.get('Anomaly_Type', 'None') != 'None'
        )
        total_slots = 24 * 7  # 24 hours * 7 days
        baseline = total_anomalies / total_slots if total_slots > 0 else 0.0
        
        # Identify clusters
        clusters = []
        for hour in range(24):
            for day in range(7):
                count = anomaly_counts[hour][day]
                if count > 0:
                    cluster_size = count
                    burst_multiplier = cluster_size / baseline if baseline > 0 else 0.0
                    is_burst = burst_multiplier >= self.burst_threshold
                    
                    anomaly_types_dict = dict(anomaly_types_by_slot[hour][day])
                    
                    cluster = TemporalCluster(
                        hour=hour,
                        day_of_week=day,
                        anomaly_count=count,
                        cluster_size=cluster_size,
                        is_burst=is_burst,
                        burst_multiplier=burst_multiplier,
                        anomaly_types=anomaly_types_dict
                    )
                    clusters.append(cluster)
        
        # Sort by burst multiplier (highest first)
        clusters.sort(key=lambda c: c.burst_multiplier, reverse=True)
        
        self.clusters = clusters
        return clusters
    
    def _get_day_of_week(self, day_name: str) -> int:
        """
        Convert day name to integer (0=Monday, 6=Sunday)
        
        Args:
            day_name: Name of day
            
        Returns:
            Integer 0-6
        """
        days = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        return days.get(day_name, 0)
    
    def get_burst_periods(self) -> List[TemporalCluster]:
        """
        Get only time slots with significant bursts
        
        Returns:
            List of TemporalCluster objects where is_burst=True
        """
        return [c for c in self.clusters if c.is_burst]
    
    def get_hourly_distribution(self) -> Dict[int, int]:
        """
        Get anomaly count distribution by hour (0-23)
        
        Returns:
            Dictionary mapping hour -> anomaly_count
        """
        hourly = defaultdict(int)
        for cluster in self.clusters:
            hourly[cluster.hour] += cluster.anomaly_count
        return dict(hourly)
    
    def get_daily_distribution(self) -> Dict[int, int]:
        """
        Get anomaly count distribution by day of week (0=Mon, 6=Sun)
        
        Returns:
            Dictionary mapping day -> anomaly_count
        """
        daily = defaultdict(int)
        for cluster in self.clusters:
            daily[cluster.day_of_week] += cluster.anomaly_count
        return dict(daily)


class GeographicHeatmapAnalyzer:
    """
    Analyze geographic anomaly patterns and create transition heatmaps
    
    Identifies high-risk city-to-city routes, analyzes distance vs severity correlation.
    """
    
    def __init__(self):
        """Initialize the geographic heatmap analyzer"""
        self.transactions: List[Dict[str, Any]] = []
        self.heatmap_data: List[GeographicHeatmap] = []
    
    def analyze(self, transactions: List[Dict[str, Any]]) -> List[GeographicHeatmap]:
        """
        Analyze geographic transitions and build heatmap
        
        Args:
            transactions: List of transaction dictionaries with anomaly labels
            
        Returns:
            List of GeographicHeatmap objects for city-to-city transitions
        """
        self.transactions = transactions
        
        # Build customer transaction sequences
        customer_sequences = defaultdict(list)
        for txn in sorted(transactions, key=lambda t: t.get('Date', '') + str(t.get('Hour', 0))):
            customer_id = txn.get('Customer_ID', 'UNKNOWN')
            customer_sequences[customer_id].append(txn)
        
        # Analyze transitions
        transition_data = defaultdict(lambda: {
            'count': 0,
            'distances': [],
            'severities': [],
            'impossible_travel': 0,
            'total_transitions': 0
        })
        
        for customer_id, txns in customer_sequences.items():
            for i in range(1, len(txns)):
                prev_txn = txns[i-1]
                curr_txn = txns[i]
                
                from_city = prev_txn.get('City', 'Unknown')
                to_city = curr_txn.get('City', 'Unknown')
                
                if from_city != to_city:  # Only track actual transitions
                    key = (from_city, to_city)
                    transition_data[key]['total_transitions'] += 1
                    
                    # Check if anomaly on this transition
                    anomaly_type = curr_txn.get('Anomaly_Type', 'None')
                    if anomaly_type == 'GEOGRAPHIC':
                        transition_data[key]['count'] += 1
                        
                        # Track distance and severity
                        distance = curr_txn.get('Distance_From_Last_Txn_km', 0.0)
                        severity = curr_txn.get('Anomaly_Severity', 0.0)
                        
                        transition_data[key]['distances'].append(distance)
                        transition_data[key]['severities'].append(severity)
                        
                        # Check if impossible travel
                        anomaly_reason = curr_txn.get('Anomaly_Reason', '')
                        if 'impossible travel' in anomaly_reason.lower():
                            transition_data[key]['impossible_travel'] += 1
        
        # Build heatmap objects
        heatmap = []
        for (from_city, to_city), data in transition_data.items():
            if data['count'] > 0:  # Only include routes with anomalies
                avg_distance = statistics.mean(data['distances']) if data['distances'] else 0.0
                avg_severity = statistics.mean(data['severities']) if data['severities'] else 0.0
                transition_rate = data['count'] / data['total_transitions'] if data['total_transitions'] > 0 else 0.0
                
                heatmap.append(GeographicHeatmap(
                    from_city=from_city,
                    to_city=to_city,
                    anomaly_count=data['count'],
                    avg_distance_km=avg_distance,
                    avg_severity=avg_severity,
                    impossible_travel_count=data['impossible_travel'],
                    transition_rate=transition_rate
                ))
        
        # Sort by anomaly count (highest first)
        heatmap.sort(key=lambda h: h.anomaly_count, reverse=True)
        
        self.heatmap_data = heatmap
        return heatmap
    
    def get_high_risk_routes(self, min_anomalies: int = 3, 
                            min_severity: float = 0.7) -> List[GeographicHeatmap]:
        """
        Get high-risk geographic routes
        
        Args:
            min_anomalies: Minimum anomaly count threshold
            min_severity: Minimum average severity threshold
            
        Returns:
            List of high-risk GeographicHeatmap objects
        """
        return [
            h for h in self.heatmap_data
            if h.anomaly_count >= min_anomalies and h.avg_severity >= min_severity
        ]
    
    def get_transition_matrix(self, cities: List[str]) -> Dict[str, Dict[str, int]]:
        """
        Build transition matrix for specified cities
        
        Args:
            cities: List of city names to include
            
        Returns:
            Nested dictionary: from_city -> to_city -> anomaly_count
        """
        matrix = {city: {c: 0 for c in cities} for city in cities}
        
        for heatmap in self.heatmap_data:
            if heatmap.from_city in cities and heatmap.to_city in cities:
                matrix[heatmap.from_city][heatmap.to_city] = heatmap.anomaly_count
        
        return matrix
    
    def analyze_distance_severity_correlation(self) -> Tuple[float, int]:
        """
        Analyze correlation between distance and severity
        
        Returns:
            Tuple of (correlation_coefficient, sample_size)
        """
        if not self.heatmap_data:
            return 0.0, 0
        
        distances = [h.avg_distance_km for h in self.heatmap_data]
        severities = [h.avg_severity for h in self.heatmap_data]
        
        if len(distances) < 2:
            return 0.0, len(distances)
        
        # Calculate Pearson correlation coefficient
        mean_dist = statistics.mean(distances)
        mean_sev = statistics.mean(severities)
        
        numerator = sum((d - mean_dist) * (s - mean_sev) for d, s in zip(distances, severities))
        
        dist_var = sum((d - mean_dist) ** 2 for d in distances)
        sev_var = sum((s - mean_sev) ** 2 for s in severities)
        
        denominator = (dist_var * sev_var) ** 0.5
        
        if denominator == 0:
            return 0.0, len(distances)
        
        correlation = numerator / denominator
        return correlation, len(distances)
