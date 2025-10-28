"""
Tests for Anomaly Analysis Module

Comprehensive test suite for anomaly-fraud correlation, severity distribution,
temporal clustering, and geographic heatmap analysis.
"""

import pytest
from src.generators.anomaly_analysis import (
    AnomalyFraudCorrelationAnalyzer,
    SeverityDistributionAnalyzer,
    TemporalClusteringAnalyzer,
    GeographicHeatmapAnalyzer,
    CorrelationResult,
    SeverityDistribution,
    TemporalCluster,
    GeographicHeatmap
)


class TestAnomalyFraudCorrelation:
    """Test anomaly-fraud correlation analysis"""
    
    def test_correlation_basic_overlap(self):
        """Test basic fraud-anomaly overlap calculation"""
        transactions = [
            {'Fraud_Type': 'Card Cloning', 'Anomaly_Type': 'GEOGRAPHIC'},
            {'Fraud_Type': 'None', 'Anomaly_Type': 'BEHAVIORAL'},
            {'Fraud_Type': 'Account Takeover', 'Anomaly_Type': 'None'},
            {'Fraud_Type': 'None', 'Anomaly_Type': 'None'},
        ]
        
        analyzer = AnomalyFraudCorrelationAnalyzer()
        result = analyzer.analyze(transactions)
        
        assert result.total_transactions == 4
        assert result.fraud_count == 2
        assert result.anomaly_count == 2
        assert result.overlap_count == 1
        assert result.overlap_rate == 0.5  # 1/2 frauds are also anomalies
        assert result.reverse_overlap_rate == 0.5  # 1/2 anomalies are also fraud
    
    def test_correlation_no_overlap(self):
        """Test case with no fraud-anomaly overlap"""
        transactions = [
            {'Fraud_Type': 'Card Cloning', 'Anomaly_Type': 'None'},
            {'Fraud_Type': 'None', 'Anomaly_Type': 'BEHAVIORAL'},
        ]
        
        analyzer = AnomalyFraudCorrelationAnalyzer()
        result = analyzer.analyze(transactions)
        
        assert result.overlap_count == 0
        assert result.overlap_rate == 0.0
        assert result.reverse_overlap_rate == 0.0
    
    def test_correlation_type_analysis(self):
        """Test correlation by fraud and anomaly types"""
        transactions = [
            {'Fraud_Type': 'Card Cloning', 'Anomaly_Type': 'GEOGRAPHIC'},
            {'Fraud_Type': 'Card Cloning', 'Anomaly_Type': 'GEOGRAPHIC'},
            {'Fraud_Type': 'Account Takeover', 'Anomaly_Type': 'BEHAVIORAL'},
            {'Fraud_Type': 'Account Takeover', 'Anomaly_Type': 'AMOUNT'},
        ]
        
        analyzer = AnomalyFraudCorrelationAnalyzer()
        result = analyzer.analyze(transactions)
        
        assert 'Card Cloning' in result.correlation_by_type
        assert result.correlation_by_type['Card Cloning']['GEOGRAPHIC'] == 2
        assert 'Account Takeover' in result.correlation_by_type
        assert result.correlation_by_type['Account Takeover']['BEHAVIORAL'] == 1
        assert result.correlation_by_type['Account Takeover']['AMOUNT'] == 1
    
    def test_correlation_strength_calculation(self):
        """Test phi coefficient calculation"""
        # Perfect positive correlation
        transactions = [
            {'Fraud_Type': 'Fraud1', 'Anomaly_Type': 'GEOGRAPHIC'},
            {'Fraud_Type': 'Fraud2', 'Anomaly_Type': 'BEHAVIORAL'},
            {'Fraud_Type': 'None', 'Anomaly_Type': 'None'},
            {'Fraud_Type': 'None', 'Anomaly_Type': 'None'},
        ]
        
        analyzer = AnomalyFraudCorrelationAnalyzer()
        result = analyzer.analyze(transactions)
        
        # Should have non-zero correlation
        assert result.correlation_strength > 0.0
        assert 0.0 <= result.correlation_strength <= 1.0
    
    def test_chi_square_significance(self):
        """Test chi-square statistical significance test"""
        # Large overlap should be significant
        transactions = [
            {'Fraud_Type': 'Fraud', 'Anomaly_Type': 'Anomaly'} for _ in range(50)
        ] + [
            {'Fraud_Type': 'None', 'Anomaly_Type': 'None'} for _ in range(50)
        ]
        
        analyzer = AnomalyFraudCorrelationAnalyzer()
        result = analyzer.analyze(transactions)
        
        assert result.chi_square_statistic > 0
        assert 0.0 <= result.p_value <= 1.0
    
    def test_high_correlation_pairs(self):
        """Test identification of high correlation pairs"""
        transactions = [
            {'Fraud_Type': 'Card Cloning', 'Anomaly_Type': 'GEOGRAPHIC'},
            {'Fraud_Type': 'Card Cloning', 'Anomaly_Type': 'GEOGRAPHIC'},
            {'Fraud_Type': 'Card Cloning', 'Anomaly_Type': 'GEOGRAPHIC'},
            {'Fraud_Type': 'Card Cloning', 'Anomaly_Type': 'None'},
        ]
        
        analyzer = AnomalyFraudCorrelationAnalyzer()
        analyzer.analyze(transactions)
        
        pairs = analyzer.get_high_correlation_pairs(threshold=0.5)
        
        assert len(pairs) > 0
        assert pairs[0][0] == 'Card Cloning'
        assert pairs[0][1] == 'GEOGRAPHIC'
        assert pairs[0][2] == 3


class TestSeverityDistribution:
    """Test severity distribution analysis"""
    
    def test_basic_distribution(self):
        """Test basic severity distribution calculation"""
        transactions = [
            {'Anomaly_Type': 'BEHAVIORAL', 'Anomaly_Severity': 0.5},
            {'Anomaly_Type': 'BEHAVIORAL', 'Anomaly_Severity': 0.6},
            {'Anomaly_Type': 'BEHAVIORAL', 'Anomaly_Severity': 0.7},
        ]
        
        analyzer = SeverityDistributionAnalyzer()
        distributions = analyzer.analyze(transactions)
        
        assert 'BEHAVIORAL' in distributions
        dist = distributions['BEHAVIORAL']
        assert dist.count == 3
        assert abs(dist.mean_severity - 0.6) < 0.01
        assert dist.min_severity == 0.5
        assert dist.max_severity == 0.7
    
    def test_severity_level_counts(self):
        """Test counting by severity levels"""
        transactions = [
            {'Anomaly_Type': 'GEOGRAPHIC', 'Anomaly_Severity': 0.2},  # Low
            {'Anomaly_Type': 'GEOGRAPHIC', 'Anomaly_Severity': 0.4},  # Medium
            {'Anomaly_Type': 'GEOGRAPHIC', 'Anomaly_Severity': 0.7},  # High
            {'Anomaly_Type': 'GEOGRAPHIC', 'Anomaly_Severity': 0.9},  # Critical
        ]
        
        analyzer = SeverityDistributionAnalyzer()
        distributions = analyzer.analyze(transactions)
        
        dist = distributions['GEOGRAPHIC']
        assert dist.low_severity_count == 1
        assert dist.medium_severity_count == 1
        assert dist.high_severity_count == 1
        assert dist.critical_severity_count == 1
    
    def test_histogram_bins(self):
        """Test severity histogram binning"""
        transactions = [
            {'Anomaly_Type': 'TEMPORAL', 'Anomaly_Severity': 0.15},
            {'Anomaly_Type': 'TEMPORAL', 'Anomaly_Severity': 0.25},
            {'Anomaly_Type': 'TEMPORAL', 'Anomaly_Severity': 0.55},
        ]
        
        analyzer = SeverityDistributionAnalyzer()
        distributions = analyzer.analyze(transactions)
        
        dist = distributions['TEMPORAL']
        assert '0.1-0.2' in dist.severity_bins
        assert '0.2-0.3' in dist.severity_bins
        assert '0.5-0.6' in dist.severity_bins
        assert dist.severity_bins['0.1-0.2'] == 1
        assert dist.severity_bins['0.2-0.3'] == 1
        assert dist.severity_bins['0.5-0.6'] == 1
    
    def test_empty_dataset(self):
        """Test handling of empty dataset"""
        transactions = []
        
        analyzer = SeverityDistributionAnalyzer()
        distributions = analyzer.analyze(transactions)
        
        assert len(distributions) == 0
    
    def test_outlier_detection(self):
        """Test IQR-based outlier detection"""
        transactions = [
            {'Anomaly_Type': 'AMOUNT', 'Anomaly_Severity': 0.5},
            {'Anomaly_Type': 'AMOUNT', 'Anomaly_Severity': 0.5},
            {'Anomaly_Type': 'AMOUNT', 'Anomaly_Severity': 0.5},
            {'Anomaly_Type': 'AMOUNT', 'Anomaly_Severity': 0.5},
            {'Anomaly_Type': 'AMOUNT', 'Anomaly_Severity': 0.95},  # Outlier
        ]
        
        analyzer = SeverityDistributionAnalyzer()
        analyzer.analyze(transactions)
        
        outliers = analyzer.get_outliers('AMOUNT', iqr_multiplier=1.5)
        
        # Should detect the 0.95 as outlier
        assert len(outliers) > 0
        assert 0.95 in outliers
    
    def test_expected_range_validation(self):
        """Test validation of expected severity ranges"""
        transactions = [
            {'Anomaly_Type': 'BEHAVIORAL', 'Anomaly_Severity': 0.6},
            {'Anomaly_Type': 'BEHAVIORAL', 'Anomaly_Severity': 0.6},
            {'Anomaly_Type': 'GEOGRAPHIC', 'Anomaly_Severity': 0.7},
            {'Anomaly_Type': 'GEOGRAPHIC', 'Anomaly_Severity': 0.7},
        ]
        
        analyzer = SeverityDistributionAnalyzer()
        analyzer.analyze(transactions)
        
        validation = analyzer.validate_expected_ranges()
        
        assert 'BEHAVIORAL' in validation
        assert 'GEOGRAPHIC' in validation
        # Both should pass validation (within expected ranges)
        assert validation['BEHAVIORAL'] is True
        assert validation['GEOGRAPHIC'] is True


class TestTemporalClustering:
    """Test temporal clustering analysis"""
    
    def test_basic_clustering(self):
        """Test basic temporal cluster detection"""
        transactions = [
            {'Anomaly_Type': 'BEHAVIORAL', 'Hour': 14, 'Day_of_Week': 'Monday'},
            {'Anomaly_Type': 'GEOGRAPHIC', 'Hour': 14, 'Day_of_Week': 'Monday'},
            {'Anomaly_Type': 'None', 'Hour': 15, 'Day_of_Week': 'Monday'},
        ]
        
        analyzer = TemporalClusteringAnalyzer(burst_threshold=1.5)
        clusters = analyzer.analyze(transactions)
        
        # Should find cluster at hour 14, Monday
        assert len(clusters) > 0
        cluster = clusters[0]
        assert cluster.hour == 14
        assert cluster.day_of_week == 0  # Monday
        assert cluster.anomaly_count == 2
    
    def test_burst_detection(self):
        """Test burst detection threshold"""
        transactions = [
            # Create burst at hour 10
            {'Anomaly_Type': 'BEHAVIORAL', 'Hour': 10, 'Day_of_Week': 'Tuesday'} for _ in range(10)
        ] + [
            # Baseline elsewhere
            {'Anomaly_Type': 'TEMPORAL', 'Hour': i, 'Day_of_Week': 'Wednesday'} for i in range(24)
        ]
        
        analyzer = TemporalClusteringAnalyzer(burst_threshold=2.0)
        clusters = analyzer.analyze(transactions)
        
        burst_clusters = analyzer.get_burst_periods()
        
        assert len(burst_clusters) > 0
        # Hour 10 on Tuesday should be a burst
        hour10_cluster = next((c for c in burst_clusters if c.hour == 10 and c.day_of_week == 1), None)
        assert hour10_cluster is not None
        assert hour10_cluster.is_burst is True
        assert hour10_cluster.burst_multiplier >= 2.0
    
    def test_hourly_distribution(self):
        """Test hourly anomaly distribution"""
        transactions = [
            {'Anomaly_Type': 'BEHAVIORAL', 'Hour': 14, 'Day_of_Week': 'Monday'},
            {'Anomaly_Type': 'BEHAVIORAL', 'Hour': 14, 'Day_of_Week': 'Tuesday'},
            {'Anomaly_Type': 'GEOGRAPHIC', 'Hour': 15, 'Day_of_Week': 'Monday'},
        ]
        
        analyzer = TemporalClusteringAnalyzer()
        analyzer.analyze(transactions)
        
        hourly = analyzer.get_hourly_distribution()
        
        assert 14 in hourly
        assert hourly[14] == 2
        assert 15 in hourly
        assert hourly[15] == 1
    
    def test_daily_distribution(self):
        """Test daily anomaly distribution"""
        transactions = [
            {'Anomaly_Type': 'BEHAVIORAL', 'Hour': 14, 'Day_of_Week': 'Monday'},
            {'Anomaly_Type': 'BEHAVIORAL', 'Hour': 15, 'Day_of_Week': 'Monday'},
            {'Anomaly_Type': 'GEOGRAPHIC', 'Hour': 14, 'Day_of_Week': 'Tuesday'},
        ]
        
        analyzer = TemporalClusteringAnalyzer()
        analyzer.analyze(transactions)
        
        daily = analyzer.get_daily_distribution()
        
        assert 0 in daily  # Monday
        assert daily[0] == 2
        assert 1 in daily  # Tuesday
        assert daily[1] == 1


class TestGeographicHeatmap:
    """Test geographic heatmap analysis"""
    
    def test_basic_transition_detection(self):
        """Test basic city-to-city transition detection"""
        transactions = [
            {
                'Customer_ID': 'CUST001',
                'Date': '2025-01-01',
                'Hour': 10,
                'City': 'Mumbai',
                'Anomaly_Type': 'None'
            },
            {
                'Customer_ID': 'CUST001',
                'Date': '2025-01-01',
                'Hour': 14,
                'City': 'Delhi',
                'Anomaly_Type': 'GEOGRAPHIC',
                'Distance_From_Last_Txn_km': 1400.0,
                'Anomaly_Severity': 0.8,
                'Anomaly_Reason': 'Impossible travel: Mumbai to Delhi'
            }
        ]
        
        analyzer = GeographicHeatmapAnalyzer()
        heatmap = analyzer.analyze(transactions)
        
        assert len(heatmap) == 1
        assert heatmap[0].from_city == 'Mumbai'
        assert heatmap[0].to_city == 'Delhi'
        assert heatmap[0].anomaly_count == 1
        assert heatmap[0].avg_distance_km == 1400.0
        assert heatmap[0].avg_severity == 0.8
    
    def test_multiple_transitions_same_route(self):
        """Test aggregation of multiple transitions on same route"""
        transactions = [
            {
                'Customer_ID': 'CUST001',
                'Date': '2025-01-01',
                'Hour': 10,
                'City': 'Mumbai',
                'Anomaly_Type': 'None'
            },
            {
                'Customer_ID': 'CUST001',
                'Date': '2025-01-01',
                'Hour': 11,
                'City': 'Pune',
                'Anomaly_Type': 'GEOGRAPHIC',
                'Distance_From_Last_Txn_km': 150.0,
                'Anomaly_Severity': 0.6,
                'Anomaly_Reason': 'Unusual location'
            },
            {
                'Customer_ID': 'CUST002',
                'Date': '2025-01-01',
                'Hour': 12,
                'City': 'Mumbai',
                'Anomaly_Type': 'None'
            },
            {
                'Customer_ID': 'CUST002',
                'Date': '2025-01-01',
                'Hour': 13,
                'City': 'Pune',
                'Anomaly_Type': 'GEOGRAPHIC',
                'Distance_From_Last_Txn_km': 150.0,
                'Anomaly_Severity': 0.7,
                'Anomaly_Reason': 'Unusual location'
            }
        ]
        
        analyzer = GeographicHeatmapAnalyzer()
        heatmap = analyzer.analyze(transactions)
        
        assert len(heatmap) == 1
        route = heatmap[0]
        assert route.from_city == 'Mumbai'
        assert route.to_city == 'Pune'
        assert route.anomaly_count == 2
        assert route.avg_distance_km == 150.0
        assert abs(route.avg_severity - 0.65) < 0.01  # Average of 0.6 and 0.7
    
    def test_high_risk_routes(self):
        """Test identification of high-risk routes"""
        transactions = [
            {
                'Customer_ID': 'CUST001',
                'Date': '2025-01-01',
                'Hour': i,
                'City': 'Mumbai' if i % 2 == 0 else 'Delhi',
                'Anomaly_Type': 'GEOGRAPHIC' if i % 2 == 1 else 'None',
                'Distance_From_Last_Txn_km': 1400.0 if i % 2 == 1 else 0.0,
                'Anomaly_Severity': 0.8 if i % 2 == 1 else 0.0,
                'Anomaly_Reason': 'Impossible travel' if i % 2 == 1 else ''
            }
            for i in range(10)
        ]
        
        analyzer = GeographicHeatmapAnalyzer()
        analyzer.analyze(transactions)
        
        high_risk = analyzer.get_high_risk_routes(min_anomalies=3, min_severity=0.7)
        
        assert len(high_risk) > 0
        route = high_risk[0]
        assert route.anomaly_count >= 3
        assert route.avg_severity >= 0.7
    
    def test_transition_matrix(self):
        """Test transition matrix generation"""
        transactions = [
            {
                'Customer_ID': 'CUST001',
                'Date': '2025-01-01',
                'Hour': 10,
                'City': 'Mumbai',
                'Anomaly_Type': 'None'
            },
            {
                'Customer_ID': 'CUST001',
                'Date': '2025-01-01',
                'Hour': 11,
                'City': 'Delhi',
                'Anomaly_Type': 'GEOGRAPHIC',
                'Distance_From_Last_Txn_km': 1400.0,
                'Anomaly_Severity': 0.8,
                'Anomaly_Reason': 'Impossible travel'
            },
            {
                'Customer_ID': 'CUST001',
                'Date': '2025-01-01',
                'Hour': 12,
                'City': 'Bangalore',
                'Anomaly_Type': 'GEOGRAPHIC',
                'Distance_From_Last_Txn_km': 1700.0,
                'Anomaly_Severity': 0.85,
                'Anomaly_Reason': 'Impossible travel'
            }
        ]
        
        analyzer = GeographicHeatmapAnalyzer()
        analyzer.analyze(transactions)
        
        cities = ['Mumbai', 'Delhi', 'Bangalore']
        matrix = analyzer.get_transition_matrix(cities)
        
        assert 'Mumbai' in matrix
        assert 'Delhi' in matrix
        assert 'Bangalore' in matrix
        assert matrix['Mumbai']['Delhi'] == 1
        assert matrix['Delhi']['Bangalore'] == 1
        assert matrix['Mumbai']['Bangalore'] == 0  # No direct transition
    
    def test_distance_severity_correlation(self):
        """Test distance-severity correlation analysis"""
        transactions = [
            {
                'Customer_ID': f'CUST{i:03d}',
                'Date': '2025-01-01',
                'Hour': 10,
                'City': 'Mumbai',
                'Anomaly_Type': 'None'
            }
            for i in range(5)
        ] + [
            {
                'Customer_ID': f'CUST{i:03d}',
                'Date': '2025-01-01',
                'Hour': 11,
                'City': 'Delhi',
                'Anomaly_Type': 'GEOGRAPHIC',
                'Distance_From_Last_Txn_km': 1000.0 + i * 100,
                'Anomaly_Severity': 0.5 + i * 0.05,
                'Anomaly_Reason': 'Travel'
            }
            for i in range(5)
        ]
        
        analyzer = GeographicHeatmapAnalyzer()
        analyzer.analyze(transactions)
        
        correlation, sample_size = analyzer.analyze_distance_severity_correlation()
        
        # Should have positive correlation (longer distance = higher severity)
        assert sample_size > 0
        assert -1.0 <= correlation <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
