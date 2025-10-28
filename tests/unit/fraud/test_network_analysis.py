"""Tests for fraud network analysis module."""

import pytest
from datetime import datetime, timedelta
from src.generators.fraud_network import (
    FraudRing,
    TemporalCluster,
    FraudNetworkAnalyzer,
)


class TestFraudRing:
    """Test FraudRing class."""
    
    def test_fraud_ring_creation(self):
        ring = FraudRing('RING-001', 'merchant')
        assert ring.ring_id == 'RING-001'
        assert ring.ring_type == 'merchant'
        assert len(ring.customer_ids) == 0
        assert ring.confidence == 0.0
    
    def test_add_transaction_to_ring(self):
        ring = FraudRing('RING-001', 'merchant')
        txn = {
            'Transaction_ID': 'TXN-001',
            'Customer_ID': 'CUST-001',
            'Amount': 1000,
            'Date': '2025-10-26',
        }
        ring.add_transaction(txn, 'CUST-001')
        
        assert len(ring.customer_ids) == 1
        assert len(ring.transaction_ids) == 1
        assert ring.total_amount == 1000
        assert ring.first_seen == '2025-10-26'
        assert ring.last_seen == '2025-10-26'
    
    def test_multiple_customers_in_ring(self):
        ring = FraudRing('RING-001', 'merchant')
        for i in range(5):
            txn = {
                'Transaction_ID': f'TXN-{i}',
                'Customer_ID': f'CUST-{i}',
                'Amount': 500,
                'Date': '2025-10-26',
            }
            ring.add_transaction(txn, f'CUST-{i}')
        
        assert len(ring.customer_ids) == 5
        assert len(ring.transaction_ids) == 5
        assert ring.total_amount == 2500
    
    def test_ring_to_dict(self):
        ring = FraudRing('RING-001', 'location')
        ring.confidence = 0.85
        txn = {
            'Transaction_ID': 'TXN-001',
            'Customer_ID': 'CUST-001',
            'Amount': 1000,
            'Date': '2025-10-26',
        }
        ring.add_transaction(txn, 'CUST-001')
        
        ring_dict = ring.to_dict()
        assert ring_dict['ring_id'] == 'RING-001'
        assert ring_dict['ring_type'] == 'location'
        assert ring_dict['customer_count'] == 1
        assert ring_dict['transaction_count'] == 1
        assert ring_dict['confidence'] == 0.85
        assert ring_dict['total_amount'] == 1000
        assert ring_dict['avg_amount_per_transaction'] == 1000


class TestTemporalCluster:
    """Test TemporalCluster class."""
    
    def test_temporal_cluster_creation(self):
        cluster = TemporalCluster('CLUSTER-001', time_window_minutes=30)
        assert cluster.cluster_id == 'CLUSTER-001'
        assert cluster.time_window_minutes == 30
        assert len(cluster.transactions) == 0
    
    def test_add_transaction_to_cluster(self):
        cluster = TemporalCluster('CLUSTER-001')
        txn = {
            'Transaction_ID': 'TXN-001',
            'Customer_ID': 'CUST-001',
            'Merchant_ID': 'MRCH-001',
            'City': 'Mumbai',
            'Date': '2025-10-26',
            'Time': '12:00:00',
        }
        cluster.add_transaction(txn, 'CUST-001')
        
        assert len(cluster.transactions) == 1
        assert len(cluster.customer_ids) == 1
        assert len(cluster.merchants) == 1
        assert len(cluster.locations) == 1
    
    def test_cluster_is_suspicious_multiple_customers(self):
        cluster = TemporalCluster('CLUSTER-001')
        for i in range(4):
            txn = {
                'Transaction_ID': f'TXN-{i}',
                'Customer_ID': f'CUST-{i}',
                'Merchant_ID': 'MRCH-001',
                'City': 'Mumbai',
                'Date': '2025-10-26',
                'Time': '12:00:00',
            }
            cluster.add_transaction(txn, f'CUST-{i}')
        
        assert cluster.is_suspicious() is True
    
    def test_cluster_is_suspicious_high_volume(self):
        cluster = TemporalCluster('CLUSTER-001')
        for i in range(12):
            txn = {
                'Transaction_ID': f'TXN-{i}',
                'Customer_ID': f'CUST-{i}',
                'Merchant_ID': f'MRCH-{i}',
                'City': f'City-{i}',
                'Date': '2025-10-26',
                'Time': '12:00:00',
            }
            cluster.add_transaction(txn, f'CUST-{i}')
        
        assert cluster.is_suspicious() is True
    
    def test_cluster_not_suspicious(self):
        cluster = TemporalCluster('CLUSTER-001')
        txn = {
            'Transaction_ID': 'TXN-001',
            'Customer_ID': 'CUST-001',
            'Merchant_ID': 'MRCH-001',
            'City': 'Mumbai',
            'Date': '2025-10-26',
            'Time': '12:00:00',
        }
        cluster.add_transaction(txn, 'CUST-001')
        
        assert cluster.is_suspicious() is False
    
    def test_cluster_to_dict(self):
        cluster = TemporalCluster('CLUSTER-001', time_window_minutes=45)
        for i in range(3):
            txn = {
                'Transaction_ID': f'TXN-{i}',
                'Customer_ID': f'CUST-{i}',
                'Merchant_ID': 'MRCH-001',
                'City': 'Mumbai',
                'Date': '2025-10-26',
                'Time': '12:00:00',
            }
            cluster.add_transaction(txn, f'CUST-{i}')
        
        cluster_dict = cluster.to_dict()
        assert cluster_dict['cluster_id'] == 'CLUSTER-001'
        assert cluster_dict['customer_count'] == 3
        assert cluster_dict['transaction_count'] == 3
        assert cluster_dict['time_window_minutes'] == 45


class TestFraudNetworkAnalyzer:
    """Test FraudNetworkAnalyzer class."""
    
    def create_fraud_transactions(self, count=10, merchant_id='MRCH-001'):
        """Helper to create fraud transactions."""
        transactions = []
        for i in range(count):
            transactions.append({
                'Transaction_ID': f'TXN-{i}',
                'Customer_ID': f'CUST-{i}',
                'Merchant_ID': merchant_id,
                'City': 'Mumbai',
                'Amount': 1000 + (i * 100),
                'Is_Fraud': 1,
                'Date': '2025-10-26',
                'Time': f'12:{i:02d}:00',
                'Device_Type': 'Mobile',
                'Channel': 'Online',
            })
        return transactions
    
    def test_analyzer_initialization(self):
        analyzer = FraudNetworkAnalyzer(seed=42)
        assert analyzer.seed == 42
        assert len(analyzer.detected_rings) == 0
        assert len(analyzer.temporal_clusters) == 0
    
    def test_analyze_merchant_networks(self):
        analyzer = FraudNetworkAnalyzer(seed=42)
        transactions = self.create_fraud_transactions(count=6, merchant_id='MRCH-FRAUD-001')
        customer_map = {}
        
        rings = analyzer.analyze_merchant_networks(transactions, customer_map, min_customers=3, min_transactions=5)
        
        assert len(rings) >= 1
        assert rings[0].ring_type == 'merchant'
        assert len(rings[0].customer_ids) >= 3
        assert rings[0].confidence > 0.0
    
    def test_merchant_network_below_threshold(self):
        analyzer = FraudNetworkAnalyzer(seed=42)
        transactions = self.create_fraud_transactions(count=2, merchant_id='MRCH-SAFE-001')
        customer_map = {}
        
        rings = analyzer.analyze_merchant_networks(transactions, customer_map, min_customers=5, min_transactions=5)
        
        assert len(rings) == 0
    
    def test_analyze_location_networks(self):
        analyzer = FraudNetworkAnalyzer(seed=42)
        transactions = []
        for i in range(5):
            transactions.append({
                'Transaction_ID': f'TXN-{i}',
                'Customer_ID': f'CUST-{i}',
                'Merchant_ID': f'MRCH-{i}',
                'City': 'High_Crime_Zone',
                'Amount': 1000,
                'Is_Fraud': 1,
                'Date': '2025-10-26',
                'Time': '12:00:00',
            })
        customer_map = {}
        
        rings = analyzer.analyze_location_networks(transactions, customer_map, min_customers=4)
        
        assert len(rings) >= 1
        assert rings[0].ring_type == 'location'
        assert rings[0].shared_attributes['is_suspicious_area'] is True
    
    def test_analyze_device_networks(self):
        analyzer = FraudNetworkAnalyzer(seed=42)
        transactions = []
        for i in range(7):
            transactions.append({
                'Transaction_ID': f'TXN-{i}',
                'Customer_ID': f'CUST-{i}',
                'Merchant_ID': f'MRCH-{i}',
                'City': f'City-{i}',
                'Device_Type': 'Suspicious_Device',
                'Channel': 'Online',
                'Amount': 1000,
                'Is_Fraud': 1,
                'Date': '2025-10-26',
                'Time': '12:00:00',
            })
        customer_map = {}
        
        rings = analyzer.analyze_device_networks(transactions, customer_map, min_customers=3)
        
        assert len(rings) >= 1
        assert rings[0].ring_type == 'device'
        assert 'device_signature' in rings[0].shared_attributes
    
    def test_detect_temporal_clusters(self):
        analyzer = FraudNetworkAnalyzer(seed=42)
        transactions = []
        for i in range(8):
            transactions.append({
                'Transaction_ID': f'TXN-{i}',
                'Customer_ID': f'CUST-{i}',
                'Merchant_ID': 'MRCH-001',
                'City': 'Mumbai',
                'Amount': 1000,
                'Is_Fraud': 1,
                'Date': '2025-10-26',
                'Time': f'12:{i:02d}:00',
            })
        
        clusters = analyzer.detect_temporal_clusters(transactions, time_window_minutes=30, min_transactions=5)
        
        assert len(clusters) >= 1
        assert len(clusters[0].transactions) >= 5
    
    def test_temporal_clusters_separate_windows(self):
        analyzer = FraudNetworkAnalyzer(seed=42)
        transactions = []
        
        # First cluster at 12:00
        for i in range(6):
            transactions.append({
                'Transaction_ID': f'TXN-A-{i}',
                'Customer_ID': f'CUST-A-{i}',
                'Merchant_ID': 'MRCH-001',
                'City': 'Mumbai',
                'Amount': 1000,
                'Is_Fraud': 1,
                'Date': '2025-10-26',
                'Time': f'12:{i:02d}:00',
            })
        
        # Second cluster at 14:00 (2 hours later, beyond 30 min window)
        for i in range(6):
            transactions.append({
                'Transaction_ID': f'TXN-B-{i}',
                'Customer_ID': f'CUST-B-{i}',
                'Merchant_ID': 'MRCH-002',
                'City': 'Delhi',
                'Amount': 2000,
                'Is_Fraud': 1,
                'Date': '2025-10-26',
                'Time': f'14:{i:02d}:00',
            })
        
        clusters = analyzer.detect_temporal_clusters(transactions, time_window_minutes=30, min_transactions=5)
        
        assert len(clusters) == 2
    
    def test_generate_network_graph(self):
        analyzer = FraudNetworkAnalyzer(seed=42)
        transactions = self.create_fraud_transactions(count=6, merchant_id='MRCH-001')
        
        analyzer.analyze_merchant_networks(transactions, {}, min_customers=3, min_transactions=5)
        analyzer.detect_temporal_clusters(transactions, time_window_minutes=30, min_transactions=5)
        
        graph = analyzer.generate_network_graph()
        
        assert 'nodes' in graph
        assert 'edges' in graph
        assert 'metadata' in graph
        assert len(graph['nodes']) > 0
        assert len(graph['edges']) > 0
    
    def test_get_network_statistics(self):
        analyzer = FraudNetworkAnalyzer(seed=42)
        transactions = self.create_fraud_transactions(count=8, merchant_id='MRCH-001')
        
        analyzer.analyze_merchant_networks(transactions, {}, min_customers=3, min_transactions=5)
        analyzer.detect_temporal_clusters(transactions, time_window_minutes=30, min_transactions=5)
        
        stats = analyzer.get_network_statistics()
        
        assert 'total_fraud_rings' in stats
        assert 'ring_types' in stats
        assert 'unique_customers_in_rings' in stats
        assert 'temporal_clusters' in stats
        assert stats['total_fraud_rings'] > 0
    
    def test_reset_analysis(self):
        analyzer = FraudNetworkAnalyzer(seed=42)
        transactions = self.create_fraud_transactions(count=6, merchant_id='MRCH-001')
        
        analyzer.analyze_merchant_networks(transactions, {}, min_customers=3, min_transactions=5)
        assert len(analyzer.detected_rings) > 0
        
        analyzer.reset_analysis()
        assert len(analyzer.detected_rings) == 0
        assert len(analyzer.temporal_clusters) == 0
    
    def test_multiple_ring_types(self):
        analyzer = FraudNetworkAnalyzer(seed=42)
        transactions = []
        
        # Create transactions for merchant ring
        for i in range(5):
            transactions.append({
                'Transaction_ID': f'TXN-M-{i}',
                'Customer_ID': f'CUST-M-{i}',
                'Merchant_ID': 'MRCH-FRAUD',
                'City': f'City-{i}',
                'Device_Type': 'Mobile',
                'Channel': 'Online',
                'Amount': 1000,
                'Is_Fraud': 1,
                'Date': '2025-10-26',
                'Time': '12:00:00',
            })
        
        # Create transactions for location ring
        for i in range(5):
            transactions.append({
                'Transaction_ID': f'TXN-L-{i}',
                'Customer_ID': f'CUST-L-{i}',
                'Merchant_ID': f'MRCH-{i}',
                'City': 'High_Crime_Zone',
                'Device_Type': f'Device-{i}',
                'Channel': 'POS',
                'Amount': 2000,
                'Is_Fraud': 1,
                'Date': '2025-10-26',
                'Time': '13:00:00',
            })
        
        analyzer.analyze_merchant_networks(transactions, {}, min_customers=3, min_transactions=5)
        analyzer.analyze_location_networks(transactions, {}, min_customers=4)
        
        stats = analyzer.get_network_statistics()
        assert stats['total_fraud_rings'] >= 2
        assert len(stats['ring_types']) >= 1
    
    def test_ring_confidence_calculation(self):
        analyzer = FraudNetworkAnalyzer(seed=42)
        transactions = self.create_fraud_transactions(count=10, merchant_id='MRCH-HIGH-CONF')
        
        rings = analyzer.analyze_merchant_networks(transactions, {}, min_customers=3, min_transactions=5)
        
        assert len(rings) > 0
        assert 0.0 <= rings[0].confidence <= 1.0
        # More customers should lead to higher confidence
        if len(rings[0].customer_ids) >= 8:
            assert rings[0].confidence >= 0.6
