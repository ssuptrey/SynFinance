"""
Fraud Network Analysis Module

This module provides tools for detecting and analyzing fraud networks, including:
- Fraud ring detection (shared merchants, locations, devices)
- Temporal clustering of coordinated attacks
- Network graph generation for visualization
- Cross-customer fraud pattern analysis
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import random


class FraudRing:
    """Represents a detected fraud ring with shared characteristics."""
    
    def __init__(self, ring_id: str, ring_type: str):
        self.ring_id = ring_id
        self.ring_type = ring_type  # 'merchant', 'location', 'device', 'temporal'
        self.customer_ids: Set[str] = set()
        self.transaction_ids: Set[str] = set()
        self.shared_attributes: Dict[str, Any] = {}
        self.confidence: float = 0.0
        self.first_seen: Optional[str] = None
        self.last_seen: Optional[str] = None
        self.total_amount: float = 0.0
        
    def add_transaction(self, transaction: Dict, customer_id: str):
        """Add a transaction to this fraud ring."""
        self.customer_ids.add(customer_id)
        self.transaction_ids.add(transaction['Transaction_ID'])
        self.total_amount += transaction.get('Amount', 0)
        
        txn_date = transaction.get('Date', '')
        if not self.first_seen or txn_date < self.first_seen:
            self.first_seen = txn_date
        if not self.last_seen or txn_date > self.last_seen:
            self.last_seen = txn_date
    
    def to_dict(self) -> Dict:
        """Convert fraud ring to dictionary representation."""
        return {
            'ring_id': self.ring_id,
            'ring_type': self.ring_type,
            'customer_count': len(self.customer_ids),
            'transaction_count': len(self.transaction_ids),
            'shared_attributes': self.shared_attributes,
            'confidence': self.confidence,
            'first_seen': self.first_seen,
            'last_seen': self.last_seen,
            'total_amount': self.total_amount,
            'avg_amount_per_transaction': self.total_amount / len(self.transaction_ids) if self.transaction_ids else 0,
        }


class TemporalCluster:
    """Represents a temporal cluster of potentially coordinated transactions."""
    
    def __init__(self, cluster_id: str, time_window_minutes: int = 30):
        self.cluster_id = cluster_id
        self.time_window_minutes = time_window_minutes
        self.transactions: List[Dict] = []
        self.customer_ids: Set[str] = set()
        self.merchants: Set[str] = set()
        self.locations: Set[str] = set()
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
    def add_transaction(self, transaction: Dict, customer_id: str):
        """Add a transaction to this temporal cluster."""
        self.transactions.append(transaction)
        self.customer_ids.add(customer_id)
        self.merchants.add(transaction.get('Merchant_ID', 'Unknown'))
        self.locations.add(transaction.get('City', 'Unknown'))
        
        txn_datetime = self._parse_datetime(transaction)
        if txn_datetime:
            if not self.start_time or txn_datetime < self.start_time:
                self.start_time = txn_datetime
            if not self.end_time or txn_datetime > self.end_time:
                self.end_time = txn_datetime
    
    def _parse_datetime(self, transaction: Dict) -> Optional[datetime]:
        """Parse transaction datetime from date and time fields."""
        try:
            date_str = transaction.get('Date', '')
            time_str = transaction.get('Time', '00:00:00')
            datetime_str = f"{date_str} {time_str}"
            return datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        except:
            return None
    
    def is_suspicious(self) -> bool:
        """Determine if this cluster shows suspicious coordination."""
        # Multiple customers, same merchant, tight time window
        if len(self.customer_ids) >= 3 and len(self.merchants) <= 2:
            return True
        # Multiple customers, same location, tight time window
        if len(self.customer_ids) >= 4 and len(self.locations) == 1:
            return True
        # High transaction volume in short time
        if len(self.transactions) >= 10:
            return True
        return False
    
    def to_dict(self) -> Dict:
        """Convert temporal cluster to dictionary representation."""
        return {
            'cluster_id': self.cluster_id,
            'customer_count': len(self.customer_ids),
            'transaction_count': len(self.transactions),
            'merchant_count': len(self.merchants),
            'location_count': len(self.locations),
            'time_window_minutes': self.time_window_minutes,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'is_suspicious': self.is_suspicious(),
        }


class FraudNetworkAnalyzer:
    """Analyzes transaction data to detect fraud networks and coordinated attacks."""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.random = random.Random(seed)
        self.detected_rings: List[FraudRing] = []
        self.temporal_clusters: List[TemporalCluster] = []
        
    def analyze_merchant_networks(
        self, 
        transactions: List[Dict], 
        customer_map: Dict[str, Any],
        min_customers: int = 3,
        min_transactions: int = 5
    ) -> List[FraudRing]:
        """Detect fraud rings based on shared merchant usage patterns."""
        merchant_customers = defaultdict(set)
        merchant_transactions = defaultdict(list)
        
        # Group transactions by merchant
        for txn in transactions:
            if txn.get('Is_Fraud') == 1:
                merchant_id = txn.get('Merchant_ID', 'Unknown')
                customer_id = txn.get('Customer_ID', 'Unknown')
                merchant_customers[merchant_id].add(customer_id)
                merchant_transactions[merchant_id].append(txn)
        
        # Identify suspicious merchant patterns
        rings = []
        for merchant_id, customers in merchant_customers.items():
            if len(customers) >= min_customers:
                txn_list = merchant_transactions[merchant_id]
                if len(txn_list) >= min_transactions:
                    ring = FraudRing(f"MERCHANT_RING_{len(rings)+1}", 'merchant')
                    ring.shared_attributes['merchant_id'] = merchant_id
                    ring.shared_attributes['merchant_focus'] = True
                    
                    for txn in txn_list:
                        ring.add_transaction(txn, txn.get('Customer_ID', 'Unknown'))
                    
                    # Confidence based on concentration
                    ring.confidence = min(0.95, 0.3 + (len(customers) * 0.1))
                    rings.append(ring)
        
        self.detected_rings.extend(rings)
        return rings
    
    def analyze_location_networks(
        self,
        transactions: List[Dict],
        customer_map: Dict[str, Any],
        min_customers: int = 4,
        suspicious_locations: Optional[List[str]] = None
    ) -> List[FraudRing]:
        """Detect fraud rings based on shared location patterns."""
        if suspicious_locations is None:
            suspicious_locations = ['Border_Area', 'High_Crime_Zone', 'Known_Fraud_Hotspot']
        
        location_customers = defaultdict(set)
        location_transactions = defaultdict(list)
        
        # Group fraud transactions by location
        for txn in transactions:
            if txn.get('Is_Fraud') == 1:
                location = txn.get('City', 'Unknown')
                customer_id = txn.get('Customer_ID', 'Unknown')
                location_customers[location].add(customer_id)
                location_transactions[location].append(txn)
        
        rings = []
        for location, customers in location_customers.items():
            if len(customers) >= min_customers:
                is_suspicious_location = any(sus in location for sus in suspicious_locations)
                
                if is_suspicious_location or len(customers) >= min_customers + 2:
                    ring = FraudRing(f"LOCATION_RING_{len(rings)+1}", 'location')
                    ring.shared_attributes['location'] = location
                    ring.shared_attributes['is_suspicious_area'] = is_suspicious_location
                    
                    for txn in location_transactions[location]:
                        ring.add_transaction(txn, txn.get('Customer_ID', 'Unknown'))
                    
                    ring.confidence = 0.4 + (0.3 if is_suspicious_location else 0.1)
                    rings.append(ring)
        
        self.detected_rings.extend(rings)
        return rings
    
    def analyze_device_networks(
        self,
        transactions: List[Dict],
        customer_map: Dict[str, Any],
        min_customers: int = 3
    ) -> List[FraudRing]:
        """Detect fraud rings based on shared device fingerprints."""
        device_customers = defaultdict(set)
        device_transactions = defaultdict(list)
        
        # Group by device type and channel combination
        for txn in transactions:
            if txn.get('Is_Fraud') == 1:
                device_type = txn.get('Device_Type', 'Unknown')
                channel = txn.get('Channel', 'Unknown')
                device_key = f"{device_type}_{channel}"
                customer_id = txn.get('Customer_ID', 'Unknown')
                
                device_customers[device_key].add(customer_id)
                device_transactions[device_key].append(txn)
        
        rings = []
        for device_key, customers in device_customers.items():
            if len(customers) >= min_customers:
                # Check for high concentration (suspicious)
                txn_list = device_transactions[device_key]
                if len(txn_list) >= min_customers * 2:
                    ring = FraudRing(f"DEVICE_RING_{len(rings)+1}", 'device')
                    ring.shared_attributes['device_signature'] = device_key
                    
                    for txn in txn_list:
                        ring.add_transaction(txn, txn.get('Customer_ID', 'Unknown'))
                    
                    ring.confidence = min(0.85, 0.35 + (len(customers) * 0.08))
                    rings.append(ring)
        
        self.detected_rings.extend(rings)
        return rings
    
    def detect_temporal_clusters(
        self,
        transactions: List[Dict],
        time_window_minutes: int = 30,
        min_transactions: int = 5
    ) -> List[TemporalCluster]:
        """Detect temporal clusters of potentially coordinated transactions."""
        # Sort transactions by datetime
        sorted_txns = sorted(
            transactions,
            key=lambda t: f"{t.get('Date', '1970-01-01')} {t.get('Time', '00:00:00')}"
        )
        
        clusters = []
        current_cluster = None
        cluster_counter = 0
        
        for txn in sorted_txns:
            if txn.get('Is_Fraud') != 1:
                continue
                
            txn_datetime = self._parse_datetime(txn)
            if not txn_datetime:
                continue
            
            # Start new cluster or add to existing
            if current_cluster is None:
                cluster_counter += 1
                current_cluster = TemporalCluster(
                    f"TEMPORAL_CLUSTER_{cluster_counter}",
                    time_window_minutes
                )
                current_cluster.add_transaction(txn, txn.get('Customer_ID', 'Unknown'))
            else:
                time_diff = (txn_datetime - current_cluster.end_time).total_seconds() / 60
                
                if time_diff <= time_window_minutes:
                    current_cluster.add_transaction(txn, txn.get('Customer_ID', 'Unknown'))
                else:
                    # Close current cluster if it meets minimum
                    if len(current_cluster.transactions) >= min_transactions:
                        clusters.append(current_cluster)
                    
                    # Start new cluster
                    cluster_counter += 1
                    current_cluster = TemporalCluster(
                        f"TEMPORAL_CLUSTER_{cluster_counter}",
                        time_window_minutes
                    )
                    current_cluster.add_transaction(txn, txn.get('Customer_ID', 'Unknown'))
        
        # Add final cluster
        if current_cluster and len(current_cluster.transactions) >= min_transactions:
            clusters.append(current_cluster)
        
        self.temporal_clusters = clusters
        return clusters
    
    def _parse_datetime(self, transaction: Dict) -> Optional[datetime]:
        """Parse transaction datetime from date and time fields."""
        try:
            date_str = transaction.get('Date', '')
            time_str = transaction.get('Time', '00:00:00')
            datetime_str = f"{date_str} {time_str}"
            return datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        except:
            return None
    
    def generate_network_graph(self) -> Dict[str, Any]:
        """Generate network graph structure for visualization."""
        nodes = []
        edges = []
        
        # Add nodes for each fraud ring
        for ring in self.detected_rings:
            nodes.append({
                'id': ring.ring_id,
                'type': 'fraud_ring',
                'ring_type': ring.ring_type,
                'size': len(ring.customer_ids),
                'confidence': ring.confidence,
            })
            
            # Add edges to customers in the ring
            for customer_id in ring.customer_ids:
                edges.append({
                    'source': ring.ring_id,
                    'target': customer_id,
                    'type': 'member_of',
                    'weight': 1.0,
                })
        
        # Add nodes for temporal clusters
        for cluster in self.temporal_clusters:
            if cluster.is_suspicious():
                nodes.append({
                    'id': cluster.cluster_id,
                    'type': 'temporal_cluster',
                    'size': len(cluster.customer_ids),
                    'time_window': cluster.time_window_minutes,
                })
                
                for customer_id in cluster.customer_ids:
                    edges.append({
                        'source': cluster.cluster_id,
                        'target': customer_id,
                        'type': 'participated_in',
                        'weight': 0.7,
                    })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'total_rings': len(self.detected_rings),
                'total_clusters': len(self.temporal_clusters),
                'suspicious_clusters': sum(1 for c in self.temporal_clusters if c.is_suspicious()),
            }
        }
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network analysis statistics."""
        total_customers_in_rings = set()
        total_transactions_in_rings = set()
        
        for ring in self.detected_rings:
            total_customers_in_rings.update(ring.customer_ids)
            total_transactions_in_rings.update(ring.transaction_ids)
        
        ring_types = defaultdict(int)
        for ring in self.detected_rings:
            ring_types[ring.ring_type] += 1
        
        return {
            'total_fraud_rings': len(self.detected_rings),
            'ring_types': dict(ring_types),
            'unique_customers_in_rings': len(total_customers_in_rings),
            'unique_transactions_in_rings': len(total_transactions_in_rings),
            'temporal_clusters': len(self.temporal_clusters),
            'suspicious_clusters': sum(1 for c in self.temporal_clusters if c.is_suspicious()),
            'avg_customers_per_ring': sum(len(r.customer_ids) for r in self.detected_rings) / len(self.detected_rings) if self.detected_rings else 0,
            'avg_transactions_per_ring': sum(len(r.transaction_ids) for r in self.detected_rings) / len(self.detected_rings) if self.detected_rings else 0,
            'total_fraud_amount_in_rings': sum(r.total_amount for r in self.detected_rings),
        }
    
    def reset_analysis(self):
        """Reset all detected rings and clusters."""
        self.detected_rings = []
        self.temporal_clusters = []
