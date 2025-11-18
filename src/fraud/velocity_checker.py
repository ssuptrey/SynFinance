"""
Velocity Checker

Detects rapid transaction sequences and velocity violations.
Tracks transactions across multiple time windows with sliding window implementation.

Week 10 Day 4: Advanced Fraud Detection
"""

import time
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Deque
import numpy as np

from src.fraud import VelocityRule, VelocityResult


class VelocityChecker:
    """
    Transaction velocity checker with time-window aggregations
    
    Features:
    - Multiple time windows (1min, 5min, 15min, 1hr, 24hr)
    - Sliding window implementation for efficiency
    - Velocity metrics: count, amount, unique merchants, distance
    - Z-score anomaly detection
    - In-memory caching for performance
    - Optional Redis support for distributed systems
    
    Performance target: <10ms per check
    """
    
    def __init__(self, use_redis: bool = False, redis_client=None):
        """
        Initialize velocity checker
        
        Args:
            use_redis: Use Redis for distributed caching
            redis_client: Redis client instance (if use_redis=True)
        """
        self.use_redis = use_redis
        self.redis_client = redis_client
        
        # In-memory transaction history (customer_id -> deque of transactions)
        self.transaction_history: Dict[str, Deque[Dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=1000)  # Keep last 1000 transactions per customer
        )
        
        # Velocity rules
        self.rules: List[VelocityRule] = []
        self._add_default_rules()
        
        # Performance tracking
        self.check_count = 0
        self.total_time_ms = 0.0
    
    def _add_default_rules(self):
        """Add default velocity rules"""
        self.rules = [
            # Transaction count rules
            VelocityRule(
                name="Rapid Transactions (1min)",
                time_window="1min",
                metric="count",
                threshold=5,
                severity="critical"
            ),
            VelocityRule(
                name="High Transaction Count (5min)",
                time_window="5min",
                metric="count",
                threshold=10,
                severity="high"
            ),
            VelocityRule(
                name="Elevated Transaction Count (1hr)",
                time_window="1hr",
                metric="count",
                threshold=20,
                severity="medium"
            ),
            
            # Amount rules
            VelocityRule(
                name="High Amount Velocity (5min)",
                time_window="5min",
                metric="amount",
                threshold=1000.0,
                severity="critical"
            ),
            VelocityRule(
                name="Elevated Amount Velocity (1hr)",
                time_window="1hr",
                metric="amount",
                threshold=5000.0,
                severity="high"
            ),
            VelocityRule(
                name="Daily Amount Limit",
                time_window="24hr",
                metric="amount",
                threshold=10000.0,
                severity="medium"
            ),
            
            # Unique merchant rules
            VelocityRule(
                name="Many Unique Merchants (1hr)",
                time_window="1hr",
                metric="unique_merchants",
                threshold=10,
                severity="high"
            ),
            VelocityRule(
                name="Excessive Merchant Diversity (24hr)",
                time_window="24hr",
                metric="unique_merchants",
                threshold=25,
                severity="medium"
            ),
        ]
    
    def add_rule(self, rule: VelocityRule):
        """
        Add custom velocity rule
        
        Args:
            rule: VelocityRule instance
        """
        self.rules.append(rule)
    
    def remove_rule(self, name: str):
        """
        Remove velocity rule by name
        
        Args:
            name: Rule name
        """
        self.rules = [r for r in self.rules if r.name != name]
    
    def check_transaction_velocity(
        self,
        customer_id: str,
        transaction: Dict[str, Any]
    ) -> VelocityResult:
        """
        Check if transaction violates velocity rules
        
        Args:
            customer_id: Customer ID
            transaction: Transaction data (amount, merchant_id, timestamp, etc.)
            
        Returns:
            VelocityResult with violations and statistics
        """
        start_time = time.time()
        
        timestamp = transaction.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        # Get recent transactions for this customer
        recent_transactions = self._get_recent_transactions(customer_id, timestamp)
        
        # Calculate velocity statistics
        velocity_stats = self._calculate_velocity_stats(
            recent_transactions,
            timestamp,
            transaction
        )
        
        # Check rules
        violated_rules = []
        for rule in self.rules:
            if self._check_rule(rule, velocity_stats):
                violated_rules.append(rule)
        
        # Determine max severity
        max_severity = "low"
        if violated_rules:
            severity_order = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
            max_severity = max(
                violated_rules,
                key=lambda r: severity_order.get(r.severity, 0)
            ).severity
        
        # Add current transaction to history
        self._add_transaction(customer_id, transaction)
        
        # Track performance
        elapsed_ms = (time.time() - start_time) * 1000
        self.check_count += 1
        self.total_time_ms += elapsed_ms
        
        return VelocityResult(
            customer_id=customer_id,
            timestamp=timestamp,
            violated_rules=violated_rules,
            velocity_stats=velocity_stats,
            is_violation=len(violated_rules) > 0,
            max_severity=max_severity
        )
    
    def _get_recent_transactions(
        self,
        customer_id: str,
        current_time: datetime,
        lookback_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Get recent transactions within lookback window
        
        Args:
            customer_id: Customer ID
            current_time: Current timestamp
            lookback_hours: Hours to look back
            
        Returns:
            List of recent transactions
        """
        if self.use_redis and self.redis_client:
            # Use Redis for distributed caching
            return self._get_from_redis(customer_id, current_time, lookback_hours)
        else:
            # Use in-memory deque
            all_transactions = list(self.transaction_history[customer_id])
            cutoff_time = current_time - timedelta(hours=lookback_hours)
            
            recent = []
            for txn in all_transactions:
                txn_time = txn.get('timestamp', datetime.now())
                if isinstance(txn_time, str):
                    txn_time = datetime.fromisoformat(txn_time)
                
                if txn_time >= cutoff_time:
                    recent.append(txn)
            
            return recent
    
    def _add_transaction(self, customer_id: str, transaction: Dict[str, Any]):
        """Add transaction to history"""
        # Ensure timestamp is datetime
        if 'timestamp' not in transaction:
            transaction['timestamp'] = datetime.now()
        elif isinstance(transaction['timestamp'], str):
            transaction['timestamp'] = datetime.fromisoformat(transaction['timestamp'])
        
        if self.use_redis and self.redis_client:
            self._add_to_redis(customer_id, transaction)
        else:
            self.transaction_history[customer_id].append(transaction)
    
    def _calculate_velocity_stats(
        self,
        recent_transactions: List[Dict[str, Any]],
        current_time: datetime,
        current_transaction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate velocity statistics across all time windows
        
        Args:
            recent_transactions: List of recent transactions
            current_time: Current timestamp
            current_transaction: Current transaction
            
        Returns:
            Dictionary with stats per time window
        """
        stats = {}
        
        time_windows = {
            '1min': 60,
            '5min': 300,
            '15min': 900,
            '1hr': 3600,
            '24hr': 86400
        }
        
        for window_name, window_seconds in time_windows.items():
            cutoff_time = current_time - timedelta(seconds=window_seconds)
            
            # Filter transactions in this window
            window_transactions = [
                txn for txn in recent_transactions
                if txn['timestamp'] >= cutoff_time
            ]
            
            # Calculate metrics
            count = len(window_transactions)
            total_amount = sum(txn.get('amount', 0.0) for txn in window_transactions)
            
            merchants = set()
            for txn in window_transactions:
                merchant_id = txn.get('merchant_id', 'UNKNOWN')
                if merchant_id:
                    merchants.add(merchant_id)
            unique_merchants = len(merchants)
            
            # Geographic distance (if location data available)
            distance = self._calculate_distance(window_transactions)
            
            stats[window_name] = {
                'count': count,
                'amount': total_amount,
                'unique_merchants': unique_merchants,
                'distance': distance
            }
        
        return stats
    
    def _calculate_distance(self, transactions: List[Dict[str, Any]]) -> float:
        """
        Calculate total geographic distance traveled
        
        Args:
            transactions: List of transactions with optional location data
            
        Returns:
            Total distance in miles
        """
        if len(transactions) < 2:
            return 0.0
        
        total_distance = 0.0
        
        # Sort by timestamp
        sorted_txns = sorted(transactions, key=lambda t: t['timestamp'])
        
        for i in range(len(sorted_txns) - 1):
            txn1 = sorted_txns[i]
            txn2 = sorted_txns[i + 1]
            
            # Check if both have location data
            if 'latitude' in txn1 and 'latitude' in txn2:
                dist = self._haversine_distance(
                    txn1['latitude'], txn1['longitude'],
                    txn2['latitude'], txn2['longitude']
                )
                total_distance += dist
        
        return total_distance
    
    def _haversine_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """
        Calculate distance between two points using Haversine formula
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in miles
        """
        # Earth radius in miles
        R = 3959.0
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _check_rule(self, rule: VelocityRule, velocity_stats: Dict[str, Any]) -> bool:
        """
        Check if rule is violated
        
        Args:
            rule: VelocityRule to check
            velocity_stats: Calculated velocity statistics
            
        Returns:
            True if rule is violated
        """
        window_stats = velocity_stats.get(rule.time_window, {})
        actual_value = window_stats.get(rule.metric, 0.0)
        
        return actual_value > rule.threshold
    
    def get_velocity_stats(
        self,
        customer_id: str,
        time_window: str = '1hr'
    ) -> Dict[str, Any]:
        """
        Get aggregated velocity statistics for a customer
        
        Args:
            customer_id: Customer ID
            time_window: Time window (1min, 5min, 15min, 1hr, 24hr)
            
        Returns:
            Dictionary with velocity statistics
        """
        current_time = datetime.now()
        recent_transactions = self._get_recent_transactions(customer_id, current_time)
        
        # Create dummy transaction for current time
        dummy_transaction = {
            'timestamp': current_time,
            'amount': 0.0,
            'merchant_id': None
        }
        
        velocity_stats = self._calculate_velocity_stats(
            recent_transactions,
            current_time,
            dummy_transaction
        )
        
        return velocity_stats.get(time_window, {})
    
    def detect_velocity_anomaly(
        self,
        customer_id: str,
        metric: str = 'count',
        time_window: str = '1hr',
        z_threshold: float = 3.0
    ) -> Optional[Dict[str, Any]]:
        """
        Detect velocity anomalies using Z-score
        
        Args:
            customer_id: Customer ID
            metric: Metric to check (count, amount, unique_merchants)
            time_window: Time window
            z_threshold: Z-score threshold (default: 3.0)
            
        Returns:
            Anomaly details if detected, None otherwise
        """
        # Get historical velocity values for this customer
        historical_values = self._get_historical_velocity(
            customer_id,
            metric,
            time_window,
            lookback_days=30
        )
        
        if len(historical_values) < 10:
            # Not enough historical data
            return None
        
        # Calculate current velocity
        current_stats = self.get_velocity_stats(customer_id, time_window)
        current_value = current_stats.get(metric, 0.0)
        
        # Calculate Z-score
        mean = np.mean(historical_values)
        std = np.std(historical_values)
        
        if std == 0:
            return None
        
        z_score = abs((current_value - mean) / std)
        
        if z_score > z_threshold:
            return {
                'customer_id': customer_id,
                'metric': metric,
                'time_window': time_window,
                'current_value': current_value,
                'historical_mean': mean,
                'historical_std': std,
                'z_score': z_score,
                'is_anomaly': True
            }
        
        return None
    
    def _get_historical_velocity(
        self,
        customer_id: str,
        metric: str,
        time_window: str,
        lookback_days: int = 30
    ) -> List[float]:
        """
        Get historical velocity values for anomaly detection
        
        Args:
            customer_id: Customer ID
            metric: Metric name
            time_window: Time window
            lookback_days: Days to look back
            
        Returns:
            List of historical values
        """
        # This is a simplified implementation
        # In production, you'd query historical data from database
        
        # For now, sample from transaction history
        all_transactions = list(self.transaction_history[customer_id])
        
        if not all_transactions:
            return []
        
        # Sample historical values (simplified)
        values = []
        for i in range(0, len(all_transactions), 10):
            sample_txns = all_transactions[max(0, i-10):i]
            if metric == 'count':
                values.append(len(sample_txns))
            elif metric == 'amount':
                values.append(sum(txn.get('amount', 0.0) for txn in sample_txns))
            elif metric == 'unique_merchants':
                merchants = set(txn.get('merchant_id') for txn in sample_txns if txn.get('merchant_id'))
                values.append(len(merchants))
        
        return values
    
    def _get_from_redis(
        self,
        customer_id: str,
        current_time: datetime,
        lookback_hours: int
    ) -> List[Dict[str, Any]]:
        """Get transactions from Redis (placeholder for Redis implementation)"""
        # TODO: Implement Redis caching
        # This would use sorted sets with timestamps as scores
        # ZRANGEBYSCORE to get transactions in time range
        return []
    
    def _add_to_redis(self, customer_id: str, transaction: Dict[str, Any]):
        """Add transaction to Redis (placeholder for Redis implementation)"""
        # TODO: Implement Redis caching
        # This would use ZADD to add transaction with timestamp score
        # ZREMRANGEBYSCORE to clean up old transactions
        pass
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_latency_ms = 0.0
        if self.check_count > 0:
            avg_latency_ms = self.total_time_ms / self.check_count
        
        return {
            'total_checks': self.check_count,
            'total_time_ms': self.total_time_ms,
            'avg_latency_ms': avg_latency_ms
        }
    
    def clear_history(self, customer_id: Optional[str] = None):
        """
        Clear transaction history
        
        Args:
            customer_id: Optional customer ID (if None, clears all)
        """
        if customer_id:
            if customer_id in self.transaction_history:
                self.transaction_history[customer_id].clear()
        else:
            self.transaction_history.clear()
