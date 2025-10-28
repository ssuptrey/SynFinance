"""
ML Feature Engineering Module for Fraud Detection

This module provides comprehensive feature engineering capabilities for training
fraud detection models. It generates aggregate, velocity, geographic, temporal,
behavioral, and network features from transaction data.

Features Generated:
- Aggregate: Daily/weekly transaction counts and amounts
- Velocity: Transaction frequency and amount velocity
- Geographic: Distance variance, travel patterns
- Temporal: Unusual hour flags, weekend patterns
- Behavioral: Category diversity, merchant loyalty
- Network: Shared merchant counts, customer proximity

Author: SynFinance Development Team
Version: 0.5.0
Date: October 26, 2025
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import math


@dataclass
class MLFeatures:
    """Container for engineered ML features."""
    
    # Transaction ID
    transaction_id: str
    
    # Aggregate Features (6 features)
    daily_txn_count: int
    weekly_txn_count: int
    daily_txn_amount: float
    weekly_txn_amount: float
    avg_daily_amount: float
    avg_weekly_amount: float
    
    # Velocity Features (6 features)
    txn_frequency_1h: int  # Transactions in last 1 hour
    txn_frequency_6h: int  # Transactions in last 6 hours
    txn_frequency_24h: int  # Transactions in last 24 hours
    amount_velocity_1h: float  # Amount spent in last 1 hour
    amount_velocity_6h: float  # Amount spent in last 6 hours
    amount_velocity_24h: float  # Amount spent in last 24 hours
    
    # Geographic Features (5 features)
    distance_from_home: float
    avg_distance_last_10: float  # Average distance of last 10 transactions
    distance_variance: float  # Variance in distances
    unique_cities_7d: int  # Unique cities visited in last 7 days
    travel_velocity_kmh: float  # Travel speed between last 2 transactions
    
    # Temporal Features (5 features)
    is_unusual_hour: bool  # Transaction at 2-5 AM
    is_weekend: bool
    is_holiday: bool
    hour_of_day: int
    day_of_week: int
    
    # Behavioral Features (6 features)
    category_diversity_score: float  # Shannon entropy of categories
    merchant_loyalty_score: float  # % of transactions at repeat merchants
    avg_merchant_reputation: float  # Average merchant rating
    new_merchant_flag: bool  # First time with this merchant
    refund_rate_30d: float  # Refund rate in last 30 days
    declined_rate_7d: float  # Declined transaction rate in last 7 days
    
    # Network Features (4 features)
    shared_merchant_count: int  # Merchants shared with other customers
    shared_location_count: int  # Locations shared with other customers
    customer_proximity_score: float  # How close to other customers
    temporal_cluster_flag: bool  # Part of temporal cluster
    
    # Label (for supervised learning)
    is_fraud: int  # 0 or 1
    fraud_type: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert features to dictionary for export."""
        return {
            'transaction_id': self.transaction_id,
            # Aggregate
            'daily_txn_count': self.daily_txn_count,
            'weekly_txn_count': self.weekly_txn_count,
            'daily_txn_amount': self.daily_txn_amount,
            'weekly_txn_amount': self.weekly_txn_amount,
            'avg_daily_amount': self.avg_daily_amount,
            'avg_weekly_amount': self.avg_weekly_amount,
            # Velocity
            'txn_frequency_1h': self.txn_frequency_1h,
            'txn_frequency_6h': self.txn_frequency_6h,
            'txn_frequency_24h': self.txn_frequency_24h,
            'amount_velocity_1h': self.amount_velocity_1h,
            'amount_velocity_6h': self.amount_velocity_6h,
            'amount_velocity_24h': self.amount_velocity_24h,
            # Geographic
            'distance_from_home': self.distance_from_home,
            'avg_distance_last_10': self.avg_distance_last_10,
            'distance_variance': self.distance_variance,
            'unique_cities_7d': self.unique_cities_7d,
            'travel_velocity_kmh': self.travel_velocity_kmh,
            # Temporal
            'is_unusual_hour': int(self.is_unusual_hour),
            'is_weekend': int(self.is_weekend),
            'is_holiday': int(self.is_holiday),
            'hour_of_day': self.hour_of_day,
            'day_of_week': self.day_of_week,
            # Behavioral
            'category_diversity_score': self.category_diversity_score,
            'merchant_loyalty_score': self.merchant_loyalty_score,
            'avg_merchant_reputation': self.avg_merchant_reputation,
            'new_merchant_flag': int(self.new_merchant_flag),
            'refund_rate_30d': self.refund_rate_30d,
            'declined_rate_7d': self.declined_rate_7d,
            # Network
            'shared_merchant_count': self.shared_merchant_count,
            'shared_location_count': self.shared_location_count,
            'customer_proximity_score': self.customer_proximity_score,
            'temporal_cluster_flag': int(self.temporal_cluster_flag),
            # Label
            'is_fraud': self.is_fraud,
            'fraud_type': self.fraud_type or 'None'
        }


class MLFeatureEngineer:
    """
    Feature engineering system for fraud detection ML models.
    
    Generates 32 features from transaction data:
    - 6 aggregate features
    - 6 velocity features
    - 5 geographic features
    - 5 temporal features
    - 6 behavioral features
    - 4 network features
    
    Example:
        engineer = MLFeatureEngineer()
        features = engineer.engineer_features(transactions, customers)
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.customer_history: Dict[str, List[Dict]] = defaultdict(list)
        self.merchant_customers: Dict[str, set] = defaultdict(set)
        self.location_customers: Dict[str, set] = defaultdict(set)
        
        # Indian holidays (simplified for example)
        self.holidays = [
            (1, 26),   # Republic Day
            (8, 15),   # Independence Day
            (10, 2),   # Gandhi Jayanti
            (10, 24),  # Diwali (approximate)
            (12, 25),  # Christmas
        ]
    
    def engineer_features(
        self,
        transaction: Dict,
        customer: Dict,
        customer_history: Optional[List[Dict]] = None
    ) -> MLFeatures:
        """
        Engineer all features for a single transaction.
        
        Args:
            transaction: Transaction dictionary
            customer: Customer dictionary
            customer_history: Previous transactions for this customer
            
        Returns:
            MLFeatures object with all engineered features
        """
        if customer_history is None:
            customer_history = []
        
        # Parse transaction timestamp
        txn_time = self._parse_timestamp(transaction)
        
        # Calculate each feature group
        agg_features = self._calculate_aggregate_features(transaction, customer_history)
        vel_features = self._calculate_velocity_features(transaction, customer_history, txn_time)
        geo_features = self._calculate_geographic_features(transaction, customer, customer_history)
        temp_features = self._calculate_temporal_features(transaction, txn_time)
        behav_features = self._calculate_behavioral_features(transaction, customer_history)
        net_features = self._calculate_network_features(transaction, customer)
        
        # Extract label
        is_fraud = transaction.get('Is_Fraud', 0)
        fraud_type = transaction.get('Fraud_Type', None)
        
        return MLFeatures(
            transaction_id=transaction['Transaction_ID'],
            # Aggregate
            daily_txn_count=agg_features['daily_txn_count'],
            weekly_txn_count=agg_features['weekly_txn_count'],
            daily_txn_amount=agg_features['daily_txn_amount'],
            weekly_txn_amount=agg_features['weekly_txn_amount'],
            avg_daily_amount=agg_features['avg_daily_amount'],
            avg_weekly_amount=agg_features['avg_weekly_amount'],
            # Velocity
            txn_frequency_1h=vel_features['txn_frequency_1h'],
            txn_frequency_6h=vel_features['txn_frequency_6h'],
            txn_frequency_24h=vel_features['txn_frequency_24h'],
            amount_velocity_1h=vel_features['amount_velocity_1h'],
            amount_velocity_6h=vel_features['amount_velocity_6h'],
            amount_velocity_24h=vel_features['amount_velocity_24h'],
            # Geographic
            distance_from_home=geo_features['distance_from_home'],
            avg_distance_last_10=geo_features['avg_distance_last_10'],
            distance_variance=geo_features['distance_variance'],
            unique_cities_7d=geo_features['unique_cities_7d'],
            travel_velocity_kmh=geo_features['travel_velocity_kmh'],
            # Temporal
            is_unusual_hour=temp_features['is_unusual_hour'],
            is_weekend=temp_features['is_weekend'],
            is_holiday=temp_features['is_holiday'],
            hour_of_day=temp_features['hour_of_day'],
            day_of_week=temp_features['day_of_week'],
            # Behavioral
            category_diversity_score=behav_features['category_diversity_score'],
            merchant_loyalty_score=behav_features['merchant_loyalty_score'],
            avg_merchant_reputation=behav_features['avg_merchant_reputation'],
            new_merchant_flag=behav_features['new_merchant_flag'],
            refund_rate_30d=behav_features['refund_rate_30d'],
            declined_rate_7d=behav_features['declined_rate_7d'],
            # Network
            shared_merchant_count=net_features['shared_merchant_count'],
            shared_location_count=net_features['shared_location_count'],
            customer_proximity_score=net_features['customer_proximity_score'],
            temporal_cluster_flag=net_features['temporal_cluster_flag'],
            # Label
            is_fraud=is_fraud,
            fraud_type=fraud_type
        )
    
    def _parse_timestamp(self, transaction: Dict) -> datetime:
        """Parse transaction timestamp from date and time fields."""
        date_str = transaction.get('Date', '2024-01-01')
        time_str = transaction.get('Time', '12:00:00')
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    
    def _calculate_aggregate_features(
        self,
        transaction: Dict,
        history: List[Dict]
    ) -> Dict:
        """Calculate aggregate transaction count and amount features."""
        txn_time = self._parse_timestamp(transaction)
        
        # Filter transactions by time windows
        daily_txns = [t for t in history if self._within_hours(t, txn_time, 24)]
        weekly_txns = [t for t in history if self._within_days(t, txn_time, 7)]
        
        daily_count = len(daily_txns)
        weekly_count = len(weekly_txns)
        
        daily_amount = sum(t.get('Amount', 0) for t in daily_txns)
        weekly_amount = sum(t.get('Amount', 0) for t in weekly_txns)
        
        avg_daily = daily_amount / max(daily_count, 1)
        avg_weekly = weekly_amount / max(weekly_count, 1)
        
        return {
            'daily_txn_count': daily_count,
            'weekly_txn_count': weekly_count,
            'daily_txn_amount': daily_amount,
            'weekly_txn_amount': weekly_amount,
            'avg_daily_amount': avg_daily,
            'avg_weekly_amount': avg_weekly
        }
    
    def _calculate_velocity_features(
        self,
        transaction: Dict,
        history: List[Dict],
        txn_time: datetime
    ) -> Dict:
        """Calculate transaction velocity features."""
        # Count transactions in different time windows
        txns_1h = [t for t in history if self._within_hours(t, txn_time, 1)]
        txns_6h = [t for t in history if self._within_hours(t, txn_time, 6)]
        txns_24h = [t for t in history if self._within_hours(t, txn_time, 24)]
        
        return {
            'txn_frequency_1h': len(txns_1h),
            'txn_frequency_6h': len(txns_6h),
            'txn_frequency_24h': len(txns_24h),
            'amount_velocity_1h': sum(t.get('Amount', 0) for t in txns_1h),
            'amount_velocity_6h': sum(t.get('Amount', 0) for t in txns_6h),
            'amount_velocity_24h': sum(t.get('Amount', 0) for t in txns_24h)
        }
    
    def _calculate_geographic_features(
        self,
        transaction: Dict,
        customer: Dict,
        history: List[Dict]
    ) -> Dict:
        """Calculate geographic distance and travel features."""
        home_city = customer.get('city', 'Mumbai')
        txn_city = transaction.get('City', 'Mumbai')
        
        # Distance from home
        distance = self._calculate_distance(home_city, txn_city)
        
        # Last 10 transactions distances
        recent_distances = []
        for t in history[-10:]:
            t_city = t.get('City', 'Mumbai')
            d = self._calculate_distance(home_city, t_city)
            recent_distances.append(d)
        
        avg_distance = sum(recent_distances) / len(recent_distances) if recent_distances else 0
        
        # Distance variance
        if len(recent_distances) > 1:
            mean_dist = sum(recent_distances) / len(recent_distances)
            variance = sum((d - mean_dist) ** 2 for d in recent_distances) / len(recent_distances)
        else:
            variance = 0
        
        # Unique cities in last 7 days
        txn_time = self._parse_timestamp(transaction)
        recent_7d = [t for t in history if self._within_days(t, txn_time, 7)]
        unique_cities = len(set(t.get('City', 'Mumbai') for t in recent_7d))
        
        # Travel velocity (speed between last 2 transactions)
        travel_velocity = 0.0
        if len(history) >= 1:
            last_txn = history[-1]
            last_city = last_txn.get('City', 'Mumbai')
            last_time = self._parse_timestamp(last_txn)
            
            dist_km = self._calculate_distance(last_city, txn_city)
            time_diff_hours = (txn_time - last_time).total_seconds() / 3600
            
            if time_diff_hours > 0:
                travel_velocity = dist_km / time_diff_hours
        
        return {
            'distance_from_home': distance,
            'avg_distance_last_10': avg_distance,
            'distance_variance': variance,
            'unique_cities_7d': unique_cities,
            'travel_velocity_kmh': travel_velocity
        }
    
    def _calculate_temporal_features(
        self,
        transaction: Dict,
        txn_time: datetime
    ) -> Dict:
        """Calculate temporal pattern features."""
        hour = txn_time.hour
        day_of_week = txn_time.weekday()  # 0=Monday, 6=Sunday
        
        # Unusual hour (2 AM - 5 AM)
        is_unusual = 2 <= hour <= 5
        
        # Weekend (Saturday=5, Sunday=6)
        is_weekend = day_of_week >= 5
        
        # Holiday check
        is_holiday = (txn_time.month, txn_time.day) in self.holidays
        
        return {
            'is_unusual_hour': is_unusual,
            'is_weekend': is_weekend,
            'is_holiday': is_holiday,
            'hour_of_day': hour,
            'day_of_week': day_of_week
        }
    
    def _calculate_behavioral_features(
        self,
        transaction: Dict,
        history: List[Dict]
    ) -> Dict:
        """Calculate behavioral pattern features."""
        # Category diversity (Shannon entropy)
        categories = [t.get('Category', 'Unknown') for t in history]
        if categories:
            category_counts = Counter(categories)
            total = len(categories)
            entropy = -sum((count / total) * math.log2(count / total) 
                          for count in category_counts.values())
        else:
            entropy = 0
        
        # Merchant loyalty
        merchants = [t.get('Merchant', 'Unknown') for t in history]
        if merchants:
            unique_merchants = len(set(merchants))
            loyalty_score = 1.0 - (unique_merchants / len(merchants))
        else:
            loyalty_score = 0
        
        # Average merchant reputation
        reputations = [t.get('Merchant_Reputation', 0.8) for t in history]
        avg_reputation = sum(reputations) / len(reputations) if reputations else 0.8
        
        # New merchant flag
        current_merchant = transaction.get('Merchant', 'Unknown')
        new_merchant = current_merchant not in merchants
        
        # Refund rate (last 30 days)
        txn_time = self._parse_timestamp(transaction)
        recent_30d = [t for t in history if self._within_days(t, txn_time, 30)]
        refunds = sum(1 for t in recent_30d if t.get('Transaction_Type') == 'Refund')
        refund_rate = refunds / len(recent_30d) if recent_30d else 0
        
        # Declined rate (last 7 days)
        recent_7d = [t for t in history if self._within_days(t, txn_time, 7)]
        declined = sum(1 for t in recent_7d if t.get('Transaction_Status') == 'Declined')
        declined_rate = declined / len(recent_7d) if recent_7d else 0
        
        return {
            'category_diversity_score': entropy,
            'merchant_loyalty_score': loyalty_score,
            'avg_merchant_reputation': avg_reputation,
            'new_merchant_flag': new_merchant,
            'refund_rate_30d': refund_rate,
            'declined_rate_7d': declined_rate
        }
    
    def _calculate_network_features(
        self,
        transaction: Dict,
        customer: Dict
    ) -> Dict:
        """Calculate network-based features."""
        customer_id = customer.get('Customer_ID', 'CUST_000')
        merchant = transaction.get('Merchant', 'Unknown')
        location = transaction.get('City', 'Mumbai')
        
        # Track merchant and location usage
        self.merchant_customers[merchant].add(customer_id)
        self.location_customers[location].add(customer_id)
        
        # Shared merchant count
        shared_merchants = len(self.merchant_customers[merchant]) - 1  # Exclude self
        
        # Shared location count
        shared_locations = len(self.location_customers[location]) - 1  # Exclude self
        
        # Customer proximity score (normalized)
        total_shared = shared_merchants + shared_locations
        proximity_score = min(total_shared / 100, 1.0)  # Cap at 1.0
        
        # Temporal cluster flag (simplified - would need more context)
        temporal_cluster = total_shared > 5
        
        return {
            'shared_merchant_count': shared_merchants,
            'shared_location_count': shared_locations,
            'customer_proximity_score': proximity_score,
            'temporal_cluster_flag': temporal_cluster
        }
    
    def _within_hours(self, transaction: Dict, reference_time: datetime, hours: int) -> bool:
        """Check if transaction is within N hours of reference time."""
        txn_time = self._parse_timestamp(transaction)
        time_diff = reference_time - txn_time
        return timedelta(0) <= time_diff <= timedelta(hours=hours)
    
    def _within_days(self, transaction: Dict, reference_time: datetime, days: int) -> bool:
        """Check if transaction is within N days of reference time."""
        txn_time = self._parse_timestamp(transaction)
        time_diff = reference_time - txn_time
        return timedelta(0) <= time_diff <= timedelta(days=days)
    
    def _calculate_distance(self, city1: str, city2: str) -> float:
        """Calculate approximate distance between two Indian cities (km)."""
        # Simplified distance matrix for major Indian cities
        distances = {
            ('Mumbai', 'Delhi'): 1400,
            ('Mumbai', 'Bangalore'): 980,
            ('Mumbai', 'Kolkata'): 2000,
            ('Mumbai', 'Chennai'): 1340,
            ('Delhi', 'Bangalore'): 2150,
            ('Delhi', 'Kolkata'): 1500,
            ('Delhi', 'Chennai'): 2200,
            ('Bangalore', 'Chennai'): 350,
            ('Bangalore', 'Kolkata'): 1900,
            ('Chennai', 'Kolkata'): 1670,
        }
        
        if city1 == city2:
            return 0
        
        # Try both orderings
        key1 = (city1, city2)
        key2 = (city2, city1)
        
        return distances.get(key1, distances.get(key2, 500))  # Default 500 km
    
    def get_feature_metadata(self) -> Dict:
        """
        Get metadata about all engineered features.
        
        Returns:
            Dictionary with feature names, types, and descriptions
        """
        return {
            'feature_count': 32,
            'features': [
                # Aggregate Features (6)
                {'name': 'daily_txn_count', 'type': 'int', 'description': 'Transaction count in last 24 hours'},
                {'name': 'weekly_txn_count', 'type': 'int', 'description': 'Transaction count in last 7 days'},
                {'name': 'daily_txn_amount', 'type': 'float', 'description': 'Total amount in last 24 hours'},
                {'name': 'weekly_txn_amount', 'type': 'float', 'description': 'Total amount in last 7 days'},
                {'name': 'avg_daily_amount', 'type': 'float', 'description': 'Average amount per transaction (24h)'},
                {'name': 'avg_weekly_amount', 'type': 'float', 'description': 'Average amount per transaction (7d)'},
                
                # Velocity Features (6)
                {'name': 'txn_frequency_1h', 'type': 'int', 'description': 'Transactions in last 1 hour'},
                {'name': 'txn_frequency_6h', 'type': 'int', 'description': 'Transactions in last 6 hours'},
                {'name': 'txn_frequency_24h', 'type': 'int', 'description': 'Transactions in last 24 hours'},
                {'name': 'amount_velocity_1h', 'type': 'float', 'description': 'Amount spent in last 1 hour'},
                {'name': 'amount_velocity_6h', 'type': 'float', 'description': 'Amount spent in last 6 hours'},
                {'name': 'amount_velocity_24h', 'type': 'float', 'description': 'Amount spent in last 24 hours'},
                
                # Geographic Features (5)
                {'name': 'distance_from_home', 'type': 'float', 'description': 'Distance from home city (km)'},
                {'name': 'avg_distance_last_10', 'type': 'float', 'description': 'Average distance of last 10 txns'},
                {'name': 'distance_variance', 'type': 'float', 'description': 'Variance in transaction distances'},
                {'name': 'unique_cities_7d', 'type': 'int', 'description': 'Unique cities visited in 7 days'},
                {'name': 'travel_velocity_kmh', 'type': 'float', 'description': 'Travel speed between last 2 txns (km/h)'},
                
                # Temporal Features (5)
                {'name': 'is_unusual_hour', 'type': 'bool', 'description': 'Transaction at 2-5 AM'},
                {'name': 'is_weekend', 'type': 'bool', 'description': 'Transaction on weekend'},
                {'name': 'is_holiday', 'type': 'bool', 'description': 'Transaction on Indian holiday'},
                {'name': 'hour_of_day', 'type': 'int', 'description': 'Hour of day (0-23)'},
                {'name': 'day_of_week', 'type': 'int', 'description': 'Day of week (0=Mon, 6=Sun)'},
                
                # Behavioral Features (6)
                {'name': 'category_diversity_score', 'type': 'float', 'description': 'Shannon entropy of categories'},
                {'name': 'merchant_loyalty_score', 'type': 'float', 'description': 'Repeat merchant percentage'},
                {'name': 'avg_merchant_reputation', 'type': 'float', 'description': 'Average merchant rating (0-1)'},
                {'name': 'new_merchant_flag', 'type': 'bool', 'description': 'First time with merchant'},
                {'name': 'refund_rate_30d', 'type': 'float', 'description': 'Refund rate in last 30 days'},
                {'name': 'declined_rate_7d', 'type': 'float', 'description': 'Declined rate in last 7 days'},
                
                # Network Features (4)
                {'name': 'shared_merchant_count', 'type': 'int', 'description': 'Customers sharing this merchant'},
                {'name': 'shared_location_count', 'type': 'int', 'description': 'Customers at this location'},
                {'name': 'customer_proximity_score', 'type': 'float', 'description': 'Proximity to other customers (0-1)'},
                {'name': 'temporal_cluster_flag', 'type': 'bool', 'description': 'Part of temporal cluster'},
                
                # Label
                {'name': 'is_fraud', 'type': 'int', 'description': 'Fraud label (0=legit, 1=fraud)'},
                {'name': 'fraud_type', 'type': 'str', 'description': 'Type of fraud (if any)'}
            ]
        }
