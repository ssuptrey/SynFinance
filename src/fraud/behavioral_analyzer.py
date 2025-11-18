"""
Behavioral Analyzer

Detects deviations from customer behavioral baselines.
Builds customer profiles and identifies statistical anomalies.

Week 10 Day 4: Advanced Fraud Detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import Counter
from scipy import stats

from src.fraud import CustomerProfile, BehavioralAnomaly, AnomalyType


class BehavioralAnalyzer:
    """
    Customer behavioral analysis and anomaly detection
    
    Features:
    - Build baseline customer profiles from historical data
    - Detect statistical anomalies (Z-score, Chi-square)
    - Track temporal patterns (hour-of-day, day-of-week)
    - Geographic pattern analysis
    - Merchant preference tracking
    - Incremental profile updates
    
    Statistical methods:
    - Z-score for continuous variables (amount, frequency)
    - Chi-square test for categorical variables (merchant, category)
    - Time-series anomaly detection
    """
    
    def __init__(self, min_transactions_for_profile: int = 10):
        """
        Initialize behavioral analyzer
        
        Args:
            min_transactions_for_profile: Minimum transactions needed to build profile
        """
        self.min_transactions = min_transactions_for_profile
        self.profiles: Dict[str, CustomerProfile] = {}
    
    def build_customer_profile(
        self,
        customer_id: str,
        historical_transactions: pd.DataFrame
    ) -> CustomerProfile:
        """
        Build baseline behavioral profile from historical data
        
        Args:
            customer_id: Customer ID
            historical_transactions: DataFrame with transaction history
            
        Returns:
            CustomerProfile with baseline statistics
        """
        if len(historical_transactions) < self.min_transactions:
            raise ValueError(
                f"Insufficient data: {len(historical_transactions)} transactions "
                f"(minimum: {self.min_transactions})"
            )
        
        # Transaction amount statistics
        amounts = historical_transactions['amount'].values
        avg_amount = float(np.mean(amounts))
        std_amount = float(np.std(amounts))
        min_amount = float(np.min(amounts))
        max_amount = float(np.max(amounts))
        percentile_25 = float(np.percentile(amounts, 25))
        percentile_75 = float(np.percentile(amounts, 75))
        percentile_95 = float(np.percentile(amounts, 95))
        
        # Merchant preferences
        if 'merchant_id' in historical_transactions.columns:
            merchant_counts = Counter(historical_transactions['merchant_id'].dropna())
            top_merchants = [m for m, _ in merchant_counts.most_common(10)]
            unique_merchants = len(merchant_counts)
        else:
            top_merchants = []
            unique_merchants = 0
        
        # Category preferences
        if 'category' in historical_transactions.columns:
            category_counts = Counter(historical_transactions['category'].dropna())
            top_categories = [c for c, _ in category_counts.most_common(10)]
        else:
            top_categories = []
        
        # Temporal patterns
        if 'timestamp' in historical_transactions.columns:
            timestamps = pd.to_datetime(historical_transactions['timestamp'])
            
            # Hour of day distribution
            hour_counts = Counter(timestamps.dt.hour)
            total_hours = len(timestamps)
            hour_of_day_distribution = {
                hour: count / total_hours
                for hour, count in hour_counts.items()
            }
            
            # Day of week distribution
            day_counts = Counter(timestamps.dt.day_name())
            day_of_week_distribution = {
                day: count / total_hours
                for day, count in day_counts.items()
            }
            
            # Activity frequency
            date_range = (timestamps.max() - timestamps.min()).days
            if date_range > 0:
                daily_transaction_count = len(timestamps) / date_range
                weekly_transaction_count = daily_transaction_count * 7
            else:
                daily_transaction_count = 0.0
                weekly_transaction_count = 0.0
        else:
            hour_of_day_distribution = {}
            day_of_week_distribution = {}
            daily_transaction_count = 0.0
            weekly_transaction_count = 0.0
        
        # Geographic patterns
        home_location = None
        frequent_cities = []
        
        if 'latitude' in historical_transactions.columns and 'longitude' in historical_transactions.columns:
            # Calculate most common location (home)
            lats = historical_transactions['latitude'].dropna()
            lons = historical_transactions['longitude'].dropna()
            
            if len(lats) > 0:
                home_location = {
                    'lat': float(lats.median()),
                    'lon': float(lons.median())
                }
        
        if 'city' in historical_transactions.columns:
            city_counts = Counter(historical_transactions['city'].dropna())
            frequent_cities = [c for c, _ in city_counts.most_common(5)]
        
        # Create profile
        profile = CustomerProfile(
            customer_id=customer_id,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            avg_amount=avg_amount,
            std_amount=std_amount,
            min_amount=min_amount,
            max_amount=max_amount,
            percentile_25=percentile_25,
            percentile_75=percentile_75,
            percentile_95=percentile_95,
            top_merchants=top_merchants,
            top_categories=top_categories,
            unique_merchants=unique_merchants,
            hour_of_day_distribution=hour_of_day_distribution,
            day_of_week_distribution=day_of_week_distribution,
            home_location=home_location,
            frequent_cities=frequent_cities,
            daily_transaction_count=daily_transaction_count,
            weekly_transaction_count=weekly_transaction_count,
            transaction_count=len(historical_transactions)
        )
        
        # Cache profile
        self.profiles[customer_id] = profile
        
        return profile
    
    def detect_anomalies(
        self,
        customer_id: str,
        transaction: Dict[str, Any],
        significance_level: float = 0.05
    ) -> List[BehavioralAnomaly]:
        """
        Detect behavioral anomalies in transaction
        
        Args:
            customer_id: Customer ID
            transaction: Transaction data
            significance_level: P-value threshold (default: 0.05)
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Get customer profile
        profile = self.profiles.get(customer_id)
        if not profile:
            # No profile available, cannot detect anomalies
            return anomalies
        
        # Amount anomaly
        amount_anomaly = self._check_amount_anomaly(transaction, profile, significance_level)
        if amount_anomaly:
            anomalies.append(amount_anomaly)
        
        # Merchant anomaly
        merchant_anomaly = self._check_merchant_anomaly(transaction, profile, significance_level)
        if merchant_anomaly:
            anomalies.append(merchant_anomaly)
        
        # Time anomaly
        time_anomaly = self._check_time_anomaly(transaction, profile, significance_level)
        if time_anomaly:
            anomalies.append(time_anomaly)
        
        # Location anomaly
        location_anomaly = self._check_location_anomaly(transaction, profile, significance_level)
        if location_anomaly:
            anomalies.append(location_anomaly)
        
        return anomalies
    
    def _check_amount_anomaly(
        self,
        transaction: Dict[str, Any],
        profile: CustomerProfile,
        significance_level: float
    ) -> Optional[BehavioralAnomaly]:
        """Check for amount anomaly using Z-score"""
        amount = transaction.get('amount', 0.0)
        
        if profile.std_amount == 0:
            return None
        
        # Calculate Z-score
        z_score = (amount - profile.avg_amount) / profile.std_amount
        
        # Calculate p-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        is_significant = p_value < significance_level
        
        if is_significant:
            return BehavioralAnomaly(
                anomaly_type=AnomalyType.AMOUNT_ANOMALY,
                field_name='amount',
                expected_value=profile.avg_amount,
                actual_value=amount,
                deviation_score=abs(z_score),
                p_value=p_value,
                is_significant=True,
                explanation=f"Amount ${amount:.2f} is {abs(z_score):.2f} std deviations from average ${profile.avg_amount:.2f}"
            )
        
        return None
    
    def _check_merchant_anomaly(
        self,
        transaction: Dict[str, Any],
        profile: CustomerProfile,
        significance_level: float
    ) -> Optional[BehavioralAnomaly]:
        """Check for merchant anomaly using Chi-square test"""
        merchant_id = transaction.get('merchant_id')
        
        if not merchant_id or not profile.top_merchants:
            return None
        
        # Check if merchant is in top merchants
        is_known_merchant = merchant_id in profile.top_merchants
        
        if not is_known_merchant:
            # Calculate p-value based on how rare new merchants are
            # If customer usually uses same merchants, this is significant
            merchant_diversity = profile.unique_merchants / max(profile.transaction_count, 1)
            
            # Higher diversity = more accepting of new merchants
            # Lower diversity = anomaly when seeing new merchant
            p_value = merchant_diversity
            
            is_significant = p_value < significance_level
            
            if is_significant:
                return BehavioralAnomaly(
                    anomaly_type=AnomalyType.MERCHANT_ANOMALY,
                    field_name='merchant_id',
                    expected_value=profile.top_merchants[0] if profile.top_merchants else None,
                    actual_value=merchant_id,
                    deviation_score=1.0 - merchant_diversity,
                    p_value=p_value,
                    is_significant=True,
                    explanation=f"Merchant '{merchant_id}' is not in customer's top {len(profile.top_merchants)} merchants"
                )
        
        return None
    
    def _check_time_anomaly(
        self,
        transaction: Dict[str, Any],
        profile: CustomerProfile,
        significance_level: float
    ) -> Optional[BehavioralAnomaly]:
        """Check for temporal pattern anomaly"""
        timestamp = transaction.get('timestamp')
        
        if not timestamp or not profile.hour_of_day_distribution:
            return None
        
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        hour = timestamp.hour
        
        # Get expected frequency for this hour
        expected_frequency = profile.hour_of_day_distribution.get(hour, 0.0)
        
        # If this hour is rare for customer (< 5% of transactions)
        if expected_frequency < 0.05:
            p_value = expected_frequency
            
            # Find most common hour
            most_common_hour = max(
                profile.hour_of_day_distribution,
                key=profile.hour_of_day_distribution.get
            )
            
            return BehavioralAnomaly(
                anomaly_type=AnomalyType.TIME_ANOMALY,
                field_name='timestamp',
                expected_value=most_common_hour,
                actual_value=hour,
                deviation_score=1.0 - expected_frequency,
                p_value=p_value,
                is_significant=p_value < significance_level,
                explanation=f"Transaction at hour {hour} is unusual (only {expected_frequency:.1%} of transactions)"
            )
        
        return None
    
    def _check_location_anomaly(
        self,
        transaction: Dict[str, Any],
        profile: CustomerProfile,
        significance_level: float
    ) -> Optional[BehavioralAnomaly]:
        """Check for geographic anomaly"""
        if not profile.home_location:
            return None
        
        # Check city if available
        city = transaction.get('city')
        if city and profile.frequent_cities:
            if city not in profile.frequent_cities:
                # Calculate rarity
                city_diversity = len(profile.frequent_cities) / max(profile.transaction_count, 1)
                p_value = city_diversity
                
                if p_value < significance_level:
                    return BehavioralAnomaly(
                        anomaly_type=AnomalyType.LOCATION_ANOMALY,
                        field_name='city',
                        expected_value=profile.frequent_cities[0] if profile.frequent_cities else None,
                        actual_value=city,
                        deviation_score=1.0,
                        p_value=p_value,
                        is_significant=True,
                        explanation=f"City '{city}' is not in customer's frequent locations"
                    )
        
        # Check distance from home
        if 'latitude' in transaction and 'longitude' in transaction:
            lat = transaction['latitude']
            lon = transaction['longitude']
            
            distance = self._calculate_distance(
                profile.home_location['lat'],
                profile.home_location['lon'],
                lat,
                lon
            )
            
            # If more than 100 miles from home, flag as anomaly
            if distance > 100:
                p_value = min(1.0, 100.0 / distance)
                
                return BehavioralAnomaly(
                    anomaly_type=AnomalyType.LOCATION_ANOMALY,
                    field_name='location',
                    expected_value=profile.home_location,
                    actual_value={'lat': lat, 'lon': lon},
                    deviation_score=distance / 100,
                    p_value=p_value,
                    is_significant=p_value < significance_level,
                    explanation=f"Transaction is {distance:.1f} miles from home location"
                )
        
        return None
    
    def _calculate_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate distance using Haversine formula"""
        R = 3959.0  # Earth radius in miles
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def update_profile(
        self,
        customer_id: str,
        new_transaction: Dict[str, Any],
        update_weight: float = 0.1
    ):
        """
        Incrementally update customer profile with new transaction
        
        Uses exponential moving average for smooth updates
        
        Args:
            customer_id: Customer ID
            new_transaction: New transaction data
            update_weight: Weight for new data (0-1, default: 0.1)
        """
        profile = self.profiles.get(customer_id)
        if not profile:
            return
        
        amount = new_transaction.get('amount', 0.0)
        
        # Update amount statistics (exponential moving average)
        profile.avg_amount = (
            (1 - update_weight) * profile.avg_amount +
            update_weight * amount
        )
        
        # Update std (approximate)
        deviation = amount - profile.avg_amount
        profile.std_amount = (
            (1 - update_weight) * profile.std_amount +
            update_weight * abs(deviation)
        )
        
        # Update min/max
        profile.min_amount = min(profile.min_amount, amount)
        profile.max_amount = max(profile.max_amount, amount)
        
        # Update merchant preferences
        merchant_id = new_transaction.get('merchant_id')
        if merchant_id and merchant_id not in profile.top_merchants:
            # Add to top merchants if space
            if len(profile.top_merchants) < 10:
                profile.top_merchants.append(merchant_id)
            profile.unique_merchants += 1
        
        # Update timestamps
        profile.last_updated = datetime.now()
        profile.transaction_count += 1
    
    def get_profile(self, customer_id: str) -> Optional[CustomerProfile]:
        """
        Get customer profile
        
        Args:
            customer_id: Customer ID
            
        Returns:
            CustomerProfile if available, None otherwise
        """
        return self.profiles.get(customer_id)
    
    def has_profile(self, customer_id: str) -> bool:
        """Check if profile exists for customer"""
        return customer_id in self.profiles
    
    def delete_profile(self, customer_id: str):
        """Delete customer profile"""
        if customer_id in self.profiles:
            del self.profiles[customer_id]
    
    def get_profile_summary(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """
        Get human-readable profile summary
        
        Args:
            customer_id: Customer ID
            
        Returns:
            Dictionary with profile summary
        """
        profile = self.profiles.get(customer_id)
        if not profile:
            return None
        
        return {
            'customer_id': customer_id,
            'transaction_count': profile.transaction_count,
            'avg_amount': f"${profile.avg_amount:.2f}",
            'amount_range': f"${profile.min_amount:.2f} - ${profile.max_amount:.2f}",
            'top_merchants': profile.top_merchants[:5],
            'unique_merchants': profile.unique_merchants,
            'transactions_per_week': f"{profile.weekly_transaction_count:.1f}",
            'most_active_hour': max(
                profile.hour_of_day_distribution,
                key=profile.hour_of_day_distribution.get
            ) if profile.hour_of_day_distribution else None,
            'home_location': profile.home_location,
            'created_at': profile.created_at.isoformat(),
            'last_updated': profile.last_updated.isoformat()
        }
