"""
Anomaly Pattern Generation System for SynFinance

This module provides anomaly detection patterns for generating realistic behavioral,
geographic, temporal, and amount anomalies in synthetic transaction data.

Anomalies vs. Fraud:
- Anomalies: Unusual but potentially legitimate behavior (e.g., vacation spending)
- Fraud: Malicious activity with criminal intent
- Some anomalies may indicate fraud, but many are benign

Anomaly Types:
1. Behavioral: Out-of-character purchases (category, merchant, payment method changes)
2. Geographic: Unusual locations, impossible travel (but not necessarily fraud)
3. Temporal: Unusual hours, schedule changes, holiday deviations
4. Amount: Spending spikes, micro-transactions, round amounts, budget changes

Severity Scoring:
- 0.0-0.3: Low severity (minor deviation)
- 0.3-0.6: Medium severity (notable deviation)
- 0.6-0.8: High severity (significant deviation)
- 0.8-1.0: Critical severity (extreme deviation)

Usage:
    from src.generators.anomaly_patterns import AnomalyPatternGenerator
    
    generator = AnomalyPatternGenerator(seed=42)
    transactions = generator.inject_anomaly_patterns(
        transactions,
        customers,
        anomaly_rate=0.05  # 5% of transactions have anomalies
    )
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import random
from datetime import datetime, timedelta
import json
import math


class AnomalyType(Enum):
    """Types of anomalies in transaction data"""
    BEHAVIORAL = "Behavioral Anomaly"
    GEOGRAPHIC = "Geographic Anomaly"
    TEMPORAL = "Temporal Anomaly"
    AMOUNT = "Amount Anomaly"
    NONE = "None"


@dataclass
class AnomalyIndicator:
    """
    Anomaly indicator with metadata for ML training
    
    Attributes:
        anomaly_type: Type of anomaly detected
        confidence: Confidence score (0.0-1.0) for anomaly detection
        reason: Human-readable explanation of the anomaly
        evidence: Dictionary of supporting evidence for the anomaly
        severity: Severity score (0.0-1.0) indicating deviation magnitude
    """
    anomaly_type: AnomalyType
    confidence: float
    reason: str
    evidence: Dict[str, Any]
    severity: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert anomaly indicator to dictionary"""
        return {
            'anomaly_type': self.anomaly_type.value,
            'confidence': round(self.confidence, 3),
            'reason': self.reason,
            'evidence': self.evidence,
            'severity': round(self.severity, 3)
        }


class AnomalyPattern:
    """Base class for anomaly pattern detection"""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize anomaly pattern
        
        Args:
            seed: Random seed for reproducibility
        """
        self.random = random.Random(seed)
        self.anomaly_type = AnomalyType.NONE
    
    def should_apply(self, transaction: Dict[str, Any], customer: Any, 
                     transaction_history: List[Dict[str, Any]]) -> bool:
        """
        Determine if this anomaly pattern should be applied
        
        Args:
            transaction: Current transaction
            customer: Customer profile
            transaction_history: List of customer's previous transactions
            
        Returns:
            True if pattern should be applied
        """
        raise NotImplementedError("Subclasses must implement should_apply()")
    
    def apply_pattern(self, transaction: Dict[str, Any], customer: Any,
                     transaction_history: List[Dict[str, Any]]) -> AnomalyIndicator:
        """
        Apply anomaly pattern to transaction
        
        Args:
            transaction: Current transaction
            customer: Customer profile
            transaction_history: List of customer's previous transactions
            
        Returns:
            AnomalyIndicator with anomaly details
        """
        raise NotImplementedError("Subclasses must implement apply_pattern()")
    
    def calculate_severity(self, deviation_ratio: float) -> float:
        """
        Calculate severity score based on deviation ratio
        
        Args:
            deviation_ratio: Ratio of deviation from normal (e.g., 1.5 = 50% higher)
            
        Returns:
            Severity score (0.0-1.0)
        """
        if deviation_ratio < 1.5:
            return 0.2  # Low severity
        elif deviation_ratio < 2.5:
            return 0.4  # Medium-low severity
        elif deviation_ratio < 4.0:
            return 0.6  # Medium-high severity
        elif deviation_ratio < 6.0:
            return 0.8  # High severity
        else:
            return 0.95  # Critical severity


class BehavioralAnomalyPattern(AnomalyPattern):
    """
    Detects out-of-character purchase behavior
    
    Indicators:
    - Category deviation: Purchasing from unusual categories
    - Amount spike: Spending 3-5x normal amount (not fraud level)
    - Merchant type change: Shopping at unusual merchant types
    - Payment method change: Using different payment methods
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.anomaly_type = AnomalyType.BEHAVIORAL
    
    def should_apply(self, transaction: Dict[str, Any], customer: Any,
                     transaction_history: List[Dict[str, Any]]) -> bool:
        """Check if behavioral anomaly should be applied"""
        # Need sufficient history to establish baseline behavior
        if len(transaction_history) < 10:
            return False
        
        # Apply to 25% of eligible transactions (will result in ~1-2% overall)
        return self.random.random() < 0.25
    
    def apply_pattern(self, transaction: Dict[str, Any], customer: Any,
                     transaction_history: List[Dict[str, Any]]) -> AnomalyIndicator:
        """Apply behavioral anomaly to transaction"""
        
        # Calculate baseline behavior from history
        recent_history = transaction_history[-30:]  # Last 30 transactions
        
        # Get category distribution
        category_counts = {}
        for txn in recent_history:
            cat = txn.get('Category', 'Unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Get average amount
        amounts = [txn.get('Amount', 0) for txn in recent_history]
        avg_amount = sum(amounts) / len(amounts) if amounts else 0
        
        # Get payment method distribution
        payment_methods = [txn.get('Payment_Mode', 'Unknown') for txn in recent_history]
        common_payment = max(set(payment_methods), key=payment_methods.count)
        
        # Choose anomaly type
        anomaly_choice = self.random.choice(['category', 'amount', 'payment_method'])
        
        if anomaly_choice == 'category':
            # Use a rare category (appears <10% of the time)
            rare_categories = [cat for cat, count in category_counts.items() 
                             if count / len(recent_history) < 0.1]
            
            if rare_categories:
                # Pick a rare category or completely new one
                if self.random.random() < 0.7 and rare_categories:
                    unusual_category = self.random.choice(rare_categories)
                else:
                    # Use a completely new category
                    all_categories = ['Groceries', 'Dining', 'Entertainment', 'Shopping',
                                    'Travel', 'Healthcare', 'Education', 'Utilities']
                    new_categories = [c for c in all_categories if c not in category_counts]
                    unusual_category = self.random.choice(new_categories) if new_categories else rare_categories[0]
                
                transaction['Category'] = unusual_category
                
                confidence = 0.6 + self.random.uniform(0, 0.2)
                severity = 0.4 + self.random.uniform(0, 0.3)
                
                return AnomalyIndicator(
                    anomaly_type=self.anomaly_type,
                    confidence=confidence,
                    reason=f"Out-of-character purchase: Rarely shops in {unusual_category} category",
                    evidence={
                        'unusual_category': unusual_category,
                        'category_frequency': category_counts.get(unusual_category, 0) / len(recent_history),
                        'transaction_count': len(recent_history)
                    },
                    severity=severity
                )
        
        elif anomaly_choice == 'amount':
            # Spending spike: 3-5x normal (not fraud level which would be 5-10x)
            multiplier = self.random.uniform(3.0, 5.0)
            new_amount = avg_amount * multiplier
            transaction['Amount'] = round(new_amount, 2)
            
            confidence = 0.5 + self.random.uniform(0, 0.25)
            severity = self.calculate_severity(multiplier)
            
            return AnomalyIndicator(
                anomaly_type=self.anomaly_type,
                confidence=confidence,
                reason=f"Unusual spending spike: {multiplier:.1f}x normal amount",
                evidence={
                    'current_amount': new_amount,
                    'avg_amount_30d': avg_amount,
                    'multiplier': multiplier
                },
                severity=severity
            )
        
        else:  # payment_method
            # Change payment method to something uncommon
            payment_modes = ['Credit Card', 'Debit Card', 'UPI', 'Net Banking', 'Wallet']
            uncommon_payments = [pm for pm in payment_modes if pm != common_payment]
            new_payment = self.random.choice(uncommon_payments)
            transaction['Payment_Mode'] = new_payment
            
            confidence = 0.5 + self.random.uniform(0, 0.2)
            severity = 0.3 + self.random.uniform(0, 0.2)
            
            return AnomalyIndicator(
                anomaly_type=self.anomaly_type,
                confidence=confidence,
                reason=f"Unusual payment method: Changed from {common_payment} to {new_payment}",
                evidence={
                    'usual_payment_method': common_payment,
                    'current_payment_method': new_payment,
                    'payment_method_change': True
                },
                severity=severity
            )


class GeographicAnomalyPattern(AnomalyPattern):
    """
    Detects unusual geographic patterns
    
    Indicators:
    - Impossible travel: >800 km/h but potentially legitimate (flights)
    - Cross-country transactions: Shopping far from home
    - Location spikes: Sudden distance increase
    - Travel frequency: Multiple cities in short time
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.anomaly_type = AnomalyType.GEOGRAPHIC
    
    def should_apply(self, transaction: Dict[str, Any], customer: Any,
                     transaction_history: List[Dict[str, Any]]) -> bool:
        """Check if geographic anomaly should be applied"""
        # Need at least one previous transaction to calculate distance
        if len(transaction_history) < 1:
            return False
        
        # Apply to 20% of eligible transactions
        return self.random.random() < 0.20
    
    def apply_pattern(self, transaction: Dict[str, Any], customer: Any,
                     transaction_history: List[Dict[str, Any]]) -> AnomalyIndicator:
        """Apply geographic anomaly to transaction"""
        
        # Get recent location
        recent_txn = transaction_history[-1]
        recent_city = recent_txn.get('City', customer.city)
        
        # Define city locations (simplified coordinates for distance calculation)
        city_coords = {
            'Mumbai': (19.0760, 72.8777),
            'Delhi': (28.7041, 77.1025),
            'Bangalore': (12.9716, 77.5946),
            'Kolkata': (22.5726, 88.3639),
            'Chennai': (13.0827, 80.2707),
            'Hyderabad': (17.3850, 78.4867),
            'Pune': (18.5204, 73.8567),
            'Ahmedabad': (23.0225, 72.5714),
            'Jaipur': (26.9124, 75.7873),
            'Lucknow': (26.8467, 80.9462),
            'Chandigarh': (30.7333, 76.7794),
            'Kochi': (9.9312, 76.2673),
            'Indore': (22.7196, 75.8577),
            'Bhopal': (23.2599, 77.4126),
            'Patna': (25.5941, 85.1376),
            'Nagpur': (21.1458, 79.0882),
            'Surat': (21.1702, 72.8311),
            'Visakhapatnam': (17.6868, 83.2185),
            'Guwahati': (26.1445, 91.7362),
            'Bhubaneswar': (20.2961, 85.8245)
        }
        
        # Calculate distance between cities
        def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
            """Calculate distance between two coordinates in kilometers"""
            lat1, lon1 = coord1
            lat2, lon2 = coord2
            
            # Convert to radians
            lat1_rad = math.radians(lat1)
            lat2_rad = math.radians(lat2)
            delta_lat = math.radians(lat2 - lat1)
            delta_lon = math.radians(lon2 - lon1)
            
            # Haversine formula
            a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            r = 6371  # Earth's radius in kilometers
            
            return c * r
        
        # Choose a distant city
        current_city = transaction.get('City', recent_city)
        distant_cities = [city for city in city_coords.keys() if city != recent_city]
        
        if distant_cities and recent_city in city_coords:
            # Pick a city that's far away
            new_city = self.random.choice(distant_cities)
            transaction['City'] = new_city
            
            # Calculate distance and travel time
            distance_km = haversine_distance(city_coords[recent_city], city_coords[new_city])
            
            # Calculate time difference
            recent_time = datetime.strptime(recent_txn.get('Transaction_Date', '2025-01-01'), '%Y-%m-%d')
            current_time = datetime.strptime(transaction.get('Transaction_Date', '2025-01-01'), '%Y-%m-%d')
            time_diff_hours = max((current_time - recent_time).total_seconds() / 3600, 0.1)
            
            # Calculate implied travel speed
            speed_kmh = distance_km / time_diff_hours if time_diff_hours > 0 else 0
            
            # Determine severity based on speed
            if speed_kmh > 2000:
                severity = 0.9
                reason = f"Impossible travel: {recent_city} to {new_city} ({distance_km:.0f}km) in {time_diff_hours:.1f} hours ({speed_kmh:.0f} km/h)"
            elif speed_kmh > 800:
                severity = 0.7
                reason = f"Very fast travel: {recent_city} to {new_city} ({distance_km:.0f}km) in {time_diff_hours:.1f} hours (possible flight)"
            else:
                severity = 0.5
                reason = f"Unusual location: Transaction in {new_city}, far from usual location {recent_city}"
            
            confidence = 0.6 + self.random.uniform(0, 0.3)
            
            return AnomalyIndicator(
                anomaly_type=self.anomaly_type,
                confidence=confidence,
                reason=reason,
                evidence={
                    'previous_city': recent_city,
                    'current_city': new_city,
                    'distance_km': round(distance_km, 1),
                    'time_diff_hours': round(time_diff_hours, 2),
                    'implied_speed_kmh': round(speed_kmh, 1)
                },
                severity=severity
            )
        
        # Fallback: generic location anomaly
        confidence = 0.5
        severity = 0.4
        return AnomalyIndicator(
            anomaly_type=self.anomaly_type,
            confidence=confidence,
            reason=f"Unusual location detected",
            evidence={'current_city': current_city},
            severity=severity
        )


class TemporalAnomalyPattern(AnomalyPattern):
    """
    Detects unusual temporal patterns
    
    Indicators:
    - Unusual hours: Transactions outside normal hours
    - Schedule changes: Weekday vs. weekend deviations
    - Holiday anomalies: Shopping on unusual days
    - Time clustering: Multiple transactions in short period
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.anomaly_type = AnomalyType.TEMPORAL
    
    def should_apply(self, transaction: Dict[str, Any], customer: Any,
                     transaction_history: List[Dict[str, Any]]) -> bool:
        """Check if temporal anomaly should be applied"""
        # Need history to establish baseline
        if len(transaction_history) < 10:
            return False
        
        # Apply to 20% of eligible transactions
        return self.random.random() < 0.20
    
    def apply_pattern(self, transaction: Dict[str, Any], customer: Any,
                     transaction_history: List[Dict[str, Any]]) -> AnomalyIndicator:
        """Apply temporal anomaly to transaction"""
        
        # Analyze historical hour patterns
        recent_history = transaction_history[-30:]
        hour_counts = {}
        for txn in recent_history:
            hour = txn.get('Hour', 12)
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        # Find uncommon hours (appear <10% of the time)
        total_txns = len(recent_history)
        uncommon_hours = [hour for hour, count in hour_counts.items() 
                         if count / total_txns < 0.1]
        
        # Also consider hours never used before
        all_hours = set(range(24))
        used_hours = set(hour_counts.keys())
        unused_hours = list(all_hours - used_hours)
        
        # Choose unusual hour
        if unused_hours and self.random.random() < 0.6:
            # Use a completely new hour (higher severity)
            unusual_hour = self.random.choice(unused_hours)
            is_new_hour = True
        elif uncommon_hours:
            # Use an uncommon hour
            unusual_hour = self.random.choice(uncommon_hours)
            is_new_hour = False
        else:
            # Fallback to late night hours
            unusual_hour = self.random.choice([2, 3, 4, 5])
            is_new_hour = False
        
        # Update transaction hour
        transaction['Hour'] = unusual_hour
        transaction['Time'] = f"{unusual_hour:02d}:{self.random.randint(0, 59):02d}:00"
        
        # Determine if it's unusual based on hour
        if unusual_hour in [0, 1, 2, 3, 4, 5]:
            hour_desc = "late night"
            severity = 0.7
        elif unusual_hour in [6, 7, 8]:
            hour_desc = "early morning"
            severity = 0.5
        elif unusual_hour in [22, 23]:
            hour_desc = "very late evening"
            severity = 0.6
        else:
            hour_desc = "unusual time"
            severity = 0.4
        
        # Boost severity if never used before
        if is_new_hour:
            severity = min(severity + 0.2, 1.0)
        
        confidence = 0.6 + self.random.uniform(0, 0.2)
        
        return AnomalyIndicator(
            anomaly_type=self.anomaly_type,
            confidence=confidence,
            reason=f"Unusual transaction time: Shopping at {hour_desc} ({unusual_hour:02d}:00)",
            evidence={
                'transaction_hour': unusual_hour,
                'hour_frequency': hour_counts.get(unusual_hour, 0) / total_txns,
                'is_new_hour': is_new_hour,
                'common_hours': sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            },
            severity=severity
        )


class AmountAnomalyPattern(AnomalyPattern):
    """
    Detects unusual transaction amount patterns
    
    Indicators:
    - Spending spikes: 3-5x normal amount (not fraud level)
    - Micro-transactions: Very small amounts (<Rs. 50)
    - Round amounts: Exact multiples (Rs. 1000, Rs. 5000)
    - Budget changes: Sudden shift in spending tier
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.anomaly_type = AnomalyType.AMOUNT
    
    def should_apply(self, transaction: Dict[str, Any], customer: Any,
                     transaction_history: List[Dict[str, Any]]) -> bool:
        """Check if amount anomaly should be applied"""
        # Need history to establish baseline
        if len(transaction_history) < 10:
            return False
        
        # Apply to 25% of eligible transactions
        return self.random.random() < 0.25
    
    def apply_pattern(self, transaction: Dict[str, Any], customer: Any,
                     transaction_history: List[Dict[str, Any]]) -> AnomalyIndicator:
        """Apply amount anomaly to transaction"""
        
        # Calculate baseline spending
        recent_history = transaction_history[-30:]
        amounts = [txn.get('Amount', 0) for txn in recent_history]
        avg_amount = sum(amounts) / len(amounts) if amounts else 1000
        min_amount = min(amounts) if amounts else 100
        max_amount = max(amounts) if amounts else 5000
        
        # Choose anomaly type
        anomaly_choice = self.random.choice(['spike', 'micro', 'round_amount'])
        
        if anomaly_choice == 'spike':
            # Spending spike: 3-5x normal (not fraud level)
            multiplier = self.random.uniform(3.0, 5.0)
            new_amount = avg_amount * multiplier
            transaction['Amount'] = round(new_amount, 2)
            
            confidence = 0.6 + self.random.uniform(0, 0.2)
            severity = self.calculate_severity(multiplier)
            
            return AnomalyIndicator(
                anomaly_type=self.anomaly_type,
                confidence=confidence,
                reason=f"Unusual spending spike: Rs. {new_amount:.2f} ({multiplier:.1f}x normal)",
                evidence={
                    'current_amount': new_amount,
                    'avg_amount_30d': avg_amount,
                    'multiplier': multiplier,
                    'previous_max': max_amount
                },
                severity=severity
            )
        
        elif anomaly_choice == 'micro':
            # Micro-transaction: Very small amount
            micro_amount = self.random.uniform(10, 50)
            transaction['Amount'] = round(micro_amount, 2)
            
            # Calculate how unusual this is
            ratio = avg_amount / micro_amount if micro_amount > 0 else 1
            severity = min(0.3 + (ratio / 100), 0.8)
            confidence = 0.5 + self.random.uniform(0, 0.2)
            
            return AnomalyIndicator(
                anomaly_type=self.anomaly_type,
                confidence=confidence,
                reason=f"Unusually small transaction: Rs. {micro_amount:.2f} (avg: Rs. {avg_amount:.2f})",
                evidence={
                    'current_amount': micro_amount,
                    'avg_amount_30d': avg_amount,
                    'ratio_to_avg': ratio,
                    'previous_min': min_amount
                },
                severity=severity
            )
        
        else:  # round_amount
            # Round amount clustering
            round_amounts = [1000, 2000, 5000, 10000, 15000, 20000, 25000, 50000]
            # Filter to amounts within reasonable range
            suitable_amounts = [amt for amt in round_amounts if avg_amount * 0.5 <= amt <= avg_amount * 4]
            
            if suitable_amounts:
                round_amount = self.random.choice(suitable_amounts)
            else:
                round_amount = round(avg_amount / 1000) * 1000
            
            transaction['Amount'] = float(round_amount)
            
            # Check how different this is from average
            ratio = abs(round_amount - avg_amount) / avg_amount if avg_amount > 0 else 1
            severity = min(0.3 + ratio, 0.7)
            confidence = 0.5 + self.random.uniform(0, 0.15)
            
            return AnomalyIndicator(
                anomaly_type=self.anomaly_type,
                confidence=confidence,
                reason=f"Unusual round amount: Rs. {round_amount} (exactly)",
                evidence={
                    'current_amount': round_amount,
                    'avg_amount_30d': avg_amount,
                    'is_round_amount': True,
                    'deviation_from_avg': abs(round_amount - avg_amount)
                },
                severity=severity
            )


class AnomalyPatternGenerator:
    """
    Main orchestration class for anomaly pattern generation
    
    This class manages multiple anomaly patterns and applies them to transactions
    based on configurable rates. It tracks statistics and ensures realistic
    anomaly distributions across behavioral, geographic, temporal, and amount categories.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize anomaly pattern generator
        
        Args:
            seed: Random seed for reproducibility
        """
        self.random = random.Random(seed)
        
        # Initialize all anomaly patterns
        self.patterns = [
            BehavioralAnomalyPattern(seed),
            GeographicAnomalyPattern(seed),
            TemporalAnomalyPattern(seed),
            AmountAnomalyPattern(seed)
        ]
        
        # Statistics tracking
        self.stats = {
            'total_transactions': 0,
            'anomaly_count': 0,
            'anomaly_rate': 0.0,
            'anomalies_by_type': {pattern.anomaly_type.value: 0 for pattern in self.patterns}
        }
    
    def inject_anomaly_patterns(self, transactions: List[Dict[str, Any]], 
                               customers: List[Any],
                               anomaly_rate: float = 0.05) -> List[Dict[str, Any]]:
        """
        Inject anomaly patterns into transactions
        
        Args:
            transactions: List of transactions to inject anomalies into
            customers: List of customer profiles
            anomaly_rate: Target rate of anomalies (default: 0.05 = 5%)
            
        Returns:
            List of transactions with anomalies injected
        """
        # Clamp anomaly rate to reasonable range
        anomaly_rate = max(0.0, min(1.0, anomaly_rate))
        
        # Build customer lookup
        customer_lookup = {customer.customer_id: customer for customer in customers}
        
        # Build transaction history per customer
        transaction_history = {}
        
        # Process transactions
        for i, transaction in enumerate(transactions):
            customer_id = transaction.get('Customer_ID')
            customer = customer_lookup.get(customer_id)
            
            if not customer:
                continue
            
            # Get customer's transaction history
            history = transaction_history.get(customer_id, [])
            
            # Decide if we should inject an anomaly
            should_inject = self.random.random() < anomaly_rate
            
            if should_inject and history:  # Need history for most anomalies
                # Choose a random applicable pattern
                applicable_patterns = [
                    pattern for pattern in self.patterns
                    if pattern.should_apply(transaction, customer, history)
                ]
                
                if applicable_patterns:
                    # Select random pattern
                    pattern = self.random.choice(applicable_patterns)
                    
                    # Apply the pattern
                    anomaly_indicator = pattern.apply_pattern(transaction, customer, history)
                    
                    # Only add anomaly if indicator was successfully created
                    if anomaly_indicator:
                        # Add anomaly fields to transaction
                        transaction['Anomaly_Type'] = anomaly_indicator.anomaly_type.value
                        transaction['Anomaly_Confidence'] = anomaly_indicator.confidence
                        transaction['Anomaly_Reason'] = anomaly_indicator.reason
                        transaction['Anomaly_Severity'] = anomaly_indicator.severity
                        transaction['Anomaly_Evidence'] = json.dumps(anomaly_indicator.evidence)
                        
                        # Update statistics
                        self.stats['anomaly_count'] += 1
                        self.stats['anomalies_by_type'][anomaly_indicator.anomaly_type.value] += 1
            
            # Update transaction history
            if customer_id not in transaction_history:
                transaction_history[customer_id] = []
            transaction_history[customer_id].append(transaction.copy())
        
        # Update statistics
        self.stats['total_transactions'] = len(transactions)
        if len(transactions) > 0:
            self.stats['anomaly_rate'] = self.stats['anomaly_count'] / len(transactions)
        
        return transactions
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get anomaly generation statistics"""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset anomaly generation statistics"""
        self.stats = {
            'total_transactions': 0,
            'anomaly_count': 0,
            'anomaly_rate': 0.0,
            'anomalies_by_type': {pattern.anomaly_type.value: 0 for pattern in self.patterns}
        }


def apply_anomaly_labels(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add anomaly label fields to transactions without anomalies
    
    Args:
        transactions: List of transactions
        
    Returns:
        List of transactions with anomaly label fields added
    """
    for transaction in transactions:
        if 'Anomaly_Type' not in transaction:
            transaction['Anomaly_Type'] = AnomalyType.NONE.value
            transaction['Anomaly_Confidence'] = 0.0
            transaction['Anomaly_Reason'] = "No anomaly detected"
            transaction['Anomaly_Severity'] = 0.0
            transaction['Anomaly_Evidence'] = json.dumps({})
    
    return transactions
