"""
Fraud Pattern Generator

This module implements sophisticated fraud detection patterns for synthetic transaction data.
Each pattern represents a realistic fraud scenario with specific characteristics, timing,
and behavioral signatures that can be used to train ML fraud detection models.

Base Patterns (Days 1-2):
1. Card Cloning - Rapid transactions in distant locations
2. Account Takeover - Sudden behavioral changes
3. Merchant Collusion - Suspicious merchant patterns
4. Velocity Abuse - Excessive transaction frequency
5. Amount Manipulation - Just-below-limit transactions
6. Refund Fraud - Suspicious refund patterns
7. Stolen Card - Stolen card usage patterns
8. Synthetic Identity - Fabricated identity indicators
9. First Party Fraud - Customer fraud behaviors
10. Friendly Fraud - Chargeback abuse patterns

Advanced Patterns (Days 3-4):
11. Transaction Replay - Duplicate transaction detection
12. Card Testing - Small test transactions before large fraud
13. Mule Account - Money laundering patterns
14. Shipping Fraud - Address manipulation detection
15. Loyalty Abuse - Points/rewards exploitation

Author: SynFinance Team
Version: 0.5.0-dev
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import random


class FraudType(Enum):
    """Enumeration of fraud pattern types."""
    CARD_CLONING = "Card Cloning"
    ACCOUNT_TAKEOVER = "Account Takeover"
    MERCHANT_COLLUSION = "Merchant Collusion"
    VELOCITY_ABUSE = "Velocity Abuse"
    AMOUNT_MANIPULATION = "Amount Manipulation"
    REFUND_FRAUD = "Refund Fraud"
    STOLEN_CARD = "Stolen Card"
    SYNTHETIC_IDENTITY = "Synthetic Identity"
    FIRST_PARTY_FRAUD = "First Party Fraud"
    FRIENDLY_FRAUD = "Friendly Fraud"
    # Advanced patterns (Week 4 Days 3-4)
    TRANSACTION_REPLAY = "Transaction Replay"
    CARD_TESTING = "Card Testing"
    MULE_ACCOUNT = "Mule Account"
    SHIPPING_FRAUD = "Shipping Fraud"
    LOYALTY_ABUSE = "Loyalty Program Abuse"
    COMBINED = "Combined Patterns"


@dataclass
class FraudIndicator:
    """
    Represents a fraud indicator with evidence and confidence.
    
    Attributes:
        fraud_type: Type of fraud detected
        confidence: Confidence score (0.0-1.0)
        reason: Detailed explanation of fraud indicators
        evidence: Dictionary of supporting evidence
        severity: Severity level (low, medium, high, critical)
    """
    fraud_type: FraudType
    confidence: float
    reason: str
    evidence: Dict[str, Any]
    severity: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fraud indicator to dictionary."""
        return {
            'fraud_type': self.fraud_type.value,
            'confidence': self.confidence,
            'reason': self.reason,
            'evidence': self.evidence,
            'severity': self.severity
        }


class FraudPattern:
    """
    Base class for fraud patterns.
    
    Each fraud pattern implements specific logic to:
    1. Detect if pattern should be applied to a transaction
    2. Modify transaction to exhibit fraud characteristics
    3. Generate fraud labels and explanations
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize fraud pattern with optional seed."""
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    def should_apply(self, customer: Any, transaction: Dict[str, Any], 
                     customer_history: List[Dict[str, Any]]) -> bool:
        """
        Determine if this fraud pattern should be applied.
        
        Args:
            customer: Customer profile
            transaction: Current transaction being generated
            customer_history: List of customer's previous transactions
            
        Returns:
            True if pattern should be applied, False otherwise
        """
        raise NotImplementedError("Subclasses must implement should_apply()")
    
    def apply_pattern(self, transaction: Dict[str, Any], 
                     customer: Any,
                     customer_history: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], FraudIndicator]:
        """
        Apply fraud pattern to transaction and generate fraud indicators.
        
        Args:
            transaction: Transaction to modify
            customer: Customer profile
            customer_history: Customer's transaction history
            
        Returns:
            Tuple of (modified transaction, fraud indicator)
        """
        raise NotImplementedError("Subclasses must implement apply_pattern()")
    
    def calculate_confidence(self, evidence: Dict[str, Any]) -> float:
        """
        Calculate confidence score based on evidence strength.
        
        Args:
            evidence: Dictionary of evidence metrics
            
        Returns:
            Confidence score (0.0-1.0)
        """
        raise NotImplementedError("Subclasses must implement calculate_confidence()")


class CardCloningPattern(FraudPattern):
    """
    Card Cloning Fraud Pattern
    
    Characteristics:
    - Multiple transactions in geographically distant locations within short timeframe
    - Impossible travel time between transactions
    - Often same amounts or round amounts
    - May occur at unusual hours
    
    Example: Transaction in Mumbai at 10:00 AM, then Delhi at 10:30 AM (impossible)
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.min_distance_km = 500  # Minimum distance for cloning suspicion
        self.max_time_hours = 2     # Maximum time between transactions
        self.impossible_speed_kmh = 800  # Speed threshold for impossible travel
    
    def should_apply(self, customer: Any, transaction: Dict[str, Any],
                     customer_history: List[Dict[str, Any]]) -> bool:
        """Card cloning typically affects any customer with recent transaction history."""
        return len(customer_history) > 0 and random.random() < 0.3
    
    def apply_pattern(self, transaction: Dict[str, Any], customer: Any,
                     customer_history: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], FraudIndicator]:
        """Apply card cloning pattern to transaction."""
        
        # Get last transaction for distance/time calculation
        last_txn = customer_history[-1]
        last_city = last_txn.get('City', customer.city)
        last_time = datetime.fromisoformat(last_txn['Date'] + ' ' + last_txn['Time'])
        current_time = datetime.fromisoformat(transaction['Date'] + ' ' + transaction['Time'])
        
        # Calculate time difference in hours
        time_diff_hours = (current_time - last_time).total_seconds() / 3600
        
        # Select distant city (different from home and last transaction city)
        distant_cities = ['Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Chennai', 
                         'Hyderabad', 'Pune', 'Ahmedabad']
        distant_cities = [c for c in distant_cities if c != last_city and c != customer.city]
        fraud_city = random.choice(distant_cities) if distant_cities else 'Mumbai'
        
        # Modify transaction to exhibit cloning characteristics
        transaction['City'] = fraud_city
        transaction['Distance_From_Home_km'] = random.uniform(800, 2000)
        
        # Often round amounts or same as previous
        if random.random() < 0.4:
            transaction['Amount'] = round(transaction['Amount'] / 1000) * 1000
        elif random.random() < 0.5:
            transaction['Amount'] = last_txn.get('Amount', transaction['Amount'])
        
        # Calculate distance (approximate)
        distance_km = transaction['Distance_From_Home_km']
        travel_speed = distance_km / time_diff_hours if time_diff_hours > 0 else 9999
        
        # Build evidence
        evidence = {
            'previous_city': last_city,
            'current_city': fraud_city,
            'distance_km': distance_km,
            'time_diff_hours': round(time_diff_hours, 2),
            'travel_speed_kmh': round(travel_speed, 2),
            'impossible_travel': travel_speed > self.impossible_speed_kmh,
            'amount_pattern': 'round' if transaction['Amount'] % 1000 == 0 else 'duplicate' if transaction['Amount'] == last_txn.get('Amount') else 'normal'
        }
        
        # Calculate confidence
        confidence = self.calculate_confidence(evidence)
        
        # Determine severity
        if travel_speed > 2000:
            severity = 'critical'
        elif travel_speed > self.impossible_speed_kmh:
            severity = 'high'
        elif distance_km > 1000 and time_diff_hours < 4:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Create fraud indicator
        reason = (f"Card used in {fraud_city} {round(time_diff_hours, 1)}h after transaction "
                 f"in {last_city} ({round(distance_km, 0)}km away). "
                 f"Required travel speed: {round(travel_speed, 0)} km/h. "
                 f"Physically impossible travel time suggests card cloning.")
        
        fraud_indicator = FraudIndicator(
            fraud_type=FraudType.CARD_CLONING,
            confidence=confidence,
            reason=reason,
            evidence=evidence,
            severity=severity
        )
        
        # Mark as fraud
        transaction['Is_Fraud'] = 1
        transaction['Fraud_Type'] = FraudType.CARD_CLONING.value
        
        return transaction, fraud_indicator
    
    def calculate_confidence(self, evidence: Dict[str, Any]) -> float:
        """Calculate confidence based on travel impossibility."""
        speed = evidence['travel_speed_kmh']
        distance = evidence['distance_km']
        time_diff = evidence['time_diff_hours']
        
        confidence = 0.5  # Base confidence
        
        # Impossible travel adds high confidence
        if speed > 2000:
            confidence += 0.4
        elif speed > self.impossible_speed_kmh:
            confidence += 0.3
        elif speed > 400:
            confidence += 0.2
        
        # Long distance in short time adds confidence
        if distance > 1000 and time_diff < 3:
            confidence += 0.1
        
        # Amount patterns add confidence
        if evidence['amount_pattern'] in ['round', 'duplicate']:
            confidence += 0.1
        
        return min(confidence, 1.0)


# ---------------------------------------------------------------------------
# Fraud Combination Generator
# ---------------------------------------------------------------------------
class FraudCombinationGenerator:
    """
    Combine multiple fraud patterns into a single, coherent modification and
    aggregated fraud indicator.

    This generator applies a list of patterns sequentially to a transaction and
    merges their evidence and confidence scores into a single FraudIndicator
    with type `FraudType.COMBINED`.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def combine_and_apply(self, transaction: Dict[str, Any],
                          patterns: List[FraudPattern],
                          customer: Any,
                          customer_history: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], FraudIndicator]:
        """Apply a set of patterns to the same transaction and return combined result.

        Args:
            transaction: base transaction dict (a copy should be provided by caller)
            patterns: list of FraudPattern instances to apply (order matters)
            customer: customer profile
            customer_history: list of historical transactions

        Returns:
            (modified_transaction, aggregated_fraud_indicator)
        """
        applied_indicators: List[FraudIndicator] = []
        modified_txn = transaction

        # Apply each pattern in sequence. Patterns mutate the transaction and
        # produce a FraudIndicator which we collect.
        for pattern in patterns:
            try:
                modified_txn, indicator = pattern.apply_pattern(modified_txn, customer, customer_history)
            except Exception:
                # If a pattern fails to apply, skip it to be robust in combination
                continue
            if indicator:
                applied_indicators.append(indicator)

        # If no indicators, return original transaction marked as no fraud
        if not applied_indicators:
            return transaction, FraudIndicator(
                fraud_type=FraudType.COMBINED,
                confidence=0.0,
                reason="No combined indicators",
                evidence={},
                severity='none'
            )

        # Merge evidence dictionaries (later indicators override earlier keys)
        merged_evidence: Dict[str, Any] = {}
        for ind in applied_indicators:
            merged_evidence.update(ind.evidence)

        # Combine confidence scores: P = 1 - product(1 - ci)
        confidences = [ind.confidence for ind in applied_indicators]
        combined_confidence = self._combine_confidences(confidences)

        # Combine severity -> pick the highest severity
        severities = [ind.severity for ind in applied_indicators]
        combined_severity = self._max_severity(severities)

        # Build combined reason string
        reasons = [f"[{ind.fraud_type.value}: {ind.reason}]" for ind in applied_indicators]
        combined_reason = " | ".join(reasons)

        combined_indicator = FraudIndicator(
            fraud_type=FraudType.COMBINED,
            confidence=round(combined_confidence, 3),
            reason=combined_reason,
            evidence=merged_evidence,
            severity=combined_severity
        )

        # Ensure transaction marked as fraud and set a combined type
        modified_txn['Is_Fraud'] = 1
        modified_txn['Fraud_Type'] = FraudType.COMBINED.value

        return modified_txn, combined_indicator

    def _combine_confidences(self, confidences: List[float]) -> float:
        """Combine confidences using probabilistic union: 1 - prod(1 - c_i)."""
        prod = 1.0
        for c in confidences:
            prod *= (1.0 - max(0.0, min(1.0, c)))
        return 1.0 - prod

    def _max_severity(self, severities: List[str]) -> str:
        """Return the maximum severity given textual severities."""
        order = ['none', 'low', 'medium', 'high', 'critical']
        max_idx = 0
        for s in severities:
            try:
                idx = order.index(s)
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                continue
        return order[max_idx]

    def apply_chained(self, transaction: Dict[str, Any],
                     pattern_sequence: List[FraudPattern],
                     customer: Any,
                     customer_history: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], FraudIndicator]:
        """
        Apply chained fraud patterns in sequence (e.g., account takeover -> velocity abuse).
        
        Chained fraud means the first pattern enables or leads to the second pattern.
        Each pattern builds on the previous one's modifications.
        
        Args:
            transaction: base transaction
            pattern_sequence: ordered list of patterns to apply in sequence
            customer: customer profile
            customer_history: transaction history
            
        Returns:
            (modified_transaction, combined_indicator)
        """
        if not pattern_sequence:
            return transaction, FraudIndicator(
                fraud_type=FraudType.COMBINED,
                confidence=0.0,
                reason="No chained patterns provided",
                evidence={},
                severity='none'
            )
        
        # Apply patterns in strict sequence
        modified_txn, combined_indicator = self.combine_and_apply(
            transaction, pattern_sequence, customer, customer_history
        )
        
        # Add chained metadata to evidence
        if combined_indicator and combined_indicator.evidence:
            combined_indicator.evidence['combination_type'] = 'chained'
            combined_indicator.evidence['chain_length'] = len(pattern_sequence)
            combined_indicator.evidence['chain_sequence'] = [
                type(p).__name__ for p in pattern_sequence
            ]
        
        # Boost confidence slightly for successful chaining
        if combined_indicator and combined_indicator.confidence > 0:
            boosted_confidence = min(1.0, combined_indicator.confidence * 1.1)
            combined_indicator = FraudIndicator(
                fraud_type=combined_indicator.fraud_type,
                confidence=round(boosted_confidence, 3),
                reason=f"[CHAINED FRAUD] {combined_indicator.reason}",
                evidence=combined_indicator.evidence,
                severity=combined_indicator.severity
            )
        
        return modified_txn, combined_indicator

    def apply_coordinated(self, transaction: Dict[str, Any],
                         patterns: List[FraudPattern],
                         customer: Any,
                         customer_history: List[Dict[str, Any]],
                         coordination_metadata: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], FraudIndicator]:
        """
        Apply coordinated fraud patterns (multiple actors/cards working together).
        
        Coordinated fraud involves multiple entities (merchants, cards, accounts)
        working together in a fraud ring or network.
        
        Args:
            transaction: base transaction
            patterns: list of patterns representing different fraud actors
            customer: customer profile
            customer_history: transaction history
            coordination_metadata: optional dict with coordination details
                (e.g., shared_merchants, shared_locations, time_clustering)
            
        Returns:
            (modified_transaction, combined_indicator)
        """
        modified_txn, combined_indicator = self.combine_and_apply(
            transaction, patterns, customer, customer_history
        )
        
        # Add coordination metadata
        if combined_indicator and combined_indicator.evidence:
            combined_indicator.evidence['combination_type'] = 'coordinated'
            combined_indicator.evidence['coordination_actors'] = len(patterns)
            
            if coordination_metadata:
                combined_indicator.evidence.update(coordination_metadata)
        
        # Increase severity for coordinated fraud (more sophisticated)
        if combined_indicator:
            current_severity = combined_indicator.severity
            severity_order = ['none', 'low', 'medium', 'high', 'critical']
            try:
                current_idx = severity_order.index(current_severity)
                # Bump up one level for coordination
                new_idx = min(len(severity_order) - 1, current_idx + 1)
                elevated_severity = severity_order[new_idx]
            except (ValueError, IndexError):
                elevated_severity = current_severity
            
            combined_indicator = FraudIndicator(
                fraud_type=combined_indicator.fraud_type,
                confidence=combined_indicator.confidence,
                reason=f"[COORDINATED FRAUD] {combined_indicator.reason}",
                evidence=combined_indicator.evidence,
                severity=elevated_severity
            )
        
        return modified_txn, combined_indicator

    def apply_progressive(self, transaction: Dict[str, Any],
                         patterns: List[FraudPattern],
                         customer: Any,
                         customer_history: List[Dict[str, Any]],
                         sophistication_level: float = 0.5) -> Tuple[Dict[str, Any], FraudIndicator]:
        """
        Apply progressive fraud patterns (escalating sophistication over time).
        
        Progressive fraud means the fraudster starts with simple patterns and
        gradually escalates to more sophisticated ones as they test defenses.
        
        Args:
            transaction: base transaction
            patterns: list of patterns ordered by increasing sophistication
            customer: customer profile
            customer_history: transaction history
            sophistication_level: 0.0-1.0 indicating how far through progression
            
        Returns:
            (modified_transaction, combined_indicator)
        """
        if not patterns:
            return transaction, FraudIndicator(
                fraud_type=FraudType.COMBINED,
                confidence=0.0,
                reason="No progressive patterns provided",
                evidence={},
                severity='none'
            )
        
        # Select subset of patterns based on sophistication level
        num_patterns = max(1, int(len(patterns) * sophistication_level))
        selected_patterns = patterns[:num_patterns]
        
        modified_txn, combined_indicator = self.combine_and_apply(
            transaction, selected_patterns, customer, customer_history
        )
        
        # Add progressive metadata
        if combined_indicator and combined_indicator.evidence:
            combined_indicator.evidence['combination_type'] = 'progressive'
            combined_indicator.evidence['sophistication_level'] = round(sophistication_level, 2)
            combined_indicator.evidence['patterns_applied'] = num_patterns
            combined_indicator.evidence['max_patterns'] = len(patterns)
            combined_indicator.evidence['progression_stage'] = (
                'early' if sophistication_level < 0.33 else
                'intermediate' if sophistication_level < 0.67 else
                'advanced'
            )
        
        # Scale confidence by sophistication level
        if combined_indicator and combined_indicator.confidence > 0:
            scaled_confidence = combined_indicator.confidence * (0.7 + 0.3 * sophistication_level)
            combined_indicator = FraudIndicator(
                fraud_type=combined_indicator.fraud_type,
                confidence=round(scaled_confidence, 3),
                reason=f"[PROGRESSIVE FRAUD] {combined_indicator.reason}",
                evidence=combined_indicator.evidence,
                severity=combined_indicator.severity
            )
        
        return modified_txn, combined_indicator



class AccountTakeoverPattern(FraudPattern):
    """
    Account Takeover Fraud Pattern
    
    Characteristics:
    - Sudden change in spending patterns (much higher amounts)
    - Different merchant types than usual
    - Transactions at unusual times
    - Different location than normal
    - Multiple rapid transactions
    
    Example: Budget-conscious customer suddenly makes luxury purchases
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.min_history_length = 10  # Need history to establish baseline
    
    def should_apply(self, customer: Any, transaction: Dict[str, Any],
                     customer_history: List[Dict[str, Any]]) -> bool:
        """Account takeover requires established customer history."""
        return len(customer_history) >= self.min_history_length and random.random() < 0.25
    
    def apply_pattern(self, transaction: Dict[str, Any], customer: Any,
                     customer_history: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], FraudIndicator]:
        """Apply account takeover pattern."""
        
        # Calculate baseline behavior
        avg_amount = sum(t.get('Amount', 0) for t in customer_history) / len(customer_history)
        common_categories = {}
        for t in customer_history:
            cat = t.get('Category', 'Unknown')
            common_categories[cat] = common_categories.get(cat, 0) + 1
        most_common_category = max(common_categories, key=common_categories.get)
        
        home_city_count = sum(1 for t in customer_history if t.get('City') == customer.city)
        home_city_ratio = home_city_count / len(customer_history)
        
        # Apply dramatic behavioral changes
        # 1. Much higher amount (3-10x average)
        multiplier = random.uniform(3, 10)
        transaction['Amount'] = avg_amount * multiplier
        
        # 2. Unusual category (luxury items if budget-conscious, or expensive electronics)
        unusual_categories = ['Electronics', 'Jewelry', 'Travel', 'Entertainment']
        unusual_categories = [c for c in unusual_categories if c != most_common_category]
        transaction['Category'] = random.choice(unusual_categories) if unusual_categories else 'Electronics'
        
        # 3. Unusual location if customer is typically home-based
        if home_city_ratio > 0.7:
            distant_cities = ['Mumbai', 'Delhi', 'Bangalore']
            transaction['City'] = random.choice([c for c in distant_cities if c != customer.city])
            transaction['Distance_From_Home_km'] = random.uniform(500, 1500)
        
        # 4. Unusual time (late night/early morning)
        if random.random() < 0.6:
            unusual_hour = random.choice([0, 1, 2, 3, 4, 23])
            transaction['Hour'] = unusual_hour
            transaction['Time'] = f"{unusual_hour:02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}"
        
        # Build evidence
        evidence = {
            'baseline_avg_amount': round(avg_amount, 2),
            'current_amount': round(transaction['Amount'], 2),
            'amount_multiplier': round(multiplier, 2),
            'baseline_category': most_common_category,
            'current_category': transaction['Category'],
            'home_city_ratio': round(home_city_ratio, 2),
            'current_city': transaction['City'],
            'transaction_hour': transaction['Hour'],
            'unusual_hour': transaction['Hour'] < 6 or transaction['Hour'] > 22
        }
        
        # Calculate confidence
        confidence = self.calculate_confidence(evidence)
        
        # Determine severity
        if multiplier > 7 and evidence['unusual_hour']:
            severity = 'critical'
        elif multiplier > 5:
            severity = 'high'
        elif multiplier > 3:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Create fraud indicator
        reason = (f"Account takeover suspected. Transaction amount ${round(transaction['Amount'], 2)} "
                 f"is {round(multiplier, 1)}x higher than baseline ${round(avg_amount, 2)}. "
                 f"Unusual category '{transaction['Category']}' (baseline: '{most_common_category}'). ")
        
        if evidence['unusual_hour']:
            reason += f"Transaction at unusual hour ({transaction['Hour']}:00). "
        if home_city_ratio > 0.7 and transaction['City'] != customer.city:
            reason += f"Transaction in {transaction['City']} (customer typically stays in {customer.city}). "
        
        fraud_indicator = FraudIndicator(
            fraud_type=FraudType.ACCOUNT_TAKEOVER,
            confidence=confidence,
            reason=reason,
            evidence=evidence,
            severity=severity
        )
        
        # Mark as fraud
        transaction['Is_Fraud'] = 1
        transaction['Fraud_Type'] = FraudType.ACCOUNT_TAKEOVER.value
        
        return transaction, fraud_indicator
    
    def calculate_confidence(self, evidence: Dict[str, Any]) -> float:
        """Calculate confidence based on behavioral deviation."""
        confidence = 0.4  # Base confidence
        
        # Amount multiplier contributes heavily
        multiplier = evidence['amount_multiplier']
        if multiplier > 7:
            confidence += 0.3
        elif multiplier > 5:
            confidence += 0.2
        elif multiplier > 3:
            confidence += 0.1
        
        # Unusual hour adds confidence
        if evidence['unusual_hour']:
            confidence += 0.15
        
        # Location change adds confidence
        if evidence['home_city_ratio'] > 0.7 and evidence['current_city'] != evidence.get('baseline_city', ''):
            confidence += 0.15
        
        return min(confidence, 1.0)


class MerchantCollusionPattern(FraudPattern):
    """
    Merchant Collusion Fraud Pattern
    
    Characteristics:
    - Multiple transactions at same merchant in short time
    - Round amounts (easier to split proceeds)
    - New or low-reputation merchant
    - High transaction amounts
    - Often just below reporting thresholds
    
    Example: Multiple Rs.49,999 transactions at newly registered merchant
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.suspicious_round_amounts = [10000, 25000, 49000, 49999, 50000, 100000]
    
    def should_apply(self, customer: Any, transaction: Dict[str, Any],
                     customer_history: List[Dict[str, Any]]) -> bool:
        """Merchant collusion can happen to any customer."""
        return random.random() < 0.2
    
    def apply_pattern(self, transaction: Dict[str, Any], customer: Any,
                     customer_history: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], FraudIndicator]:
        """Apply merchant collusion pattern."""
        
        # Select round amount just below limit
        amount = random.choice(self.suspicious_round_amounts)
        transaction['Amount'] = amount
        
        # Use low-reputation or new merchant
        transaction['Merchant_Rating'] = random.uniform(0.5, 2.5)
        transaction['Merchant_Years_Operating'] = random.randint(0, 2)
        
        # Often in person (POS) to avoid digital trail
        transaction['Channel'] = 'POS'
        transaction['Device_Type'] = 'POS'
        
        # Check if customer has been to this merchant before
        same_merchant_count = sum(1 for t in customer_history 
                                 if t.get('Merchant_ID') == transaction.get('Merchant_ID'))
        
        # Build evidence
        evidence = {
            'amount': amount,
            'round_amount': amount in self.suspicious_round_amounts,
            'just_below_limit': amount in [49000, 49999],
            'merchant_rating': round(transaction['Merchant_Rating'], 2),
            'merchant_years': transaction['Merchant_Years_Operating'],
            'new_merchant': same_merchant_count == 0,
            'repeat_merchant_count': same_merchant_count,
            'channel': transaction['Channel']
        }
        
        # Calculate confidence
        confidence = self.calculate_confidence(evidence)
        
        # Determine severity
        if amount >= 49000 and evidence['new_merchant'] and evidence['merchant_years'] == 0:
            severity = 'critical'
        elif amount >= 25000 and evidence['new_merchant']:
            severity = 'high'
        elif evidence['round_amount']:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Create fraud indicator
        reason = (f"Merchant collusion suspected. Transaction of Rs.{amount:,} ")
        
        if evidence['just_below_limit']:
            reason += "is just below reporting threshold. "
        if evidence['round_amount']:
            reason += "Round amount suggests coordinated fraud. "
        if evidence['new_merchant']:
            reason += f"New merchant relationship (first transaction). "
        if evidence['merchant_years'] <= 1:
            reason += f"Merchant only {evidence['merchant_years']} years old. "
        if evidence['merchant_rating'] < 2.0:
            reason += f"Low merchant rating ({evidence['merchant_rating']:.1f}/5). "
        
        fraud_indicator = FraudIndicator(
            fraud_type=FraudType.MERCHANT_COLLUSION,
            confidence=confidence,
            reason=reason,
            evidence=evidence,
            severity=severity
        )
        
        # Mark as fraud
        transaction['Is_Fraud'] = 1
        transaction['Fraud_Type'] = FraudType.MERCHANT_COLLUSION.value
        
        return transaction, fraud_indicator
    
    def calculate_confidence(self, evidence: Dict[str, Any]) -> float:
        """Calculate confidence based on merchant and amount patterns."""
        confidence = 0.3  # Base confidence
        
        # Just below limit is highly suspicious
        if evidence['just_below_limit']:
            confidence += 0.3
        
        # Round amounts add confidence
        if evidence['round_amount']:
            confidence += 0.15
        
        # New + low rating merchant
        if evidence['new_merchant'] and evidence['merchant_rating'] < 2.0:
            confidence += 0.2
        
        # Very new merchant
        if evidence['merchant_years'] == 0:
            confidence += 0.1
        
        return min(confidence, 1.0)


class VelocityAbusePattern(FraudPattern):
    """
    Velocity Abuse Fraud Pattern
    
    Characteristics:
    - Many transactions in very short time period
    - Exceeds normal customer behavior
    - Often testing card limits
    - May be at different merchants
    - Can be online or in-person
    
    Example: 10 transactions in 15 minutes
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.velocity_threshold = 5  # Transactions per hour considered suspicious
        self.time_window_hours = 1.0
    
    def should_apply(self, customer: Any, transaction: Dict[str, Any],
                     customer_history: List[Dict[str, Any]]) -> bool:
        """Velocity abuse can affect any active customer."""
        return len(customer_history) > 0 and random.random() < 0.2
    
    def apply_pattern(self, transaction: Dict[str, Any], customer: Any,
                     customer_history: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], FraudIndicator]:
        """Apply velocity abuse pattern."""
        
        # Calculate recent transaction velocity
        current_time = datetime.fromisoformat(transaction['Date'] + ' ' + transaction['Time'])
        recent_txns = []
        
        for hist_txn in reversed(customer_history):
            hist_time = datetime.fromisoformat(hist_txn['Date'] + ' ' + hist_txn['Time'])
            hours_ago = (current_time - hist_time).total_seconds() / 3600
            
            if hours_ago <= self.time_window_hours:
                recent_txns.append(hist_txn)
            else:
                break
        
        velocity = len(recent_txns) + 1  # Include current transaction
        
        # Often small amounts to test card
        transaction['Amount'] = random.uniform(50, 500)
        
        # Mix of merchants
        transaction['Is_First_Transaction_with_Merchant'] = random.random() < 0.7
        
        # Often online for speed
        if random.random() < 0.6:
            transaction['Channel'] = 'Online'
            transaction['Device_Type'] = 'Mobile' if random.random() < 0.5 else 'Web'
        
        # Build evidence
        evidence = {
            'transactions_in_hour': velocity,
            'velocity_threshold': self.velocity_threshold,
            'time_window_hours': self.time_window_hours,
            'avg_amount': round(transaction['Amount'], 2),
            'new_merchant': transaction['Is_First_Transaction_with_Merchant'],
            'channel': transaction['Channel']
        }
        
        # Calculate confidence
        confidence = self.calculate_confidence(evidence)
        
        # Determine severity
        if velocity > 10:
            severity = 'critical'
        elif velocity > 7:
            severity = 'high'
        elif velocity > 5:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Create fraud indicator
        reason = (f"Velocity abuse detected. {velocity} transactions in last "
                 f"{self.time_window_hours} hour(s) exceeds threshold of {self.velocity_threshold}. "
                 f"Small amount (Rs.{round(transaction['Amount'], 2)}) suggests card testing. ")
        
        if evidence['new_merchant']:
            reason += "Transaction with new merchant adds to suspicion. "
        
        fraud_indicator = FraudIndicator(
            fraud_type=FraudType.VELOCITY_ABUSE,
            confidence=confidence,
            reason=reason,
            evidence=evidence,
            severity=severity
        )
        
        # Mark as fraud
        transaction['Is_Fraud'] = 1
        transaction['Fraud_Type'] = FraudType.VELOCITY_ABUSE.value
        
        return transaction, fraud_indicator
    
    def calculate_confidence(self, evidence: Dict[str, Any]) -> float:
        """Calculate confidence based on transaction velocity."""
        velocity = evidence['transactions_in_hour']
        threshold = evidence['velocity_threshold']
        
        confidence = 0.3  # Base confidence
        
        # Velocity above threshold
        if velocity > threshold * 2:
            confidence += 0.4
        elif velocity > threshold * 1.5:
            confidence += 0.3
        elif velocity > threshold:
            confidence += 0.2
        
        # New merchant pattern
        if evidence['new_merchant']:
            confidence += 0.1
        
        # Small amounts (testing)
        if evidence['avg_amount'] < 500:
            confidence += 0.1
        
        return min(confidence, 1.0)


class AmountManipulationPattern(FraudPattern):
    """
    Amount Manipulation Fraud Pattern
    
    Characteristics:
    - Transactions just below reporting limits
    - Structuring (breaking large amount into smaller ones)
    - Avoiding detection thresholds
    - Often cash or UPI transactions
    - Same merchant or category
    
    Example: Rs.49,999 multiple times instead of Rs.2,00,000 once
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        # Common reporting thresholds in India
        self.thresholds = [10000, 20000, 50000, 100000, 200000]
        self.just_below_margin = 1000  # Amount below threshold
    
    def should_apply(self, customer: Any, transaction: Dict[str, Any],
                     customer_history: List[Dict[str, Any]]) -> bool:
        """Amount manipulation typically happens with established customers."""
        return len(customer_history) > 3 and random.random() < 0.15
    
    def apply_pattern(self, transaction: Dict[str, Any], customer: Any,
                     customer_history: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], FraudIndicator]:
        """Apply amount manipulation pattern."""
        
        # Select threshold and set amount just below it
        threshold = random.choice(self.thresholds)
        margin = random.uniform(1, self.just_below_margin)
        transaction['Amount'] = threshold - margin
        
        # Prefer payment modes that are harder to track
        if random.random() < 0.5:
            transaction['Payment_Mode'] = 'UPI'
        else:
            transaction['Payment_Mode'] = 'Cash'
        
        # Count similar amounts in recent history
        similar_count = 0
        for hist_txn in customer_history[-20:]:  # Last 20 transactions
            if abs(hist_txn.get('Amount', 0) - transaction['Amount']) < 2000:
                similar_count += 1
        
        # Build evidence
        evidence = {
            'amount': round(transaction['Amount'], 2),
            'threshold': threshold,
            'margin_below_threshold': round(margin, 2),
            'just_below_threshold': margin < self.just_below_margin,
            'payment_mode': transaction['Payment_Mode'],
            'similar_amount_count': similar_count,
            'structuring_suspected': similar_count >= 2
        }
        
        # Calculate confidence
        confidence = self.calculate_confidence(evidence)
        
        # Determine severity
        if evidence['structuring_suspected'] and margin < 100:
            severity = 'critical'
        elif evidence['structuring_suspected']:
            severity = 'high'
        elif margin < 500:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Create fraud indicator
        reason = (f"Amount manipulation detected. Transaction of Rs.{round(transaction['Amount'], 2)} "
                 f"is Rs.{round(margin, 2)} below reporting threshold of Rs.{threshold:,}. ")
        
        if evidence['structuring_suspected']:
            reason += f"Similar amounts in {similar_count} recent transactions suggests structuring. "
        
        if transaction['Payment_Mode'] in ['UPI', 'Cash']:
            reason += f"{transaction['Payment_Mode']} payment harder to track. "
        
        fraud_indicator = FraudIndicator(
            fraud_type=FraudType.AMOUNT_MANIPULATION,
            confidence=confidence,
            reason=reason,
            evidence=evidence,
            severity=severity
        )
        
        # Mark as fraud
        transaction['Is_Fraud'] = 1
        transaction['Fraud_Type'] = FraudType.AMOUNT_MANIPULATION.value
        
        return transaction, fraud_indicator
    
    def calculate_confidence(self, evidence: Dict[str, Any]) -> float:
        """Calculate confidence based on threshold proximity and patterns."""
        margin = evidence['margin_below_threshold']
        
        confidence = 0.3  # Base confidence
        
        # Very close to threshold is highly suspicious
        if margin < 100:
            confidence += 0.3
        elif margin < 500:
            confidence += 0.2
        elif margin < 1000:
            confidence += 0.1
        
        # Structuring pattern
        if evidence['structuring_suspected']:
            confidence += 0.3
        
        # Hard-to-track payment mode
        if evidence['payment_mode'] in ['UPI', 'Cash']:
            confidence += 0.1
        
        return min(confidence, 1.0)


class RefundFraudPattern(FraudPattern):
    """
    Refund Fraud Pattern
    
    Characteristics:
    - Multiple refunds in short period
    - High refund rate compared to normal customers
    - Often same merchant or category
    - May involve collusion with merchant
    - Typically online purchases
    
    Example: Buy expensive items, claim not received, get refund
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.normal_refund_rate = 0.02  # 2% normal refund rate
    
    def should_apply(self, customer: Any, transaction: Dict[str, Any],
                     customer_history: List[Dict[str, Any]]) -> bool:
        """Refund fraud requires purchase history."""
        return len(customer_history) > 5 and random.random() < 0.1
    
    def apply_pattern(self, transaction: Dict[str, Any], customer: Any,
                     customer_history: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], FraudIndicator]:
        """Apply refund fraud pattern."""
        
        # Mark as potential refund fraud
        # (In real system, would track actual refunds separately)
        transaction['Transaction_Status'] = 'Approved'
        
        # Online purchases easier to claim not received
        transaction['Channel'] = 'Online'
        transaction['Device_Type'] = random.choice(['Mobile', 'Web'])
        
        # Often high-value items
        transaction['Amount'] = random.uniform(5000, 50000)
        
        # Categories prone to refund fraud
        refund_prone_categories = ['Electronics', 'Fashion', 'Shopping', 'Travel']
        transaction['Category'] = random.choice(refund_prone_categories)
        
        # Count refund pattern in history (simulated)
        # In real implementation, would track actual refunds
        refund_count = sum(1 for t in customer_history 
                          if t.get('Transaction_Status') == 'Pending' 
                          or t.get('Category') in refund_prone_categories)
        
        refund_rate = refund_count / len(customer_history) if customer_history else 0
        
        # Build evidence
        evidence = {
            'amount': round(transaction['Amount'], 2),
            'category': transaction['Category'],
            'channel': transaction['Channel'],
            'historical_refund_count': refund_count,
            'historical_refund_rate': round(refund_rate, 3),
            'normal_refund_rate': self.normal_refund_rate,
            'elevated_refund_rate': refund_rate > self.normal_refund_rate * 3
        }
        
        # Calculate confidence
        confidence = self.calculate_confidence(evidence)
        
        # Determine severity
        if refund_rate > 0.15 and transaction['Amount'] > 20000:
            severity = 'critical'
        elif refund_rate > 0.10:
            severity = 'high'
        elif refund_rate > 0.05:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Create fraud indicator
        reason = (f"Refund fraud suspected. Customer's refund rate ({round(refund_rate * 100, 1)}%) "
                 f"is {round(refund_rate / self.normal_refund_rate, 1)}x higher than normal ({round(self.normal_refund_rate * 100, 1)}%). "
                 f"High-value {transaction['Category']} purchase (Rs.{round(transaction['Amount'], 2)}) "
                 f"via {transaction['Channel']} channel. ")
        
        fraud_indicator = FraudIndicator(
            fraud_type=FraudType.REFUND_FRAUD,
            confidence=confidence,
            reason=reason,
            evidence=evidence,
            severity=severity
        )
        
        # Mark as fraud
        transaction['Is_Fraud'] = 1
        transaction['Fraud_Type'] = FraudType.REFUND_FRAUD.value
        
        return transaction, fraud_indicator
    
    def calculate_confidence(self, evidence: Dict[str, Any]) -> float:
        """Calculate confidence based on refund patterns."""
        refund_rate = evidence['historical_refund_rate']
        normal_rate = evidence['normal_refund_rate']
        
        confidence = 0.2  # Base confidence
        
        # Elevated refund rate
        if refund_rate > normal_rate * 5:
            confidence += 0.4
        elif refund_rate > normal_rate * 3:
            confidence += 0.3
        elif refund_rate > normal_rate * 2:
            confidence += 0.2
        
        # High-value transaction
        if evidence['amount'] > 20000:
            confidence += 0.15
        
        # Prone category
        if evidence['category'] in ['Electronics', 'Travel']:
            confidence += 0.1
        
        return min(confidence, 1.0)


class StolenCardPattern(FraudPattern):
    """
    Stolen Card Fraud Pattern
    
    Characteristics:
    - Sudden spike in transactions after period of inactivity
    - Different location from usual
    - High-value purchases
    - Multiple rapid transactions
    - Often cash-equivalent purchases (gift cards, jewelry)
    - May have failed PIN attempts
    
    Example: Card inactive for days, then sudden Rs.50,000 purchase in different city
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.inactivity_days = 3  # Days of inactivity before theft
    
    def should_apply(self, customer: Any, transaction: Dict[str, Any],
                     customer_history: List[Dict[str, Any]]) -> bool:
        """Stolen card pattern requires history to show sudden change."""
        if len(customer_history) < 5:
            return False
        
        # Check for inactivity period
        if customer_history:
            last_txn = customer_history[-1]
            last_time = datetime.fromisoformat(last_txn['Date'] + ' ' + last_txn['Time'])
            current_time = datetime.fromisoformat(transaction['Date'] + ' ' + transaction['Time'])
            days_since_last = (current_time - last_time).days
            
            return days_since_last >= self.inactivity_days and random.random() < 0.15
        return False
    
    def apply_pattern(self, transaction: Dict[str, Any], customer: Any,
                     customer_history: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], FraudIndicator]:
        """Apply stolen card pattern."""
        
        # Calculate inactivity period
        last_txn = customer_history[-1]
        last_time = datetime.fromisoformat(last_txn['Date'] + ' ' + last_txn['Time'])
        current_time = datetime.fromisoformat(transaction['Date'] + ' ' + transaction['Time'])
        inactivity_days = (current_time - last_time).days
        
        # High-value purchase
        transaction['Amount'] = random.uniform(10000, 80000)
        
        # Cash-equivalent categories
        cash_equiv_categories = ['Electronics', 'Jewelry', 'Shopping', 'Travel']
        transaction['Category'] = random.choice(cash_equiv_categories)
        
        # Different location
        other_cities = ['Mumbai', 'Delhi', 'Bangalore', 'Pune', 'Hyderabad']
        transaction['City'] = random.choice([c for c in other_cities if c != customer.city])
        transaction['Distance_From_Home_km'] = random.uniform(300, 1500)
        
        # Often in-person to avoid digital footprint
        transaction['Channel'] = 'POS'
        transaction['Device_Type'] = 'POS'
        
        # Calculate baseline
        if customer_history:
            avg_amount = sum(t.get('Amount', 0) for t in customer_history) / len(customer_history)
        else:
            avg_amount = 5000
        
        amount_multiplier = transaction['Amount'] / avg_amount if avg_amount > 0 else 1
        
        # Build evidence
        evidence = {
            'inactivity_days': inactivity_days,
            'amount': round(transaction['Amount'], 2),
            'avg_amount': round(avg_amount, 2),
            'amount_multiplier': round(amount_multiplier, 2),
            'category': transaction['Category'],
            'cash_equivalent': transaction['Category'] in cash_equiv_categories,
            'different_city': transaction['City'] != customer.city,
            'distance_km': round(transaction['Distance_From_Home_km'], 2),
            'channel': transaction['Channel']
        }
        
        # Calculate confidence
        confidence = self.calculate_confidence(evidence)
        
        # Determine severity
        if inactivity_days > 7 and amount_multiplier > 10:
            severity = 'critical'
        elif inactivity_days > 5 and amount_multiplier > 5:
            severity = 'high'
        elif inactivity_days > 3:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Create fraud indicator
        reason = (f"Stolen card suspected. Card inactive for {inactivity_days} days, then sudden "
                 f"Rs.{round(transaction['Amount'], 2)} purchase ({round(amount_multiplier, 1)}x baseline). "
                 f"Transaction in {transaction['City']} ({round(transaction['Distance_From_Home_km'], 0)}km from home). ")
        
        if evidence['cash_equivalent']:
            reason += f"Cash-equivalent category ({transaction['Category']}) common in card theft. "
        
        fraud_indicator = FraudIndicator(
            fraud_type=FraudType.STOLEN_CARD,
            confidence=confidence,
            reason=reason,
            evidence=evidence,
            severity=severity
        )
        
        # Mark as fraud
        transaction['Is_Fraud'] = 1
        transaction['Fraud_Type'] = FraudType.STOLEN_CARD.value
        
        return transaction, fraud_indicator
    
    def calculate_confidence(self, evidence: Dict[str, Any]) -> float:
        """Calculate confidence based on inactivity and behavioral change."""
        inactivity = evidence['inactivity_days']
        multiplier = evidence['amount_multiplier']
        
        confidence = 0.3  # Base confidence
        
        # Long inactivity period
        if inactivity > 7:
            confidence += 0.3
        elif inactivity > 5:
            confidence += 0.2
        elif inactivity > 3:
            confidence += 0.1
        
        # High amount compared to baseline
        if multiplier > 10:
            confidence += 0.2
        elif multiplier > 5:
            confidence += 0.15
        
        # Cash-equivalent category
        if evidence['cash_equivalent']:
            confidence += 0.1
        
        # Different location
        if evidence['different_city']:
            confidence += 0.1
        
        return min(confidence, 1.0)


class SyntheticIdentityPattern(FraudPattern):
    """
    Synthetic Identity Fraud Pattern
    
    Characteristics:
    - New customer with little history
    - Inconsistent demographic information
    - Rapid credit building behavior
    - Multiple accounts or cards
    - Similar patterns across "different" customers
    - Gradually increasing transaction amounts
    
    Example: New customer with no history suddenly making large purchases
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.max_history_for_synthetic = 15  # Limited history
    
    def should_apply(self, customer: Any, transaction: Dict[str, Any],
                     customer_history: List[Dict[str, Any]]) -> bool:
        """Synthetic identity has limited history."""
        return len(customer_history) < self.max_history_for_synthetic and random.random() < 0.1
    
    def apply_pattern(self, transaction: Dict[str, Any], customer: Any,
                     customer_history: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], FraudIndicator]:
        """Apply synthetic identity pattern."""
        
        history_length = len(customer_history)
        
        # Gradually increasing amounts (building credit)
        if history_length > 0:
            prev_amounts = [t.get('Amount', 0) for t in customer_history]
            avg_prev = sum(prev_amounts) / len(prev_amounts)
            # Each transaction slightly higher
            transaction['Amount'] = avg_prev * random.uniform(1.2, 1.5)
        else:
            transaction['Amount'] = random.uniform(1000, 5000)
        
        # New customer characteristics
        transaction['Customer_Lifetime_Days'] = random.randint(1, 90)
        
        # Often online for anonymity
        transaction['Channel'] = 'Online'
        transaction['Device_Type'] = random.choice(['Mobile', 'Web'])
        
        # Build evidence
        evidence = {
            'customer_history_length': history_length,
            'customer_lifetime_days': transaction.get('Customer_Lifetime_Days', 0),
            'amount': round(transaction['Amount'], 2),
            'limited_history': history_length < 10,
            'new_customer': transaction.get('Customer_Lifetime_Days', 999) < 90,
            'gradual_increase': history_length > 0,
            'channel': transaction['Channel']
        }
        
        # Calculate average transaction growth rate
        if history_length > 2:
            amounts = [t.get('Amount', 0) for t in customer_history] + [transaction['Amount']]
            growth_rates = []
            for i in range(1, len(amounts)):
                if amounts[i-1] > 0:
                    growth = (amounts[i] - amounts[i-1]) / amounts[i-1]
                    growth_rates.append(growth)
            avg_growth = sum(growth_rates) / len(growth_rates) if growth_rates else 0
            evidence['avg_transaction_growth_rate'] = round(avg_growth, 3)
            evidence['consistent_growth'] = avg_growth > 0.15
        
        # Calculate confidence
        confidence = self.calculate_confidence(evidence)
        
        # Determine severity
        if history_length < 5 and evidence.get('consistent_growth', False):
            severity = 'high'
        elif history_length < 10:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Create fraud indicator
        reason = (f"Synthetic identity suspected. Customer has only {history_length} transactions "
                 f"in {evidence['customer_lifetime_days']} days. ")
        
        if evidence.get('consistent_growth', False):
            growth_rate = evidence.get('avg_transaction_growth_rate', 0)
            reason += f"Consistent transaction growth ({round(growth_rate * 100, 1)}%) suggests credit building. "
        
        if evidence['limited_history'] and transaction['Amount'] > 5000:
            reason += f"Large transaction (Rs.{round(transaction['Amount'], 2)}) for limited history. "
        
        fraud_indicator = FraudIndicator(
            fraud_type=FraudType.SYNTHETIC_IDENTITY,
            confidence=confidence,
            reason=reason,
            evidence=evidence,
            severity=severity
        )
        
        # Mark as fraud
        transaction['Is_Fraud'] = 1
        transaction['Fraud_Type'] = FraudType.SYNTHETIC_IDENTITY.value
        
        return transaction, fraud_indicator
    
    def calculate_confidence(self, evidence: Dict[str, Any]) -> float:
        """Calculate confidence based on history and growth patterns."""
        confidence = 0.2  # Base confidence
        
        # Very limited history
        history_len = evidence['customer_history_length']
        if history_len < 5:
            confidence += 0.3
        elif history_len < 10:
            confidence += 0.2
        
        # New customer
        if evidence['new_customer']:
            confidence += 0.15
        
        # Consistent growth pattern
        if evidence.get('consistent_growth', False):
            confidence += 0.25
        
        return min(confidence, 1.0)


class FirstPartyFraudPattern(FraudPattern):
    """
    First Party Fraud Pattern
    
    Characteristics:
    - Customer intentionally misrepresenting information
    - Application fraud (false income, employment)
    - Bust-out fraud (max out credit then disappear)
    - Large purchases followed by non-payment
    - May involve identity manipulation
    
    Example: Customer maxes out all credit then stops paying
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.bust_out_threshold = 0.8  # Using 80% of available credit
    
    def should_apply(self, customer: Any, transaction: Dict[str, Any],
                     customer_history: List[Dict[str, Any]]) -> bool:
        """First party fraud happens after building some trust."""
        return len(customer_history) > 10 and random.random() < 0.08
    
    def apply_pattern(self, transaction: Dict[str, Any], customer: Any,
                     customer_history: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], FraudIndicator]:
        """Apply first party fraud pattern."""
        
        # Sudden large transaction (bust-out)
        if customer_history:
            avg_amount = sum(t.get('Amount', 0) for t in customer_history) / len(customer_history)
            transaction['Amount'] = avg_amount * random.uniform(5, 15)
        else:
            transaction['Amount'] = random.uniform(30000, 100000)
        
        # Often high-value categories
        bust_out_categories = ['Electronics', 'Jewelry', 'Travel', 'Shopping']
        transaction['Category'] = random.choice(bust_out_categories)
        
        # Calculate spending velocity increase
        recent_txns = customer_history[-30:] if len(customer_history) > 30 else customer_history
        recent_total = sum(t.get('Amount', 0) for t in recent_txns)
        
        # Build evidence
        evidence = {
            'amount': round(transaction['Amount'], 2),
            'category': transaction['Category'],
            'history_length': len(customer_history),
            'recent_transactions': len(recent_txns),
            'recent_total_spending': round(recent_total, 2),
            'sudden_large_purchase': True,
            'high_value_category': transaction['Category'] in bust_out_categories
        }
        
        if customer_history:
            avg_amount = sum(t.get('Amount', 0) for t in customer_history) / len(customer_history)
            evidence['baseline_avg_amount'] = round(avg_amount, 2)
            evidence['amount_multiplier'] = round(transaction['Amount'] / avg_amount, 2) if avg_amount > 0 else 1
        
        # Calculate confidence
        confidence = self.calculate_confidence(evidence)
        
        # Determine severity
        if transaction['Amount'] > 50000 and evidence.get('amount_multiplier', 1) > 10:
            severity = 'critical'
        elif transaction['Amount'] > 30000:
            severity = 'high'
        else:
            severity = 'medium'
        
        # Create fraud indicator
        reason = (f"First party fraud suspected. Customer with {evidence['history_length']} transactions "
                 f"suddenly makes Rs.{round(transaction['Amount'], 2)} purchase ")
        
        if evidence.get('amount_multiplier'):
            reason += f"({round(evidence['amount_multiplier'], 1)}x baseline). "
        
        reason += f"High-value {transaction['Category']} category. Pattern consistent with bust-out fraud."
        
        fraud_indicator = FraudIndicator(
            fraud_type=FraudType.FIRST_PARTY_FRAUD,
            confidence=confidence,
            reason=reason,
            evidence=evidence,
            severity=severity
        )
        
        # Mark as fraud
        transaction['Is_Fraud'] = 1
        transaction['Fraud_Type'] = FraudType.FIRST_PARTY_FRAUD.value
        
        return transaction, fraud_indicator
    
    def calculate_confidence(self, evidence: Dict[str, Any]) -> float:
        """Calculate confidence based on spending pattern change."""
        confidence = 0.3  # Base confidence
        
        # Large multiplier indicates bust-out
        multiplier = evidence.get('amount_multiplier', 1)
        if multiplier > 10:
            confidence += 0.3
        elif multiplier > 7:
            confidence += 0.2
        elif multiplier > 5:
            confidence += 0.1
        
        # High-value category
        if evidence['high_value_category']:
            confidence += 0.15
        
        # Established history makes sudden change more suspicious
        if evidence['history_length'] > 20:
            confidence += 0.1
        
        return min(confidence, 1.0)


class FriendlyFraudPattern(FraudPattern):
    """
    Friendly Fraud Pattern (Chargeback Abuse)
    
    Characteristics:
    - Customer makes purchase, receives goods, then disputes charge
    - Claims item not received or unauthorized
    - Pattern of chargebacks across multiple transactions
    - Often online purchases
    - May target specific merchants
    
    Example: Buy item online, receive it, claim never arrived, get refund
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.normal_dispute_rate = 0.01  # 1% normal dispute rate
    
    def should_apply(self, customer: Any, transaction: Dict[str, Any],
                     customer_history: List[Dict[str, Any]]) -> bool:
        """Friendly fraud requires purchase history to show pattern."""
        return len(customer_history) > 8 and random.random() < 0.07
    
    def apply_pattern(self, transaction: Dict[str, Any], customer: Any,
                     customer_history: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], FraudIndicator]:
        """Apply friendly fraud pattern."""
        
        # Online purchase (easier to dispute)
        transaction['Channel'] = 'Online'
        transaction['Device_Type'] = random.choice(['Mobile', 'Web'])
        
        # Categories commonly targeted
        chargeback_prone = ['Electronics', 'Fashion', 'Shopping', 'Entertainment']
        transaction['Category'] = random.choice(chargeback_prone)
        
        # Moderate to high amounts
        transaction['Amount'] = random.uniform(2000, 25000)
        
        # Transaction appears normal initially
        transaction['Transaction_Status'] = 'Approved'
        
        # Calculate dispute rate from history
        # (In real system, would track actual disputes)
        online_txns = [t for t in customer_history if t.get('Channel') == 'Online']
        dispute_count = len([t for t in online_txns if t.get('Category') in chargeback_prone])
        
        dispute_rate = dispute_count / len(customer_history) if customer_history else 0
        
        # Build evidence
        evidence = {
            'amount': round(transaction['Amount'], 2),
            'category': transaction['Category'],
            'channel': transaction['Channel'],
            'historical_online_transactions': len(online_txns),
            'historical_dispute_prone_transactions': dispute_count,
            'dispute_rate': round(dispute_rate, 3),
            'normal_dispute_rate': self.normal_dispute_rate,
            'elevated_dispute_rate': dispute_rate > self.normal_dispute_rate * 3,
            'chargeback_prone_category': transaction['Category'] in chargeback_prone
        }
        
        # Calculate confidence
        confidence = self.calculate_confidence(evidence)
        
        # Determine severity
        if dispute_rate > 0.20:
            severity = 'critical'
        elif dispute_rate > 0.10:
            severity = 'high'
        elif dispute_rate > 0.05:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Create fraud indicator
        reason = (f"Friendly fraud (chargeback abuse) suspected. Customer's dispute rate "
                 f"({round(dispute_rate * 100, 1)}%) is {round(dispute_rate / self.normal_dispute_rate, 1)}x "
                 f"higher than normal ({round(self.normal_dispute_rate * 100, 1)}%). "
                 f"Online {transaction['Category']} purchase (Rs.{round(transaction['Amount'], 2)}) "
                 f"in chargeback-prone category.")
        
        fraud_indicator = FraudIndicator(
            fraud_type=FraudType.FRIENDLY_FRAUD,
            confidence=confidence,
            reason=reason,
            evidence=evidence,
            severity=severity
        )
        
        # Mark as fraud
        transaction['Is_Fraud'] = 1
        transaction['Fraud_Type'] = FraudType.FRIENDLY_FRAUD.value
        
        return transaction, fraud_indicator
    
    def calculate_confidence(self, evidence: Dict[str, Any]) -> float:
        """Calculate confidence based on dispute pattern."""
        dispute_rate = evidence['dispute_rate']
        normal_rate = evidence['normal_dispute_rate']
        
        confidence = 0.2  # Base confidence
        
        # Elevated dispute rate
        if dispute_rate > normal_rate * 5:
            confidence += 0.4
        elif dispute_rate > normal_rate * 3:
            confidence += 0.3
        elif dispute_rate > normal_rate * 2:
            confidence += 0.2
        
        # Chargeback-prone category
        if evidence['chargeback_prone_category']:
            confidence += 0.15
        
        # Online channel (easier to dispute)
        if evidence['channel'] == 'Online':
            confidence += 0.1
        
        return min(confidence, 1.0)


# ============================================================================
# ADVANCED FRAUD PATTERNS (Week 4 Days 3-4)
# ============================================================================


class TransactionReplayPattern(FraudPattern):
    """
    Transaction Replay Pattern (Duplicate Transaction Attack)
    
    Characteristics:
    - Exact or near-exact duplicate of legitimate transaction
    - Same merchant, amount, and timing pattern
    - Often occurs within minutes/hours of original
    - May have slightly different device or location
    - Exploits systems without proper deduplication
    
    Example: Legitimate Rs.5,000 transaction replayed 3 times within 30 minutes
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.replay_window_minutes = 120  # 2 hour window for replay attacks
    
    def should_apply(self, customer: Any, transaction: Dict[str, Any],
                     customer_history: List[Dict[str, Any]]) -> bool:
        """Replay requires recent transaction history."""
        return len(customer_history) > 3 and random.random() < 0.06
    
    def apply_pattern(self, transaction: Dict[str, Any], customer: Any,
                     customer_history: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], FraudIndicator]:
        """Apply transaction replay pattern."""
        
        # Find a recent transaction to replay
        recent_txns = customer_history[-10:] if len(customer_history) >= 10 else customer_history
        if not recent_txns:
            # Fallback: create suspicious duplicate pattern
            base_txn = transaction.copy()
        else:
            base_txn = random.choice(recent_txns)
        
        # Replay the transaction with slight variations
        transaction['Merchant_ID'] = base_txn.get('Merchant_ID', transaction['Merchant_ID'])
        transaction['Amount'] = base_txn.get('Amount', transaction['Amount'])
        transaction['Category'] = base_txn.get('Category', transaction['Category'])
        transaction['City'] = base_txn.get('City', transaction['City'])
        
        # Small variations to evade simple duplicate detection
        # Device might change (fraudster using different device)
        if random.random() < 0.6:
            transaction['Device_Type'] = random.choice(['Mobile', 'Web', 'POS'])
        else:
            transaction['Device_Type'] = base_txn.get('Device_Type', 'Web')
        
        # Location might be slightly different
        if random.random() < 0.4:
            base_location = base_txn.get('Location', transaction.get('Location', 'Unknown'))
            transaction['Location'] = base_location + " (duplicate)" if base_location else 'Unknown'
        
        # Amount might be exact or slightly different
        amount_variance = random.uniform(0.98, 1.02) if random.random() < 0.3 else 1.0
        transaction['Amount'] = round(base_txn.get('Amount', transaction['Amount']) * amount_variance, 2)
        
        # Count similar recent transactions
        similar_count = sum(1 for t in recent_txns 
                          if abs(t.get('Amount', 0) - transaction['Amount']) < 10 
                          and t.get('Merchant_ID') == transaction['Merchant_ID'])
        
        # Build evidence
        evidence = {
            'amount': round(transaction['Amount'], 2),
            'original_amount': round(base_txn.get('Amount', transaction['Amount']), 2),
            'merchant_id': transaction['Merchant_ID'],
            'category': transaction['Category'],
            'similar_transactions_count': similar_count,
            'replay_window_minutes': self.replay_window_minutes,
            'device_changed': transaction['Device_Type'] != base_txn.get('Device_Type'),
            'exact_amount_match': amount_variance == 1.0,
            'amount_variance_ratio': round(amount_variance, 3)
        }
        
        # Calculate confidence
        confidence = self.calculate_confidence(evidence)
        
        # Determine severity based on count
        if similar_count >= 4:
            severity = 'critical'
        elif similar_count >= 3:
            severity = 'high'
        elif similar_count >= 2:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Create fraud indicator
        reason = (f"Transaction replay attack detected. {similar_count} similar transactions "
                 f"to {transaction['Merchant_ID']} for Rs.{round(transaction['Amount'], 2)} "
                 f"within replay window. ")
        if evidence['device_changed']:
            reason += "Device type changed between transactions (evasion indicator). "
        if evidence['exact_amount_match']:
            reason += "Exact amount match suggests automated replay."
        
        fraud_indicator = FraudIndicator(
            fraud_type=FraudType.TRANSACTION_REPLAY,
            confidence=confidence,
            reason=reason,
            evidence=evidence,
            severity=severity
        )
        
        # Mark as fraud
        transaction['Is_Fraud'] = 1
        transaction['Fraud_Type'] = FraudType.TRANSACTION_REPLAY.value
        
        return transaction, fraud_indicator
    
    def calculate_confidence(self, evidence: Dict[str, Any]) -> float:
        """Calculate confidence based on replay indicators."""
        confidence = 0.3  # Base confidence for replay pattern
        
        # Multiple similar transactions
        similar_count = evidence['similar_transactions_count']
        if similar_count >= 4:
            confidence += 0.4
        elif similar_count >= 3:
            confidence += 0.3
        elif similar_count >= 2:
            confidence += 0.2
        
        # Exact amount match
        if evidence['exact_amount_match']:
            confidence += 0.2
        
        # Device changed (evasion)
        if evidence['device_changed']:
            confidence += 0.1
        
        return min(confidence, 1.0)


class CardTestingPattern(FraudPattern):
    """
    Card Testing Pattern (Card Validation Fraud)
    
    Characteristics:
    - Series of small test transactions (often under Rs.100)
    - Rapid succession (minutes apart)
    - Testing multiple cards or one card repeatedly
    - Often at online merchants
    - Followed by larger fraudulent transactions if successful
    
    Example: Rs.10, Rs.25, Rs.50 transactions within 5 minutes at online merchant
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.test_amount_threshold = 100  # Below Rs.100 considered test
        self.test_sequence_minutes = 15  # Test sequence within 15 minutes
    
    def should_apply(self, customer: Any, transaction: Dict[str, Any],
                     customer_history: List[Dict[str, Any]]) -> bool:
        """Card testing can occur at any time."""
        return random.random() < 0.05
    
    def apply_pattern(self, transaction: Dict[str, Any], customer: Any,
                     customer_history: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], FraudIndicator]:
        """Apply card testing pattern."""
        
        # Small test amount
        transaction['Amount'] = random.uniform(5, 95)
        
        # Online channel (easier for testing)
        transaction['Channel'] = 'Online'
        transaction['Device_Type'] = random.choice(['Web', 'Mobile'])
        
        # Often testing at specific merchant types
        test_merchants = ['Shopping', 'Entertainment', 'Electronics', 'Online Services']
        transaction['Category'] = random.choice(test_merchants)
        
        # Transaction usually approved (that's the test)
        transaction['Transaction_Status'] = 'Approved'
        
        # Count recent small transactions
        recent_small_txns = sum(1 for t in customer_history[-20:] 
                               if t.get('Amount', 0) < self.test_amount_threshold 
                               and t.get('Channel') == 'Online')
        
        # Calculate average transaction amount for customer
        if customer_history:
            avg_amount = sum(t.get('Amount', 0) for t in customer_history) / len(customer_history)
        else:
            avg_amount = transaction['Amount'] * 50  # Make it look anomalous
        
        # Build evidence
        evidence = {
            'test_amount': round(transaction['Amount'], 2),
            'test_threshold': self.test_amount_threshold,
            'channel': transaction['Channel'],
            'category': transaction['Category'],
            'recent_small_transactions': recent_small_txns,
            'customer_avg_amount': round(avg_amount, 2),
            'amount_ratio_to_avg': round(transaction['Amount'] / avg_amount if avg_amount > 0 else 0, 3),
            'sequence_window_minutes': self.test_sequence_minutes,
            'rapid_succession': recent_small_txns >= 2
        }
        
        # Calculate confidence
        confidence = self.calculate_confidence(evidence)
        
        # Determine severity
        if recent_small_txns >= 5:
            severity = 'critical'
        elif recent_small_txns >= 3:
            severity = 'high'
        elif recent_small_txns >= 2:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Create fraud indicator
        reason = (f"Card testing fraud detected. Small online transaction (Rs.{round(transaction['Amount'], 2)}) "
                 f"is {round(transaction['Amount'] / avg_amount * 100 if avg_amount > 0 else 0, 1)}% "
                 f"of customer's average (Rs.{round(avg_amount, 2)}). ")
        if recent_small_txns >= 2:
            reason += f"{recent_small_txns} similar test transactions in recent history. "
        reason += "Likely testing card validity before larger fraud."
        
        fraud_indicator = FraudIndicator(
            fraud_type=FraudType.CARD_TESTING,
            confidence=confidence,
            reason=reason,
            evidence=evidence,
            severity=severity
        )
        
        # Mark as fraud
        transaction['Is_Fraud'] = 1
        transaction['Fraud_Type'] = FraudType.CARD_TESTING.value
        
        return transaction, fraud_indicator
    
    def calculate_confidence(self, evidence: Dict[str, Any]) -> float:
        """Calculate confidence based on testing indicators."""
        confidence = 0.25  # Base confidence
        
        # Very small amount compared to average
        ratio = evidence['amount_ratio_to_avg']
        if ratio < 0.05:
            confidence += 0.35
        elif ratio < 0.10:
            confidence += 0.25
        elif ratio < 0.20:
            confidence += 0.15
        
        # Multiple recent small transactions
        recent_count = evidence['recent_small_transactions']
        if recent_count >= 5:
            confidence += 0.3
        elif recent_count >= 3:
            confidence += 0.2
        elif recent_count >= 2:
            confidence += 0.1
        
        # Online channel (easier for testing)
        if evidence['channel'] == 'Online':
            confidence += 0.1
        
        return min(confidence, 1.0)


class MuleAccountPattern(FraudPattern):
    """
    Mule Account Pattern (Money Laundering)
    
    Characteristics:
    - Rapid inflow followed by immediate outflow
    - High velocity of transactions
    - Large round amounts
    - Multiple transfers to different accounts
    - Short-lived activity burst
    - Often new or recently dormant accounts
    
    Example: Rs.50,000 received, then Rs.48,000 sent out in 4 transactions within 1 hour
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.rapid_turnover_threshold = 0.9  # 90% of funds moved quickly
        self.velocity_threshold = 5  # 5+ transactions in short period
    
    def should_apply(self, customer: Any, transaction: Dict[str, Any],
                     customer_history: List[Dict[str, Any]]) -> bool:
        """Mule accounts show specific velocity patterns."""
        return len(customer_history) > 2 and random.random() < 0.04
    
    def apply_pattern(self, transaction: Dict[str, Any], customer: Any,
                     customer_history: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], FraudIndicator]:
        """Apply mule account pattern."""
        
        # Large round amount (typical for layering)
        round_amounts = [10000, 25000, 50000, 75000, 100000]
        transaction['Amount'] = random.choice(round_amounts)
        
        # Online transfers (money movement)
        transaction['Channel'] = 'Online'
        transaction['Device_Type'] = random.choice(['Web', 'Mobile'])
        transaction['Category'] = 'Transfer'
        
        # Transaction approved (mule accounts are real accounts)
        transaction['Transaction_Status'] = 'Approved'
        
        # Calculate velocity metrics
        recent_txns = customer_history[-10:] if len(customer_history) >= 10 else customer_history
        total_recent_amount = sum(t.get('Amount', 0) for t in recent_txns)
        transfer_count = sum(1 for t in recent_txns if t.get('Category') == 'Transfer')
        
        # Calculate turnover ratio
        if customer_history:
            total_inflow = sum(t.get('Amount', 0) for t in customer_history if t.get('Amount', 0) > 0)
            turnover_ratio = total_recent_amount / total_inflow if total_inflow > 0 else 0
        else:
            turnover_ratio = 0.95  # High turnover indicator
        
        # Build evidence
        evidence = {
            'transaction_amount': round(transaction['Amount'], 2),
            'is_round_amount': transaction['Amount'] in round_amounts,
            'category': transaction['Category'],
            'recent_transfer_count': transfer_count,
            'recent_total_amount': round(total_recent_amount, 2),
            'turnover_ratio': round(turnover_ratio, 3),
            'turnover_threshold': self.rapid_turnover_threshold,
            'velocity_threshold': self.velocity_threshold,
            'high_velocity': transfer_count >= self.velocity_threshold,
            'rapid_turnover': turnover_ratio >= self.rapid_turnover_threshold,
            'account_age_days': len(customer_history)  # Proxy for account age
        }
        
        # Calculate confidence
        confidence = self.calculate_confidence(evidence)
        
        # Determine severity
        if turnover_ratio > 0.95 and transfer_count >= 7:
            severity = 'critical'
        elif turnover_ratio > 0.90 and transfer_count >= 5:
            severity = 'high'
        elif turnover_ratio > 0.85:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Create fraud indicator
        reason = (f"Mule account (money laundering) suspected. Rapid fund turnover "
                 f"({round(turnover_ratio * 100, 1)}%) with {transfer_count} recent transfers. "
                 f"Large round amount transfer (Rs.{round(transaction['Amount'], 2)}). ")
        if evidence['account_age_days'] < 30:
            reason += "New/recently activated account increases suspicion."
        
        fraud_indicator = FraudIndicator(
            fraud_type=FraudType.MULE_ACCOUNT,
            confidence=confidence,
            reason=reason,
            evidence=evidence,
            severity=severity
        )
        
        # Mark as fraud
        transaction['Is_Fraud'] = 1
        transaction['Fraud_Type'] = FraudType.MULE_ACCOUNT.value
        
        return transaction, fraud_indicator
    
    def calculate_confidence(self, evidence: Dict[str, Any]) -> float:
        """Calculate confidence based on mule account indicators."""
        confidence = 0.2  # Base confidence
        
        # Rapid turnover
        turnover = evidence['turnover_ratio']
        if turnover >= 0.95:
            confidence += 0.4
        elif turnover >= 0.90:
            confidence += 0.3
        elif turnover >= 0.85:
            confidence += 0.2
        
        # High velocity
        transfer_count = evidence['recent_transfer_count']
        if transfer_count >= 7:
            confidence += 0.25
        elif transfer_count >= 5:
            confidence += 0.15
        elif transfer_count >= 3:
            confidence += 0.1
        
        # Round amounts
        if evidence['is_round_amount']:
            confidence += 0.1
        
        # New account
        if evidence['account_age_days'] < 30:
            confidence += 0.05
        
        return min(confidence, 1.0)


class ShippingFraudPattern(FraudPattern):
    """
    Shipping Fraud Pattern (Address Manipulation)
    
    Characteristics:
    - Sudden change to shipping address
    - Rush/expedited shipping requested
    - High-value items (electronics, jewelry)
    - Different from billing address
    - Often to unusual locations
    - May use freight forwarding addresses
    
    Example: Rs.50,000 electronics order with new shipping address and rush delivery
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.high_value_threshold = 15000  # Above Rs.15,000 considered high value
        self.rush_shipping_indicator = True
    
    def should_apply(self, customer: Any, transaction: Dict[str, Any],
                     customer_history: List[Dict[str, Any]]) -> bool:
        """Shipping fraud requires some purchase history."""
        return len(customer_history) > 5 and random.random() < 0.05
    
    def apply_pattern(self, transaction: Dict[str, Any], customer: Any,
                     customer_history: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], FraudIndicator]:
        """Apply shipping fraud pattern."""
        
        # High-value items commonly targeted
        fraud_categories = ['Electronics', 'Fashion', 'Jewelry', 'Luxury']
        transaction['Category'] = random.choice(fraud_categories)
        
        # High amount
        transaction['Amount'] = random.uniform(self.high_value_threshold, 80000)
        
        # Online purchase (requires shipping)
        transaction['Channel'] = 'Online'
        transaction['Device_Type'] = random.choice(['Web', 'Mobile'])
        
        # Transaction usually approved initially
        transaction['Transaction_Status'] = 'Approved'
        
        # Different city than customer's usual location
        customer_cities = [t.get('City') for t in customer_history if t.get('City')]
        if customer_cities:
            common_city = max(set(customer_cities), key=customer_cities.count)
            # Ship to different city
            unusual_cities = ['Border Town', 'Freight Hub', 'Forward Address']
            transaction['City'] = random.choice(unusual_cities)
            address_changed = transaction['City'] != common_city
        else:
            address_changed = True
            common_city = 'Unknown'
        
        # Calculate shipping address change pattern
        recent_cities = [t.get('City') for t in customer_history[-10:]]
        city_changes = len(set(recent_cities)) if recent_cities else 0
        
        # Build evidence
        evidence = {
            'transaction_amount': round(transaction['Amount'], 2),
            'high_value_threshold': self.high_value_threshold,
            'category': transaction['Category'],
            'shipping_city': transaction['City'],
            'usual_city': common_city,
            'address_changed': address_changed,
            'recent_city_changes': city_changes,
            'rush_shipping': self.rush_shipping_indicator,
            'online_channel': transaction['Channel'] == 'Online',
            'high_value_item': transaction['Amount'] >= self.high_value_threshold
        }
        
        # Calculate confidence
        confidence = self.calculate_confidence(evidence)
        
        # Determine severity
        if transaction['Amount'] > 50000 and address_changed:
            severity = 'critical'
        elif transaction['Amount'] > 30000:
            severity = 'high'
        elif transaction['Amount'] > 15000:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Create fraud indicator
        reason = (f"Shipping fraud suspected. High-value {transaction['Category']} purchase "
                 f"(Rs.{round(transaction['Amount'], 2)}) with ")
        if address_changed:
            reason += f"shipping address changed from {common_city} to {transaction['City']}. "
        reason += f"Rush delivery requested for {transaction['Category']} (common fraud indicator)."
        
        fraud_indicator = FraudIndicator(
            fraud_type=FraudType.SHIPPING_FRAUD,
            confidence=confidence,
            reason=reason,
            evidence=evidence,
            severity=severity
        )
        
        # Mark as fraud
        transaction['Is_Fraud'] = 1
        transaction['Fraud_Type'] = FraudType.SHIPPING_FRAUD.value
        
        return transaction, fraud_indicator
    
    def calculate_confidence(self, evidence: Dict[str, Any]) -> float:
        """Calculate confidence based on shipping fraud indicators."""
        confidence = 0.25  # Base confidence
        
        # High-value item
        if evidence['high_value_item']:
            confidence += 0.25
        
        # Address changed
        if evidence['address_changed']:
            confidence += 0.25
        
        # Multiple recent address changes
        if evidence['recent_city_changes'] >= 3:
            confidence += 0.15
        elif evidence['recent_city_changes'] >= 2:
            confidence += 0.1
        
        # Rush shipping
        if evidence['rush_shipping']:
            confidence += 0.1
        
        # Online channel
        if evidence['online_channel']:
            confidence += 0.05
        
        return min(confidence, 1.0)


class LoyaltyAbusePattern(FraudPattern):
    """
    Loyalty Program Abuse Pattern (Points/Rewards Exploitation)
    
    Characteristics:
    - Exploiting loyalty program vulnerabilities
    - Rapid accumulation of points through fake purchases
    - Redemption patterns that avoid detection
    - Return fraud combined with point retention
    - Multiple small transactions to maximize rewards
    - Point transfer to multiple accounts
    
    Example: 10 transactions of Rs.1,999 each to stay below threshold while maximizing points
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.points_threshold = 2000  # Suspicious if optimizing around thresholds
        self.redemption_rate_normal = 0.15  # Normal customers redeem 15% of points
    
    def should_apply(self, customer: Any, transaction: Dict[str, Any],
                     customer_history: List[Dict[str, Any]]) -> bool:
        """Loyalty abuse shows in transaction patterns."""
        return len(customer_history) > 8 and random.random() < 0.04
    
    def apply_pattern(self, transaction: Dict[str, Any], customer: Any,
                     customer_history: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], FraudIndicator]:
        """Apply loyalty abuse pattern."""
        
        # Amount just below threshold to maximize points
        threshold_amounts = [1999, 4999, 9999, 19999]
        transaction['Amount'] = random.choice(threshold_amounts) + random.uniform(-50, 50)
        
        # Categories with high loyalty rewards
        loyalty_categories = ['Shopping', 'Fashion', 'Electronics', 'Dining', 'Travel']
        transaction['Category'] = random.choice(loyalty_categories)
        
        # Mix of channels
        transaction['Channel'] = random.choice(['Online', 'POS'])
        transaction['Device_Type'] = 'Mobile' if transaction['Channel'] == 'Online' else 'POS'
        
        # Usually approved (real transactions)
        transaction['Transaction_Status'] = 'Approved'
        
        # Count threshold-optimized transactions
        threshold_txns = sum(1 for t in customer_history 
                           if any(abs(t.get('Amount', 0) - thresh) < 100 for thresh in threshold_amounts))
        
        # Calculate points-earning efficiency
        recent_txns = customer_history[-20:] if len(customer_history) >= 20 else customer_history
        loyalty_category_count = sum(1 for t in recent_txns if t.get('Category') in loyalty_categories)
        loyalty_ratio = loyalty_category_count / len(recent_txns) if recent_txns else 0
        
        # Build evidence
        evidence = {
            'transaction_amount': round(transaction['Amount'], 2),
            'nearest_threshold': min(threshold_amounts, key=lambda x: abs(transaction['Amount'] - x)),
            'distance_from_threshold': round(abs(transaction['Amount'] - 
                                             min(threshold_amounts, key=lambda x: abs(transaction['Amount'] - x))), 2),
            'category': transaction['Category'],
            'threshold_optimized_transactions': threshold_txns,
            'loyalty_category_ratio': round(loyalty_ratio, 3),
            'recent_transaction_count': len(recent_txns),
            'points_optimization_detected': abs(transaction['Amount'] - 
                                               min(threshold_amounts, key=lambda x: abs(transaction['Amount'] - x))) < 100,
            'high_loyalty_focus': loyalty_ratio > 0.7
        }
        
        # Calculate confidence
        confidence = self.calculate_confidence(evidence)
        
        # Determine severity
        if threshold_txns >= 8 and loyalty_ratio > 0.8:
            severity = 'critical'
        elif threshold_txns >= 5 and loyalty_ratio > 0.7:
            severity = 'high'
        elif threshold_txns >= 3:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Create fraud indicator
        reason = (f"Loyalty program abuse detected. Transaction amount (Rs.{round(transaction['Amount'], 2)}) "
                 f"is Rs.{round(evidence['distance_from_threshold'], 2)} from threshold "
                 f"(Rs.{evidence['nearest_threshold']}). ")
        if threshold_txns >= 3:
            reason += f"{threshold_txns} similar threshold-optimized transactions detected. "
        if loyalty_ratio > 0.7:
            reason += f"{round(loyalty_ratio * 100, 1)}% of transactions in high-reward categories."
        
        fraud_indicator = FraudIndicator(
            fraud_type=FraudType.LOYALTY_ABUSE,
            confidence=confidence,
            reason=reason,
            evidence=evidence,
            severity=severity
        )
        
        # Mark as fraud
        transaction['Is_Fraud'] = 1
        transaction['Fraud_Type'] = FraudType.LOYALTY_ABUSE.value
        
        return transaction, fraud_indicator
    
    def calculate_confidence(self, evidence: Dict[str, Any]) -> float:
        """Calculate confidence based on loyalty abuse indicators."""
        confidence = 0.2  # Base confidence
        
        # Threshold optimization
        if evidence['points_optimization_detected']:
            confidence += 0.3
        
        # Multiple threshold-optimized transactions
        threshold_count = evidence['threshold_optimized_transactions']
        if threshold_count >= 8:
            confidence += 0.3
        elif threshold_count >= 5:
            confidence += 0.2
        elif threshold_count >= 3:
            confidence += 0.1
        
        # High loyalty category focus
        if evidence['high_loyalty_focus']:
            confidence += 0.2
        
        return min(confidence, 1.0)


class FraudPatternGenerator:
    """
    Main fraud pattern generator that coordinates all fraud patterns.
    
    Manages fraud injection rates, pattern selection, and ensures realistic
    distribution of fraud across customer segments and time periods.
    
    Usage:
        fraud_gen = FraudPatternGenerator(fraud_rate=0.02, seed=42)
        transaction, fraud_info = fraud_gen.maybe_apply_fraud(transaction, customer, history)
    """
    
    def __init__(self, fraud_rate: float = 0.02, seed: Optional[int] = None):
        """
        Initialize fraud pattern generator.
        
        Args:
            fraud_rate: Probability of fraud (0.0-1.0). Default 2% (0.02)
            seed: Random seed for reproducibility
        """
        self.fraud_rate = max(0.0, min(1.0, fraud_rate))  # Clamp between 0 and 1
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        
        # Initialize all fraud pattern types
        self.patterns = {
            # Base patterns (Days 1-2)
            FraudType.CARD_CLONING: CardCloningPattern(seed),
            FraudType.ACCOUNT_TAKEOVER: AccountTakeoverPattern(seed),
            FraudType.MERCHANT_COLLUSION: MerchantCollusionPattern(seed),
            FraudType.VELOCITY_ABUSE: VelocityAbusePattern(seed),
            FraudType.AMOUNT_MANIPULATION: AmountManipulationPattern(seed),
            FraudType.REFUND_FRAUD: RefundFraudPattern(seed),
            FraudType.STOLEN_CARD: StolenCardPattern(seed),
            FraudType.SYNTHETIC_IDENTITY: SyntheticIdentityPattern(seed),
            FraudType.FIRST_PARTY_FRAUD: FirstPartyFraudPattern(seed),
            FraudType.FRIENDLY_FRAUD: FriendlyFraudPattern(seed),
            # Advanced patterns (Days 3-4)
            FraudType.TRANSACTION_REPLAY: TransactionReplayPattern(seed),
            FraudType.CARD_TESTING: CardTestingPattern(seed),
            FraudType.MULE_ACCOUNT: MuleAccountPattern(seed),
            FraudType.SHIPPING_FRAUD: ShippingFraudPattern(seed),
            FraudType.LOYALTY_ABUSE: LoyaltyAbusePattern(seed),
        }
        
        # Track fraud statistics
        self.fraud_counts = {fraud_type: 0 for fraud_type in FraudType}
        self.total_transactions = 0
        self.total_fraud = 0
        
        # Cross-pattern statistics (Days 3-4 Task 4)
        from collections import defaultdict
        self.pattern_co_occurrences = defaultdict(lambda: defaultdict(int))
        self.fraud_cascades = []  # Track sequential fraud patterns
        self.pattern_isolation_stats = defaultdict(int)  # Track isolated patterns
    
    def maybe_apply_fraud(self, transaction: Dict[str, Any], customer: Any,
                         customer_history: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Optional[FraudIndicator]]:
        """
        Decide whether to apply fraud pattern to transaction.
        
        Args:
            transaction: Transaction dictionary to potentially modify
            customer: Customer profile
            customer_history: List of customer's previous transactions
            
        Returns:
            Tuple of (transaction, fraud_indicator or None)
        """
        self.total_transactions += 1
        
        # Decide if this transaction should be fraudulent
        if random.random() >= self.fraud_rate:
            return transaction, None
        
        # Select applicable fraud patterns
        applicable_patterns = []
        for fraud_type, pattern in self.patterns.items():
            if pattern.should_apply(customer, transaction, customer_history):
                applicable_patterns.append((fraud_type, pattern))
        
        # If no patterns applicable, skip fraud injection
        if not applicable_patterns:
            return transaction, None

        # If multiple patterns are applicable, prefer applying a combined pattern
        # to create realistic multi-pattern fraud scenarios.
        if len(applicable_patterns) >= 2:
            # Build list of pattern objects to apply
            patterns_to_apply = [p for (_ft, p) in applicable_patterns]
            combiner = FraudCombinationGenerator(seed=self.seed)
            modified_transaction, fraud_indicator = combiner.combine_and_apply(
                transaction.copy(), patterns_to_apply, customer, customer_history
            )
            # Update counts for each involved fraud type
            fraud_types_involved = []
            for ft, _ in applicable_patterns:
                self.fraud_counts[ft] = self.fraud_counts.get(ft, 0) + 1
                fraud_types_involved.append(ft)
            
            # Track co-occurrences
            self._track_co_occurrences(fraud_types_involved)
            
            self.total_fraud += 1
            return modified_transaction, fraud_indicator

        # Otherwise apply one randomly selected pattern
        fraud_type, pattern = random.choice(applicable_patterns)

        # Apply pattern
        modified_transaction, fraud_indicator = pattern.apply_pattern(
            transaction.copy(), customer, customer_history
        )

        # Update statistics
        self.fraud_counts[fraud_type] += 1
        self.total_fraud += 1

        return modified_transaction, fraud_indicator
    
    def get_fraud_statistics(self) -> Dict[str, Any]:
        """
        Get fraud injection statistics.
        
        Returns:
            Dictionary with fraud statistics
        """
        return {
            'total_transactions': self.total_transactions,
            'total_fraud': self.total_fraud,
            'fraud_rate': self.total_fraud / self.total_transactions if self.total_transactions > 0 else 0.0,
            'target_fraud_rate': self.fraud_rate,
            'fraud_by_type': {
                fraud_type.value: count 
                for fraud_type, count in self.fraud_counts.items() 
                if count > 0
            },
            'fraud_type_distribution': {
                fraud_type.value: count / self.total_fraud if self.total_fraud > 0 else 0.0
                for fraud_type, count in self.fraud_counts.items()
            }
        }
    
    def reset_statistics(self):
        """Reset fraud statistics counters."""
        self.fraud_counts = {fraud_type: 0 for fraud_type in FraudType}
        self.total_transactions = 0
        self.total_fraud = 0
        # Reset cross-pattern statistics
        from collections import defaultdict
        self.pattern_co_occurrences = defaultdict(lambda: defaultdict(int))
        self.fraud_cascades = []
        self.pattern_isolation_stats = defaultdict(int)
    
    def set_fraud_rate(self, new_rate: float):
        """
        Update fraud injection rate.
        
        Args:
            new_rate: New fraud rate (0.0-1.0)
        """
        self.fraud_rate = max(0.0, min(1.0, new_rate))
    
    def _track_co_occurrences(self, fraud_types: List[FraudType]):
        """
        Track co-occurrence of fraud patterns.
        
        Args:
            fraud_types: List of fraud types that occurred together
        """
        # Record co-occurrence for each pair
        for i, ft1 in enumerate(fraud_types):
            for ft2 in fraud_types[i+1:]:
                self.pattern_co_occurrences[ft1][ft2] += 1
                self.pattern_co_occurrences[ft2][ft1] += 1
        
        # Track if pattern appeared isolated (alone)
        if len(fraud_types) == 1:
            self.pattern_isolation_stats[fraud_types[0]] += 1
    
    def get_pattern_co_occurrence_matrix(self) -> Dict[str, Dict[str, int]]:
        """
        Get co-occurrence matrix showing which patterns appear together.
        
        Returns:
            Dictionary mapping fraud type pairs to co-occurrence counts
        """
        matrix = {}
        for ft1 in FraudType:
            matrix[ft1.value] = {}
            for ft2 in FraudType:
                if ft1 in self.pattern_co_occurrences and ft2 in self.pattern_co_occurrences[ft1]:
                    matrix[ft1.value][ft2.value] = self.pattern_co_occurrences[ft1][ft2]
                else:
                    matrix[ft1.value][ft2.value] = 0
        return matrix
    
    def get_pattern_isolation_stats(self) -> Dict[str, Any]:
        """
        Get statistics on pattern isolation.
        
        Returns:
            Dictionary with isolation statistics for each pattern
        """
        isolation_stats = {}
        for fraud_type in FraudType:
            total_occurrences = self.fraud_counts.get(fraud_type, 0)
            isolated_occurrences = self.pattern_isolation_stats.get(fraud_type, 0)
            
            if total_occurrences > 0:
                isolation_rate = isolated_occurrences / total_occurrences
            else:
                isolation_rate = 0.0
            
            isolation_stats[fraud_type.value] = {
                'total_occurrences': total_occurrences,
                'isolated_occurrences': isolated_occurrences,
                'combined_occurrences': total_occurrences - isolated_occurrences,
                'isolation_rate': round(isolation_rate, 3),
            }
        
        return isolation_stats
    
    def get_cross_pattern_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cross-pattern statistics.
        
        Returns:
            Dictionary with co-occurrence matrix, isolation stats, and pattern analysis
        """
        # Calculate overall isolation rate
        total_isolated = sum(self.pattern_isolation_stats.values())
        total_patterns_applied = sum(self.fraud_counts.values())
        overall_isolation_rate = total_isolated / total_patterns_applied if total_patterns_applied > 0 else 0.0
        
        # Find most common co-occurrences
        co_occurrence_pairs = []
        for ft1 in self.pattern_co_occurrences:
            for ft2, count in self.pattern_co_occurrences[ft1].items():
                if ft1.value < ft2.value:  # Avoid duplicates
                    co_occurrence_pairs.append({
                        'pattern_1': ft1.value,
                        'pattern_2': ft2.value,
                        'count': count,
                    })
        
        co_occurrence_pairs.sort(key=lambda x: x['count'], reverse=True)
        
        return {
            'co_occurrence_matrix': self.get_pattern_co_occurrence_matrix(),
            'isolation_stats': self.get_pattern_isolation_stats(),
            'overall_isolation_rate': round(overall_isolation_rate, 3),
            'total_isolated_patterns': total_isolated,
            'total_combined_patterns': total_patterns_applied - total_isolated,
            'most_common_combinations': co_occurrence_pairs[:10],  # Top 10
            'patterns_meeting_isolation_target': [
                ft.value for ft, stats in self.get_pattern_isolation_stats().items()
                if stats.get('isolation_rate', 0) >= 0.95  # 95%+ isolation target
            ],
        }


def apply_fraud_labels(transaction: Dict[str, Any], fraud_indicator: Optional[FraudIndicator]) -> Dict[str, Any]:
    """
    Apply fraud labels and explanations to transaction.
    
    Adds the following fields to transaction:
    - Fraud_Type: Type of fraud detected
    - Fraud_Confidence: Confidence score (0.0-1.0)
    - Fraud_Reason: Detailed explanation
    - Fraud_Severity: Severity level (low, medium, high, critical)
    - Fraud_Evidence: JSON string of evidence dictionary
    
    Args:
        transaction: Transaction dictionary
        fraud_indicator: Fraud indicator object (None if no fraud)
        
    Returns:
        Transaction with fraud labels added
    """
    if fraud_indicator is None:
        # No fraud - set default values
        transaction['Fraud_Type'] = 'None'
        transaction['Fraud_Confidence'] = 0.0
        transaction['Fraud_Reason'] = 'No fraud detected'
        transaction['Fraud_Severity'] = 'none'
        transaction['Fraud_Evidence'] = '{}'
    else:
        # Apply fraud labels
        transaction['Fraud_Type'] = fraud_indicator.fraud_type.value
        transaction['Fraud_Confidence'] = round(fraud_indicator.confidence, 3)
        transaction['Fraud_Reason'] = fraud_indicator.reason
        transaction['Fraud_Severity'] = fraud_indicator.severity
        
        # Convert evidence to JSON-compatible format
        import json
        transaction['Fraud_Evidence'] = json.dumps(fraud_indicator.evidence)
    
    return transaction


# Convenience function for quick fraud injection
def inject_fraud_into_dataset(transactions: List[Dict[str, Any]], 
                              customers: List[Any],
                              fraud_rate: float = 0.02,
                              seed: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Inject fraud patterns into existing transaction dataset.
    
    Args:
        transactions: List of transaction dictionaries
        customers: List of customer profiles
        fraud_rate: Target fraud rate (0.0-1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (modified transactions, fraud statistics)
    """
    fraud_gen = FraudPatternGenerator(fraud_rate=fraud_rate, seed=seed)
    
    # Build customer history map
    customer_history = {}
    for customer in customers:
        customer_history[customer.customer_id] = []
    
    # Process transactions
    modified_transactions = []
    for txn in transactions:
        customer_id = txn.get('Customer_ID')
        customer = next((c for c in customers if c.customer_id == customer_id), None)
        
        if customer is None:
            modified_transactions.append(txn)
            continue
        
        # Get customer's history up to this point
        history = customer_history.get(customer_id, [])
        
        # Maybe apply fraud
        modified_txn, fraud_indicator = fraud_gen.maybe_apply_fraud(txn, customer, history)
        
        # Apply labels
        modified_txn = apply_fraud_labels(modified_txn, fraud_indicator)
        
        # Add to history
        customer_history[customer_id].append(modified_txn)
        modified_transactions.append(modified_txn)
    
    # Get statistics
    stats = fraud_gen.get_fraud_statistics()
    
    return modified_transactions, stats
