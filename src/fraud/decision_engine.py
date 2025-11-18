"""
Decision Engine

Automated fraud decision-making with configurable business rules.
Multi-tier risk classification and false positive minimization.

Week 10 Day 4: Advanced Fraud Detection
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from src.fraud import FraudDecision, DecisionType, RiskScore, CustomerProfile


class DecisionEngine:
    """
    Automated fraud decision engine
    
    Features:
    - Multi-tier risk classification (APPROVE, REVIEW_LOW, REVIEW_HIGH, REVIEW_URGENT, DECLINE)
    - Configurable thresholds per customer segment, amount tier, merchant category
    - Whitelist management for trusted entities
    - False positive tracking and adaptive thresholds
    - Manual review queue prioritization
    - Explainable decisions with reasoning
    
    Decision tiers:
    - APPROVE (0-30): Auto-approve, no review
    - REVIEW_LOW (31-50): Flag for periodic audit
    - REVIEW_HIGH (51-70): Manual review within 24 hours
    - REVIEW_URGENT (71-85): Immediate manual review
    - DECLINE (86-100): Auto-decline, alert customer
    """
    
    def __init__(self):
        """Initialize decision engine"""
        # Default thresholds (can be customized)
        self.thresholds = {
            'default': {
                DecisionType.APPROVE: (0, 30),
                DecisionType.REVIEW_LOW: (31, 50),
                DecisionType.REVIEW_HIGH: (51, 70),
                DecisionType.REVIEW_URGENT: (71, 85),
                DecisionType.DECLINE: (86, 100)
            }
        }
        
        # Whitelists
        self.whitelisted_customers: Set[str] = set()
        self.whitelisted_merchants: Set[str] = set()
        self.whitelisted_pairs: Set[Tuple[str, str]] = set()  # (customer_id, merchant_id)
        
        # False positive tracking
        self.false_positive_history = defaultdict(list)
        
        # Decision statistics
        self.decision_counts = defaultdict(int)
        self.total_decisions = 0
    
    def make_decision(
        self,
        risk_score: RiskScore,
        transaction: Dict[str, Any],
        customer_profile: Optional[CustomerProfile] = None
    ) -> FraudDecision:
        """
        Generate automated decision with reasoning
        
        Args:
            risk_score: Calculated risk score
            transaction: Transaction data
            customer_profile: Optional customer profile
            
        Returns:
            FraudDecision with decision type and reasoning
        """
        customer_id = transaction.get('customer_id', 'UNKNOWN')
        transaction_id = transaction.get('transaction_id', 'UNKNOWN')
        amount = transaction.get('amount', 0.0)
        merchant_id = transaction.get('merchant_id', 'UNKNOWN')
        
        reasoning = []
        
        # Check whitelists first
        if self._is_whitelisted(customer_id, merchant_id):
            reasoning.append("Customer or merchant is whitelisted")
            
            decision = FraudDecision(
                transaction_id=transaction_id,
                customer_id=customer_id,
                decision=DecisionType.APPROVE,
                risk_score=risk_score.score,
                confidence=1.0,
                reasoning=reasoning,
                recommended_action="Auto-approve (whitelisted)",
                manual_review_priority=None
            )
            
            self._track_decision(decision.decision)
            return decision
        
        # Get appropriate thresholds
        thresholds = self._get_thresholds(transaction, customer_profile)
        
        # Determine decision based on risk score
        decision_type = self._classify_risk(risk_score.score, thresholds)
        
        # Build reasoning
        reasoning.append(f"Risk score: {risk_score.score:.1f}/100 ({risk_score.get_risk_level()})")
        reasoning.append(f"Confidence: {risk_score.confidence:.1%}")
        
        # Add top risk factors to reasoning
        for factor in risk_score.get_top_factors(3):
            reasoning.append(f"- {factor.name}: {factor.explanation}")
        
        # Add customer-specific reasoning
        if customer_profile:
            if customer_profile.transaction_count < 10:
                reasoning.append("New customer with limited transaction history")
            else:
                reasoning.append(f"Established customer ({customer_profile.transaction_count} transactions)")
        
        # Add amount-specific reasoning
        if amount > 1000:
            reasoning.append(f"High-value transaction: ${amount:.2f}")
        
        # Determine recommended action and priority
        if decision_type == DecisionType.APPROVE:
            recommended_action = "Auto-approve transaction"
            manual_review_priority = None
        elif decision_type == DecisionType.REVIEW_LOW:
            recommended_action = "Flag for periodic audit (low priority)"
            manual_review_priority = 3
        elif decision_type == DecisionType.REVIEW_HIGH:
            recommended_action = "Queue for manual review within 24 hours"
            manual_review_priority = 7
        elif decision_type == DecisionType.REVIEW_URGENT:
            recommended_action = "Immediate manual review required"
            manual_review_priority = 10
        else:  # DECLINE
            recommended_action = "Decline transaction and alert customer"
            manual_review_priority = None
        
        decision = FraudDecision(
            transaction_id=transaction_id,
            customer_id=customer_id,
            decision=decision_type,
            risk_score=risk_score.score,
            confidence=risk_score.confidence,
            reasoning=reasoning,
            recommended_action=recommended_action,
            manual_review_priority=manual_review_priority
        )
        
        self._track_decision(decision_type)
        
        return decision
    
    def _classify_risk(
        self,
        score: float,
        thresholds: Dict[DecisionType, Tuple[float, float]]
    ) -> DecisionType:
        """Classify risk score into decision type"""
        for decision_type, (min_score, max_score) in thresholds.items():
            if min_score <= score <= max_score:
                return decision_type
        
        # Default to REVIEW_HIGH if no match
        return DecisionType.REVIEW_HIGH
    
    def _get_thresholds(
        self,
        transaction: Dict[str, Any],
        customer_profile: Optional[CustomerProfile]
    ) -> Dict[DecisionType, Tuple[float, float]]:
        """
        Get appropriate thresholds based on transaction and customer
        
        Can be customized per:
        - Customer segment (new, established, VIP)
        - Transaction amount tier
        - Merchant category
        - Geographic region
        """
        # Start with default thresholds
        thresholds = self.thresholds['default'].copy()
        
        # Adjust for customer segment
        if customer_profile:
            if customer_profile.transaction_count < 10:
                # New customer: stricter thresholds
                segment_key = 'new_customer'
            elif customer_profile.transaction_count > 100 and customer_profile.avg_amount > 500:
                # VIP customer: more lenient thresholds
                segment_key = 'vip_customer'
            else:
                segment_key = 'default'
            
            if segment_key in self.thresholds:
                thresholds = self.thresholds[segment_key].copy()
        
        # Adjust for transaction amount
        amount = transaction.get('amount', 0.0)
        if amount > 5000:
            # High-value: stricter thresholds
            amount_tier_key = 'high_value'
            if amount_tier_key in self.thresholds:
                thresholds = self.thresholds[amount_tier_key].copy()
        
        return thresholds
    
    def configure_thresholds(
        self,
        tier: str,
        thresholds: Dict[DecisionType, Tuple[float, float]]
    ):
        """
        Set custom decision thresholds
        
        Args:
            tier: Threshold tier name (e.g., 'new_customer', 'vip_customer', 'high_value')
            thresholds: Dictionary mapping decision types to (min, max) score ranges
        """
        self.thresholds[tier] = thresholds
    
    def _is_whitelisted(self, customer_id: str, merchant_id: str) -> bool:
        """Check if customer or merchant is whitelisted"""
        return (
            customer_id in self.whitelisted_customers or
            merchant_id in self.whitelisted_merchants or
            (customer_id, merchant_id) in self.whitelisted_pairs
        )
    
    def add_to_whitelist(
        self,
        customer_id: Optional[str] = None,
        merchant_id: Optional[str] = None,
        reason: str = ""
    ):
        """
        Add entity to whitelist
        
        Args:
            customer_id: Customer ID to whitelist (optional)
            merchant_id: Merchant ID to whitelist (optional)
            reason: Reason for whitelisting
        """
        if customer_id:
            self.whitelisted_customers.add(customer_id)
        
        if merchant_id:
            self.whitelisted_merchants.add(merchant_id)
        
        if customer_id and merchant_id:
            self.whitelisted_pairs.add((customer_id, merchant_id))
    
    def remove_from_whitelist(
        self,
        customer_id: Optional[str] = None,
        merchant_id: Optional[str] = None
    ):
        """Remove entity from whitelist"""
        if customer_id and customer_id in self.whitelisted_customers:
            self.whitelisted_customers.remove(customer_id)
        
        if merchant_id and merchant_id in self.whitelisted_merchants:
            self.whitelisted_merchants.remove(merchant_id)
        
        if customer_id and merchant_id:
            pair = (customer_id, merchant_id)
            if pair in self.whitelisted_pairs:
                self.whitelisted_pairs.remove(pair)
    
    def track_false_positive(
        self,
        transaction_id: str,
        customer_id: str,
        original_decision: DecisionType,
        actual_fraud: bool = False
    ):
        """
        Track false positive for adaptive threshold adjustment
        
        Args:
            transaction_id: Transaction ID
            customer_id: Customer ID
            original_decision: Original decision made
            actual_fraud: Whether transaction was actually fraud
        """
        self.false_positive_history[customer_id].append({
            'transaction_id': transaction_id,
            'original_decision': original_decision,
            'actual_fraud': actual_fraud,
            'timestamp': datetime.now()
        })
    
    def get_false_positive_rate(
        self,
        customer_id: Optional[str] = None,
        lookback_days: int = 30
    ) -> float:
        """
        Calculate false positive rate
        
        Args:
            customer_id: Optional customer ID (if None, overall rate)
            lookback_days: Days to look back
            
        Returns:
            False positive rate (0-1)
        """
        cutoff_time = datetime.now() - timedelta(days=lookback_days)
        
        if customer_id:
            history = self.false_positive_history.get(customer_id, [])
        else:
            history = []
            for customer_history in self.false_positive_history.values():
                history.extend(customer_history)
        
        # Filter by time window
        recent_history = [
            h for h in history
            if h['timestamp'] >= cutoff_time
        ]
        
        if not recent_history:
            return 0.0
        
        # Count false positives (flagged as fraud but wasn't)
        false_positives = sum(
            1 for h in recent_history
            if h['original_decision'] != DecisionType.APPROVE and not h['actual_fraud']
        )
        
        total_flagged = sum(
            1 for h in recent_history
            if h['original_decision'] != DecisionType.APPROVE
        )
        
        if total_flagged == 0:
            return 0.0
        
        return false_positives / total_flagged
    
    def _track_decision(self, decision_type: DecisionType):
        """Track decision statistics"""
        self.decision_counts[decision_type] += 1
        self.total_decisions += 1
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get decision statistics"""
        stats = {
            'total_decisions': self.total_decisions,
            'decision_breakdown': dict(self.decision_counts),
            'approval_rate': 0.0,
            'decline_rate': 0.0,
            'review_rate': 0.0
        }
        
        if self.total_decisions > 0:
            stats['approval_rate'] = self.decision_counts.get(DecisionType.APPROVE, 0) / self.total_decisions
            stats['decline_rate'] = self.decision_counts.get(DecisionType.DECLINE, 0) / self.total_decisions
            
            review_count = (
                self.decision_counts.get(DecisionType.REVIEW_LOW, 0) +
                self.decision_counts.get(DecisionType.REVIEW_HIGH, 0) +
                self.decision_counts.get(DecisionType.REVIEW_URGENT, 0)
            )
            stats['review_rate'] = review_count / self.total_decisions
        
        return stats
    
    def get_review_queue(
        self,
        priority_threshold: int = 5
    ) -> List[FraudDecision]:
        """
        Get manual review queue sorted by priority
        
        Args:
            priority_threshold: Minimum priority (1-10)
            
        Returns:
            List of decisions needing review, sorted by priority
        """
        # This would typically query a database
        # For now, return empty list as placeholder
        return []
    
    def optimize_thresholds(
        self,
        target_false_positive_rate: float = 0.02,
        historical_data: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Auto-optimize thresholds to achieve target false positive rate
        
        Args:
            target_false_positive_rate: Desired FPR (default: 2%)
            historical_data: Historical decisions with outcomes
        """
        if not historical_data:
            return
        
        # This is a simplified optimization
        # In production, you'd use more sophisticated methods (ROC curve analysis, etc.)
        
        current_fpr = self.get_false_positive_rate()
        
        if current_fpr > target_false_positive_rate:
            # Too many false positives, raise thresholds
            adjustment = 5.0  # Increase threshold by 5 points
            
            for tier_name, tier_thresholds in self.thresholds.items():
                adjusted_thresholds = {}
                for decision_type, (min_score, max_score) in tier_thresholds.items():
                    adjusted_thresholds[decision_type] = (
                        min(100, min_score + adjustment),
                        min(100, max_score + adjustment)
                    )
                self.thresholds[tier_name] = adjusted_thresholds
        
        elif current_fpr < target_false_positive_rate * 0.5:
            # Too few false positives (maybe missing fraud), lower thresholds
            adjustment = -5.0  # Decrease threshold by 5 points
            
            for tier_name, tier_thresholds in self.thresholds.items():
                adjusted_thresholds = {}
                for decision_type, (min_score, max_score) in tier_thresholds.items():
                    adjusted_thresholds[decision_type] = (
                        max(0, min_score + adjustment),
                        max(0, max_score + adjustment)
                    )
                self.thresholds[tier_name] = adjusted_thresholds
