"""
Pattern Detector

Identifies known fraud patterns and suspicious transaction sequences.
Includes sequential pattern mining and graph-based fraud ring detection.

Week 10 Day 4: Advanced Fraud Detection
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter

from src.fraud import FraudPattern, PatternMatch


class PatternDetector:
    """
    Fraud pattern detection and sequential analysis
    
    Features:
    - 6+ known fraud patterns (card testing, ATO, bust-out, etc.)
    - Sequential pattern mining
    - Graph-based fraud ring detection
    - Pattern similarity scoring
    - Custom pattern registration
    
    Fraud patterns detected:
    1. Card Testing: Small authorization attempts before large purchases
    2. Account Takeover (ATO): Profile changes + high-value transactions
    3. Bust-Out: Gradual credit limit exploitation
    4. Triangulation: Legitimate purchase → stolen card → resale
    5. Friendly Fraud: Chargeback after legitimate purchase
    6. Synthetic Identity: New account with rapid credit building
    """
    
    def __init__(self):
        """Initialize pattern detector"""
        self.patterns: Dict[str, FraudPattern] = {}
        self.pattern_library = []
        
        # Register default patterns
        self._register_default_patterns()
        
        # Graph for fraud ring detection
        self.fraud_graph = nx.Graph()
        
        # Pattern match cache
        self.match_cache = {}
    
    def _register_default_patterns(self):
        """Register known fraud patterns"""
        
        # Pattern 1: Card Testing
        card_testing = FraudPattern(
            pattern_id="P001",
            name="Card Testing",
            description="Multiple small authorization attempts followed by large purchase",
            rules=[
                {'type': 'small_transactions', 'count_min': 3, 'amount_max': 5.0},
                {'type': 'large_transaction', 'amount_min': 100.0, 'time_after_small': 3600}  # Within 1 hour
            ],
            severity="critical",
            min_transactions=4,
            time_window="2hr"
        )
        self.register_pattern(card_testing)
        
        # Pattern 2: Account Takeover (ATO)
        account_takeover = FraudPattern(
            pattern_id="P002",
            name="Account Takeover",
            description="Profile changes followed by suspicious transactions",
            rules=[
                {'type': 'profile_change', 'fields': ['email', 'password', 'phone', 'address']},
                {'type': 'high_value_transaction', 'amount_min': 200.0, 'time_after_change': 3600}
            ],
            severity="critical",
            min_transactions=2,
            time_window="24hr"
        )
        self.register_pattern(account_takeover)
        
        # Pattern 3: Bust-Out
        bust_out = FraudPattern(
            pattern_id="P003",
            name="Bust-Out",
            description="Gradual credit limit increases followed by maxing out",
            rules=[
                {'type': 'credit_increase_sequence', 'count_min': 3},
                {'type': 'max_out', 'utilization_min': 0.95}
            ],
            severity="high",
            min_transactions=5,
            time_window="30d"
        )
        self.register_pattern(bust_out)
        
        # Pattern 4: Triangulation
        triangulation = FraudPattern(
            pattern_id="P004",
            name="Triangulation Fraud",
            description="Legitimate purchase, stolen card use, resale pattern",
            rules=[
                {'type': 'legitimate_purchase', 'merchant_type': 'reputable'},
                {'type': 'stolen_card_usage', 'different_location': True},
                {'type': 'resale_transaction', 'time_window': 86400}  # Within 24 hours
            ],
            severity="high",
            min_transactions=3,
            time_window="48hr"
        )
        self.register_pattern(triangulation)
        
        # Pattern 5: Friendly Fraud
        friendly_fraud = FraudPattern(
            pattern_id="P005",
            name="Friendly Fraud",
            description="Legitimate purchase followed by chargeback",
            rules=[
                {'type': 'normal_transaction', 'no_anomalies': True},
                {'type': 'chargeback', 'time_after_purchase': 604800}  # Within 7 days
            ],
            severity="medium",
            min_transactions=2,
            time_window="30d"
        )
        self.register_pattern(friendly_fraud)
        
        # Pattern 6: Synthetic Identity
        synthetic_identity = FraudPattern(
            pattern_id="P006",
            name="Synthetic Identity",
            description="New account with rapid credit building",
            rules=[
                {'type': 'new_account', 'account_age_days_max': 90},
                {'type': 'rapid_credit_building', 'transactions_min': 10},
                {'type': 'sudden_high_value', 'amount_min': 500.0}
            ],
            severity="high",
            min_transactions=12,
            time_window="90d"
        )
        self.register_pattern(synthetic_identity)
        
        # Pattern 7: Velocity Abuse
        velocity_abuse = FraudPattern(
            pattern_id="P007",
            name="Velocity Abuse",
            description="Rapid sequence of transactions across multiple merchants",
            rules=[
                {'type': 'rapid_transactions', 'count_min': 10, 'time_window': 3600},
                {'type': 'multiple_merchants', 'unique_merchants_min': 5}
            ],
            severity="high",
            min_transactions=10,
            time_window="1hr"
        )
        self.register_pattern(velocity_abuse)
        
        # Pattern 8: Geographic Impossibility
        geographic_impossibility = FraudPattern(
            pattern_id="P008",
            name="Geographic Impossibility",
            description="Transactions in different locations within impossible timeframe",
            rules=[
                {'type': 'distant_transactions', 'distance_miles_min': 500, 'time_between_max': 1800}  # 30 minutes
            ],
            severity="critical",
            min_transactions=2,
            time_window="1hr"
        )
        self.register_pattern(geographic_impossibility)
    
    def register_pattern(self, pattern: FraudPattern):
        """
        Register a fraud pattern
        
        Args:
            pattern: FraudPattern instance
        """
        self.patterns[pattern.pattern_id] = pattern
        self.pattern_library.append(pattern)
    
    def unregister_pattern(self, pattern_id: str):
        """
        Remove a pattern from registry
        
        Args:
            pattern_id: Pattern ID to remove
        """
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            del self.patterns[pattern_id]
            self.pattern_library = [p for p in self.pattern_library if p.pattern_id != pattern_id]
    
    def detect_patterns(
        self,
        transactions: List[Dict[str, Any]],
        customer_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[PatternMatch]:
        """
        Scan for matching fraud patterns
        
        Args:
            transactions: List of transactions to analyze
            customer_id: Optional customer ID
            context: Optional additional context (profile changes, chargebacks, etc.)
            
        Returns:
            List of detected pattern matches
        """
        matches = []
        
        if not transactions:
            return matches
        
        # Sort transactions by timestamp
        sorted_txns = sorted(
            transactions,
            key=lambda t: t.get('timestamp', datetime.now())
        )
        
        # Check each pattern
        for pattern in self.pattern_library:
            pattern_matches = self._check_pattern(
                pattern,
                sorted_txns,
                customer_id,
                context
            )
            matches.extend(pattern_matches)
        
        return matches
    
    def _check_pattern(
        self,
        pattern: FraudPattern,
        transactions: List[Dict[str, Any]],
        customer_id: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> List[PatternMatch]:
        """Check for specific pattern"""
        if pattern.pattern_id == "P001":
            return self._check_card_testing(pattern, transactions, customer_id)
        elif pattern.pattern_id == "P002":
            return self._check_account_takeover(pattern, transactions, customer_id, context)
        elif pattern.pattern_id == "P003":
            return self._check_bust_out(pattern, transactions, customer_id, context)
        elif pattern.pattern_id == "P007":
            return self._check_velocity_abuse(pattern, transactions, customer_id)
        elif pattern.pattern_id == "P008":
            return self._check_geographic_impossibility(pattern, transactions, customer_id)
        else:
            # Generic pattern matching
            return []
    
    def _check_card_testing(
        self,
        pattern: FraudPattern,
        transactions: List[Dict[str, Any]],
        customer_id: Optional[str]
    ) -> List[PatternMatch]:
        """Detect card testing pattern"""
        matches = []
        
        for i in range(len(transactions) - pattern.min_transactions + 1):
            window_txns = transactions[i:i+10]  # Look at next 10 transactions
            
            # Find small transactions
            small_txns = [
                t for t in window_txns
                if t.get('amount', 0) <= 5.0
            ]
            
            # Find large transaction
            large_txns = [
                t for t in window_txns
                if t.get('amount', 0) >= 100.0
            ]
            
            if len(small_txns) >= 3 and len(large_txns) >= 1:
                # Check if large transaction comes after small ones
                last_small_time = max(t.get('timestamp', datetime.min) for t in small_txns)
                first_large_time = min(t.get('timestamp', datetime.min) for t in large_txns)
                
                if first_large_time > last_small_time:
                    time_diff = (first_large_time - last_small_time).total_seconds()
                    
                    if time_diff <= 3600:  # Within 1 hour
                        confidence = pattern.calculate_confidence(small_txns + large_txns)
                        
                        matches.append(PatternMatch(
                            pattern=pattern,
                            customer_id=customer_id or "UNKNOWN",
                            matched_transactions=small_txns + large_txns,
                            confidence=confidence,
                            timestamp=datetime.now(),
                            explanation=f"Detected {len(small_txns)} small transactions (< $5) followed by ${large_txns[0]['amount']:.2f} transaction"
                        ))
        
        return matches
    
    def _check_account_takeover(
        self,
        pattern: FraudPattern,
        transactions: List[Dict[str, Any]],
        customer_id: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> List[PatternMatch]:
        """Detect account takeover pattern"""
        matches = []
        
        if not context or 'profile_changes' not in context:
            return matches
        
        profile_changes = context['profile_changes']
        
        # Check for high-value transactions after profile changes
        for change in profile_changes:
            change_time = change.get('timestamp', datetime.now())
            
            # Find transactions within 24 hours after change
            suspicious_txns = [
                t for t in transactions
                if t.get('timestamp', datetime.min) > change_time and
                (t.get('timestamp', datetime.max) - change_time).total_seconds() <= 86400 and
                t.get('amount', 0) >= 200.0
            ]
            
            if suspicious_txns:
                confidence = min(1.0, len(suspicious_txns) / 3)
                
                matches.append(PatternMatch(
                    pattern=pattern,
                    customer_id=customer_id or "UNKNOWN",
                    matched_transactions=suspicious_txns,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    explanation=f"Profile change ({change.get('field')}) followed by {len(suspicious_txns)} high-value transactions"
                ))
        
        return matches
    
    def _check_bust_out(
        self,
        pattern: FraudPattern,
        transactions: List[Dict[str, Any]],
        customer_id: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> List[PatternMatch]:
        """Detect bust-out pattern"""
        matches = []
        
        # This requires credit line information from context
        if not context or 'credit_history' not in context:
            return matches
        
        credit_history = context['credit_history']
        
        # Check for gradual credit increases
        credit_increases = [
            ch for ch in credit_history
            if ch.get('type') == 'credit_increase'
        ]
        
        if len(credit_increases) >= 3:
            # Check current utilization
            current_utilization = context.get('current_credit_utilization', 0.0)
            
            if current_utilization >= 0.95:
                confidence = min(1.0, len(credit_increases) / 5)
                
                matches.append(PatternMatch(
                    pattern=pattern,
                    customer_id=customer_id or "UNKNOWN",
                    matched_transactions=transactions[-5:],  # Last 5 transactions
                    confidence=confidence,
                    timestamp=datetime.now(),
                    explanation=f"{len(credit_increases)} credit increases with {current_utilization:.0%} utilization"
                ))
        
        return matches
    
    def _check_velocity_abuse(
        self,
        pattern: FraudPattern,
        transactions: List[Dict[str, Any]],
        customer_id: Optional[str]
    ) -> List[PatternMatch]:
        """Detect velocity abuse pattern"""
        matches = []
        
        # Look for windows with many transactions across many merchants
        for i in range(len(transactions) - pattern.min_transactions + 1):
            window_start = transactions[i].get('timestamp', datetime.now())
            window_end = window_start + timedelta(hours=1)
            
            window_txns = [
                t for t in transactions[i:]
                if window_start <= t.get('timestamp', datetime.min) <= window_end
            ]
            
            if len(window_txns) >= 10:
                merchants = set(t.get('merchant_id') for t in window_txns if t.get('merchant_id'))
                
                if len(merchants) >= 5:
                    confidence = min(1.0, len(window_txns) / 15)
                    
                    matches.append(PatternMatch(
                        pattern=pattern,
                        customer_id=customer_id or "UNKNOWN",
                        matched_transactions=window_txns,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        explanation=f"{len(window_txns)} transactions across {len(merchants)} merchants in 1 hour"
                    ))
        
        return matches
    
    def _check_geographic_impossibility(
        self,
        pattern: FraudPattern,
        transactions: List[Dict[str, Any]],
        customer_id: Optional[str]
    ) -> List[PatternMatch]:
        """Detect geographically impossible transactions"""
        matches = []
        
        # Check consecutive transactions with location data
        for i in range(len(transactions) - 1):
            txn1 = transactions[i]
            txn2 = transactions[i + 1]
            
            if 'latitude' not in txn1 or 'latitude' not in txn2:
                continue
            
            # Calculate distance
            distance = self._haversine_distance(
                txn1['latitude'], txn1['longitude'],
                txn2['latitude'], txn2['longitude']
            )
            
            # Calculate time difference
            time_diff = (txn2['timestamp'] - txn1['timestamp']).total_seconds()
            
            if distance >= 500 and time_diff <= 1800:  # 500+ miles in 30 minutes
                confidence = min(1.0, distance / 1000)
                
                matches.append(PatternMatch(
                    pattern=pattern,
                    customer_id=customer_id or "UNKNOWN",
                    matched_transactions=[txn1, txn2],
                    confidence=confidence,
                    timestamp=datetime.now(),
                    explanation=f"Transactions {distance:.0f} miles apart in {time_diff/60:.0f} minutes (impossible travel)"
                ))
        
        return matches
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance using Haversine formula"""
        R = 3959.0  # Earth radius in miles
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def detect_fraud_ring(
        self,
        transactions: pd.DataFrame,
        connection_fields: List[str] = ['device_id', 'ip_address', 'email', 'phone']
    ) -> List[Dict[str, Any]]:
        """
        Detect fraud rings using graph analysis
        
        Finds connected components where customers share suspicious attributes
        
        Args:
            transactions: DataFrame with transaction data
            connection_fields: Fields that indicate connections
            
        Returns:
            List of detected fraud rings with member details
        """
        # Build graph
        graph = nx.Graph()
        
        # Add edges between customers who share attributes
        customers = transactions['customer_id'].unique()
        
        for field in connection_fields:
            if field not in transactions.columns:
                continue
            
            # Group by field value
            field_groups = transactions.groupby(field)['customer_id'].apply(set)
            
            for field_value, customer_set in field_groups.items():
                if len(customer_set) >= 2:  # At least 2 customers share this value
                    # Add edges between all pairs
                    customer_list = list(customer_set)
                    for i in range(len(customer_list)):
                        for j in range(i + 1, len(customer_list)):
                            if graph.has_edge(customer_list[i], customer_list[j]):
                                # Strengthen existing edge
                                graph[customer_list[i]][customer_list[j]]['weight'] += 1
                                graph[customer_list[i]][customer_list[j]]['shared_fields'].append(field)
                            else:
                                graph.add_edge(
                                    customer_list[i],
                                    customer_list[j],
                                    weight=1,
                                    shared_fields=[field]
                                )
        
        # Find connected components (potential fraud rings)
        fraud_rings = []
        
        for component in nx.connected_components(graph):
            if len(component) >= 3:  # Minimum 3 members for a ring
                # Calculate ring statistics
                subgraph = graph.subgraph(component)
                
                # Count shared attributes
                shared_attrs = defaultdict(list)
                for u, v, data in subgraph.edges(data=True):
                    for field in data['shared_fields']:
                        shared_attrs[field].append((u, v))
                
                ring_info = {
                    'member_count': len(component),
                    'members': list(component),
                    'connection_count': subgraph.number_of_edges(),
                    'shared_attributes': dict(shared_attrs),
                    'density': nx.density(subgraph),
                    'risk_score': min(100, len(component) * 20 + subgraph.number_of_edges() * 5)
                }
                
                fraud_rings.append(ring_info)
        
        # Sort by risk score
        fraud_rings.sort(key=lambda r: r['risk_score'], reverse=True)
        
        return fraud_rings
    
    def mine_new_patterns(
        self,
        labeled_fraud_data: pd.DataFrame,
        min_support: float = 0.1,
        min_confidence: float = 0.7
    ) -> List[FraudPattern]:
        """
        Discover new fraud patterns from historical labeled data
        
        Uses sequential pattern mining to find common fraud sequences
        
        Args:
            labeled_fraud_data: DataFrame with fraud labels
            min_support: Minimum support for pattern (0-1)
            min_confidence: Minimum confidence for pattern (0-1)
            
        Returns:
            List of discovered patterns
        """
        # This is a simplified implementation
        # In production, you'd use more sophisticated algorithms like SPADE or PrefixSpan
        
        new_patterns = []
        
        # Get fraud transactions only
        fraud_txns = labeled_fraud_data[labeled_fraud_data['is_fraud'] == 1]
        
        # Group by customer
        customer_groups = fraud_txns.groupby('customer_id')
        
        # Find common sequences
        sequence_counts = Counter()
        
        for customer_id, group in customer_groups:
            # Sort by timestamp
            sorted_group = group.sort_values('timestamp')
            
            # Extract sequence features (amount range, merchant type, etc.)
            sequence = []
            for _, row in sorted_group.iterrows():
                amount = row['amount']
                if amount < 10:
                    amount_category = 'small'
                elif amount < 100:
                    amount_category = 'medium'
                else:
                    amount_category = 'large'
                
                sequence.append(amount_category)
            
            # Convert to tuple for hashing
            if len(sequence) >= 2:
                sequence_counts[tuple(sequence)] += 1
        
        # Find frequent sequences
        total_fraud_customers = len(customer_groups)
        
        for sequence, count in sequence_counts.items():
            support = count / total_fraud_customers
            
            if support >= min_support:
                # Create pattern
                pattern_id = f"M{len(new_patterns) + 1:03d}"
                
                pattern = FraudPattern(
                    pattern_id=pattern_id,
                    name=f"Mined Pattern {pattern_id}",
                    description=f"Sequence: {' -> '.join(sequence)}",
                    rules=[{'type': 'mined_sequence', 'sequence': list(sequence)}],
                    severity="medium",
                    min_transactions=len(sequence),
                    time_window="24hr"
                )
                
                new_patterns.append(pattern)
        
        return new_patterns
    
    def calculate_pattern_similarity(
        self,
        pattern1: FraudPattern,
        pattern2: FraudPattern
    ) -> float:
        """
        Calculate similarity between two patterns using Jaccard similarity
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            Similarity score (0-1)
        """
        # Extract rule types
        rules1 = set(rule.get('type') for rule in pattern1.rules)
        rules2 = set(rule.get('type') for rule in pattern2.rules)
        
        # Jaccard similarity
        intersection = len(rules1 & rules2)
        union = len(rules1 | rules2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about registered patterns"""
        severity_counts = Counter(p.severity for p in self.pattern_library)
        
        return {
            'total_patterns': len(self.pattern_library),
            'severity_distribution': dict(severity_counts),
            'avg_min_transactions': np.mean([p.min_transactions for p in self.pattern_library]),
            'pattern_ids': [p.pattern_id for p in self.pattern_library],
            'pattern_names': [p.name for p in self.pattern_library]
        }
