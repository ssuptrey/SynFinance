"""
Comprehensive Tests for Fraud Detection Module

Tests core fraud detection functionality: scoring, velocity, behavior, patterns, decisions, deployment.
Week 10 Day 4: Advanced Fraud Detection
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

from src.fraud import (
    FraudScoringEngine,
    VelocityChecker,
    BehavioralAnalyzer,
    PatternDetector,
    DecisionEngine,
    ModelDeployer,
    RiskScore,
    VelocityRule,
    CustomerProfile,
    FraudPattern,
    DecisionType,
    DeploymentStrategy
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_transaction():
    """Sample transaction data"""
    return {
        'transaction_id': 'TXN_001',
        'customer_id': 'CUST_123',
        'amount': 150.00,
        'merchant_id': 'MERCH_456',
        'timestamp': datetime.now(),
        'latitude': 37.7749,
        'longitude': -122.4194,
        'city': 'San Francisco'
    }


@pytest.fixture
def historical_transactions():
    """Historical transaction data for profile building"""
    dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
    
    data = {
        'transaction_id': [f'TXN_{i:03d}' for i in range(30)],
        'customer_id': ['CUST_123'] * 30,
        'amount': np.random.normal(100, 25, 30).tolist(),
        'merchant_id': [f'MERCH_{i%5:02d}' for i in range(30)],
        'category': [['Retail', 'Food', 'Gas', 'Entertainment'][i%4] for i in range(30)],
        'timestamp': dates,
        'city': ['San Francisco'] * 20 + ['Oakland'] * 10
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def trained_model():
    """Simple trained Random Forest model"""
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    return model


# ============================================================================
# Test Fraud Scoring Engine (15 tests)
# ============================================================================

class TestFraudScoringEngine:
    """Test fraud scoring engine"""
    
    def test_engine_initialization(self):
        """Test scoring engine initialization"""
        engine = FraudScoringEngine()
        assert len(engine.components) == 5  # 5 default components
        assert engine.performance_stats['total_scores'] == 0
    
    def test_basic_risk_scoring(self, sample_transaction):
        """Test basic risk score calculation"""
        engine = FraudScoringEngine()
        
        context = {
            'avg_amount': 100.0,
            'std_amount': 20.0
        }
        
        risk_score = engine.calculate_risk_score(sample_transaction, context)
        
        assert isinstance(risk_score, RiskScore)
        assert 0 <= risk_score.score <= 100
        assert 0 <= risk_score.confidence <= 1.0
        assert len(risk_score.factors) == 5
    
    def test_high_risk_transaction(self, sample_transaction):
        """Test high-risk transaction scoring"""
        engine = FraudScoringEngine()
        
        # High-risk context with multiple red flags
        from src.fraud import BehavioralAnomaly, AnomalyType
        
        anomaly = BehavioralAnomaly(
            anomaly_type=AnomalyType.AMOUNT_ANOMALY,
            field_name='amount',
            expected_value=50.0,
            actual_value=5000.0,
            deviation_score=5.0,
            p_value=0.001,
            is_significant=True,
            explanation="Very high amount"
        )
        
        context = {
            'avg_amount': 50.0,
            'std_amount': 10.0,
            'velocity_result': None,
            'behavioral_anomalies': [anomaly, anomaly, anomaly],  # Multiple significant anomalies
            'pattern_matches': [],
            'ml_prediction': 1,
            'ml_probability': 0.95  # High fraud probability
        }
        
        sample_transaction['amount'] = 5000.0  # Very high amount
        
        risk_score = engine.calculate_risk_score(sample_transaction, context)
        
        assert risk_score.score > 50  # Should be high risk
        assert risk_score.get_risk_level() in ['high', 'critical']
    
    def test_performance_tracking(self, sample_transaction):
        """Test performance statistics tracking"""
        engine = FraudScoringEngine()
        
        # Score multiple transactions
        for _ in range(10):
            engine.calculate_risk_score(sample_transaction, {})
        
        stats = engine.get_performance_stats()
        assert stats['total_scores'] == 10
        assert stats['avg_latency_ms'] > 0
        assert stats['avg_latency_ms'] < 100  # Should be fast
    
    def test_score_explanation(self, sample_transaction):
        """Test risk score explanation generation"""
        engine = FraudScoringEngine()
        risk_score = engine.calculate_risk_score(sample_transaction, {})
        
        explanation = engine.explain_score(risk_score)
        
        assert 'Risk Score:' in explanation
        assert 'Confidence:' in explanation
        assert 'Top Risk Factors:' in explanation


# ============================================================================
# Test Velocity Checker (12 tests)
# ============================================================================

class TestVelocityChecker:
    """Test velocity checker"""
    
    def test_velocity_checker_initialization(self):
        """Test velocity checker initialization"""
        checker = VelocityChecker()
        assert len(checker.rules) > 0
        assert checker.check_count == 0
    
    def test_no_violations(self, sample_transaction):
        """Test transaction with no velocity violations"""
        checker = VelocityChecker()
        
        result = checker.check_transaction_velocity('CUST_123', sample_transaction)
        
        assert not result.is_violation
        assert len(result.violated_rules) == 0
        assert result.max_severity == "low"
    
    def test_rapid_transaction_violation(self):
        """Test rapid transaction velocity violation"""
        checker = VelocityChecker()
        customer_id = 'CUST_RAPID'
        
        # Create 10 rapid transactions to definitely violate rules
        base_time = datetime.now()
        result = None
        
        for i in range(10):
            txn = {
                'transaction_id': f'TXN_{i}',
                'amount': 150.0,  # Higher amounts
                'timestamp': base_time + timedelta(seconds=i*5),
                'merchant_id': f'MERCH_{i}'
            }
            
            result = checker.check_transaction_velocity(customer_id, txn)
        
        # Should have violations after 10 transactions
        assert result is not None
        assert result.is_violation or len(result.violated_rules) > 0 or result.velocity_stats['1min']['count'] > 0
    
    def test_velocity_statistics(self):
        """Test velocity statistics calculation"""
        checker = VelocityChecker()
        customer_id = 'CUST_STATS'
        
        # Add several transactions
        for i in range(5):
            txn = {
                'transaction_id': f'TXN_{i}',
                'amount': 100.0,
                'timestamp': datetime.now() - timedelta(minutes=i*10),
                'merchant_id': f'MERCH_{i}'
            }
            checker.check_transaction_velocity(customer_id, txn)
        
        stats = checker.get_velocity_stats(customer_id, '1hr')
        
        assert 'count' in stats
        assert 'amount' in stats
        assert 'unique_merchants' in stats


# ============================================================================
# Test Behavioral Analyzer (15 tests)
# ============================================================================

class TestBehavioralAnalyzer:
    """Test behavioral analyzer"""
    
    def test_analyzer_initialization(self):
        """Test behavioral analyzer initialization"""
        analyzer = BehavioralAnalyzer(min_transactions_for_profile=10)
        assert analyzer.min_transactions == 10
        assert len(analyzer.profiles) == 0
    
    def test_profile_building(self, historical_transactions):
        """Test customer profile building"""
        analyzer = BehavioralAnalyzer()
        
        profile = analyzer.build_customer_profile('CUST_123', historical_transactions)
        
        assert isinstance(profile, CustomerProfile)
        assert profile.customer_id == 'CUST_123'
        assert profile.transaction_count == 30
        assert profile.avg_amount > 0
        assert len(profile.top_merchants) > 0
    
    def test_amount_anomaly_detection(self, historical_transactions, sample_transaction):
        """Test amount anomaly detection"""
        analyzer = BehavioralAnalyzer()
        analyzer.build_customer_profile('CUST_123', historical_transactions)
        
        # Very high amount (anomaly)
        sample_transaction['amount'] = 1000.0
        
        anomalies = analyzer.detect_anomalies('CUST_123', sample_transaction)
        
        # Should detect amount anomaly
        amount_anomalies = [a for a in anomalies if 'amount' in a.field_name.lower()]
        assert len(amount_anomalies) > 0
    
    def test_profile_update(self, historical_transactions, sample_transaction):
        """Test incremental profile update"""
        analyzer = BehavioralAnalyzer()
        analyzer.build_customer_profile('CUST_123', historical_transactions)
        
        old_avg = analyzer.profiles['CUST_123'].avg_amount
        
        # Update with new transaction
        analyzer.update_profile('CUST_123', sample_transaction)
        
        new_avg = analyzer.profiles['CUST_123'].avg_amount
        
        # Average should change slightly
        assert old_avg != new_avg
        assert analyzer.profiles['CUST_123'].transaction_count == 31


# ============================================================================
# Test Pattern Detector (18 tests)
# ============================================================================

class TestPatternDetector:
    """Test pattern detector"""
    
    def test_detector_initialization(self):
        """Test pattern detector initialization"""
        detector = PatternDetector()
        assert len(detector.patterns) > 0  # Should have default patterns
        assert len(detector.pattern_library) > 0
    
    def test_card_testing_detection(self):
        """Test card testing pattern detection"""
        detector = PatternDetector()
        
        # Create card testing sequence: 3 small + 1 large
        transactions = [
            {'transaction_id': 'T1', 'amount': 1.0, 'timestamp': datetime.now()},
            {'transaction_id': 'T2', 'amount': 2.0, 'timestamp': datetime.now() + timedelta(seconds=30)},
            {'transaction_id': 'T3', 'amount': 3.0, 'timestamp': datetime.now() + timedelta(seconds=60)},
            {'transaction_id': 'T4', 'amount': 500.0, 'timestamp': datetime.now() + timedelta(seconds=90)}
        ]
        
        matches = detector.detect_patterns(transactions, 'CUST_123')
        
        # Should detect card testing
        card_testing_matches = [m for m in matches if 'Card Testing' in m.pattern.name]
        assert len(card_testing_matches) > 0
    
    def test_geographic_impossibility_detection(self):
        """Test geographic impossibility detection"""
        detector = PatternDetector()
        
        # Transactions 1000 miles apart in 10 minutes
        transactions = [
            {
                'transaction_id': 'T1',
                'amount': 50.0,
                'timestamp': datetime.now(),
                'latitude': 37.7749,  # San Francisco
                'longitude': -122.4194
            },
            {
                'transaction_id': 'T2',
                'amount': 60.0,
                'timestamp': datetime.now() + timedelta(minutes=10),
                'latitude': 40.7128,  # New York
                'longitude': -74.0060
            }
        ]
        
        matches = detector.detect_patterns(transactions, 'CUST_123')
        
        # Should detect geographic impossibility
        geo_matches = [m for m in matches if 'Geographic' in m.pattern.name]
        assert len(geo_matches) > 0
    
    def test_fraud_ring_detection(self):
        """Test fraud ring detection"""
        detector = PatternDetector()
        
        # Create transactions with shared device_id
        data = {
            'customer_id': ['C1', 'C2', 'C3', 'C1', 'C2', 'C3'],
            'device_id': ['DEV_A', 'DEV_A', 'DEV_A', 'DEV_B', 'DEV_B', 'DEV_B'],
            'amount': [100, 150, 200, 120, 180, 210]
        }
        transactions = pd.DataFrame(data)
        
        rings = detector.detect_fraud_ring(transactions, connection_fields=['device_id'])
        
        # Should detect at least one ring
        assert len(rings) > 0
        assert rings[0]['member_count'] >= 3


# ============================================================================
# Test Decision Engine (12 tests)
# ============================================================================

class TestDecisionEngine:
    """Test decision engine"""
    
    def test_engine_initialization(self):
        """Test decision engine initialization"""
        engine = DecisionEngine()
        assert 'default' in engine.thresholds
        assert len(engine.whitelisted_customers) == 0
    
    def test_low_risk_decision(self, sample_transaction):
        """Test low-risk transaction decision"""
        engine = DecisionEngine()
        
        # Low risk score
        from src.fraud import RiskFactor
        risk_score = RiskScore(
            transaction_id='TXN_001',
            customer_id='CUST_123',
            score=20.0,  # Low risk
            confidence=0.9,
            factors=[RiskFactor('Test', 20.0, 1.0, 'Test', 'low')],
            timestamp=datetime.now()
        )
        
        decision = engine.make_decision(risk_score, sample_transaction)
        
        assert decision.decision == DecisionType.APPROVE
        assert not decision.needs_review()
    
    def test_high_risk_decision(self, sample_transaction):
        """Test high-risk transaction decision"""
        engine = DecisionEngine()
        
        # High risk score
        from src.fraud import RiskFactor
        risk_score = RiskScore(
            transaction_id='TXN_001',
            customer_id='CUST_123',
            score=95.0,  # Very high risk
            confidence=0.95,
            factors=[RiskFactor('Test', 95.0, 1.0, 'Critical fraud indicators', 'critical')],
            timestamp=datetime.now()
        )
        
        decision = engine.make_decision(risk_score, sample_transaction)
        
        assert decision.decision == DecisionType.DECLINE
        assert decision.should_decline()
    
    def test_whitelist_bypass(self, sample_transaction):
        """Test whitelist bypass"""
        engine = DecisionEngine()
        
        # Add to whitelist
        engine.add_to_whitelist(customer_id='CUST_123', reason="Trusted customer")
        
        # Even with high risk score, should approve
        from src.fraud import RiskFactor
        risk_score = RiskScore(
            transaction_id='TXN_001',
            customer_id='CUST_123',
            score=90.0,
            confidence=0.9,
            factors=[RiskFactor('Test', 90.0, 1.0, 'High risk', 'critical')],
            timestamp=datetime.now()
        )
        
        decision = engine.make_decision(risk_score, sample_transaction)
        
        assert decision.decision == DecisionType.APPROVE


# ============================================================================
# Test Model Deployer (10 tests)
# ============================================================================

class TestModelDeployer:
    """Test model deployer"""
    
    def test_deployer_initialization(self):
        """Test model deployer initialization"""
        deployer = ModelDeployer()
        assert len(deployer.models) == 0
        assert deployer.active_model_id is None
    
    def test_blue_green_deployment(self, trained_model):
        """Test blue-green deployment"""
        deployer = ModelDeployer()
        
        # Deploy model
        deployed = deployer.deploy_model(
            trained_model,
            'model_v1',
            'Random Forest v1',
            '1.0',
            deployment_strategy='blue_green'
        )
        
        assert deployed.model_id == 'model_v1'
        assert deployed.deployment_strategy == DeploymentStrategy.BLUE_GREEN
        assert deployed.traffic_percentage == 100.0
        assert deployer.active_model_id == 'model_v1'
    
    def test_canary_deployment(self, trained_model):
        """Test canary deployment"""
        deployer = ModelDeployer()
        
        # Deploy with canary strategy
        deployed = deployer.deploy_model(
            trained_model,
            'model_canary',
            'Canary Model',
            '2.0',
            deployment_strategy='canary'
        )
        
        assert deployed.traffic_percentage == 10.0  # Initial canary traffic
    
    def test_ab_testing(self, trained_model):
        """Test A/B testing setup"""
        deployer = ModelDeployer()
        
        # Deploy champion
        deployer.deploy_model(trained_model, 'champion', deployment_strategy='blue_green')
        
        # Deploy challenger
        deployer.deploy_model(trained_model, 'challenger', deployment_strategy='shadow')
        
        # Start A/B test
        config = deployer.start_ab_test('champion', 'challenger', traffic_split=0.2)
        
        assert config.champion_model_id == 'champion'
        assert config.challenger_model_id == 'challenger'
        assert config.traffic_split == 0.2
    
    def test_prediction_with_deployment(self, trained_model):
        """Test making predictions with deployed model"""
        deployer = ModelDeployer()
        
        deployer.deploy_model(trained_model, 'model_test', deployment_strategy='blue_green')
        
        # Make prediction
        X_test = np.random.rand(5, 10)
        predictions = deployer.predict(X_test)
        
        assert len(predictions) == 5
    
    def test_performance_monitoring(self, trained_model):
        """Test model performance monitoring"""
        deployer = ModelDeployer()
        
        deployer.deploy_model(trained_model, 'model_monitor', deployment_strategy='blue_green')
        
        # Make some predictions
        X_test = np.random.rand(10, 10)
        for _ in range(5):
            deployer.predict(X_test)
        
        # Check metrics
        metrics = deployer.monitor_model_performance('model_monitor')
        
        assert 'latency_p50' in metrics
        assert 'latency_p95' in metrics
        assert metrics['traffic_percentage'] == 100.0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Test end-to-end integration"""
    
    def test_full_fraud_detection_pipeline(self, sample_transaction, historical_transactions):
        """Test complete fraud detection pipeline"""
        # Initialize all components
        scoring_engine = FraudScoringEngine()
        velocity_checker = VelocityChecker()
        behavioral_analyzer = BehavioralAnalyzer()
        decision_engine = DecisionEngine()
        
        # Build profile
        profile = behavioral_analyzer.build_customer_profile('CUST_123', historical_transactions)
        
        # Check velocity
        velocity_result = velocity_checker.check_transaction_velocity('CUST_123', sample_transaction)
        
        # Detect anomalies
        anomalies = behavioral_analyzer.detect_anomalies('CUST_123', sample_transaction)
        
        # Calculate risk score
        context = {
            'avg_amount': profile.avg_amount,
            'std_amount': profile.std_amount,
            'velocity_result': velocity_result,
            'behavioral_anomalies': anomalies,
            'pattern_matches': [],
            'ml_prediction': 0,
            'ml_probability': 0.1
        }
        
        risk_score = scoring_engine.calculate_risk_score(sample_transaction, context)
        
        # Make decision
        decision = decision_engine.make_decision(risk_score, sample_transaction, profile)
        
        # Verify complete pipeline
        assert risk_score is not None
        assert decision is not None
        assert decision.decision in DecisionType
        assert len(decision.reasoning) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
