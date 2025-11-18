"""
Fraud Detection System Demo

Demonstrates end-to-end fraud scoring, pattern detection, and decision-making.
Week 10 Day 4: Advanced Fraud Detection & Real-Time Analytics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

from src.fraud import (
    FraudScoringEngine,
    VelocityChecker,
    BehavioralAnalyzer,
    PatternDetector,
    DecisionEngine,
    ModelDeployer
)


def print_section(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demo_fraud_scoring():
    """Demo: Real-time fraud scoring"""
    print_section("SCENARIO 1: Real-Time Fraud Scoring")
    
    # Initialize components
    scoring_engine = FraudScoringEngine()
    velocity_checker = VelocityChecker()
    behavioral_analyzer = BehavioralAnalyzer()
    decision_engine = DecisionEngine()
    
    # Build customer profile from historical data
    print("[*] Building customer profile from 30-day history...")
    
    dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
    historical_data = pd.DataFrame({
        'transaction_id': [f'TXN_{i:03d}' for i in range(30)],
        'customer_id': ['CUST_123'] * 30,
        'amount': np.random.normal(100, 20, 30).tolist(),
        'merchant_id': [f'MERCH_{i%5:02d}' for i in range(30)],
        'category': [['Retail', 'Food', 'Gas'][i%3] for i in range(30)],
        'timestamp': dates,
        'city': ['San Francisco'] * 30
    })
    
    profile = behavioral_analyzer.build_customer_profile('CUST_123', historical_data)
    
    print(f"    Customer: {profile.customer_id}")
    print(f"    Transaction Count: {profile.transaction_count}")
    print(f"    Average Amount: ${profile.avg_amount:.2f}")
    print(f"    Amount Range: ${profile.min_amount:.2f} - ${profile.max_amount:.2f}")
    print(f"    Top Merchants: {', '.join(profile.top_merchants[:3])}")
    
    # Test 3 scenarios
    scenarios = [
        {
            'name': "Normal Transaction",
            'transaction': {
                'transaction_id': 'TXN_NEW_001',
                'customer_id': 'CUST_123',
                'amount': 105.00,  # Normal amount
                'merchant_id': 'MERCH_00',  # Known merchant
                'timestamp': datetime.now(),
                'city': 'San Francisco'
            }
        },
        {
            'name': "Suspicious Transaction",
            'transaction': {
                'transaction_id': 'TXN_NEW_002',
                'customer_id': 'CUST_123',
                'amount': 850.00,  # High amount
                'merchant_id': 'MERCH_99',  # Unknown merchant
                'timestamp': datetime.now(),
                'city': 'Los Angeles'  # Different city
            }
        },
        {
            'name': "High-Risk Transaction",
            'transaction': {
                'transaction_id': 'TXN_NEW_003',
                'customer_id': 'CUST_123',
                'amount': 2500.00,  # Very high amount
                'merchant_id': 'MERCH_UNKNOWN',
                'timestamp': datetime.now(),
                'city': 'Miami'  # Far away
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        transaction = scenario['transaction']
        
        # Check velocity
        velocity_result = velocity_checker.check_transaction_velocity(
            transaction['customer_id'],
            transaction
        )
        
        # Detect behavioral anomalies
        anomalies = behavioral_analyzer.detect_anomalies(
            transaction['customer_id'],
            transaction
        )
        
        # Calculate risk score
        context = {
            'avg_amount': profile.avg_amount,
            'std_amount': profile.std_amount,
            'velocity_result': velocity_result,
            'behavioral_anomalies': anomalies,
            'pattern_matches': [],
            'ml_prediction': 0 if transaction['amount'] < 200 else 1,
            'ml_probability': 0.1 if transaction['amount'] < 200 else 0.6
        }
        
        risk_score = scoring_engine.calculate_risk_score(transaction, context)
        
        # Make decision
        decision = decision_engine.make_decision(risk_score, transaction, profile)
        
        # Display results
        print(f"  Transaction ID: {transaction['transaction_id']}")
        print(f"  Amount: ${transaction['amount']:.2f}")
        print(f"  Risk Score: {risk_score.score:.1f}/100 ({risk_score.get_risk_level().upper()})")
        print(f"  Confidence: {risk_score.confidence:.1%}")
        print(f"  Decision: {decision.decision.value}")
        print(f"  Action: {decision.recommended_action}")
        
        if anomalies:
            print(f"  Anomalies Detected: {len(anomalies)}")
            for anomaly in anomalies[:2]:
                print(f"    - {anomaly.anomaly_type.value}: {anomaly.explanation}")
    
    # Performance stats
    stats = scoring_engine.get_performance_stats()
    print(f"\n[OK] Scoring Performance:")
    print(f"    Total Scores: {stats['total_scores']}")
    print(f"    Average Latency: {stats['avg_latency_ms']:.2f}ms")
    print(f"    Max Latency: {stats['max_latency_ms']:.2f}ms")


def demo_pattern_detection():
    """Demo: Fraud pattern detection"""
    print_section("SCENARIO 2: Fraud Pattern Detection")
    
    detector = PatternDetector()
    
    print(f"[*] Loaded {len(detector.patterns)} fraud patterns")
    print("    Patterns: " + ", ".join([p.name for p in detector.pattern_library[:5]]))
    
    # Scenario A: Card Testing
    print("\n--- Pattern A: Card Testing ---")
    card_testing_txns = [
        {'transaction_id': 'CT1', 'amount': 1.00, 'timestamp': datetime.now()},
        {'transaction_id': 'CT2', 'amount': 2.50, 'timestamp': datetime.now() + timedelta(seconds=20)},
        {'transaction_id': 'CT3', 'amount': 5.00, 'timestamp': datetime.now() + timedelta(seconds=40)},
        {'transaction_id': 'CT4', 'amount': 500.00, 'timestamp': datetime.now() + timedelta(seconds=60)}
    ]
    
    matches = detector.detect_patterns(card_testing_txns, 'CUST_FRAUD_001')
    
    if matches:
        for match in matches:
            print(f"  [DETECTED] {match.pattern.name}")
            print(f"    Confidence: {match.confidence:.1%}")
            print(f"    Risk Score: {match.get_risk_score():.1f}/100")
            print(f"    Explanation: {match.explanation}")
            print(f"    Matched {len(match.matched_transactions)} transactions")
    else:
        print("  [OK] No patterns detected")
    
    # Scenario B: Geographic Impossibility
    print("\n--- Pattern B: Geographic Impossibility ---")
    geo_txns = [
        {
            'transaction_id': 'GEO1',
            'amount': 75.00,
            'timestamp': datetime.now(),
            'latitude': 37.7749,  # San Francisco
            'longitude': -122.4194
        },
        {
            'transaction_id': 'GEO2',
            'amount': 85.00,
            'timestamp': datetime.now() + timedelta(minutes=15),
            'latitude': 40.7128,  # New York (3000 miles away)
            'longitude': -74.0060
        }
    ]
    
    matches = detector.detect_patterns(geo_txns, 'CUST_FRAUD_002')
    
    if matches:
        for match in matches:
            print(f"  [DETECTED] {match.pattern.name}")
            print(f"    Confidence: {match.confidence:.1%}")
            print(f"    Risk Score: {match.get_risk_score():.1f}/100")
            print(f"    Explanation: {match.explanation}")
    else:
        print("  [OK] No patterns detected")
    
    # Scenario C: Fraud Ring Detection
    print("\n--- Pattern C: Fraud Ring Detection ---")
    
    ring_data = pd.DataFrame({
        'customer_id': ['C1', 'C2', 'C3', 'C4', 'C5'] * 3,
        'device_id': ['DEV_A'] * 5 + ['DEV_B'] * 5 + ['DEV_C'] * 5,
        'ip_address': ['192.168.1.1'] * 5 + ['192.168.1.2'] * 10,
        'amount': np.random.uniform(100, 500, 15).tolist()
    })
    
    rings = detector.detect_fraud_ring(ring_data, connection_fields=['device_id', 'ip_address'])
    
    if rings:
        print(f"  [DETECTED] {len(rings)} potential fraud rings")
        for i, ring in enumerate(rings[:2], 1):
            print(f"\n  Ring {i}:")
            print(f"    Members: {ring['member_count']} customers")
            print(f"    Connections: {ring['connection_count']} edges")
            print(f"    Risk Score: {ring['risk_score']:.1f}/100")
            print(f"    Density: {ring['density']:.2f}")
            print(f"    Customers: {', '.join(ring['members'][:5])}")
    else:
        print("  [OK] No fraud rings detected")


def demo_model_deployment():
    """Demo: ML model deployment and A/B testing"""
    print_section("SCENARIO 3: ML Model Deployment & A/B Testing")
    
    deployer = ModelDeployer()
    
    # Train simple models
    print("[*] Training models...")
    
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    
    champion_model = RandomForestClassifier(n_estimators=50, random_state=42)
    champion_model.fit(X_train, y_train)
    
    challenger_model = RandomForestClassifier(n_estimators=100, random_state=43)
    challenger_model.fit(X_train, y_train)
    
    # Blue-Green Deployment
    print("\n--- Strategy 1: Blue-Green Deployment ---")
    
    deployed_champion = deployer.deploy_model(
        champion_model,
        'champion_rf_v1',
        'Random Forest Champion v1',
        '1.0',
        deployment_strategy='blue_green'
    )
    
    print(f"  [OK] Deployed: {deployed_champion.model_name}")
    print(f"      Model ID: {deployed_champion.model_id}")
    print(f"      Version: {deployed_champion.version}")
    print(f"      Strategy: {deployed_champion.deployment_strategy.value}")
    print(f"      Traffic: {deployed_champion.traffic_percentage:.0f}%")
    
    # Make predictions
    X_test = np.random.rand(10, 10)
    predictions = deployer.predict(X_test)
    print(f"  [OK] Made {len(predictions)} predictions")
    
    # A/B Testing
    print("\n--- Strategy 2: A/B Testing ---")
    
    deployed_challenger = deployer.deploy_model(
        challenger_model,
        'challenger_rf_v2',
        'Random Forest Challenger v2',
        '2.0',
        deployment_strategy='shadow'
    )
    
    ab_config = deployer.start_ab_test(
        'champion_rf_v1',
        'challenger_rf_v2',
        traffic_split=0.2  # 20% to challenger
    )
    
    print(f"  [OK] Started A/B Test: {ab_config.test_id}")
    print(f"      Champion: {ab_config.champion_model_id} ({deployer.models[ab_config.champion_model_id].traffic_percentage:.0f}%)")
    print(f"      Challenger: {ab_config.challenger_model_id} ({deployer.models[ab_config.challenger_model_id].traffic_percentage:.0f}%)")
    print(f"      Duration: {ab_config.duration_hours} hours")
    print(f"      Min Improvement Required: {ab_config.min_improvement:.1%}")
    
    # Performance Monitoring
    print("\n--- Performance Monitoring ---")
    
    # Make several predictions to generate metrics
    for _ in range(20):
        X_batch = np.random.rand(5, 10)
        deployer.predict(X_batch)
    
    # Monitor champion
    champion_metrics = deployer.monitor_model_performance('champion_rf_v1')
    
    print(f"  Champion Model Metrics:")
    print(f"    Latency (p50): {champion_metrics.get('latency_p50', 0):.2f}ms")
    print(f"    Latency (p95): {champion_metrics.get('latency_p95', 0):.2f}ms")
    print(f"    Latency (p99): {champion_metrics.get('latency_p99', 0):.2f}ms")
    print(f"    Traffic: {champion_metrics['traffic_percentage']:.0f}%")
    
    # Deployment Status
    print("\n--- Overall Deployment Status ---")
    status = deployer.get_deployment_status()
    
    print(f"  Total Models: {status['total_models']}")
    print(f"  Active Model: {status['active_model']}")
    print(f"  Active A/B Tests: {status['active_ab_tests']}")
    print(f"  Total Rollbacks: {status['total_rollbacks']}")
    
    print("\n  Deployed Models:")
    for model_id, info in status['models'].items():
        print(f"    - {model_id}")
        print(f"        Version: {info['version']}")
        print(f"        Traffic: {info['traffic_percentage']:.0f}%")
        print(f"        Strategy: {info['strategy']}")


def main():
    """Run all demo scenarios"""
    print("\n" + "=" * 80)
    print("  FRAUD DETECTION SYSTEM DEMO")
    print("  Week 10 Day 4: Advanced Fraud Detection & Real-Time Analytics")
    print("=" * 80)
    
    try:
        # Run demos
        demo_fraud_scoring()
        demo_pattern_detection()
        demo_model_deployment()
        
        # Summary
        print_section("DEMO COMPLETE")
        print("[OK] All scenarios completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  1. Multi-factor fraud scoring (amount, velocity, behavior, patterns, ML)")
        print("  2. Behavioral profiling and anomaly detection")
        print("  3. Fraud pattern recognition (card testing, geographic impossibility)")
        print("  4. Fraud ring detection using graph analysis")
        print("  5. Automated decision engine with configurable thresholds")
        print("  6. ML model deployment strategies (blue-green, A/B testing)")
        print("  7. Real-time performance monitoring")
        print("\n[OK] Fraud Detection System Ready for Production!")
        
    except Exception as e:
        print(f"\n[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
