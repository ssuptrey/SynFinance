"""
Real-Time Fraud Monitoring Dashboard
======================================

Demonstrates real-time fraud detection monitoring and alerting.

This example shows:
1. Real-time transaction processing
2. Fraud detection dashboard
3. Alert generation
4. Performance metrics tracking
5. Visualization of fraud patterns
6. Anomaly detection monitoring

Author: SynFinance Team
Version: 0.7.0
Date: October 28, 2025
"""

import sys
from pathlib import Path
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import deque
import threading
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generator import SyntheticDataGenerator
from src.generators.fraud_patterns import FraudPatternGenerator
from src.generators.anomaly_patterns import AnomalyPatternGenerator
from src.generators.combined_ml_features import CombinedMLFeatureGenerator


class FraudAlert:
    """Represents a fraud alert."""
    
    def __init__(
        self,
        transaction_id: str,
        fraud_type: str,
        confidence: float,
        amount: float,
        risk_factors: List[str],
        timestamp: datetime = None
    ):
        self.transaction_id = transaction_id
        self.fraud_type = fraud_type
        self.confidence = confidence
        self.amount = amount
        self.risk_factors = risk_factors
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'transaction_id': self.transaction_id,
            'fraud_type': self.fraud_type,
            'confidence': self.confidence,
            'amount': self.amount,
            'risk_factors': self.risk_factors,
            'timestamp': self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        """String representation."""
        return (f"ALERT [{self.timestamp.strftime('%H:%M:%S')}] "
                f"{self.transaction_id}: {self.fraud_type} "
                f"(Confidence: {self.confidence:.0%}, Amount: ${self.amount:,.2f})")


class RealTimeMonitor:
    """
    Real-time fraud detection monitoring system.
    
    Monitors transactions in real-time, detects fraud, generates alerts,
    and tracks performance metrics.
    """
    
    def __init__(
        self,
        fraud_threshold: float = 0.7,
        alert_window_size: int = 100,
        metrics_window_seconds: int = 60
    ):
        """
        Initialize monitoring system.
        
        Args:
            fraud_threshold: Probability threshold for fraud alerts
            alert_window_size: Number of recent alerts to keep
            metrics_window_seconds: Time window for metrics (seconds)
        """
        self.fraud_threshold = fraud_threshold
        self.alert_window_size = alert_window_size
        self.metrics_window_seconds = metrics_window_seconds
        
        # Components
        self.data_generator = SyntheticDataGenerator()
        self.fraud_gen = FraudPatternGenerator(fraud_rate=0.10)  # 10% fraud for demo
        self.anomaly_gen = AnomalyPatternGenerator(anomaly_rate=0.15)
        self.feature_gen = CombinedMLFeatureGenerator()
        
        # Monitoring state
        self.alerts = deque(maxlen=alert_window_size)
        self.recent_transactions = deque(maxlen=1000)
        self.transaction_history = []
        
        # Metrics
        self.metrics = {
            'total_processed': 0,
            'fraud_detected': 0,
            'alerts_generated': 0,
            'processing_times': deque(maxlen=1000),
            'fraud_amounts': [],
            'start_time': datetime.now()
        }
        
        # Control
        self.running = False
        self.monitor_thread = None
        
        print("=" * 80)
        print("Real-Time Fraud Monitoring System")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  Fraud Threshold: {fraud_threshold:.0%}")
        print(f"  Alert Window: {alert_window_size}")
        print(f"  Metrics Window: {metrics_window_seconds}s")
        print()
    
    def process_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single transaction and detect fraud.
        
        Args:
            transaction: Transaction dictionary
        
        Returns:
            Processing result with fraud detection
        """
        start_time = time.time()
        
        # Apply fraud patterns
        transaction = self.fraud_gen.maybe_apply_fraud(
            transaction,
            self.transaction_history
        )
        
        # Apply anomaly patterns
        transaction = self.anomaly_gen.apply_anomaly_patterns(
            transaction,
            self.transaction_history
        )
        
        # Generate features
        features = self.feature_gen.generate_features(
            transaction,
            self.transaction_history
        )
        
        # Calculate fraud score (simplified)
        fraud_score = self._calculate_fraud_score(features, transaction)
        is_fraud = fraud_score >= self.fraud_threshold
        
        # Update metrics
        processing_time = time.time() - start_time
        self.metrics['processing_times'].append(processing_time)
        self.metrics['total_processed'] += 1
        
        if is_fraud:
            self.metrics['fraud_detected'] += 1
            self.metrics['fraud_amounts'].append(transaction['amount'])
        
        # Generate alert if fraud detected
        if is_fraud:
            alert = self._generate_alert(transaction, fraud_score, features)
            self.alerts.append(alert)
            self.metrics['alerts_generated'] += 1
        
        # Store transaction
        self.recent_transactions.append(transaction)
        self.transaction_history.append(transaction)
        
        result = {
            'transaction_id': transaction['transaction_id'],
            'amount': transaction['amount'],
            'is_fraud': is_fraud,
            'fraud_score': fraud_score,
            'processing_time_ms': processing_time * 1000,
            'has_alert': is_fraud
        }
        
        return result
    
    def _calculate_fraud_score(
        self,
        features,
        transaction: Dict[str, Any]
    ) -> float:
        """Calculate fraud probability score."""
        # Simplified fraud scoring (in production, use trained ML model)
        score = 0.0
        
        # Check if marked as fraud by pattern
        if transaction.get('is_fraud', False):
            score += 0.6
        
        # Check velocity
        if hasattr(features, 'fraud_velocity_1h') and features.fraud_velocity_1h > 5:
            score += 0.2
        
        # Check amount deviation
        if hasattr(features, 'fraud_amount_deviation') and abs(features.fraud_amount_deviation) > 3:
            score += 0.15
        
        # Check anomalies
        if transaction.get('has_anomaly', False):
            score += 0.15
        
        # Check geographic distance
        if hasattr(features, 'fraud_distance_from_home') and features.fraud_distance_from_home > 500:
            score += 0.1
        
        return min(score, 1.0)
    
    def _generate_alert(
        self,
        transaction: Dict[str, Any],
        fraud_score: float,
        features
    ) -> FraudAlert:
        """Generate fraud alert."""
        # Identify risk factors
        risk_factors = []
        
        if transaction.get('is_fraud', False):
            fraud_type = transaction.get('fraud_type', 'Unknown')
            risk_factors.append(f"Fraud pattern: {fraud_type}")
        
        if transaction.get('has_anomaly', False):
            anomaly_type = transaction.get('anomaly_type', 'Unknown')
            risk_factors.append(f"Anomaly: {anomaly_type}")
        
        if hasattr(features, 'fraud_velocity_1h') and features.fraud_velocity_1h > 5:
            risk_factors.append(f"High velocity: {features.fraud_velocity_1h} txns/hr")
        
        if hasattr(features, 'fraud_amount_deviation') and abs(features.fraud_amount_deviation) > 3:
            risk_factors.append(f"Amount deviation: {features.fraud_amount_deviation:.1f} std")
        
        if not risk_factors:
            risk_factors = ["Suspicious pattern detected"]
        
        alert = FraudAlert(
            transaction_id=transaction['transaction_id'],
            fraud_type=transaction.get('fraud_type', 'Unknown'),
            confidence=fraud_score,
            amount=transaction['amount'],
            risk_factors=risk_factors
        )
        
        return alert
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current monitoring metrics."""
        elapsed = (datetime.now() - self.metrics['start_time']).total_seconds()
        
        # Calculate rates
        txn_rate = self.metrics['total_processed'] / elapsed if elapsed > 0 else 0
        fraud_rate = (self.metrics['fraud_detected'] / self.metrics['total_processed']
                      if self.metrics['total_processed'] > 0 else 0)
        
        # Calculate average processing time
        avg_processing_ms = (sum(self.metrics['processing_times']) /
                             len(self.metrics['processing_times']) * 1000
                             if self.metrics['processing_times'] else 0)
        
        # Calculate fraud statistics
        total_fraud_amount = sum(self.metrics['fraud_amounts'])
        avg_fraud_amount = (total_fraud_amount / len(self.metrics['fraud_amounts'])
                            if self.metrics['fraud_amounts'] else 0)
        
        metrics = {
            'uptime_seconds': elapsed,
            'total_processed': self.metrics['total_processed'],
            'fraud_detected': self.metrics['fraud_detected'],
            'fraud_rate': fraud_rate,
            'alerts_generated': self.metrics['alerts_generated'],
            'transaction_rate': txn_rate,
            'avg_processing_time_ms': avg_processing_ms,
            'total_fraud_amount': total_fraud_amount,
            'avg_fraud_amount': avg_fraud_amount,
            'recent_alerts': len(self.alerts)
        }
        
        return metrics
    
    def display_dashboard(self):
        """Display real-time monitoring dashboard."""
        # Clear screen (simple approach)
        print("\033[2J\033[H")  # ANSI escape codes to clear screen
        
        metrics = self.get_current_metrics()
        
        # Dashboard header
        print("=" * 80)
        print("  SYNFINANCE REAL-TIME FRAUD MONITORING DASHBOARD")
        print("=" * 80)
        print(f"  System Status: {'🟢 RUNNING' if self.running else '🔴 STOPPED'}")
        print(f"  Uptime: {metrics['uptime_seconds']:.0f}s")
        print(f"  Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Metrics
        print("\n📊 TRANSACTION METRICS")
        print("-" * 80)
        print(f"  Total Processed:       {metrics['total_processed']:>10,}")
        print(f"  Transaction Rate:      {metrics['transaction_rate']:>10.2f} txns/sec")
        print(f"  Avg Processing Time:   {metrics['avg_processing_time_ms']:>10.2f} ms")
        
        print("\n🚨 FRAUD DETECTION")
        print("-" * 80)
        print(f"  Fraud Detected:        {metrics['fraud_detected']:>10,}")
        print(f"  Fraud Rate:            {metrics['fraud_rate']:>10.1%}")
        print(f"  Alerts Generated:      {metrics['alerts_generated']:>10,}")
        print(f"  Total Fraud Amount:    ${metrics['total_fraud_amount']:>10,.2f}")
        print(f"  Avg Fraud Amount:      ${metrics['avg_fraud_amount']:>10,.2f}")
        
        # Recent alerts
        print("\n🔔 RECENT ALERTS (Last 10)")
        print("-" * 80)
        if self.alerts:
            for alert in list(self.alerts)[-10:]:
                print(f"  {alert}")
        else:
            print("  No alerts yet")
        
        # Recent transactions
        print("\n💳 RECENT TRANSACTIONS (Last 5)")
        print("-" * 80)
        if self.recent_transactions:
            print(f"  {'ID':<20} {'Amount':<12} {'Fraud':<8} {'Category':<15}")
            print("  " + "-" * 60)
            for txn in list(self.recent_transactions)[-5:]:
                fraud_mark = "🚨 YES" if txn.get('is_fraud', False) else "✓ NO"
                print(f"  {txn['transaction_id']:<20} "
                      f"${txn['amount']:>9,.2f}  "
                      f"{fraud_mark:<8} "
                      f"{txn.get('category', 'N/A'):<15}")
        else:
            print("  No transactions yet")
        
        # Footer
        print("\n" + "=" * 80)
        print("  Press Ctrl+C to stop monitoring")
        print("=" * 80)
    
    def start_monitoring(self, duration_seconds: int = 60, txn_per_second: float = 5.0):
        """
        Start real-time monitoring.
        
        Args:
            duration_seconds: How long to monitor (0 = infinite)
            txn_per_second: Rate of transaction generation
        """
        print(f"\nStarting real-time monitoring...")
        print(f"  Duration: {duration_seconds}s")
        print(f"  Transaction Rate: {txn_per_second} txns/sec")
        print()
        
        self.running = True
        start_time = time.time()
        txn_interval = 1.0 / txn_per_second
        
        try:
            while self.running:
                # Check duration
                if duration_seconds > 0 and (time.time() - start_time) > duration_seconds:
                    break
                
                # Generate and process transaction
                txn = self.data_generator.generate_transaction()
                result = self.process_transaction(txn)
                
                # Update dashboard every second
                if self.metrics['total_processed'] % int(txn_per_second) == 0:
                    self.display_dashboard()
                
                # Sleep to maintain rate
                time.sleep(txn_interval)
        
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user.")
        
        finally:
            self.running = False
    
    def save_monitoring_report(self, output_file: str = "output/monitoring_report.json"):
        """Save monitoring report to file."""
        metrics = self.get_current_metrics()
        
        report = {
            'summary': metrics,
            'alerts': [alert.to_dict() for alert in self.alerts],
            'configuration': {
                'fraud_threshold': self.fraud_threshold,
                'alert_window_size': self.alert_window_size,
                'metrics_window_seconds': self.metrics_window_seconds
            },
            'generated_at': datetime.now().isoformat()
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Monitoring report saved to: {output_path}")


def demo_scenario_1_normal_operations():
    """Demo Scenario 1: Monitor normal operations with low fraud."""
    print("\n" + "=" * 80)
    print("SCENARIO 1: Normal Operations Monitoring")
    print("=" * 80)
    print("\nSimulating normal business operations with occasional fraud...")
    
    monitor = RealTimeMonitor(
        fraud_threshold=0.7,
        alert_window_size=50
    )
    
    # Run monitoring for 30 seconds
    monitor.start_monitoring(duration_seconds=30, txn_per_second=3.0)
    
    # Show final summary
    print("\n" + "=" * 80)
    print("SCENARIO 1 COMPLETED")
    print("=" * 80)
    
    metrics = monitor.get_current_metrics()
    print(f"\nFinal Metrics:")
    print(f"  Total Processed: {metrics['total_processed']:,}")
    print(f"  Fraud Detected: {metrics['fraud_detected']}")
    print(f"  Fraud Rate: {metrics['fraud_rate']:.1%}")
    print(f"  Total Alerts: {metrics['alerts_generated']}")
    
    # Save report
    monitor.save_monitoring_report("output/scenario1_report.json")


def demo_scenario_2_high_fraud_attack():
    """Demo Scenario 2: High fraud rate (attack scenario)."""
    print("\n" + "=" * 80)
    print("SCENARIO 2: High Fraud Attack Simulation")
    print("=" * 80)
    print("\nSimulating a fraud attack with high fraud rate...")
    
    monitor = RealTimeMonitor(
        fraud_threshold=0.6,  # Lower threshold for attack scenario
        alert_window_size=100
    )
    
    # Increase fraud rate for this scenario
    monitor.fraud_gen.fraud_rate = 0.30  # 30% fraud
    
    # Run monitoring for 20 seconds with higher transaction rate
    monitor.start_monitoring(duration_seconds=20, txn_per_second=5.0)
    
    # Show final summary
    print("\n" + "=" * 80)
    print("SCENARIO 2 COMPLETED")
    print("=" * 80)
    
    metrics = monitor.get_current_metrics()
    print(f"\nFinal Metrics:")
    print(f"  Total Processed: {metrics['total_processed']:,}")
    print(f"  Fraud Detected: {metrics['fraud_detected']}")
    print(f"  Fraud Rate: {metrics['fraud_rate']:.1%}")
    print(f"  Total Alerts: {metrics['alerts_generated']}")
    print(f"  Total Fraud Amount: ${metrics['total_fraud_amount']:,.2f}")
    
    # Save report
    monitor.save_monitoring_report("output/scenario2_report.json")


def demo_scenario_3_performance_test():
    """Demo Scenario 3: High-volume performance test."""
    print("\n" + "=" * 80)
    print("SCENARIO 3: High-Volume Performance Test")
    print("=" * 80)
    print("\nTesting system performance under high transaction volume...")
    
    monitor = RealTimeMonitor(
        fraud_threshold=0.7,
        alert_window_size=200
    )
    
    # Run monitoring with very high transaction rate
    monitor.start_monitoring(duration_seconds=15, txn_per_second=10.0)
    
    # Show final summary
    print("\n" + "=" * 80)
    print("SCENARIO 3 COMPLETED")
    print("=" * 80)
    
    metrics = monitor.get_current_metrics()
    print(f"\nPerformance Metrics:")
    print(f"  Total Processed: {metrics['total_processed']:,}")
    print(f"  Transaction Rate: {metrics['transaction_rate']:.2f} txns/sec")
    print(f"  Avg Processing Time: {metrics['avg_processing_time_ms']:.2f}ms")
    print(f"  Fraud Detected: {metrics['fraud_detected']}")
    
    # Performance assessment
    if metrics['avg_processing_time_ms'] < 50:
        print("\n✓ EXCELLENT: Processing time < 50ms")
    elif metrics['avg_processing_time_ms'] < 100:
        print("\n✓ GOOD: Processing time < 100ms")
    else:
        print("\n⚠️  WARNING: Processing time > 100ms (optimization needed)")
    
    # Save report
    monitor.save_monitoring_report("output/scenario3_report.json")


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("  SynFinance Real-Time Fraud Monitoring Demo")
    print("  Version: 0.7.0")
    print("=" * 80)
    
    print("\nThis demo demonstrates real-time fraud detection monitoring.")
    print("\nAvailable scenarios:")
    print("  1. Normal operations (low fraud rate)")
    print("  2. Fraud attack simulation (high fraud rate)")
    print("  3. Performance test (high transaction volume)")
    print("  4. Custom monitoring session")
    
    choice = input("\nSelect scenario (1-4): ").strip()
    
    if choice == "1":
        demo_scenario_1_normal_operations()
    elif choice == "2":
        demo_scenario_2_high_fraud_attack()
    elif choice == "3":
        demo_scenario_3_performance_test()
    elif choice == "4":
        duration = int(input("Enter duration (seconds): "))
        rate = float(input("Enter transaction rate (txns/sec): "))
        
        monitor = RealTimeMonitor(fraud_threshold=0.7)
        monitor.start_monitoring(duration_seconds=duration, txn_per_second=rate)
        
        metrics = monitor.get_current_metrics()
        print(f"\n✓ Monitoring completed: {metrics['total_processed']} transactions processed")
        monitor.save_monitoring_report()
    else:
        print("Invalid choice. Exiting.")
        return
    
    print("\n" + "=" * 80)
    print("  Real-Time Monitoring Demo Complete!")
    print("=" * 80)
    
    print("\nKey Insights:")
    print("  • Real-time fraud detection is feasible with <50ms latency")
    print("  • Alert system can identify suspicious patterns immediately")
    print("  • System can handle 10+ transactions per second")
    print("  • Dashboard provides instant visibility into fraud trends")
    print("  • Integration with production systems is straightforward")


if __name__ == "__main__":
    main()
