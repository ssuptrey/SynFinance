"""
Load Tester

Load testing framework using Locust for performance testing.
Provides load test scenarios, stress testing, and result analysis.
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import json
import os
import time
import statistics


try:
    from locust import HttpUser, task, between, events
    from locust.env import Environment
    from locust.stats import stats_printer, stats_history
    from locust.log import setup_logging
    LOCUST_AVAILABLE = True
except ImportError:
    LOCUST_AVAILABLE = False
    # Create placeholder classes for when Locust is not installed
    class HttpUser:
        wait_time = None
        host = None
    
    def task(weight=1):
        def decorator(func):
            return func
        return decorator
    
    def between(min_wait, max_wait):
        return None


class LoadTestResult:
    """Load test execution result."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.timestamp = datetime.now()
        self.duration_seconds: float = 0.0
        self.total_requests: int = 0
        self.failed_requests: int = 0
        self.requests_per_second: float = 0.0
        self.response_times: Dict[str, float] = {}
        self.error_rate: float = 0.0
        self.user_count: int = 0
        self.spawn_rate: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'test_name': self.test_name,
            'timestamp': self.timestamp.isoformat(),
            'duration_seconds': self.duration_seconds,
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'requests_per_second': self.requests_per_second,
            'response_times': self.response_times,
            'error_rate': self.error_rate,
            'user_count': self.user_count,
            'spawn_rate': self.spawn_rate
        }


class FraudDetectionUser(HttpUser):
    """Simulated user for fraud detection load testing."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Setup before tests (login, etc.)."""
        # Example: self.client.post("/login", json={"username": "test", "password": "test"})
        pass
    
    @task(10)  # Weight: 10 (most common operation)
    def check_fraud_score(self):
        """Test fraud scoring endpoint."""
        transaction = {
            'customer_id': f'CUST_{int(time.time() * 1000) % 10000}',
            'amount': 100.00 + (int(time.time() * 100) % 500),
            'merchant_id': f'MERCH_{int(time.time() * 100) % 100}',
            'timestamp': datetime.now().isoformat()
        }
        
        with self.client.post(
            '/api/v1/fraud/score',
            json=transaction,
            catch_response=True
        ) as response:
            if response.elapsed.total_seconds() > 0.1:  # 100ms SLA
                response.failure(f"Too slow: {response.elapsed.total_seconds():.3f}s")
            elif response.status_code != 200:
                response.failure(f"Got status {response.status_code}")
            else:
                response.success()
    
    @task(5)  # Weight: 5
    def get_customer_profile(self):
        """Test customer profile endpoint."""
        customer_id = f'CUST_{int(time.time() * 1000) % 10000}'
        
        self.client.get(
            f'/api/v1/customers/{customer_id}/profile',
            name='/api/v1/customers/[id]/profile'
        )
    
    @task(3)  # Weight: 3
    def detect_patterns(self):
        """Test pattern detection endpoint."""
        customer_id = f'CUST_{int(time.time() * 1000) % 10000}'
        
        self.client.get(
            f'/api/v1/fraud/patterns/{customer_id}',
            name='/api/v1/fraud/patterns/[id]'
        )
    
    @task(1)  # Weight: 1 (least common)
    def generate_report(self):
        """Test report generation endpoint."""
        params = {
            'start_date': '2025-10-01',
            'end_date': '2025-11-01',
            'format': 'json'
        }
        
        self.client.get('/api/v1/reports/fraud', params=params)


class TransactionProcessingUser(HttpUser):
    """Simulated user for transaction processing."""
    
    wait_time = between(0.5, 2)
    
    @task
    def process_transaction(self):
        """Test transaction processing."""
        transaction = {
            'customer_id': f'CUST_{int(time.time() * 1000) % 10000}',
            'amount': 50.00 + (int(time.time() * 100) % 200),
            'merchant_id': f'MERCH_{int(time.time() * 100) % 100}',
            'category': ['Retail', 'Food', 'Gas', 'Online'][int(time.time()) % 4]
        }
        
        self.client.post('/api/v1/transactions', json=transaction)


class LoadTester:
    """Load testing framework."""
    
    def __init__(self, base_url: str = 'http://localhost:8000'):
        """
        Initialize load tester.
        
        Args:
            base_url: Base URL for API testing
        """
        self.base_url = base_url
        self.results: List[LoadTestResult] = []
        self.output_dir = 'load_test_results'
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        if not LOCUST_AVAILABLE:
            print("Warning: Locust not installed. Load testing will be simulated.")
    
    def run_load_test(self, users: int = 100, spawn_rate: int = 10,
                     duration: int = 300, user_class=None) -> LoadTestResult:
        """
        Run load test with specified parameters.
        
        Args:
            users: Number of concurrent users
            spawn_rate: Users spawned per second
            duration: Test duration in seconds
            user_class: User class to use (defaults to FraudDetectionUser)
            
        Returns:
            Load test result
        """
        if not LOCUST_AVAILABLE:
            return self._simulate_load_test(users, spawn_rate, duration)
        
        user_class = user_class or FraudDetectionUser
        user_class.host = self.base_url
        
        result = LoadTestResult(f"load_test_{users}_users")
        result.user_count = users
        result.spawn_rate = spawn_rate
        result.duration_seconds = duration
        
        # Setup Locust environment
        setup_logging("INFO", None)
        env = Environment(user_classes=[user_class])
        
        # Start test
        env.runner.start(users, spawn_rate)
        
        # Run for duration
        time.sleep(duration)
        
        # Stop test
        env.runner.quit()
        
        # Collect results
        stats = env.runner.stats
        result.total_requests = stats.total.num_requests
        result.failed_requests = stats.total.num_failures
        result.requests_per_second = stats.total.total_rps
        result.error_rate = (
            stats.total.num_failures / stats.total.num_requests
            if stats.total.num_requests > 0 else 0
        )
        
        result.response_times = {
            'median': stats.total.median_response_time,
            'p95': stats.total.get_response_time_percentile(0.95),
            'p99': stats.total.get_response_time_percentile(0.99),
            'avg': stats.total.avg_response_time,
            'min': stats.total.min_response_time,
            'max': stats.total.max_response_time
        }
        
        self.results.append(result)
        self._save_result(result)
        
        return result
    
    def _simulate_load_test(self, users: int, spawn_rate: int,
                           duration: int) -> LoadTestResult:
        """Simulate load test when Locust is not available."""
        result = LoadTestResult(f"simulated_load_test_{users}_users")
        result.user_count = users
        result.spawn_rate = spawn_rate
        result.duration_seconds = duration
        
        # Simulate realistic metrics
        result.total_requests = int(users * duration * 2)  # ~2 RPS per user
        result.failed_requests = int(result.total_requests * 0.001)  # 0.1% error rate
        result.requests_per_second = result.total_requests / duration
        result.error_rate = result.failed_requests / result.total_requests
        
        result.response_times = {
            'median': 25.0,
            'p95': 75.0,
            'p99': 120.0,
            'avg': 35.0,
            'min': 10.0,
            'max': 200.0
        }
        
        self.results.append(result)
        print(f"[SIMULATED] Load test completed: {users} users, {duration}s")
        
        return result
    
    def run_stress_test(self, max_users: int = 10000, increment: int = 100,
                       step_duration: int = 60) -> List[LoadTestResult]:
        """
        Run stress test to find breaking point.
        
        Args:
            max_users: Maximum users to test
            increment: User increment per step
            step_duration: Duration of each step in seconds
            
        Returns:
            List of load test results
        """
        results = []
        
        for user_count in range(increment, max_users + 1, increment):
            print(f"Testing with {user_count} users...")
            
            result = self.run_load_test(
                users=user_count,
                spawn_rate=increment,
                duration=step_duration
            )
            
            results.append(result)
            
            # Check if system is failing
            if result.error_rate > 0.05:  # 5% error rate
                print(f"Breaking point found at {user_count} users")
                break
            
            # Small break between steps
            time.sleep(5)
        
        return results
    
    def run_spike_test(self, baseline_users: int = 100,
                      spike_users: int = 5000,
                      spike_duration: int = 60) -> Dict[str, LoadTestResult]:
        """
        Run spike test (sudden traffic surge).
        
        Args:
            baseline_users: Normal user count
            spike_users: Spike user count
            spike_duration: Spike duration in seconds
            
        Returns:
            Dict with baseline and spike results
        """
        # Baseline
        print("Running baseline test...")
        baseline_result = self.run_load_test(
            users=baseline_users,
            spawn_rate=10,
            duration=60
        )
        
        time.sleep(10)
        
        # Spike
        print(f"Running spike test ({spike_users} users)...")
        spike_result = self.run_load_test(
            users=spike_users,
            spawn_rate=spike_users,  # Instant spike
            duration=spike_duration
        )
        
        return {
            'baseline': baseline_result,
            'spike': spike_result
        }
    
    def analyze_results(self, result: LoadTestResult) -> Dict[str, Any]:
        """
        Analyze load test results.
        
        Args:
            result: Load test result to analyze
            
        Returns:
            Analysis dictionary
        """
        analysis = {
            'test_name': result.test_name,
            'performance_grade': 'Unknown',
            'issues': [],
            'recommendations': []
        }
        
        # Grade performance
        if result.error_rate < 0.001 and result.response_times.get('p95', 0) < 100:
            analysis['performance_grade'] = 'A (Excellent)'
        elif result.error_rate < 0.01 and result.response_times.get('p95', 0) < 200:
            analysis['performance_grade'] = 'B (Good)'
        elif result.error_rate < 0.05 and result.response_times.get('p95', 0) < 500:
            analysis['performance_grade'] = 'C (Fair)'
        else:
            analysis['performance_grade'] = 'D (Poor)'
        
        # Identify issues
        if result.error_rate > 0.01:
            analysis['issues'].append(f"High error rate: {result.error_rate:.2%}")
            analysis['recommendations'].append("Investigate error logs and add error handling")
        
        if result.response_times.get('p95', 0) > 100:
            analysis['issues'].append(f"Slow p95 latency: {result.response_times['p95']:.1f}ms")
            analysis['recommendations'].append("Optimize slow endpoints and add caching")
        
        if result.response_times.get('p99', 0) > result.response_times.get('p95', 0) * 2:
            analysis['issues'].append("Large p99 vs p95 variance")
            analysis['recommendations'].append("Investigate outliers and add timeouts")
        
        if result.requests_per_second < result.user_count:
            analysis['issues'].append("Low throughput per user")
            analysis['recommendations'].append("Check for bottlenecks and optimize connections")
        
        return analysis
    
    def compare_results(self, result1: LoadTestResult,
                       result2: LoadTestResult) -> Dict[str, Any]:
        """
        Compare two load test results.
        
        Args:
            result1: First result (baseline)
            result2: Second result (current)
            
        Returns:
            Comparison dictionary
        """
        comparison = {
            'baseline': result1.test_name,
            'current': result2.test_name,
            'improvements': [],
            'regressions': []
        }
        
        # Compare RPS
        rps_change = ((result2.requests_per_second - result1.requests_per_second) / 
                     result1.requests_per_second * 100 if result1.requests_per_second > 0 else 0)
        
        if rps_change > 10:
            comparison['improvements'].append(f"RPS improved by {rps_change:.1f}%")
        elif rps_change < -10:
            comparison['regressions'].append(f"RPS decreased by {abs(rps_change):.1f}%")
        
        # Compare p95 latency
        p95_change = ((result2.response_times.get('p95', 0) - result1.response_times.get('p95', 0)) / 
                     result1.response_times.get('p95', 1) * 100)
        
        if p95_change < -10:
            comparison['improvements'].append(f"p95 latency improved by {abs(p95_change):.1f}%")
        elif p95_change > 10:
            comparison['regressions'].append(f"p95 latency increased by {p95_change:.1f}%")
        
        # Compare error rate
        error_change = result2.error_rate - result1.error_rate
        
        if error_change < -0.005:
            comparison['improvements'].append(f"Error rate improved by {abs(error_change):.2%}")
        elif error_change > 0.005:
            comparison['regressions'].append(f"Error rate increased by {error_change:.2%}")
        
        return comparison
    
    def _save_result(self, result: LoadTestResult) -> None:
        """Save result to JSON file."""
        timestamp = result.timestamp.strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(
            self.output_dir,
            f"{result.test_name}_{timestamp}.json"
        )
        
        with open(filename, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def generate_report(self, output_file: str = 'load_test_report.html') -> None:
        """
        Generate HTML load test report.
        
        Args:
            output_file: Output HTML filename
        """
        if not self.results:
            print("No results to report")
            return
        
        html = self._generate_html_report()
        
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w') as f:
            f.write(html)
        
        print(f"Report generated: {output_path}")
    
    def _generate_html_report(self) -> str:
        """Generate HTML report content."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Load Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        .good { color: green; }
        .bad { color: red; }
    </style>
</head>
<body>
    <h1>Load Test Report</h1>
    <p>Generated: {}</p>
    
    <h2>Test Results</h2>
    <table>
        <tr>
            <th>Test Name</th>
            <th>Users</th>
            <th>Duration (s)</th>
            <th>Total Requests</th>
            <th>RPS</th>
            <th>Error Rate</th>
            <th>p95 (ms)</th>
            <th>p99 (ms)</th>
        </tr>
        {}
    </table>
</body>
</html>
        """.format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            self._generate_table_rows()
        )
        
        return html
    
    def _generate_table_rows(self) -> str:
        """Generate table rows for HTML report."""
        rows = []
        
        for result in self.results:
            error_class = 'good' if result.error_rate < 0.01 else 'bad'
            p95_class = 'good' if result.response_times.get('p95', 0) < 100 else 'bad'
            
            row = f"""
        <tr>
            <td>{result.test_name}</td>
            <td>{result.user_count}</td>
            <td>{result.duration_seconds}</td>
            <td>{result.total_requests}</td>
            <td>{result.requests_per_second:.1f}</td>
            <td class="{error_class}">{result.error_rate:.2%}</td>
            <td class="{p95_class}">{result.response_times.get('p95', 0):.1f}</td>
            <td>{result.response_times.get('p99', 0):.1f}</td>
        </tr>
            """
            rows.append(row)
        
        return '\n'.join(rows)
