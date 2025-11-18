"""
Metrics Collector

Real-time performance metrics collection and monitoring.
Collects system metrics, application metrics, and business metrics.
"""

import psutil
import time
from typing import Dict, Any, List, Optional
from collections import deque
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, field


@dataclass
class MetricSnapshot:
    """Single metric snapshot."""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Performance alert."""
    metric: str
    value: float
    threshold: float
    severity: str
    timestamp: datetime
    message: str


class MetricsCollector:
    """Real-time performance metrics collection."""
    
    def __init__(self, window_size: int = 3600):
        """
        Initialize metrics collector.
        
        Args:
            window_size: Time window for metrics in seconds (default: 1 hour)
        """
        self.window_size = window_size
        
        # Metrics storage (time-series)
        self.metrics: Dict[str, deque] = {}
        
        # Request metrics by endpoint
        self.request_metrics: Dict[str, Dict[str, deque]] = {}
        
        # Alerts
        self.alerts: List[Alert] = []
        
        # Collection start time
        self.start_time = datetime.now()
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """
        Collect CPU, memory, disk, network metrics.
        
        Returns:
            System metrics dictionary
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': psutil.cpu_percent(interval=0.1),
                'per_core': psutil.cpu_percent(interval=0.1, percpu=True),
                'count': psutil.cpu_count(),
                'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent,
                'used': psutil.virtual_memory().used,
                'swap_total': psutil.swap_memory().total,
                'swap_used': psutil.swap_memory().used,
                'swap_percent': psutil.swap_memory().percent
            },
            'disk': {
                'read_bytes': psutil.disk_io_counters().read_bytes if psutil.disk_io_counters() else 0,
                'write_bytes': psutil.disk_io_counters().write_bytes if psutil.disk_io_counters() else 0,
                'read_count': psutil.disk_io_counters().read_count if psutil.disk_io_counters() else 0,
                'write_count': psutil.disk_io_counters().write_count if psutil.disk_io_counters() else 0
            },
            'network': {
                'bytes_sent': psutil.net_io_counters().bytes_sent,
                'bytes_recv': psutil.net_io_counters().bytes_recv,
                'packets_sent': psutil.net_io_counters().packets_sent,
                'packets_recv': psutil.net_io_counters().packets_recv
            }
        }
        
        # Store metrics
        self._store_metric('cpu_percent', metrics['cpu']['percent'])
        self._store_metric('memory_percent', metrics['memory']['percent'])
        
        return metrics
    
    def _store_metric(self, name: str, value: float) -> None:
        """Store metric value with timestamp."""
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=self.window_size)
        
        snapshot = MetricSnapshot(
            timestamp=datetime.now(),
            value=value
        )
        self.metrics[name].append(snapshot)
    
    def collect_application_metrics(self, custom_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Collect application-specific metrics.
        
        Args:
            custom_metrics: Optional custom metrics to include
            
        Returns:
            Application metrics dictionary
        """
        process = psutil.Process()
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'process': {
                'cpu_percent': process.cpu_percent(),
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'num_threads': process.num_threads(),
                'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0,
                'connections': len(process.connections())
            }
        }
        
        # Add custom metrics
        if custom_metrics:
            metrics['custom'] = custom_metrics
            
            for name, value in custom_metrics.items():
                self._store_metric(f"custom_{name}", value)
        
        return metrics
    
    def record_request(self, endpoint: str, duration_ms: float,
                      status_code: int, method: str = 'GET') -> None:
        """
        Record API request metrics.
        
        Args:
            endpoint: API endpoint
            duration_ms: Request duration in milliseconds
            status_code: HTTP status code
            method: HTTP method
        """
        if endpoint not in self.request_metrics:
            self.request_metrics[endpoint] = {
                'requests': deque(maxlen=self.window_size),
                'durations': deque(maxlen=self.window_size),
                'errors': deque(maxlen=self.window_size),
                'methods': {}
            }
        
        endpoint_metrics = self.request_metrics[endpoint]
        
        # Record request
        endpoint_metrics['requests'].append(datetime.now())
        endpoint_metrics['durations'].append(duration_ms)
        
        # Record method
        if method not in endpoint_metrics['methods']:
            endpoint_metrics['methods'][method] = 0
        endpoint_metrics['methods'][method] += 1
        
        # Record error
        if status_code >= 400:
            endpoint_metrics['errors'].append({
                'timestamp': datetime.now(),
                'status_code': status_code,
                'duration_ms': duration_ms
            })
    
    def calculate_percentile(self, metric: str, percentile: float) -> float:
        """
        Calculate percentile for metric.
        
        Args:
            metric: Metric name
            percentile: Percentile (0-100)
            
        Returns:
            Percentile value
        """
        if metric not in self.metrics or not self.metrics[metric]:
            return 0.0
        
        values = [snapshot.value for snapshot in self.metrics[metric]]
        return float(np.percentile(values, percentile))
    
    def get_metrics_summary(self, time_window: int = 60) -> Dict[str, Any]:
        """
        Get metrics summary for time window.
        
        Args:
            time_window: Time window in seconds
            
        Returns:
            Metrics summary
        """
        cutoff_time = datetime.now() - timedelta(seconds=time_window)
        
        summary = {
            'time_window_seconds': time_window,
            'system': {},
            'requests': {}
        }
        
        # System metrics summary
        for metric_name in ['cpu_percent', 'memory_percent']:
            if metric_name in self.metrics:
                recent_values = [
                    snapshot.value
                    for snapshot in self.metrics[metric_name]
                    if snapshot.timestamp >= cutoff_time
                ]
                
                if recent_values:
                    summary['system'][metric_name] = {
                        'current': recent_values[-1],
                        'avg': np.mean(recent_values),
                        'min': np.min(recent_values),
                        'max': np.max(recent_values),
                        'p95': np.percentile(recent_values, 95)
                    }
        
        # Request metrics summary
        for endpoint, metrics in self.request_metrics.items():
            recent_requests = [
                ts for ts in metrics['requests']
                if ts >= cutoff_time
            ]
            
            recent_durations = list(metrics['durations'])[-len(recent_requests):]
            
            recent_errors = [
                err for err in metrics['errors']
                if err['timestamp'] >= cutoff_time
            ]
            
            if recent_requests:
                summary['requests'][endpoint] = {
                    'total_requests': len(recent_requests),
                    'rps': len(recent_requests) / time_window,
                    'error_count': len(recent_errors),
                    'error_rate': len(recent_errors) / len(recent_requests) if recent_requests else 0,
                    'latency': {
                        'avg': np.mean(recent_durations) if recent_durations else 0,
                        'p50': np.percentile(recent_durations, 50) if recent_durations else 0,
                        'p95': np.percentile(recent_durations, 95) if recent_durations else 0,
                        'p99': np.percentile(recent_durations, 99) if recent_durations else 0
                    }
                }
        
        return summary
    
    def check_alerts(self, thresholds: Dict[str, float]) -> List[Alert]:
        """
        Check if any metrics exceed thresholds.
        
        Args:
            thresholds: Threshold values for each metric
            
        Returns:
            List of active alerts
        """
        new_alerts = []
        
        # Check CPU threshold
        if 'cpu_percent' in thresholds and 'cpu_percent' in self.metrics:
            if self.metrics['cpu_percent']:
                current_cpu = self.metrics['cpu_percent'][-1].value
                
                if current_cpu > thresholds['cpu_percent']:
                    alert = Alert(
                        metric='cpu_percent',
                        value=current_cpu,
                        threshold=thresholds['cpu_percent'],
                        severity='warning' if current_cpu < thresholds['cpu_percent'] * 1.1 else 'critical',
                        timestamp=datetime.now(),
                        message=f"CPU usage ({current_cpu:.1f}%) exceeds threshold ({thresholds['cpu_percent']:.1f}%)"
                    )
                    new_alerts.append(alert)
        
        # Check memory threshold
        if 'memory_percent' in thresholds and 'memory_percent' in self.metrics:
            if self.metrics['memory_percent']:
                current_memory = self.metrics['memory_percent'][-1].value
                
                if current_memory > thresholds['memory_percent']:
                    alert = Alert(
                        metric='memory_percent',
                        value=current_memory,
                        threshold=thresholds['memory_percent'],
                        severity='warning' if current_memory < thresholds['memory_percent'] * 1.1 else 'critical',
                        timestamp=datetime.now(),
                        message=f"Memory usage ({current_memory:.1f}%) exceeds threshold ({thresholds['memory_percent']:.1f}%)"
                    )
                    new_alerts.append(alert)
        
        # Check request latency thresholds
        if 'p95_latency_ms' in thresholds:
            for endpoint, metrics in self.request_metrics.items():
                if metrics['durations']:
                    p95_latency = np.percentile(list(metrics['durations']), 95)
                    
                    if p95_latency > thresholds['p95_latency_ms']:
                        alert = Alert(
                            metric=f'{endpoint}_p95_latency',
                            value=p95_latency,
                            threshold=thresholds['p95_latency_ms'],
                            severity='warning',
                            timestamp=datetime.now(),
                            message=f"Endpoint {endpoint} p95 latency ({p95_latency:.1f}ms) exceeds threshold"
                        )
                        new_alerts.append(alert)
        
        # Check error rate thresholds
        if 'error_rate' in thresholds:
            for endpoint, metrics in self.request_metrics.items():
                if metrics['requests']:
                    recent_requests = len(metrics['requests'])
                    recent_errors = len(metrics['errors'])
                    error_rate = recent_errors / recent_requests if recent_requests > 0 else 0
                    
                    if error_rate > thresholds['error_rate']:
                        alert = Alert(
                            metric=f'{endpoint}_error_rate',
                            value=error_rate,
                            threshold=thresholds['error_rate'],
                            severity='critical',
                            timestamp=datetime.now(),
                            message=f"Endpoint {endpoint} error rate ({error_rate:.2%}) exceeds threshold"
                        )
                        new_alerts.append(alert)
        
        # Store alerts
        self.alerts.extend(new_alerts)
        
        return new_alerts
    
    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Prometheus-formatted metrics
        """
        lines = []
        
        # System metrics
        if 'cpu_percent' in self.metrics and self.metrics['cpu_percent']:
            lines.append(f"# HELP synfinance_cpu_percent CPU usage percentage")
            lines.append(f"# TYPE synfinance_cpu_percent gauge")
            lines.append(f"synfinance_cpu_percent {self.metrics['cpu_percent'][-1].value}")
        
        if 'memory_percent' in self.metrics and self.metrics['memory_percent']:
            lines.append(f"# HELP synfinance_memory_percent Memory usage percentage")
            lines.append(f"# TYPE synfinance_memory_percent gauge")
            lines.append(f"synfinance_memory_percent {self.metrics['memory_percent'][-1].value}")
        
        # Request metrics
        for endpoint, metrics in self.request_metrics.items():
            endpoint_safe = endpoint.replace('/', '_').replace('-', '_')
            
            # Request count
            lines.append(f"# HELP synfinance_requests_total Total requests")
            lines.append(f"# TYPE synfinance_requests_total counter")
            lines.append(f'synfinance_requests_total{{endpoint="{endpoint}"}} {len(metrics["requests"])}')
            
            # Request duration
            if metrics['durations']:
                p50 = np.percentile(list(metrics['durations']), 50)
                p95 = np.percentile(list(metrics['durations']), 95)
                p99 = np.percentile(list(metrics['durations']), 99)
                
                lines.append(f"# HELP synfinance_request_duration_ms Request duration")
                lines.append(f"# TYPE synfinance_request_duration_ms summary")
                lines.append(f'synfinance_request_duration_ms{{endpoint="{endpoint}",quantile="0.5"}} {p50}')
                lines.append(f'synfinance_request_duration_ms{{endpoint="{endpoint}",quantile="0.95"}} {p95}')
                lines.append(f'synfinance_request_duration_ms{{endpoint="{endpoint}",quantile="0.99"}} {p99}')
            
            # Error count
            lines.append(f"# HELP synfinance_errors_total Total errors")
            lines.append(f"# TYPE synfinance_errors_total counter")
            lines.append(f'synfinance_errors_total{{endpoint="{endpoint}"}} {len(metrics["errors"])}')
        
        return '\n'.join(lines)
    
    def get_uptime(self) -> Dict[str, Any]:
        """Get system uptime information."""
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'start_time': self.start_time.isoformat(),
            'uptime_seconds': uptime_seconds,
            'uptime_hours': uptime_seconds / 3600,
            'uptime_days': uptime_seconds / 86400
        }
    
    def reset(self) -> None:
        """Reset all metrics and alerts."""
        self.metrics = {}
        self.request_metrics = {}
        self.alerts = []
        self.start_time = datetime.now()
