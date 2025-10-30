"""
Structured Logging Framework for SynFinance

Provides comprehensive structured logging with:
- JSON-formatted log output
- Contextual information (request ID, user ID, correlation ID)
- Thread-safe context management
- Log level management
- Performance tracking
- Security event logging
- Integration with standard Python logging

Week 7 Day 4: Enhanced Observability
"""

import logging
import json
import sys
import threading
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict


class LogLevel(Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """Log categories for classification"""
    SYSTEM = "system"
    API = "api"
    DATA_GENERATION = "data_generation"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL = "model"
    QUALITY = "quality"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DATABASE = "database"
    CACHE = "cache"


@dataclass
class LogContext:
    """
    Thread-local logging context
    
    Attributes:
        request_id: Unique request identifier
        correlation_id: Correlation ID for distributed tracing
        user_id: User identifier
        session_id: Session identifier
        transaction_id: Transaction being processed
        customer_id: Customer being processed
        operation: Current operation name
        metadata: Additional context metadata
    """
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    transaction_id: Optional[str] = None
    customer_id: Optional[str] = None
    operation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        result = {}
        for key, value in asdict(self).items():
            if value is not None and value != {}:
                result[key] = value
        return result
    
    def update(self, **kwargs) -> None:
        """Update context fields"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.metadata[key] = value


class ContextManager:
    """Thread-safe context manager for logging"""
    
    def __init__(self):
        self._local = threading.local()
    
    def get_context(self) -> LogContext:
        """Get current thread's log context"""
        if not hasattr(self._local, 'context'):
            self._local.context = LogContext()
        return self._local.context
    
    def set_context(self, context: LogContext) -> None:
        """Set current thread's log context"""
        self._local.context = context
    
    def clear_context(self) -> None:
        """Clear current thread's log context"""
        if hasattr(self._local, 'context'):
            del self._local.context
    
    def update_context(self, **kwargs) -> None:
        """Update current thread's log context"""
        context = self.get_context()
        context.update(**kwargs)


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter
    
    Formats log records as JSON with structured fields
    """
    
    def __init__(self, context_manager: ContextManager):
        super().__init__()
        self.context_manager = context_manager
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        # Build base log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add context
        context = self.context_manager.get_context()
        context_dict = context.to_dict()
        if context_dict:
            log_entry['context'] = context_dict
        
        # Add location information
        log_entry['location'] = {
            'file': record.pathname,
            'line': record.lineno,
            'function': record.funcName,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info),
            }
        
        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'message', 'pathname', 'process', 'processName',
                          'relativeCreated', 'thread', 'threadName', 'exc_info',
                          'exc_text', 'stack_info']:
                extra_fields[key] = value
        
        if extra_fields:
            log_entry['extra'] = extra_fields
        
        return json.dumps(log_entry)


class StructuredLogger:
    """
    Structured logging system
    
    Provides JSON-formatted logging with context management,
    performance tracking, and security event logging.
    
    Examples:
        >>> logger = StructuredLogger("myapp")
        >>> logger.info("User logged in", user_id="123")
        >>> 
        >>> with logger.context(request_id="req-456"):
        ...     logger.debug("Processing request")
        ...     logger.error("Request failed", error_code="E001")
    """
    
    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        output_file: Optional[str] = None,
        console_output: bool = True
    ):
        """
        Initialize structured logger
        
        Args:
            name: Logger name
            level: Minimum log level
            output_file: Optional file path for log output
            console_output: Whether to output to console
        """
        self.name = name
        self.context_manager = ContextManager()
        
        # Create Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.value))
        self.logger.handlers.clear()  # Remove existing handlers
        
        # Create JSON formatter
        formatter = JSONFormatter(self.context_manager)
        
        # Add console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Add file handler
        if output_file:
            file_handler = logging.FileHandler(output_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Performance tracking
        self._operation_timings: Dict[str, List[float]] = {}
    
    def debug(self, message: str, category: Optional[LogCategory] = None, **kwargs) -> None:
        """Log debug message"""
        self._log(LogLevel.DEBUG, message, category, **kwargs)
    
    def info(self, message: str, category: Optional[LogCategory] = None, **kwargs) -> None:
        """Log info message"""
        self._log(LogLevel.INFO, message, category, **kwargs)
    
    def warning(self, message: str, category: Optional[LogCategory] = None, **kwargs) -> None:
        """Log warning message"""
        self._log(LogLevel.WARNING, message, category, **kwargs)
    
    def error(self, message: str, category: Optional[LogCategory] = None, **kwargs) -> None:
        """Log error message"""
        self._log(LogLevel.ERROR, message, category, **kwargs)
    
    def critical(self, message: str, category: Optional[LogCategory] = None, **kwargs) -> None:
        """Log critical message"""
        self._log(LogLevel.CRITICAL, message, category, **kwargs)
    
    def exception(self, message: str, category: Optional[LogCategory] = None, **kwargs) -> None:
        """Log exception with traceback"""
        kwargs['exc_info'] = True
        self._log(LogLevel.ERROR, message, category, **kwargs)
    
    def _log(
        self,
        level: LogLevel,
        message: str,
        category: Optional[LogCategory] = None,
        **kwargs
    ) -> None:
        """Internal logging method"""
        # Add category to kwargs if provided
        if category:
            kwargs['category'] = category.value
        
        # Log with Python logger
        log_func = getattr(self.logger, level.value.lower())
        log_func(message, extra=kwargs)
    
    @contextmanager
    def context(self, **kwargs):
        """
        Context manager for temporary logging context
        
        Args:
            **kwargs: Context fields to set
        
        Example:
            >>> with logger.context(request_id="123", user_id="456"):
            ...     logger.info("Processing request")
        """
        # Save current context
        old_context = LogContext(**self.context_manager.get_context().__dict__)
        
        # Update context
        self.context_manager.update_context(**kwargs)
        
        try:
            yield
        finally:
            # Restore old context
            self.context_manager.set_context(old_context)
    
    def set_context(self, **kwargs) -> None:
        """
        Set logging context fields
        
        Args:
            **kwargs: Context fields to set
        """
        self.context_manager.update_context(**kwargs)
    
    def clear_context(self) -> None:
        """Clear logging context"""
        self.context_manager.clear_context()
    
    def get_context(self) -> LogContext:
        """Get current logging context"""
        return self.context_manager.get_context()
    
    @contextmanager
    def operation(self, operation_name: str, **context_kwargs):
        """
        Context manager for tracking operation performance
        
        Args:
            operation_name: Name of the operation
            **context_kwargs: Additional context fields
        
        Example:
            >>> with logger.operation("generate_transactions", customer_id="123"):
            ...     # Generate transactions
            ...     pass
        """
        start_time = datetime.now()
        
        # Set operation context
        with self.context(operation=operation_name, **context_kwargs):
            self.debug(f"Starting operation: {operation_name}")
            
            try:
                yield
                
                # Log success
                duration = (datetime.now() - start_time).total_seconds()
                self._record_timing(operation_name, duration)
                self.info(
                    f"Completed operation: {operation_name}",
                    category=LogCategory.PERFORMANCE,
                    duration_seconds=duration
                )
                
            except Exception as e:
                # Log failure
                duration = (datetime.now() - start_time).total_seconds()
                self.exception(
                    f"Failed operation: {operation_name}",
                    category=LogCategory.SYSTEM,
                    duration_seconds=duration,
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                raise
    
    def _record_timing(self, operation: str, duration: float) -> None:
        """Record operation timing"""
        if operation not in self._operation_timings:
            self._operation_timings[operation] = []
        self._operation_timings[operation].append(duration)
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance statistics for tracked operations
        
        Returns:
            Dictionary mapping operation names to statistics
        """
        stats = {}
        for operation, timings in self._operation_timings.items():
            if timings:
                stats[operation] = {
                    'count': len(timings),
                    'min': min(timings),
                    'max': max(timings),
                    'mean': sum(timings) / len(timings),
                    'total': sum(timings)
                }
        return stats
    
    def log_security_event(
        self,
        event_type: str,
        severity: LogLevel,
        message: str,
        **kwargs
    ) -> None:
        """
        Log security event
        
        Args:
            event_type: Type of security event
            severity: Event severity
            message: Event description
            **kwargs: Additional event details
        """
        kwargs['event_type'] = event_type
        kwargs['security_event'] = True
        self._log(severity, message, LogCategory.SECURITY, **kwargs)
    
    def log_api_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        **kwargs
    ) -> None:
        """
        Log API request
        
        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            duration_ms: Request duration in milliseconds
            **kwargs: Additional request details
        """
        kwargs.update({
            'method': method,
            'path': path,
            'status_code': status_code,
            'duration_ms': duration_ms,
        })
        
        level = LogLevel.INFO if status_code < 400 else LogLevel.ERROR
        self._log(
            level,
            f"{method} {path} - {status_code}",
            LogCategory.API,
            **kwargs
        )
    
    def set_level(self, level: LogLevel) -> None:
        """Set minimum log level"""
        self.logger.setLevel(getattr(logging, level.value))


# Global logger instance
_global_logger: Optional[StructuredLogger] = None


def get_logger(name: Optional[str] = None) -> StructuredLogger:
    """
    Get or create structured logger
    
    Args:
        name: Logger name (uses default if None)
    
    Returns:
        StructuredLogger instance
    """
    global _global_logger
    
    if name is None:
        if _global_logger is None:
            _global_logger = StructuredLogger("synfinance")
        return _global_logger
    
    return StructuredLogger(name)


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    output_file: Optional[str] = None,
    console_output: bool = True
) -> StructuredLogger:
    """
    Configure global logging
    
    Args:
        level: Minimum log level
        output_file: Optional file path for log output
        console_output: Whether to output to console
    
    Returns:
        Configured global logger
    """
    global _global_logger
    _global_logger = StructuredLogger(
        "synfinance",
        level=level,
        output_file=output_file,
        console_output=console_output
    )
    return _global_logger
