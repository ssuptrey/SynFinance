"""
Structured JSON logging configuration for SynFinance.

Provides:
- JSON formatted logs for easy parsing
- Request ID tracking for correlation
- Contextual fields (user, tenant, trace_id)
- Log level filtering
- Compatible with ELK, Loki, CloudWatch
"""

import logging
import sys
import os
from pythonjsonlogger import jsonlogger
from typing import Optional
import uuid
from contextvars import ContextVar

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
trace_id_var: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
tenant_id_var: ContextVar[Optional[str]] = ContextVar('tenant_id', default=None)


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter that adds contextual fields to every log record.
    """
    
    def add_fields(self, log_record, record, message_dict):
        """Add custom fields to log record."""
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        
        # Add standard fields
        log_record['timestamp'] = record.created
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        # Add service info
        log_record['service'] = 'synfinance-api'
        log_record['version'] = os.getenv('APP_VERSION', '2.15.0')
        log_record['environment'] = os.getenv('ENVIRONMENT', 'development')
        
        # Add request context if available
        request_id = request_id_var.get()
        if request_id:
            log_record['request_id'] = request_id
        
        trace_id = trace_id_var.get()
        if trace_id:
            log_record['trace_id'] = trace_id
        
        user_id = user_id_var.get()
        if user_id:
            log_record['user_id'] = user_id
        
        tenant_id = tenant_id_var.get()
        if tenant_id:
            log_record['tenant_id'] = tenant_id


def setup_logging(level: str = None, json_logs: bool = None) -> logging.Logger:
    """
    Setup structured JSON logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: Whether to use JSON format (default: True in production)
    
    Returns:
        Configured logger
    """
    # Determine log level
    if level is None:
        level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Determine format
    if json_logs is None:
        # Use JSON in production, human-readable in development
        json_logs = os.getenv('ENVIRONMENT', 'development') != 'development'
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    
    if json_logs:
        # JSON format for production
        formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )
    else:
        # Human-readable format for development
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))
    root_logger.addHandler(handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('fastapi').setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_request_context(
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None
):
    """
    Set request context for logging.
    
    This should be called at the start of each request to add
    contextual fields to all subsequent logs.
    
    Args:
        request_id: Unique request ID
        trace_id: Distributed trace ID
        user_id: User ID making the request
        tenant_id: Tenant ID for multi-tenancy
    """
    if request_id:
        request_id_var.set(request_id)
    if trace_id:
        trace_id_var.set(trace_id)
    if user_id:
        user_id_var.set(user_id)
    if tenant_id:
        tenant_id_var.set(tenant_id)


def clear_request_context():
    """Clear request context after request completes."""
    request_id_var.set(None)
    trace_id_var.set(None)
    user_id_var.set(None)
    tenant_id_var.set(None)


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


# Example usage for business events
def log_business_event(
    logger: logging.Logger,
    event_type: str,
    details: dict,
    level: str = 'INFO'
):
    """
    Log a business event with structured data.
    
    Args:
        logger: Logger instance
        event_type: Type of business event (e.g., 'transaction_created', 'fraud_detected')
        details: Event details as dictionary
        level: Log level
    """
    logger.log(
        getattr(logging, level.upper()),
        f"Business event: {event_type}",
        extra={
            'event_type': event_type,
            'event_details': details
        }
    )
