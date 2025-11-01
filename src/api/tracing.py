"""
OpenTelemetry distributed tracing configuration for SynFinance.

Provides:
- Automatic instrumentation for FastAPI
- Custom spans for ML operations
- Trace export to Jaeger/Tempo
- W3C Trace Context propagation
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
import os
from typing import Optional

def setup_tracing(
    service_name: str = "synfinance-api",
    service_version: str = "2.15.0",
    jaeger_endpoint: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
    sample_rate: float = 1.0
) -> trace.Tracer:
    """
    Setup distributed tracing with OpenTelemetry.
    
    Args:
        service_name: Name of the service
        service_version: Version of the service
        jaeger_endpoint: Jaeger collector endpoint (e.g., "jaeger:14268")
        otlp_endpoint: OTLP endpoint (e.g., "tempo:4317")
        sample_rate: Sampling rate (0.0 to 1.0, default 1.0 = 100%)
    
    Returns:
        Tracer instance
    """
    # Create resource with service information
    resource = Resource.create({
        "service.name": service_name,
        "service.version": service_version,
        "deployment.environment": os.getenv("ENVIRONMENT", "development")
    })
    
    # Create tracer provider with sampling
    tracer_provider = TracerProvider(
        resource=resource,
        sampler=TraceIdRatioBased(sample_rate)
    )
    
    # Add exporters based on configuration
    if jaeger_endpoint or os.getenv("JAEGER_ENDPOINT"):
        endpoint = jaeger_endpoint or os.getenv("JAEGER_ENDPOINT")
        jaeger_exporter = JaegerExporter(
            agent_host_name=endpoint.split(':')[0],
            agent_port=int(endpoint.split(':')[1]) if ':' in endpoint else 6831,
        )
        tracer_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
    
    if otlp_endpoint or os.getenv("OTLP_ENDPOINT"):
        endpoint = otlp_endpoint or os.getenv("OTLP_ENDPOINT")
        otlp_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    
    # Set as global tracer provider
    trace.set_tracer_provider(tracer_provider)
    
    # Return tracer instance
    return trace.get_tracer(__name__)


def instrument_fastapi_app(app):
    """
    Instrument FastAPI application with OpenTelemetry.
    
    This adds automatic tracing for all HTTP requests.
    
    Args:
        app: FastAPI application instance
    """
    FastAPIInstrumentor.instrument_app(app)


def get_current_span() -> Optional[trace.Span]:
    """Get the current active span."""
    return trace.get_current_span()


def add_span_attributes(**attributes):
    """
    Add attributes to the current span.
    
    Example:
        add_span_attributes(user_id="123", transaction_amount=1500.0)
    """
    span = get_current_span()
    if span and span.is_recording():
        for key, value in attributes.items():
            span.set_attribute(key, value)


def record_exception(exception: Exception):
    """Record an exception in the current span."""
    span = get_current_span()
    if span and span.is_recording():
        span.record_exception(exception)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))
