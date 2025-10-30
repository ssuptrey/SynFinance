"""
SynFinance Observability Framework

Comprehensive observability tools for monitoring, debugging, and optimizing
the SynFinance fraud detection system.

Components:
- StructuredLogger: JSON logging with context management
- PerformanceProfiler: CPU, memory, and I/O profiling
- DebugInspector: Object introspection and debugging

Week 7 Day 4: Enhanced Observability
"""

from .structured_logger import (
    LogLevel,
    LogCategory,
    LogContext,
    StructuredLogger,
    get_logger,
    configure_logging
)

from .profiling import (
    ProfileResult,
    FunctionStats,
    CPUProfiler,
    MemoryProfiler,
    IOProfiler,
    PerformanceProfiler,
    profile_operation
)

from .inspector import (
    InspectionResult,
    ObjectInspector,
    TransactionInspector,
    FeatureInspector,
    ModelInspector,
    DebugInspector,
    inspect_object
)

__all__ = [
    # Logging
    'LogLevel',
    'LogCategory',
    'LogContext',
    'StructuredLogger',
    'get_logger',
    'configure_logging',
    
    # Profiling
    'ProfileResult',
    'FunctionStats',
    'CPUProfiler',
    'MemoryProfiler',
    'IOProfiler',
    'PerformanceProfiler',
    'profile_operation',
    
    # Inspection
    'InspectionResult',
    'ObjectInspector',
    'TransactionInspector',
    'FeatureInspector',
    'ModelInspector',
    'DebugInspector',
    'inspect_object',
]

__version__ = "1.0.0"
