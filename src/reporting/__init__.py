"""
Reporting Package

Provides comprehensive reporting and dataset comparison capabilities for SynFinance.

Modules:
- html_generator: HTML report generation with Jinja2 templates
- pdf_exporter: PDF export (requires WeasyPrint - optional on Windows)
- excel_generator: Excel dashboard creation with openpyxl
- dataset_comparator: Multi-dataset comparison with statistical tests

Author: SynFinance Development Team
Date: November 2, 2025
Version: 2.17.0
"""

from src.reporting.html_generator import HTMLReportGenerator
from src.reporting.excel_generator import ExcelDashboardGenerator
from src.reporting.dataset_comparator import DatasetComparator, ComparisonResult, FieldComparison

# PDF exporter is optional (requires WeasyPrint/GTK on Windows)
try:
    from src.reporting.pdf_exporter import PDFExporter
    PDF_AVAILABLE = True
except (ImportError, OSError):
    PDFExporter = None
    PDF_AVAILABLE = False

__version__ = "2.17.0"

__all__ = [
    'HTMLReportGenerator',
    'ExcelDashboardGenerator',
    'DatasetComparator',
    'ComparisonResult',
    'FieldComparison',
    'PDFExporter',
    'PDF_AVAILABLE'
]
