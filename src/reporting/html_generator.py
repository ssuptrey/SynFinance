"""
HTML Report Generator Module

This module provides HTML report generation capabilities using Jinja2 templates.
Supports multiple report types with embedded charts and professional styling.

Author: SynFinance Development Team
Date: November 2, 2025
Version: 2.17.0
"""

import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


class HTMLReportGenerator:
    """
    Generates professional HTML reports using Jinja2 templates.
    
    Supports multiple report types:
    - Executive Summary
    - Technical Analysis
    - Fraud Detection
    - Data Quality
    """
    
    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize the HTML report generator.
        
        Args:
            template_dir: Directory containing Jinja2 templates.
                         Defaults to src/reporting/templates/
        """
        if template_dir is None:
            # Default to templates directory relative to this file
            template_dir = Path(__file__).parent / "templates"
        
        self.template_dir = Path(template_dir)
        
        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
        # Register custom filters
        self._register_filters()
    
    def _register_filters(self):
        """Register custom Jinja2 filters for formatting."""
        
        def number_format(value: Union[int, float], decimals: int = 0) -> str:
            """Format number with thousand separators."""
            try:
                if decimals == 0:
                    return f"{int(value):,}"
                return f"{float(value):,.{decimals}f}"
            except (ValueError, TypeError):
                return str(value)
        
        def percentage(value: Union[int, float], decimals: int = 1) -> str:
            """Format number as percentage."""
            try:
                return f"{float(value) * 100:.{decimals}f}%"
            except (ValueError, TypeError):
                return str(value)
        
        def currency(value: Union[int, float], symbol: str = "₹") -> str:
            """Format number as currency."""
            try:
                return f"{symbol}{float(value):,.2f}"
            except (ValueError, TypeError):
                return str(value)
        
        def date_format(value: Union[str, datetime], format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
            """Format datetime object."""
            if isinstance(value, str):
                return value
            try:
                return value.strftime(format_str)
            except AttributeError:
                return str(value)
        
        # Register filters
        self.env.filters['number_format'] = number_format
        self.env.filters['percentage'] = percentage
        self.env.filters['currency'] = currency
        self.env.filters['date_format'] = date_format
    
    def _figure_to_base64(self, fig: plt.Figure) -> str:
        """
        Convert matplotlib figure to base64 string for embedding.
        
        Args:
            fig: Matplotlib figure object
            
        Returns:
            Base64 encoded PNG image string
        """
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        plt.close(fig)
        return image_base64
    
    def generate_executive_report(
        self,
        data: pd.DataFrame,
        metrics: Dict[str, Any],
        charts: Optional[Dict[str, plt.Figure]] = None,
        findings: Optional[List[Dict[str, str]]] = None,
        recommendations: Optional[List[Dict[str, str]]] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Generate an executive summary report.
        
        Args:
            data: Transaction data DataFrame
            metrics: Dictionary of key metrics (total_transactions, fraud_rate, etc.)
            charts: Dictionary of matplotlib figures to embed
            findings: List of key findings with title, description, severity
            recommendations: List of recommendations
            output_path: Optional path to save HTML file
            
        Returns:
            HTML string
        """
        # Calculate default metrics if not provided
        if 'total_transactions' not in metrics:
            metrics['total_transactions'] = len(data)
        
        if 'total_amount' not in metrics and 'Transaction_Amount' in data.columns:
            metrics['total_amount'] = data['Transaction_Amount'].sum()
            metrics['avg_amount'] = data['Transaction_Amount'].mean()
        
        if 'fraud_rate' not in metrics and 'Fraud_Type' in data.columns:
            fraud_count = data['Fraud_Type'].notna().sum()
            metrics['fraud_count'] = fraud_count
            metrics['fraud_rate'] = fraud_count / len(data) if len(data) > 0 else 0
        
        # Convert charts to base64
        chart_dict = {}
        if charts:
            for name, fig in charts.items():
                if isinstance(fig, plt.Figure):
                    chart_dict[name] = self._figure_to_base64(fig)
        
        # Load and render template
        template = self.env.get_template('executive_report.html')
        html = template.render(
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            report_type="Executive Summary",
            version="2.17.0",
            metrics=metrics,
            charts=chart_dict,
            findings=findings or [],
            recommendations=recommendations or []
        )
        
        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html, encoding='utf-8')
        
        return html
    
    def generate_technical_report(
        self,
        data: pd.DataFrame,
        statistics: Dict[str, Any],
        correlations: Optional[pd.DataFrame] = None,
        charts: Optional[Dict[str, plt.Figure]] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Generate a technical analysis report with detailed statistics.
        
        Args:
            data: Transaction data DataFrame
            statistics: Detailed statistical analysis results
            correlations: Correlation matrix DataFrame
            charts: Dictionary of matplotlib figures
            output_path: Optional path to save HTML file
            
        Returns:
            HTML string
        """
        # Convert charts to base64
        chart_dict = {}
        if charts:
            for name, fig in charts.items():
                if isinstance(fig, plt.Figure):
                    chart_dict[name] = self._figure_to_base64(fig)
        
        # Prepare correlation data
        correlation_html = ""
        if correlations is not None:
            correlation_html = correlations.to_html(
                classes='table table-striped',
                float_format=lambda x: f'{x:.3f}'
            )
        
        # Load template (will create this next)
        template = self.env.get_template('base.html')  # Using base for now
        html = template.render(
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            report_type="Technical Analysis",
            version="2.17.0",
            statistics=statistics,
            charts=chart_dict,
            correlation_html=correlation_html
        )
        
        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html, encoding='utf-8')
        
        return html
    
    def generate_fraud_report(
        self,
        data: pd.DataFrame,
        fraud_statistics: Dict[str, Any],
        pattern_analysis: Optional[Dict[str, Any]] = None,
        charts: Optional[Dict[str, plt.Figure]] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Generate a fraud detection analysis report.
        
        Args:
            data: Transaction data DataFrame
            fraud_statistics: Fraud pattern statistics
            pattern_analysis: Detailed pattern analysis results
            charts: Dictionary of matplotlib figures
            output_path: Optional path to save HTML file
            
        Returns:
            HTML string
        """
        # Calculate fraud metrics
        fraud_data = data[data['Fraud_Type'].notna()] if 'Fraud_Type' in data.columns else pd.DataFrame()
        
        fraud_stats = {
            'total_fraud': len(fraud_data),
            'fraud_rate': len(fraud_data) / len(data) if len(data) > 0 else 0,
            'total_loss': fraud_data['Transaction_Amount'].sum() if len(fraud_data) > 0 and 'Transaction_Amount' in fraud_data.columns else 0,
            **fraud_statistics
        }
        
        # Convert charts to base64
        chart_dict = {}
        if charts:
            for name, fig in charts.items():
                if isinstance(fig, plt.Figure):
                    chart_dict[name] = self._figure_to_base64(fig)
        
        # Load template (will create specific template)
        template = self.env.get_template('base.html')
        html = template.render(
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            report_type="Fraud Detection Analysis",
            version="2.17.0",
            fraud_stats=fraud_stats,
            pattern_analysis=pattern_analysis or {},
            charts=chart_dict
        )
        
        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html, encoding='utf-8')
        
        return html
    
    def generate_quality_report(
        self,
        data: pd.DataFrame,
        quality_metrics: Dict[str, Any],
        field_analysis: Optional[Dict[str, Any]] = None,
        charts: Optional[Dict[str, plt.Figure]] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Generate a data quality analysis report.
        
        Args:
            data: Transaction data DataFrame
            quality_metrics: Data quality metrics
            field_analysis: Field-by-field quality analysis
            charts: Dictionary of matplotlib figures
            output_path: Optional path to save HTML file
            
        Returns:
            HTML string
        """
        # Calculate quality metrics
        total_fields = len(data.columns)
        missing_values = data.isnull().sum().sum()
        total_values = len(data) * len(data.columns)
        completeness = 1 - (missing_values / total_values) if total_values > 0 else 0
        
        quality_stats = {
            'completeness': completeness,
            'total_records': len(data),
            'total_fields': total_fields,
            'missing_values': missing_values,
            **quality_metrics
        }
        
        # Convert charts to base64
        chart_dict = {}
        if charts:
            for name, fig in charts.items():
                if isinstance(fig, plt.Figure):
                    chart_dict[name] = self._figure_to_base64(fig)
        
        # Load template
        template = self.env.get_template('base.html')
        html = template.render(
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            report_type="Data Quality Analysis",
            version="2.17.0",
            quality_stats=quality_stats,
            field_analysis=field_analysis or {},
            charts=chart_dict
        )
        
        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html, encoding='utf-8')
        
        return html
    
    def save_report(self, html_content: str, output_path: Union[str, Path]):
        """
        Save HTML report to file.
        
        Args:
            html_content: HTML string
            output_path: Path to save the HTML file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding='utf-8')


# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = pd.DataFrame({
        'Transaction_Amount': np.random.lognormal(8, 1.5, 100),
        'Fraud_Type': [None] * 95 + ['Card Cloning'] * 5
    })
    
    # Generate report
    generator = HTMLReportGenerator()
    
    metrics = {
        'total_transactions': len(sample_data),
        'fraud_rate': 0.05,
        'fraud_count': 5,
        'total_amount': sample_data['Transaction_Amount'].sum(),
        'avg_amount': sample_data['Transaction_Amount'].mean()
    }
    
    html = generator.generate_executive_report(
        data=sample_data,
        metrics=metrics,
        output_path="test_executive_report.html"
    )
    
    print(f"✅ Generated executive report ({len(html)} characters)")
