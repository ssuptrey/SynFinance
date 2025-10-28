"""
HTML Dashboard Generator for Fraud Detection Analytics

Generates comprehensive, interactive HTML dashboards with embedded charts,
statistics, and model performance metrics.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import base64
from io import BytesIO
import json
from datetime import datetime
import numpy as np
from matplotlib.figure import Figure

from .advanced_analytics import AnalyticsReport, ModelMetrics, FeatureImportanceResult


class HTMLDashboardGenerator:
    """
    Generates comprehensive HTML analytics dashboards
    """
    
    def __init__(self):
        """Initialize HTML dashboard generator"""
        self.template = self._load_template()
    
    def _load_template(self) -> str:
        """Load HTML template"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .header .timestamp {{
            margin-top: 15px;
            font-size: 0.9em;
            opacity: 0.8;
        }}
        
        .nav {{
            background: #f8f9fa;
            padding: 15px 40px;
            border-bottom: 2px solid #e9ecef;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        
        .nav-button {{
            padding: 10px 20px;
            background: white;
            border: 2px solid #667eea;
            border-radius: 5px;
            cursor: pointer;
            font-weight: 600;
            color: #667eea;
            transition: all 0.3s;
        }}
        
        .nav-button:hover {{
            background: #667eea;
            color: white;
        }}
        
        .nav-button.active {{
            background: #667eea;
            color: white;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 50px;
        }}
        
        .section-title {{
            font-size: 2em;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
            transition: transform 0.3s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
        }}
        
        .chart-container {{
            margin: 30px 0;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .chart-container img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        
        .chart-title {{
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 15px;
            color: #495057;
        }}
        
        .table-container {{
            overflow-x: auto;
            margin: 20px 0;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 10px;
            overflow: hidden;
        }}
        
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e9ecef;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .metric-good {{
            color: #28a745;
            font-weight: 600;
        }}
        
        .metric-warning {{
            color: #ffc107;
            font-weight: 600;
        }}
        
        .metric-bad {{
            color: #dc3545;
            font-weight: 600;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 30px 40px;
            text-align: center;
            color: #6c757d;
            border-top: 2px solid #e9ecef;
        }}
        
        .alert {{
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }}
        
        .alert-info {{
            background: #e7f3ff;
            border-color: #0066cc;
            color: #004085;
        }}
        
        .alert-warning {{
            background: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }}
        
        .alert-success {{
            background: #d4edda;
            border-color: #28a745;
            color: #155724;
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            
            .nav {{
                display: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <div class="subtitle">{subtitle}</div>
            <div class="timestamp">Generated: {timestamp}</div>
        </div>
        
        {navigation}
        
        <div class="content">
            {content}
        </div>
        
        <div class="footer">
            <p>SynFinance Fraud Detection Analytics Dashboard</p>
            <p>© 2025 SynFinance. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
"""
    
    def _figure_to_base64(self, fig: Figure) -> str:
        """Convert matplotlib figure to base64 string"""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return f"data:image/png;base64,{img_base64}"
    
    def _generate_navigation(self, sections: List[str]) -> str:
        """Generate navigation buttons"""
        nav_html = '<div class="nav">\n'
        for section in sections:
            nav_html += f'  <button class="nav-button" onclick="location.href=\'#{section.lower().replace(" ", "-")}\'">{section}</button>\n'
        nav_html += '</div>'
        return nav_html
    
    def _generate_stats_grid(self, stats: Dict[str, Any]) -> str:
        """Generate statistics grid"""
        html = '<div class="stats-grid">\n'
        for label, value in stats.items():
            html += f'''  <div class="stat-card">
    <div class="stat-label">{label}</div>
    <div class="stat-value">{value}</div>
  </div>\n'''
        html += '</div>\n'
        return html
    
    def _generate_chart(self, title: str, fig: Figure) -> str:
        """Generate chart HTML"""
        img_data = self._figure_to_base64(fig)
        return f'''<div class="chart-container">
  <div class="chart-title">{title}</div>
  <img src="{img_data}" alt="{title}">
</div>\n'''
    
    def _generate_table(self, headers: List[str], rows: List[List[Any]]) -> str:
        """Generate HTML table"""
        html = '<div class="table-container">\n<table>\n  <thead>\n    <tr>\n'
        for header in headers:
            html += f'      <th>{header}</th>\n'
        html += '    </tr>\n  </thead>\n  <tbody>\n'
        
        for row in rows:
            html += '    <tr>\n'
            for cell in row:
                # Format cell value
                if isinstance(cell, float):
                    formatted = f'{cell:.4f}'
                    # Color code metrics
                    if cell >= 0.8:
                        css_class = 'metric-good'
                    elif cell >= 0.6:
                        css_class = 'metric-warning'
                    else:
                        css_class = 'metric-bad'
                    html += f'      <td class="{css_class}">{formatted}</td>\n'
                else:
                    html += f'      <td>{cell}</td>\n'
            html += '    </tr>\n'
        
        html += '  </tbody>\n</table>\n</div>\n'
        return html
    
    def generate_overview_section(
        self,
        dataset_info: Dict[str, Any],
        model_metrics: Optional[ModelMetrics] = None
    ) -> str:
        """Generate overview section"""
        html = '<section class="section" id="overview">\n'
        html += '  <h2 class="section-title">📊 Dataset Overview</h2>\n'
        
        # Dataset statistics
        stats = {
            "Total Transactions": f"{dataset_info.get('total_transactions', 0):,}",
            "Fraud Transactions": f"{dataset_info.get('fraud_transactions', 0):,}",
            "Fraud Rate": f"{dataset_info.get('fraud_rate', 0):.2%}",
            "Total Features": f"{dataset_info.get('total_features', 0)}",
        }
        html += self._generate_stats_grid(stats)
        
        # Model performance (if available)
        if model_metrics:
            html += '  <h3 class="section-title" style="font-size: 1.5em; margin-top: 30px;">🎯 Model Performance</h3>\n'
            perf_stats = {
                "Accuracy": f"{model_metrics.accuracy:.2%}",
                "Precision": f"{model_metrics.precision:.2%}",
                "Recall": f"{model_metrics.recall:.2%}",
                "F1-Score": f"{model_metrics.f1:.2%}",
                "ROC-AUC": f"{model_metrics.roc_auc:.4f}",
                "Avg Precision": f"{model_metrics.average_precision:.4f}",
            }
            html += self._generate_stats_grid(perf_stats)
        
        html += '</section>\n'
        return html
    
    def generate_feature_importance_section(
        self,
        importance_results: List[FeatureImportanceResult],
        charts: Dict[str, Figure]
    ) -> str:
        """Generate feature importance section"""
        html = '<section class="section" id="feature-importance">\n'
        html += '  <h2 class="section-title">⭐ Feature Importance Analysis</h2>\n'
        
        for result in importance_results:
            html += f'  <h3 style="margin-top: 30px; color: #495057;">{result.method.title()} Importance</h3>\n'
            
            # Top 10 features table
            top_features = result.get_top_features(10)
            headers = ['Rank', 'Feature', 'Importance']
            rows = [[i+1, feat, imp] for i, (feat, imp) in enumerate(top_features)]
            html += self._generate_table(headers, rows)
            
            # Chart (if available)
            chart_key = f'{result.method}_importance'
            if chart_key in charts:
                html += self._generate_chart(
                    f'{result.method.title()} Feature Importance',
                    charts[chart_key]
                )
        
        html += '</section>\n'
        return html
    
    def generate_correlation_section(
        self,
        correlation_results,
        charts: Dict[str, Figure]
    ) -> str:
        """Generate correlation analysis section"""
        html = '<section class="section" id="correlation">\n'
        html += '  <h2 class="section-title">🔗 Correlation Analysis</h2>\n'
        
        if correlation_results:
            # High correlations alert
            high_corr = correlation_results.get_highly_correlated_pairs()
            if high_corr:
                html += f'  <div class="alert alert-warning">\n'
                html += f'    <strong>⚠️ High Correlations Found:</strong> {len(high_corr)} feature pairs have correlation above {correlation_results.threshold}\n'
                html += '  </div>\n'
                
                # Table of high correlations
                if len(high_corr) > 0:
                    headers = ['Feature 1', 'Feature 2', 'Correlation']
                    rows = [[f1, f2, corr] for f1, f2, corr in high_corr[:20]]
                    html += '  <h3>Top Correlated Feature Pairs</h3>\n'
                    html += self._generate_table(headers, rows)
            else:
                html += '  <div class="alert alert-success">\n'
                html += f'    <strong>✅ No High Correlations:</strong> All feature correlations are below {correlation_results.threshold}\n'
                html += '  </div>\n'
            
            # Correlation heatmap
            if 'correlation_heatmap' in charts:
                html += self._generate_chart(
                    'Feature Correlation Heatmap',
                    charts['correlation_heatmap']
                )
        
        html += '</section>\n'
        return html
    
    def generate_model_performance_section(
        self,
        model_metrics: ModelMetrics,
        charts: Dict[str, Figure]
    ) -> str:
        """Generate model performance section"""
        html = '<section class="section" id="model-performance">\n'
        html += '  <h2 class="section-title">🎯 Model Performance Metrics</h2>\n'
        
        # Confusion matrix
        if 'confusion_matrix' in charts:
            html += self._generate_chart(
                'Confusion Matrix',
                charts['confusion_matrix']
            )
        
        # ROC curve
        if 'roc_curve' in charts:
            html += self._generate_chart(
                'ROC Curve',
                charts['roc_curve']
            )
        
        # Precision-Recall curve
        if 'pr_curve' in charts:
            html += self._generate_chart(
                'Precision-Recall Curve',
                charts['pr_curve']
            )
        
        # Classification report
        html += '  <h3 style="margin-top: 30px;">Classification Report</h3>\n'
        html += f'  <pre style="background: #f8f9fa; padding: 20px; border-radius: 5px; overflow-x: auto;">{model_metrics.classification_report}</pre>\n'
        
        html += '</section>\n'
        return html
    
    def generate_anomaly_section(
        self,
        anomaly_stats: Dict[str, Any],
        charts: Dict[str, Figure]
    ) -> str:
        """Generate anomaly detection section"""
        html = '<section class="section" id="anomaly-detection">\n'
        html += '  <h2 class="section-title">🔍 Anomaly Detection Analysis</h2>\n'
        
        # Anomaly statistics
        stats = {
            "Total Anomalies": f"{anomaly_stats.get('total_anomalies', 0):,}",
            "Anomaly Rate": f"{anomaly_stats.get('anomaly_rate', 0):.2%}",
            "Avg Severity": f"{anomaly_stats.get('avg_severity', 0):.3f}",
            "High Severity": f"{anomaly_stats.get('high_severity_count', 0):,}",
        }
        html += self._generate_stats_grid(stats)
        
        # Anomaly type distribution
        if 'anomaly_types' in anomaly_stats:
            headers = ['Anomaly Type', 'Count', 'Percentage']
            rows = []
            total = sum(anomaly_stats['anomaly_types'].values())
            for atype, count in anomaly_stats['anomaly_types'].items():
                pct = count / total if total > 0 else 0
                rows.append([atype, count, f'{pct:.2%}'])
            html += '  <h3>Anomaly Type Distribution</h3>\n'
            html += self._generate_table(headers, rows)
        
        # Charts
        if 'anomaly_distribution' in charts:
            html += self._generate_chart(
                'Anomaly Distribution',
                charts['anomaly_distribution']
            )
        
        html += '</section>\n'
        return html
    
    def generate_dashboard(
        self,
        output_path: str,
        title: str = "Fraud Detection Analytics Dashboard",
        subtitle: str = "Comprehensive Analysis Report",
        dataset_info: Dict[str, Any] = None,
        model_metrics: Optional[ModelMetrics] = None,
        importance_results: Optional[List[FeatureImportanceResult]] = None,
        correlation_results = None,
        anomaly_stats: Optional[Dict[str, Any]] = None,
        charts: Optional[Dict[str, Figure]] = None
    ):
        """
        Generate complete HTML dashboard
        
        Args:
            output_path: Path to save HTML file
            title: Dashboard title
            subtitle: Dashboard subtitle
            dataset_info: Dataset statistics
            model_metrics: Model performance metrics
            importance_results: Feature importance results
            correlation_results: Correlation analysis results
            anomaly_stats: Anomaly detection statistics
            charts: Dictionary of {chart_name: matplotlib_figure}
        """
        dataset_info = dataset_info or {}
        charts = charts or {}
        
        # Generate sections
        sections = []
        content = ""
        
        # Overview
        sections.append("Overview")
        content += self.generate_overview_section(dataset_info, model_metrics)
        
        # Feature Importance
        if importance_results:
            sections.append("Feature Importance")
            content += self.generate_feature_importance_section(importance_results, charts)
        
        # Correlation Analysis
        if correlation_results:
            sections.append("Correlation")
            content += self.generate_correlation_section(correlation_results, charts)
        
        # Model Performance
        if model_metrics:
            sections.append("Model Performance")
            content += self.generate_model_performance_section(model_metrics, charts)
        
        # Anomaly Detection
        if anomaly_stats:
            sections.append("Anomaly Detection")
            content += self.generate_anomaly_section(anomaly_stats, charts)
        
        # Generate navigation
        navigation = self._generate_navigation(sections)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Fill template
        html = self.template.format(
            title=title,
            subtitle=subtitle,
            timestamp=timestamp,
            navigation=navigation,
            content=content
        )
        
        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding='utf-8')
        
        print(f"✅ Dashboard saved to: {output_path}")
        print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
