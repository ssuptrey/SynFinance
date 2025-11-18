"""
Visualization gallery providing pre-configured dashboards and chart collections.

Provides high-level functions to generate comprehensive visualization suites
for common analysis scenarios.
"""

from typing import Optional, List, Dict, Any
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from .static_charts import StaticCharts
from .interactive_charts import InteractiveCharts
from .statistical_plots import StatisticalPlots
from .export import ExportManager


class VisualizationGallery:
    """Gallery of pre-configured visualization dashboards."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        theme: str = "default",
        output_dir: Optional[str] = None
    ):
        """
        Initialize visualization gallery.
        
        Args:
            data: DataFrame containing data to visualize
            theme: Theme name for static charts
            output_dir: Default output directory for saved visualizations
        """
        self.data = data
        self.theme = theme
        self.output_dir = output_dir or 'visualizations'
        
        # Initialize generators
        self.static = StaticCharts(theme=theme)
        self.interactive = InteractiveCharts(template="plotly_white")
        self.statistical = StatisticalPlots(theme=theme)
        self.export_manager = ExportManager(output_dir=output_dir)
    
    def generate_overview_dashboard(
        self,
        output_dir: Optional[str] = None,
        save: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive dataset overview dashboard.
        
        Args:
            output_dir: Output directory for saved charts
            save: Whether to save charts to files
            
        Returns:
            Dictionary of chart names and figure objects
        """
        output_dir = output_dir or os.path.join(self.output_dir, 'overview')
        if save:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        charts = {}
        
        # Get numeric columns
        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        
        # Distribution plots for numeric columns (first 4)
        for i, col in enumerate(numeric_cols[:4]):
            fig = self.static.plot_distribution(
                self.data,
                field=col,
                title=f'Distribution of {col}',
                kde=True
            )
            charts[f'distribution_{col}'] = fig
            if save:
                self.export_manager.export_to_png(
                    fig,
                    os.path.join(output_dir, f'distribution_{col}.png'),
                    dpi=300
                )
                plt.close(fig)
        
        # Correlation heatmap (if enough numeric columns)
        if len(numeric_cols) > 1:
            fig = self.static.plot_correlation_heatmap(
                self.data[numeric_cols],
                title='Feature Correlations'
            )
            charts['correlation_heatmap'] = fig
            if save:
                self.export_manager.export_to_png(
                    fig,
                    os.path.join(output_dir, 'correlation_heatmap.png'),
                    dpi=300
                )
                plt.close(fig)
        
        # Box plots for outlier detection (first 4 numeric columns)
        for i, col in enumerate(numeric_cols[:4]):
            fig = self.static.plot_boxplot(
                self.data,
                y=col,
                title=f'Outliers in {col}'
            )
            charts[f'boxplot_{col}'] = fig
            if save:
                self.export_manager.export_to_png(
                    fig,
                    os.path.join(output_dir, f'boxplot_{col}.png'),
                    dpi=300
                )
                plt.close(fig)
        
        return charts
    
    def generate_correlation_dashboard(
        self,
        output_dir: Optional[str] = None,
        save: bool = True,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Generate correlation analysis dashboard.
        
        Args:
            output_dir: Output directory for saved charts
            save: Whether to save charts to files
            top_n: Number of top correlations to visualize
            
        Returns:
            Dictionary of chart names and figure objects
        """
        output_dir = output_dir or os.path.join(self.output_dir, 'correlation')
        if save:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        charts = {}
        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            return charts
        
        # Correlation heatmaps for different methods
        for method in ['pearson', 'spearman']:
            fig = self.static.plot_correlation_heatmap(
                self.data[numeric_cols],
                method=method,
                title=f'{method.capitalize()} Correlation Matrix'
            )
            charts[f'correlation_{method}'] = fig
            if save:
                self.export_manager.export_to_png(
                    fig,
                    os.path.join(output_dir, f'correlation_{method}.png'),
                    dpi=300
                )
                plt.close(fig)
        
        # Scatter plots for top correlations
        corr_matrix = self.data[numeric_cols].corr()
        
        # Find top correlations (excluding diagonal)
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                correlations.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'corr': abs(corr_matrix.iloc[i, j])
                })
        
        top_corr = sorted(correlations, key=lambda x: x['corr'], reverse=True)[:min(top_n, len(correlations))]
        
        for item in top_corr[:4]:  # Limit to 4 scatter plots
            fig = self.static.plot_scatter(
                self.data,
                x=item['var1'],
                y=item['var2'],
                title=f"{item['var1']} vs {item['var2']} (r={item['corr']:.3f})",
                add_regression=True
            )
            charts[f"scatter_{item['var1']}_vs_{item['var2']}"] = fig
            if save:
                self.export_manager.export_to_png(
                    fig,
                    os.path.join(output_dir, f"scatter_{item['var1']}_vs_{item['var2']}.png"),
                    dpi=300
                )
                plt.close(fig)
        
        return charts
    
    def generate_distribution_dashboard(
        self,
        output_dir: Optional[str] = None,
        save: bool = True
    ) -> Dict[str, Any]:
        """
        Generate distribution analysis dashboard.
        
        Args:
            output_dir: Output directory for saved charts
            save: Whether to save charts to files
            
        Returns:
            Dictionary of chart names and figure objects
        """
        output_dir = output_dir or os.path.join(self.output_dir, 'distribution')
        if save:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        charts = {}
        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        
        # Distribution plots
        for col in numeric_cols[:6]:  # Limit to 6 columns
            # Histogram with KDE
            fig = self.static.plot_distribution(
                self.data,
                field=col,
                title=f'Distribution of {col}',
                kde=True
            )
            charts[f'hist_{col}'] = fig
            if save:
                self.export_manager.export_to_png(
                    fig, 
                    os.path.join(output_dir, f'hist_{col}.png'),
                    dpi=300
                )
                plt.close(fig)
            
            # Q-Q plot
            fig = self.static.plot_qq_plot(
                self.data[col].dropna(),
                title=f'Q-Q Plot for {col}'
            )
            charts[f'qq_{col}'] = fig
            if save:
                self.export_manager.export_to_png(
                    fig,
                    os.path.join(output_dir, f'qq_{col}.png'),
                    dpi=300
                )
                plt.close(fig)
            
            # Box plot
            fig = self.static.plot_boxplot(
                self.data,
                y=col,
                title=f'Box Plot for {col}'
            )
            charts[f'box_{col}'] = fig
            if save:
                self.export_manager.export_to_png(
                    fig,
                    os.path.join(output_dir, f'box_{col}.png'),
                    dpi=300
                )
                plt.close(fig)
        
        return charts
    
    def generate_time_series_dashboard(
        self,
        time_col: str,
        value_cols: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        save: bool = True
    ) -> Dict[str, Any]:
        """
        Generate time series analysis dashboard.
        
        Args:
            time_col: Column containing time/date values
            value_cols: Columns to plot over time (uses all numeric if None)
            output_dir: Output directory for saved charts
            save: Whether to save charts to files
            
        Returns:
            Dictionary of chart names and figure objects
        """
        output_dir = output_dir or os.path.join(self.output_dir, 'time_series')
        if save:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        charts = {}
        
        if value_cols is None:
            value_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        
        # Time series plots
        for col in value_cols[:4]:  # Limit to 4 series
            fig = self.static.plot_time_series(
                self.data,
                x=time_col,
                y=col,
                title=f'{col} Over Time',
                add_trend=True
            )
            charts[f'timeseries_{col}'] = fig
            if save:
                self.export_manager.export_to_png(
                    fig,
                    os.path.join(output_dir, f'timeseries_{col}.png'),
                    dpi=300
                )
                plt.close(fig)
        
        # Multi-line chart
        if len(value_cols) > 1:
            fig = self.static.plot_time_series(
                self.data,
                x=time_col,
                y=value_cols[:5],  # Limit to 5 lines
                title='Multiple Time Series'
            )
            charts['timeseries_multi'] = fig
            if save:
                self.export_manager.export_to_png(
                    fig,
                    os.path.join(output_dir, 'timeseries_multi.png'),
                    dpi=300
                )
                plt.close(fig)
        
        return charts
    
    def generate_fraud_analysis_dashboard(
        self,
        fraud_col: str = 'is_fraud',
        amount_col: str = 'amount',
        output_dir: Optional[str] = None,
        save: bool = True
    ) -> Dict[str, Any]:
        """
        Generate fraud-specific analysis dashboard.
        
        Args:
            fraud_col: Column indicating fraud (binary)
            amount_col: Column containing transaction amounts
            output_dir: Output directory for saved charts
            save: Whether to save charts to files
            
        Returns:
            Dictionary of chart names and figure objects
        """
        output_dir = output_dir or os.path.join(self.output_dir, 'fraud_analysis')
        if save:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        charts = {}
        
        if fraud_col not in self.data.columns:
            return charts
        
        # Fraud rate pie chart
        fraud_counts = self.data[fraud_col].value_counts()
        fig = self.static.plot_pie_chart(
            fraud_counts,
            title='Fraud vs Legitimate Transactions'
        )
        charts['fraud_rate_pie'] = fig
        if save:
            self.export_manager.export_to_png(
                fig,
                os.path.join(output_dir, 'fraud_rate_pie.png'),
                dpi=300
            )
            plt.close(fig)
        
        # Amount distribution by fraud status
        if amount_col in self.data.columns:
            fig = self.static.plot_violin(
                self.data,
                x=fraud_col,
                y=amount_col,
                title='Transaction Amount by Fraud Status'
            )
            charts['amount_by_fraud'] = fig
            if save:
                self.export_manager.export_to_png(
                    fig,
                    os.path.join(output_dir, 'amount_by_fraud.png'),
                    dpi=300
                )
                plt.close(fig)
        
        # Fraud rate by categorical variables
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols[:3]:  # Limit to 3 categorical columns
            if col != fraud_col:
                fraud_by_category = self.data.groupby(col)[fraud_col].mean().sort_values(ascending=False).head(10)
                fig = self.static.plot_bar_chart(
                    pd.DataFrame({col: fraud_by_category.index, 'fraud_rate': fraud_by_category.values}),
                    x=col,
                    y='fraud_rate',
                    title=f'Fraud Rate by {col}'
                )
                charts[f'fraud_by_{col}'] = fig
                if save:
                    self.export_manager.export_to_png(
                        fig,
                        os.path.join(output_dir, f'fraud_by_{col}.png'),
                        dpi=300
                    )
                    plt.close(fig)
        
        return charts
    
    def save_gallery(
        self,
        output_dir: Optional[str] = None,
        formats: List[str] = ['png']
    ) -> Dict[str, List[str]]:
        """
        Save all generated charts to files.
        
        Args:
            output_dir: Output directory
            formats: List of formats to save
            
        Returns:
            Dictionary mapping chart types to lists of saved file paths
        """
        output_dir = output_dir or self.output_dir
        
        saved_files = {}
        
        # Generate all dashboards
        overview = self.generate_overview_dashboard(
            output_dir=os.path.join(output_dir, 'overview'),
            save=True
        )
        saved_files['overview'] = list(overview.keys())
        
        correlation = self.generate_correlation_dashboard(
            output_dir=os.path.join(output_dir, 'correlation'),
            save=True
        )
        saved_files['correlation'] = list(correlation.keys())
        
        distribution = self.generate_distribution_dashboard(
            output_dir=os.path.join(output_dir, 'distribution'),
            save=True
        )
        saved_files['distribution'] = list(distribution.keys())
        
        return saved_files
    
    def create_html_report(
        self,
        output_file: str,
        include_static: bool = True,
        include_interactive: bool = True,
        title: str = "Data Analysis Report"
    ) -> str:
        """
        Create HTML report with all visualizations.
        
        Args:
            output_file: Output HTML file path
            include_static: Whether to include static charts
            include_interactive: Whether to include interactive charts
            title: Report title
            
        Returns:
            Path to created HTML file
        """
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                }}
                h2 {{
                    color: #555;
                    border-bottom: 2px solid #ddd;
                    padding-bottom: 10px;
                }}
                .chart-container {{
                    background-color: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Dataset Overview</h2>
            <div class="chart-container">
                <p><strong>Records:</strong> {len(self.data)}</p>
                <p><strong>Columns:</strong> {len(self.data.columns)}</p>
                <p><strong>Numeric Columns:</strong> {len(self.data.select_dtypes(include=['number']).columns)}</p>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        return output_file
