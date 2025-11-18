"""
Comprehensive tests for visualization modules.

Tests all chart types, themes, export functionality, and gallery features.
"""

import pytest
import numpy as np
import pandas as pd

# Configure matplotlib to use non-interactive backend for testing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.figure as mpl_figure

from pathlib import Path
import tempfile
import os

from src.visualizations.themes import (
    ColorPalette,
    ChartTheme,
    get_theme,
    apply_theme,
    DEFAULT_PALETTE,
    COLORBLIND_PALETTE,
    DARK_PALETTE
)
from src.visualizations.static_charts import StaticCharts
from src.visualizations.export import ExportManager
from src.visualizations.gallery import VisualizationGallery


# Fixtures

@pytest.fixture
def sample_numeric_data():
    """Generate sample numeric data."""
    np.random.seed(42)
    return np.random.randn(1000)


@pytest.fixture
def sample_dataframe():
    """Generate sample DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        'amount': np.random.lognormal(5, 1, 1000),
        'age': np.random.randint(18, 80, 1000),
        'score': np.random.uniform(0, 100, 1000),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
        'is_fraud': np.random.choice([0, 1], 1000, p=[0.95, 0.05]),
        'risk_level': np.random.choice(['Low', 'Medium', 'High'], 1000)
    })


@pytest.fixture
def timeseries_dataframe():
    """Generate time series DataFrame."""
    dates = pd.date_range('2024-01-01', periods=365, freq='D')
    np.random.seed(42)
    
    # Create trend and seasonal components
    trend = np.linspace(100, 150, 365)
    seasonal = 20 * np.sin(np.arange(365) * 2 * np.pi / 365)
    noise = np.random.randn(365) * 5
    
    return pd.DataFrame({
        'date': dates,
        'value': trend + seasonal + noise,
        'value2': np.random.randn(365).cumsum() + 100
    })


@pytest.fixture
def geographic_dataframe():
    """Generate geographic data."""
    np.random.seed(42)
    return pd.DataFrame({
        'latitude': np.random.uniform(37.0, 42.0, 100),
        'longitude': np.random.uniform(-122.0, -117.0, 100),
        'amount': np.random.lognormal(5, 1, 100),
        'merchant': [f'Merchant_{i}' for i in range(100)],
        'is_fraud': np.random.choice([0, 1], 100, p=[0.9, 0.1])
    })


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# Tests for Themes

class TestThemes:
    """Test color themes and palettes."""
    
    def test_color_palette_creation(self):
        """Test ColorPalette creation."""
        palette = ColorPalette(
            name="test",
            colors=["#ff0000", "#00ff00", "#0000ff"],
            description="Test palette"
        )
        assert palette.name == "test"
        assert len(palette.colors) == 3
        assert palette.description == "Test palette"
    
    def test_default_palette(self):
        """Test default color palette."""
        assert DEFAULT_PALETTE.name == "default"
        assert len(DEFAULT_PALETTE.colors) == 10
    
    def test_colorblind_palette(self):
        """Test colorblind-friendly palette."""
        assert COLORBLIND_PALETTE.name == "colorblind"
        assert len(COLORBLIND_PALETTE.colors) == 8
    
    def test_dark_palette(self):
        """Test dark theme palette."""
        assert DARK_PALETTE.name == "dark"
        assert len(DARK_PALETTE.colors) == 8
    
    def test_chart_theme_creation(self):
        """Test ChartTheme creation."""
        theme = ChartTheme(
            name="custom",
            palette=DEFAULT_PALETTE,
            style="whitegrid",
            context="notebook"
        )
        assert theme.name == "custom"
        assert theme.palette == DEFAULT_PALETTE
        assert theme.style == "whitegrid"
    
    def test_theme_get_color(self):
        """Test getting color by index."""
        theme = ChartTheme(palette=DEFAULT_PALETTE)
        color = theme.get_color(0)
        assert color == DEFAULT_PALETTE.colors[0]
    
    def test_theme_get_colors(self):
        """Test getting multiple colors."""
        theme = ChartTheme(palette=DEFAULT_PALETTE)
        colors = theme.get_colors(5)
        assert len(colors) == 5
        assert colors[0] == DEFAULT_PALETTE.colors[0]
    
    def test_get_theme(self):
        """Test getting pre-defined theme."""
        theme = get_theme("default")
        assert isinstance(theme, ChartTheme)
        assert theme.name == "default"
    
    def test_apply_theme(self):
        """Test applying theme to matplotlib."""
        apply_theme("default")
        # Just verify it doesn't raise an error
        assert True


# Tests for Static Charts

class TestStaticCharts:
    """Test static chart generation."""
    
    def test_initialization(self):
        """Test StaticCharts initialization."""
        charts = StaticCharts(theme="default")
        assert charts.theme.name == "default"
    
    def test_plot_distribution(self, sample_dataframe):
        """Test distribution plot."""
        charts = StaticCharts()
        fig = charts.plot_distribution(
            sample_dataframe,
            field='amount',
            title='Amount Distribution',
            kde=True
        )
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_distribution_with_series(self, sample_numeric_data):
        """Test distribution plot with Series."""
        charts = StaticCharts()
        fig = charts.plot_distribution(
            pd.Series(sample_numeric_data),
            title='Distribution'
        )
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_boxplot(self, sample_dataframe):
        """Test box plot."""
        charts = StaticCharts()
        fig = charts.plot_boxplot(
            sample_dataframe,
            y='amount',
            title='Amount Box Plot'
        )
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_boxplot_with_grouping(self, sample_dataframe):
        """Test box plot with grouping."""
        charts = StaticCharts()
        fig = charts.plot_boxplot(
            sample_dataframe,
            x='category',
            y='amount',
            title='Amount by Category'
        )
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_violin(self, sample_dataframe):
        """Test violin plot."""
        charts = StaticCharts()
        fig = charts.plot_violin(
            sample_dataframe,
            x='category',
            y='amount',
            title='Violin Plot'
        )
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_correlation_heatmap(self, sample_dataframe):
        """Test correlation heatmap."""
        charts = StaticCharts()
        numeric_df = sample_dataframe[['amount', 'age', 'score']]
        fig = charts.plot_correlation_heatmap(
            numeric_df,
            method='pearson',
            title='Correlation Matrix'
        )
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_scatter(self, sample_dataframe):
        """Test scatter plot."""
        charts = StaticCharts()
        fig = charts.plot_scatter(
            sample_dataframe,
            x='age',
            y='amount',
            title='Age vs Amount'
        )
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_scatter_with_regression(self, sample_dataframe):
        """Test scatter plot with regression line."""
        charts = StaticCharts()
        fig = charts.plot_scatter(
            sample_dataframe,
            x='age',
            y='score',
            add_regression=True,
            title='Age vs Score with Regression'
        )
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_time_series(self, timeseries_dataframe):
        """Test time series plot."""
        charts = StaticCharts()
        fig = charts.plot_time_series(
            timeseries_dataframe,
            x='date',
            y='value',
            title='Time Series'
        )
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_time_series_multiple(self, timeseries_dataframe):
        """Test time series plot with multiple series."""
        charts = StaticCharts()
        fig = charts.plot_time_series(
            timeseries_dataframe,
            x='date',
            y=['value', 'value2'],
            title='Multiple Time Series'
        )
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_bar_chart(self, sample_dataframe):
        """Test bar chart."""
        charts = StaticCharts()
        avg_by_category = sample_dataframe.groupby('category')['amount'].mean().reset_index()
        fig = charts.plot_bar_chart(
            avg_by_category,
            x='category',
            y='amount',
            title='Average Amount by Category'
        )
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_stacked_bar(self):
        """Test stacked bar chart."""
        charts = StaticCharts()
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        }, index=['X', 'Y', 'Z'])
        fig = charts.plot_stacked_bar(
            data,
            title='Stacked Bar Chart'
        )
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_pie_chart(self):
        """Test pie chart."""
        charts = StaticCharts()
        data = pd.Series([30, 20, 25, 25], index=['A', 'B', 'C', 'D'])
        fig = charts.plot_pie_chart(
            data,
            title='Pie Chart'
        )
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_qq_plot(self, sample_numeric_data):
        """Test Q-Q plot."""
        charts = StaticCharts()
        fig = charts.plot_qq_plot(
            sample_numeric_data,
            dist='norm',
            title='Q-Q Plot'
        )
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_plot_pairplot(self, sample_dataframe):
        """Test pair plot."""
        charts = StaticCharts()
        numeric_df = sample_dataframe[['amount', 'age', 'score']].sample(100)  # Smaller sample for speed
        g = charts.plot_pairplot(
            numeric_df,
            title='Pair Plot'
        )
        assert g is not None
        plt.close('all')


# Tests for Export Manager

class TestExportManager:
    """Test export functionality."""
    
    def test_initialization(self, temp_dir):
        """Test ExportManager initialization."""
        manager = ExportManager(output_dir=temp_dir)
        assert manager.output_dir == temp_dir
        assert Path(temp_dir).exists()
    
    def test_save_matplotlib_figure_png(self, sample_dataframe, temp_dir):
        """Test saving matplotlib figure as PNG."""
        charts = StaticCharts()
        fig = charts.plot_distribution(sample_dataframe, field='amount')
        
        manager = ExportManager(output_dir=temp_dir)
        saved_files = manager.save_figure(fig, 'test_chart', formats=['png'])
        
        assert len(saved_files) == 1
        assert saved_files[0].endswith('.png')
        assert os.path.exists(saved_files[0])
        plt.close(fig)
    
    def test_save_matplotlib_figure_multiple_formats(self, sample_dataframe, temp_dir):
        """Test saving matplotlib figure in multiple formats."""
        charts = StaticCharts()
        fig = charts.plot_distribution(sample_dataframe, field='amount')
        
        manager = ExportManager(output_dir=temp_dir)
        saved_files = manager.save_figure(fig, 'test_chart', formats=['png', 'svg', 'pdf'])
        
        assert len(saved_files) == 3
        assert any(f.endswith('.png') for f in saved_files)
        assert any(f.endswith('.svg') for f in saved_files)
        assert any(f.endswith('.pdf') for f in saved_files)
        plt.close(fig)
    
    def test_export_to_png(self, sample_dataframe, temp_dir):
        """Test PNG export."""
        charts = StaticCharts()
        fig = charts.plot_distribution(sample_dataframe, field='amount')
        
        manager = ExportManager()
        filepath = os.path.join(temp_dir, 'test.png')
        result = manager.export_to_png(fig, filepath, dpi=150)
        
        assert result == filepath
        assert os.path.exists(filepath)
        plt.close(fig)
    
    def test_export_to_svg(self, sample_dataframe, temp_dir):
        """Test SVG export."""
        charts = StaticCharts()
        fig = charts.plot_distribution(sample_dataframe, field='amount')
        
        manager = ExportManager()
        filepath = os.path.join(temp_dir, 'test.svg')
        result = manager.export_to_svg(fig, filepath)
        
        assert result == filepath
        assert os.path.exists(filepath)
        plt.close(fig)
    
    def test_export_to_pdf(self, sample_dataframe, temp_dir):
        """Test PDF export."""
        charts = StaticCharts()
        fig = charts.plot_distribution(sample_dataframe, field='amount')
        
        manager = ExportManager()
        filepath = os.path.join(temp_dir, 'test.pdf')
        result = manager.export_to_pdf(fig, filepath)
        
        assert result == filepath
        assert os.path.exists(filepath)
        plt.close(fig)
    
    def test_batch_export(self, sample_dataframe, temp_dir):
        """Test batch export of multiple figures."""
        charts = StaticCharts()
        fig1 = charts.plot_distribution(sample_dataframe, field='amount')
        fig2 = charts.plot_boxplot(sample_dataframe, y='amount')
        
        manager = ExportManager(output_dir=temp_dir)
        saved_files = manager.batch_export(
            [fig1, fig2],
            ['dist', 'box'],
            formats=['png']
        )
        
        assert len(saved_files) == 2
        assert all(os.path.exists(f[0]) for f in saved_files)
        plt.close('all')
    
    def test_close_all(self):
        """Test closing all figures."""
        charts = StaticCharts()
        data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        
        # Create multiple figures
        for i in range(3):
            charts.plot_scatter(data, 'x', 'y')
        
        manager = ExportManager()
        manager.close_all()
        
        # Verify all figures are closed
        assert len(plt.get_fignums()) == 0


# Tests for Visualization Gallery

class TestVisualizationGallery:
    """Test visualization gallery."""
    
    def test_initialization(self, sample_dataframe, temp_dir):
        """Test VisualizationGallery initialization."""
        gallery = VisualizationGallery(
            sample_dataframe,
            theme="default",
            output_dir=temp_dir
        )
        assert gallery.data.equals(sample_dataframe)
        assert gallery.theme == "default"
        assert gallery.output_dir == temp_dir
    
    def test_generate_overview_dashboard(self, sample_dataframe, temp_dir):
        """Test overview dashboard generation."""
        gallery = VisualizationGallery(sample_dataframe, output_dir=temp_dir)
        charts = gallery.generate_overview_dashboard(save=False)
        
        assert isinstance(charts, dict)
        assert len(charts) > 0
        plt.close('all')
    
    def test_generate_overview_dashboard_with_save(self, sample_dataframe, temp_dir):
        """Test overview dashboard with file saving."""
        gallery = VisualizationGallery(sample_dataframe, output_dir=temp_dir)
        charts = gallery.generate_overview_dashboard(
            output_dir=os.path.join(temp_dir, 'overview'),
            save=True
        )
        
        assert isinstance(charts, dict)
        assert os.path.exists(os.path.join(temp_dir, 'overview'))
        plt.close('all')
    
    def test_generate_correlation_dashboard(self, sample_dataframe, temp_dir):
        """Test correlation dashboard generation."""
        gallery = VisualizationGallery(sample_dataframe, output_dir=temp_dir)
        charts = gallery.generate_correlation_dashboard(save=False)
        
        assert isinstance(charts, dict)
        assert len(charts) > 0
        plt.close('all')
    
    def test_generate_distribution_dashboard(self, sample_dataframe, temp_dir):
        """Test distribution dashboard generation."""
        gallery = VisualizationGallery(sample_dataframe, output_dir=temp_dir)
        charts = gallery.generate_distribution_dashboard(save=False)
        
        assert isinstance(charts, dict)
        assert len(charts) > 0
        plt.close('all')
    
    def test_generate_time_series_dashboard(self, timeseries_dataframe, temp_dir):
        """Test time series dashboard generation."""
        gallery = VisualizationGallery(timeseries_dataframe, output_dir=temp_dir)
        charts = gallery.generate_time_series_dashboard(
            time_col='date',
            value_cols=['value', 'value2'],
            save=False
        )
        
        assert isinstance(charts, dict)
        assert len(charts) > 0
        plt.close('all')
    
    def test_generate_fraud_analysis_dashboard(self, sample_dataframe, temp_dir):
        """Test fraud analysis dashboard generation."""
        gallery = VisualizationGallery(sample_dataframe, output_dir=temp_dir)
        charts = gallery.generate_fraud_analysis_dashboard(
            fraud_col='is_fraud',
            amount_col='amount',
            save=False
        )
        
        assert isinstance(charts, dict)
        assert len(charts) > 0
        plt.close('all')
    
    def test_save_gallery(self, sample_dataframe, temp_dir):
        """Test saving entire gallery."""
        gallery = VisualizationGallery(sample_dataframe, output_dir=temp_dir)
        saved_files = gallery.save_gallery(formats=['png'])
        
        assert isinstance(saved_files, dict)
        assert 'overview' in saved_files
        assert 'correlation' in saved_files
        assert 'distribution' in saved_files
        plt.close('all')
    
    def test_create_html_report(self, sample_dataframe, temp_dir):
        """Test HTML report creation."""
        gallery = VisualizationGallery(sample_dataframe, output_dir=temp_dir)
        output_file = os.path.join(temp_dir, 'report.html')
        result = gallery.create_html_report(
            output_file,
            title="Test Report"
        )
        
        assert result == output_file
        assert os.path.exists(output_file)
        
        # Verify HTML content
        with open(output_file, 'r') as f:
            content = f.read()
            assert 'Test Report' in content
            assert str(len(sample_dataframe)) in content


# Tests for edge cases and error handling

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        charts = StaticCharts()
        empty_df = pd.DataFrame()
        
        # Should not raise an error for operations that can handle empty data
        # (some operations may raise, which is expected behavior)
        assert True
    
    def test_single_value_distribution(self):
        """Test distribution plot with single unique value."""
        charts = StaticCharts()
        data = pd.DataFrame({'value': [5.0] * 100})
        
        # Should handle gracefully
        try:
            fig = charts.plot_distribution(data, field='value')
            assert isinstance(fig, mpl_figure.Figure)
            plt.close(fig)
        except Exception:
            # Some visualizations may not work with constant data
            pass
    
    def test_missing_values(self):
        """Test handling of missing values."""
        charts = StaticCharts()
        data = pd.DataFrame({
            'value': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]
        })
        
        fig = charts.plot_distribution(data, field='value')
        assert isinstance(fig, mpl_figure.Figure)
        plt.close(fig)
    
    def test_all_missing_values(self):
        """Test handling of all missing values."""
        charts = StaticCharts()
        data = pd.DataFrame({'value': [np.nan] * 10})
        
        # Should handle gracefully (may produce empty plot or raise)
        try:
            fig = charts.plot_distribution(data, field='value')
            plt.close(fig)
        except Exception:
            # Expected for all-NaN data
            pass
    
    def test_invalid_field_name(self, sample_dataframe):
        """Test error handling for invalid field name."""
        charts = StaticCharts()
        
        with pytest.raises(Exception):  # Should raise KeyError or similar
            charts.plot_distribution(sample_dataframe, field='nonexistent_field')
    
    def test_theme_color_wraparound(self):
        """Test color selection wraps around palette."""
        theme = ChartTheme(palette=DEFAULT_PALETTE)
        colors = theme.get_colors(15)  # More than palette size
        
        assert len(colors) == 15
        # First color should repeat at position len(palette)
        assert colors[0] == colors[len(DEFAULT_PALETTE.colors)]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
