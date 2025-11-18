"""
Tests for interactive visualizations (Plotly and Folium).

These tests require plotly and folium to be installed.
Tests are marked with pytest.mark.skipif to skip when dependencies are missing.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os

# Try to import optional dependencies
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

if PLOTLY_AVAILABLE:
    from src.visualizations.interactive_charts import InteractiveCharts

if FOLIUM_AVAILABLE:
    from src.visualizations.geographic_maps import GeographicMaps


# Fixtures

@pytest.fixture
def sample_dataframe():
    """Generate sample DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        'amount': np.random.lognormal(5, 1, 500),
        'age': np.random.randint(18, 80, 500),
        'score': np.random.uniform(0, 100, 500),
        'category': np.random.choice(['A', 'B', 'C'], 500),
        'is_fraud': np.random.choice([0, 1], 500, p=[0.9, 0.1]),
        'month': np.random.choice(['Jan', 'Feb', 'Mar'], 500)
    })


@pytest.fixture
def timeseries_dataframe():
    """Generate time series DataFrame."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    return pd.DataFrame({
        'date': dates,
        'value': np.random.randn(100).cumsum() + 100,
        'category': np.random.choice(['A', 'B'], 100)
    })


@pytest.fixture
def hierarchical_dataframe():
    """Generate hierarchical data for sunburst/treemap."""
    return pd.DataFrame({
        'region': ['North', 'North', 'South', 'South', 'East', 'East'],
        'city': ['NY', 'Boston', 'Miami', 'Atlanta', 'Philadelphia', 'Baltimore'],
        'sales': [100, 80, 120, 90, 70, 60]
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
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# Tests for Interactive Charts (Plotly)

@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
class TestInteractiveCharts:
    """Test interactive chart generation with Plotly."""
    
    def test_initialization(self):
        """Test InteractiveCharts initialization."""
        charts = InteractiveCharts(template="plotly_white")
        assert charts.template == "plotly_white"
    
    def test_create_interactive_histogram(self, sample_dataframe):
        """Test interactive histogram creation."""
        charts = InteractiveCharts()
        fig = charts.create_interactive_histogram(
            sample_dataframe,
            field='amount',
            title='Amount Distribution'
        )
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == 'Amount Distribution'
    
    def test_create_interactive_histogram_with_stats(self, sample_dataframe):
        """Test interactive histogram with mean and median lines."""
        charts = InteractiveCharts()
        fig = charts.create_interactive_histogram(
            sample_dataframe,
            field='amount',
            show_mean=True,
            show_median=True
        )
        assert isinstance(fig, go.Figure)
    
    def test_create_scatter_matrix(self, sample_dataframe):
        """Test scatter matrix creation."""
        charts = InteractiveCharts()
        fig = charts.create_scatter_matrix(
            sample_dataframe,
            dimensions=['amount', 'age', 'score'],
            color='category',
            title='Scatter Matrix'
        )
        assert isinstance(fig, go.Figure)
    
    def test_create_parallel_coordinates(self, sample_dataframe):
        """Test parallel coordinates plot."""
        charts = InteractiveCharts()
        fig = charts.create_parallel_coordinates(
            sample_dataframe.head(100),  # Smaller sample
            dimensions=['amount', 'age', 'score'],
            color='is_fraud',
            title='Parallel Coordinates'
        )
        assert isinstance(fig, go.Figure)
    
    def test_create_3d_scatter(self, sample_dataframe):
        """Test 3D scatter plot."""
        charts = InteractiveCharts()
        fig = charts.create_3d_scatter(
            sample_dataframe,
            x='amount',
            y='age',
            z='score',
            color='category',
            title='3D Scatter'
        )
        assert isinstance(fig, go.Figure)
    
    def test_create_animated_time_series(self, timeseries_dataframe):
        """Test animated time series."""
        charts = InteractiveCharts()
        # Add month column for animation
        df = timeseries_dataframe.copy()
        df['month'] = df['date'].dt.month
        
        fig = charts.create_animated_time_series(
            df,
            x='date',
            y='value',
            animation_frame='month',
            title='Animated Time Series'
        )
        assert isinstance(fig, go.Figure)
    
    def test_create_sunburst(self, hierarchical_dataframe):
        """Test sunburst chart."""
        charts = InteractiveCharts()
        fig = charts.create_sunburst(
            hierarchical_dataframe,
            path=['region', 'city'],
            values='sales',
            title='Sunburst Chart'
        )
        assert isinstance(fig, go.Figure)
    
    def test_create_treemap(self, hierarchical_dataframe):
        """Test treemap chart."""
        charts = InteractiveCharts()
        fig = charts.create_treemap(
            hierarchical_dataframe,
            path=['region', 'city'],
            values='sales',
            title='Treemap'
        )
        assert isinstance(fig, go.Figure)
    
    def test_create_sankey(self):
        """Test Sankey diagram."""
        charts = InteractiveCharts()
        fig = charts.create_sankey(
            source=[0, 0, 1, 1, 2],
            target=[2, 3, 3, 4, 4],
            value=[10, 20, 15, 25, 30],
            labels=['A', 'B', 'C', 'D', 'E'],
            title='Sankey Diagram'
        )
        assert isinstance(fig, go.Figure)
    
    def test_create_waterfall(self):
        """Test waterfall chart."""
        charts = InteractiveCharts()
        data = pd.DataFrame({
            'category': ['Start', 'Revenue', 'Costs', 'Profit'],
            'value': [100, 50, -30, 20]
        })
        fig = charts.create_waterfall(
            data,
            x='category',
            y='value',
            title='Waterfall Chart'
        )
        assert isinstance(fig, go.Figure)
    
    def test_create_funnel(self):
        """Test funnel chart."""
        charts = InteractiveCharts()
        data = pd.DataFrame({
            'stage': ['Visit', 'Add to Cart', 'Checkout', 'Purchase'],
            'count': [1000, 500, 200, 100]
        })
        fig = charts.create_funnel(
            data,
            x='count',
            y='stage',
            title='Funnel Chart'
        )
        assert isinstance(fig, go.Figure)
    
    def test_create_box_plot(self, sample_dataframe):
        """Test interactive box plot."""
        charts = InteractiveCharts()
        fig = charts.create_box_plot(
            sample_dataframe,
            y='amount',
            x='category',
            title='Box Plot'
        )
        assert isinstance(fig, go.Figure)
    
    def test_create_heatmap(self):
        """Test interactive heatmap."""
        charts = InteractiveCharts()
        data = pd.DataFrame(
            np.random.randn(10, 10),
            columns=[f'Col{i}' for i in range(10)],
            index=[f'Row{i}' for i in range(10)]
        )
        fig = charts.create_heatmap(
            data,
            title='Heatmap'
        )
        assert isinstance(fig, go.Figure)
    
    def test_create_multi_line(self, timeseries_dataframe):
        """Test multi-line chart."""
        charts = InteractiveCharts()
        # Create second value column
        df = timeseries_dataframe.copy()
        df['value2'] = df['value'] * 1.2
        
        fig = charts.create_multi_line(
            df,
            x='date',
            y_columns=['value', 'value2'],
            title='Multi-line Chart'
        )
        assert isinstance(fig, go.Figure)
    
    def test_save_html(self, sample_dataframe, temp_dir):
        """Test saving figure to HTML."""
        charts = InteractiveCharts()
        fig = charts.create_interactive_histogram(
            sample_dataframe,
            field='amount'
        )
        
        filepath = os.path.join(temp_dir, 'test.html')
        charts.save_html(fig, filepath)
        
        assert os.path.exists(filepath)
    
    def test_save_image(self, sample_dataframe, temp_dir):
        """Test saving figure to static image (requires kaleido)."""
        charts = InteractiveCharts()
        fig = charts.create_interactive_histogram(
            sample_dataframe,
            field='amount'
        )
        
        filepath = os.path.join(temp_dir, 'test.png')
        
        try:
            charts.save_image(fig, filepath, width=800, height=600)
            assert os.path.exists(filepath)
        except Exception:
            # Kaleido may not be installed, skip
            pytest.skip("Kaleido not available for image export")


# Tests for Geographic Maps (Folium)

@pytest.mark.skipif(not FOLIUM_AVAILABLE, reason="Folium not installed")
class TestGeographicMaps:
    """Test geographic visualization with Folium."""
    
    def test_initialization(self):
        """Test GeographicMaps initialization."""
        maps = GeographicMaps(tiles='OpenStreetMap', zoom_start=10)
        assert maps.tiles == 'OpenStreetMap'
        assert maps.zoom_start == 10
    
    def test_create_point_map(self, geographic_dataframe):
        """Test point map creation."""
        maps = GeographicMaps()
        m = maps.create_point_map(
            geographic_dataframe,
            lat_col='latitude',
            lon_col='longitude',
            popup_fields=['amount', 'merchant']
        )
        assert isinstance(m, folium.Map)
    
    def test_create_point_map_with_custom_center(self, geographic_dataframe):
        """Test point map with custom center."""
        maps = GeographicMaps()
        m = maps.create_point_map(
            geographic_dataframe.head(10),
            lat_col='latitude',
            lon_col='longitude',
            center=(39.0, -120.0)
        )
        assert isinstance(m, folium.Map)
    
    def test_create_heatmap(self, geographic_dataframe):
        """Test heatmap creation."""
        maps = GeographicMaps()
        m = maps.create_heatmap(
            geographic_dataframe,
            lat_col='latitude',
            lon_col='longitude',
            intensity_col='amount'
        )
        assert isinstance(m, folium.Map)
    
    def test_create_heatmap_without_intensity(self, geographic_dataframe):
        """Test heatmap without intensity values."""
        maps = GeographicMaps()
        m = maps.create_heatmap(
            geographic_dataframe,
            lat_col='latitude',
            lon_col='longitude'
        )
        assert isinstance(m, folium.Map)
    
    def test_create_cluster_map(self, geographic_dataframe):
        """Test cluster map creation."""
        maps = GeographicMaps()
        m = maps.create_cluster_map(
            geographic_dataframe,
            lat_col='latitude',
            lon_col='longitude',
            popup_fields=['amount', 'merchant'],
            tooltip_fields=['is_fraud']
        )
        assert isinstance(m, folium.Map)
    
    def test_create_route_map(self, geographic_dataframe):
        """Test route map creation."""
        maps = GeographicMaps()
        route_data = geographic_dataframe.head(10).copy()
        route_data['order'] = range(10)
        
        m = maps.create_route_map(
            route_data,
            lat_col='latitude',
            lon_col='longitude',
            order_col='order'
        )
        assert isinstance(m, folium.Map)
    
    def test_add_custom_markers(self, geographic_dataframe):
        """Test adding custom markers to map."""
        maps = GeographicMaps()
        m = folium.Map(location=[39.0, -120.0], zoom_start=10)
        
        locations = [(39.0, -120.0), (39.5, -119.5)]
        popups = ['Location 1', 'Location 2']
        colors = ['red', 'blue']
        
        m = maps.add_custom_markers(m, locations, popups=popups, colors=colors)
        assert isinstance(m, folium.Map)
    
    def test_add_circle_markers(self, geographic_dataframe):
        """Test adding circle markers."""
        maps = GeographicMaps()
        m = folium.Map(location=[39.0, -120.0], zoom_start=10)
        
        m = maps.add_circle_markers(
            m,
            geographic_dataframe.head(20),
            lat_col='latitude',
            lon_col='longitude',
            radius_col='amount',
            popup_fields=['merchant']
        )
        assert isinstance(m, folium.Map)
    
    def test_save_map(self, geographic_dataframe, temp_dir):
        """Test saving map to HTML."""
        maps = GeographicMaps()
        m = maps.create_point_map(
            geographic_dataframe.head(10),
            lat_col='latitude',
            lon_col='longitude'
        )
        
        filepath = os.path.join(temp_dir, 'map.html')
        maps.save(m, filepath)
        
        assert os.path.exists(filepath)
    
    def test_handle_missing_coordinates(self):
        """Test handling of missing coordinate values."""
        maps = GeographicMaps()
        data = pd.DataFrame({
            'latitude': [37.0, np.nan, 38.0],
            'longitude': [-120.0, -119.0, np.nan],
            'value': [100, 200, 300]
        })
        
        m = maps.create_point_map(
            data,
            lat_col='latitude',
            lon_col='longitude'
        )
        assert isinstance(m, folium.Map)


# Integration tests

@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
class TestInteractiveIntegration:
    """Integration tests for interactive visualizations."""
    
    def test_multiple_chart_creation(self, sample_dataframe):
        """Test creating multiple charts in sequence."""
        charts = InteractiveCharts()
        
        fig1 = charts.create_interactive_histogram(sample_dataframe, field='amount')
        fig2 = charts.create_box_plot(sample_dataframe, y='amount', x='category')
        fig3 = charts.create_3d_scatter(sample_dataframe, x='amount', y='age', z='score')
        
        assert isinstance(fig1, go.Figure)
        assert isinstance(fig2, go.Figure)
        assert isinstance(fig3, go.Figure)
    
    def test_chart_customization(self, sample_dataframe):
        """Test chart template customization."""
        charts_light = InteractiveCharts(template="plotly_white")
        charts_dark = InteractiveCharts(template="plotly_dark")
        
        fig_light = charts_light.create_interactive_histogram(sample_dataframe, field='amount')
        fig_dark = charts_dark.create_interactive_histogram(sample_dataframe, field='amount')
        
        assert fig_light.layout.template.layout.plot_bgcolor != fig_dark.layout.template.layout.plot_bgcolor


@pytest.mark.skipif(not FOLIUM_AVAILABLE, reason="Folium not installed")
class TestGeographicIntegration:
    """Integration tests for geographic visualizations."""
    
    def test_multiple_layer_map(self, geographic_dataframe):
        """Test creating map with multiple layers."""
        maps = GeographicMaps()
        
        # Create base map with cluster
        m = maps.create_cluster_map(
            geographic_dataframe,
            lat_col='latitude',
            lon_col='longitude'
        )
        
        # Add circle markers on top
        m = maps.add_circle_markers(
            m,
            geographic_dataframe.head(5),
            lat_col='latitude',
            lon_col='longitude',
            default_color='red'
        )
        
        assert isinstance(m, folium.Map)
    
    def test_different_tile_providers(self, geographic_dataframe):
        """Test using different tile providers."""
        for tiles in ['OpenStreetMap', 'CartoDB positron', 'CartoDB dark_matter']:
            maps = GeographicMaps(tiles=tiles)
            m = maps.create_point_map(
                geographic_dataframe.head(5),
                lat_col='latitude',
                lon_col='longitude'
            )
            assert isinstance(m, folium.Map)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
