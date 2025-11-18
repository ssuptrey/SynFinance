"""
Interactive chart generation using Plotly.

Provides interactive visualizations with hover, zoom, pan, and other
dynamic features for exploratory data analysis.
"""

from typing import Optional, List, Union, Dict, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class InteractiveCharts:
    """Interactive chart generator using Plotly."""
    
    def __init__(self, template: str = "plotly_white"):
        """
        Initialize interactive charts generator.
        
        Args:
            template: Plotly template name (plotly, plotly_white, plotly_dark, etc.)
        """
        self.template = template
    
    def create_interactive_histogram(
        self,
        data: Union[pd.DataFrame, pd.Series, np.ndarray],
        field: Optional[str] = None,
        nbins: Optional[int] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: str = "Count",
        color: Optional[str] = None,
        show_mean: bool = True,
        show_median: bool = True
    ) -> go.Figure:
        """
        Create interactive histogram with statistics overlay.
        
        Args:
            data: Data to plot
            field: Field name if data is DataFrame
            nbins: Number of bins
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            color: Bar color
            show_mean: Whether to show mean line
            show_median: Whether to show median line
            
        Returns:
            Plotly Figure object
        """
        # Extract data
        if isinstance(data, pd.DataFrame):
            if field is None:
                raise ValueError("field required when data is DataFrame")
            values = data[field].dropna()
            if xlabel is None:
                xlabel = field
        else:
            values = pd.Series(data).dropna()
        
        # Create histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=values,
            nbinsx=nbins,
            name='Distribution',
            marker_color=color or '#1f77b4',
            hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
        ))
        
        # Add mean line
        if show_mean:
            mean_val = values.mean()
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_val:.2f}",
                annotation_position="top right"
            )
        
        # Add median line
        if show_median:
            median_val = values.median()
            fig.add_vline(
                x=median_val,
                line_dash="dot",
                line_color="green",
                annotation_text=f"Median: {median_val:.2f}",
                annotation_position="bottom right"
            )
        
        # Update layout
        fig.update_layout(
            title=title or "Distribution",
            xaxis_title=xlabel or "Value",
            yaxis_title=ylabel,
            template=self.template,
            hovermode='closest'
        )
        
        return fig
    
    def create_scatter_matrix(
        self,
        data: pd.DataFrame,
        dimensions: List[str],
        color: Optional[str] = None,
        title: Optional[str] = None,
        size: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive scatter plot matrix.
        
        Args:
            data: DataFrame containing data
            dimensions: Columns to include
            color: Column for color mapping
            title: Plot title
            size: Column for marker size
            
        Returns:
            Plotly Figure object
        """
        fig = px.scatter_matrix(
            data,
            dimensions=dimensions,
            color=color,
            size=size,
            title=title or "Scatter Matrix",
            template=self.template
        )
        
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
        
        return fig
    
    def create_parallel_coordinates(
        self,
        data: pd.DataFrame,
        dimensions: List[str],
        color: Optional[str] = None,
        title: Optional[str] = None,
        color_continuous_scale: str = 'Viridis'
    ) -> go.Figure:
        """
        Create parallel coordinates plot for high-dimensional data.
        
        Args:
            data: DataFrame containing data
            dimensions: Columns to include as axes
            color: Column for color mapping
            title: Plot title
            color_continuous_scale: Colorscale name
            
        Returns:
            Plotly Figure object
        """
        fig = px.parallel_coordinates(
            data,
            dimensions=dimensions,
            color=color,
            title=title or "Parallel Coordinates",
            color_continuous_scale=color_continuous_scale,
            template=self.template
        )
        
        return fig
    
    def create_3d_scatter(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        z: str,
        color: Optional[str] = None,
        size: Optional[str] = None,
        title: Optional[str] = None,
        hover_data: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create 3D scatter plot.
        
        Args:
            data: DataFrame containing data
            x: Column for x-axis
            y: Column for y-axis
            z: Column for z-axis
            color: Column for color mapping
            size: Column for marker size
            title: Plot title
            hover_data: Additional columns for hover tooltip
            
        Returns:
            Plotly Figure object
        """
        fig = px.scatter_3d(
            data,
            x=x,
            y=y,
            z=z,
            color=color,
            size=size,
            title=title or "3D Scatter Plot",
            hover_data=hover_data,
            template=self.template
        )
        
        fig.update_traces(marker=dict(line=dict(width=0)))
        
        return fig
    
    def create_animated_time_series(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        animation_frame: str,
        color: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create animated time series visualization.
        
        Args:
            data: DataFrame containing data
            x: Column for x-axis (time)
            y: Column for y-axis (values)
            animation_frame: Column for animation frames
            color: Column for color grouping
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        fig = px.line(
            data,
            x=x,
            y=y,
            animation_frame=animation_frame,
            color=color,
            title=title or "Animated Time Series",
            template=self.template
        )
        
        fig.update_xaxes(rangeslider_visible=True)
        
        return fig
    
    def create_sunburst(
        self,
        data: pd.DataFrame,
        path: List[str],
        values: str,
        title: Optional[str] = None,
        color: Optional[str] = None
    ) -> go.Figure:
        """
        Create sunburst chart for hierarchical data.
        
        Args:
            data: DataFrame containing data
            path: Columns defining hierarchy (outer to inner)
            values: Column for slice sizes
            title: Plot title
            color: Column for color mapping
            
        Returns:
            Plotly Figure object
        """
        fig = px.sunburst(
            data,
            path=path,
            values=values,
            color=color,
            title=title or "Sunburst Chart",
            template=self.template
        )
        
        fig.update_traces(textinfo="label+percent parent")
        
        return fig
    
    def create_treemap(
        self,
        data: pd.DataFrame,
        path: List[str],
        values: str,
        title: Optional[str] = None,
        color: Optional[str] = None
    ) -> go.Figure:
        """
        Create treemap for hierarchical proportions.
        
        Args:
            data: DataFrame containing data
            path: Columns defining hierarchy
            values: Column for rectangle sizes
            title: Plot title
            color: Column for color mapping
            
        Returns:
            Plotly Figure object
        """
        fig = px.treemap(
            data,
            path=path,
            values=values,
            color=color,
            title=title or "Treemap",
            template=self.template
        )
        
        fig.update_traces(textinfo="label+value+percent parent")
        
        return fig
    
    def create_sankey(
        self,
        source: List[int],
        target: List[int],
        value: List[float],
        labels: List[str],
        title: Optional[str] = None,
        colors: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create Sankey diagram for flow visualization.
        
        Args:
            source: List of source node indices
            target: List of target node indices
            value: List of flow values
            labels: List of node labels
            title: Plot title
            colors: List of node colors
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=colors
            ),
            link=dict(
                source=source,
                target=target,
                value=value
            )
        )])
        
        fig.update_layout(
            title=title or "Sankey Diagram",
            template=self.template
        )
        
        return fig
    
    def create_waterfall(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        title: Optional[str] = None,
        measure: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create waterfall chart for cumulative effect visualization.
        
        Args:
            data: DataFrame containing data
            x: Column for x-axis (categories)
            y: Column for y-axis (values)
            title: Plot title
            measure: List of measure types ('relative', 'total', 'absolute')
            
        Returns:
            Plotly Figure object
        """
        if measure is None:
            measure = ['relative'] * len(data)
        
        fig = go.Figure(go.Waterfall(
            name="",
            orientation="v",
            measure=measure,
            x=data[x],
            y=data[y],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title=title or "Waterfall Chart",
            template=self.template,
            showlegend=False
        )
        
        return fig
    
    def create_funnel(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        title: Optional[str] = None,
        color: Optional[str] = None
    ) -> go.Figure:
        """
        Create funnel chart for conversion analysis.
        
        Args:
            data: DataFrame containing data
            x: Column for x-axis (values)
            y: Column for y-axis (stages)
            title: Plot title
            color: Column for color mapping
            
        Returns:
            Plotly Figure object
        """
        fig = px.funnel(
            data,
            x=x,
            y=y,
            color=color,
            title=title or "Funnel Chart",
            template=self.template
        )
        
        return fig
    
    def create_box_plot(
        self,
        data: pd.DataFrame,
        y: str,
        x: Optional[str] = None,
        color: Optional[str] = None,
        title: Optional[str] = None,
        points: str = 'outliers'
    ) -> go.Figure:
        """
        Create interactive box plot.
        
        Args:
            data: DataFrame containing data
            y: Column for y-axis (values)
            x: Column for x-axis (categories)
            color: Column for color grouping
            title: Plot title
            points: Point display mode ('outliers', 'all', False)
            
        Returns:
            Plotly Figure object
        """
        fig = px.box(
            data,
            x=x,
            y=y,
            color=color,
            title=title or "Box Plot",
            points=points,
            template=self.template
        )
        
        return fig
    
    def create_heatmap(
        self,
        data: pd.DataFrame,
        title: Optional[str] = None,
        colorscale: str = 'RdBu_r',
        zmin: Optional[float] = None,
        zmax: Optional[float] = None
    ) -> go.Figure:
        """
        Create interactive heatmap.
        
        Args:
            data: DataFrame with values to plot
            title: Plot title
            colorscale: Colorscale name
            zmin: Minimum value for colorscale
            zmax: Maximum value for colorscale
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            hoverongaps=False,
            hovertemplate='%{y} - %{x}<br>Value: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title or "Heatmap",
            template=self.template
        )
        
        return fig
    
    def create_multi_line(
        self,
        data: pd.DataFrame,
        x: str,
        y_columns: List[str],
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None
    ) -> go.Figure:
        """
        Create multi-line chart.
        
        Args:
            data: DataFrame containing data
            x: Column for x-axis
            y_columns: Columns for y-axis (multiple lines)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        for col in y_columns:
            fig.add_trace(go.Scatter(
                x=data[x],
                y=data[col],
                mode='lines',
                name=col,
                hovertemplate=f'{col}: %{{y:.2f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title or "Time Series",
            xaxis_title=xlabel or x,
            yaxis_title=ylabel or "Value",
            template=self.template,
            hovermode='x unified'
        )
        
        return fig
    
    def save_html(self, fig: go.Figure, filepath: str, auto_open: bool = False) -> None:
        """
        Save figure to HTML file.
        
        Args:
            fig: Plotly figure to save
            filepath: Output file path
            auto_open: Whether to open in browser
        """
        fig.write_html(filepath, auto_open=auto_open)
    
    def save_image(
        self,
        fig: go.Figure,
        filepath: str,
        width: int = 1200,
        height: int = 800,
        scale: float = 2.0
    ) -> None:
        """
        Save figure to static image (requires kaleido).
        
        Args:
            fig: Plotly figure to save
            filepath: Output file path
            width: Image width in pixels
            height: Image height in pixels
            scale: Image scale factor
        """
        fig.write_image(filepath, width=width, height=height, scale=scale)
