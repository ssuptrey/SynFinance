"""
SynFinance Visualization Suite

This package provides comprehensive visualization capabilities for financial data analysis,
including static charts, interactive visualizations, geographic maps, and statistical plots.

Week 10 Day 2: Visualization Suite
"""

from typing import TYPE_CHECKING

# Lazy imports for performance
if TYPE_CHECKING:
    from .static_charts import StaticCharts
    from .interactive_charts import InteractiveCharts
    from .geographic_maps import GeographicMaps
    from .statistical_plots import StatisticalPlots
    from .gallery import VisualizationGallery
    from .export import ExportManager
    from .themes import ChartTheme, ColorPalette

__all__ = [
    "StaticCharts",
    "InteractiveCharts",
    "GeographicMaps",
    "StatisticalPlots",
    "VisualizationGallery",
    "ExportManager",
    "ChartTheme",
    "ColorPalette",
]

__version__ = "2.17.0"


def __getattr__(name: str):
    """Lazy load visualization modules."""
    if name == "StaticCharts":
        from .static_charts import StaticCharts
        return StaticCharts
    elif name == "InteractiveCharts":
        from .interactive_charts import InteractiveCharts
        return InteractiveCharts
    elif name == "GeographicMaps":
        from .geographic_maps import GeographicMaps
        return GeographicMaps
    elif name == "StatisticalPlots":
        from .statistical_plots import StatisticalPlots
        return StatisticalPlots
    elif name == "VisualizationGallery":
        from .gallery import VisualizationGallery
        return VisualizationGallery
    elif name == "ExportManager":
        from .export import ExportManager
        return ExportManager
    elif name == "ChartTheme":
        from .themes import ChartTheme
        return ChartTheme
    elif name == "ColorPalette":
        from .themes import ColorPalette
        return ColorPalette
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
