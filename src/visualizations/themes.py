"""
Color themes and palettes for visualizations.

Provides professional color schemes, themes, and styling configurations
for consistent and accessible visualizations.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class ColorPalette:
    """Color palette definition."""
    
    name: str
    colors: List[str]
    description: str = ""
    
    def as_matplotlib_cmap(self):
        """Convert to matplotlib colormap."""
        from matplotlib.colors import ListedColormap
        return ListedColormap(self.colors, name=self.name)
    
    def as_seaborn_palette(self):
        """Convert to seaborn palette."""
        return sns.color_palette(self.colors)


# Default color palette
DEFAULT_PALETTE = ColorPalette(
    name="default",
    colors=[
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf",  # Cyan
    ],
    description="Default SynFinance color palette"
)


# Colorblind-friendly palette (based on ColorBrewer)
COLORBLIND_PALETTE = ColorPalette(
    name="colorblind",
    colors=[
        "#0173b2",  # Blue
        "#de8f05",  # Orange
        "#029e73",  # Green
        "#cc78bc",  # Pink
        "#ca9161",  # Tan
        "#949494",  # Gray
        "#ece133",  # Yellow
        "#56b4e9",  # Sky blue
    ],
    description="Colorblind-friendly palette"
)


# Dark theme palette
DARK_PALETTE = ColorPalette(
    name="dark",
    colors=[
        "#4a9eff",  # Bright blue
        "#ffb84d",  # Bright orange
        "#4dff88",  # Bright green
        "#ff6b6b",  # Bright red
        "#b695ff",  # Bright purple
        "#ffd93d",  # Bright yellow
        "#ff85c1",  # Bright pink
        "#6bcfff",  # Bright cyan
    ],
    description="High-contrast palette for dark backgrounds"
)


# Sequential palettes for heatmaps
SEQUENTIAL_PALETTES = {
    "blues": ColorPalette(
        name="blues",
        colors=["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"],
        description="Sequential blue palette"
    ),
    "greens": ColorPalette(
        name="greens",
        colors=["#f7fcf5", "#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", "#238b45", "#006d2c", "#00441b"],
        description="Sequential green palette"
    ),
    "reds": ColorPalette(
        name="reds",
        colors=["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#a50f15", "#67000d"],
        description="Sequential red palette"
    ),
}


# Diverging palettes for correlations
DIVERGING_PALETTES = {
    "rdbu": ColorPalette(
        name="rdbu",
        colors=["#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061"],
        description="Red-Blue diverging palette"
    ),
    "rdylgn": ColorPalette(
        name="rdylgn",
        colors=["#a50026", "#d73027", "#f46d43", "#fdae61", "#fee08b", "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850", "#006837"],
        description="Red-Yellow-Green diverging palette"
    ),
}


class ChartTheme:
    """Chart theme configuration."""
    
    def __init__(
        self,
        name: str = "default",
        palette: Optional[ColorPalette] = None,
        style: str = "whitegrid",
        context: str = "notebook",
        font_scale: float = 1.0,
        font_family: str = "sans-serif"
    ):
        """
        Initialize chart theme.
        
        Args:
            name: Theme name
            palette: Color palette to use
            style: Seaborn style (whitegrid, darkgrid, white, dark, ticks)
            context: Seaborn context (paper, notebook, talk, poster)
            font_scale: Font size scaling factor
            font_family: Font family to use
        """
        self.name = name
        self.palette = palette or DEFAULT_PALETTE
        self.style = style
        self.context = context
        self.font_scale = font_scale
        self.font_family = font_family
        
    def apply(self) -> None:
        """Apply the theme to matplotlib and seaborn."""
        # Apply seaborn style
        sns.set_style(self.style)
        sns.set_context(self.context, font_scale=self.font_scale)
        sns.set_palette(self.palette.as_seaborn_palette())
        
        # Apply matplotlib settings
        plt.rcParams['font.family'] = self.font_family
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        
    def get_color(self, index: int = 0) -> str:
        """Get color by index from palette."""
        return self.palette.colors[index % len(self.palette.colors)]
    
    def get_colors(self, n: int) -> List[str]:
        """Get n colors from palette."""
        if n <= len(self.palette.colors):
            return self.palette.colors[:n]
        # Repeat colors if more needed
        return [self.palette.colors[i % len(self.palette.colors)] for i in range(n)]


# Pre-defined themes
THEMES = {
    "default": ChartTheme(
        name="default",
        palette=DEFAULT_PALETTE,
        style="whitegrid",
        context="notebook"
    ),
    "dark": ChartTheme(
        name="dark",
        palette=DARK_PALETTE,
        style="darkgrid",
        context="notebook"
    ),
    "colorblind": ChartTheme(
        name="colorblind",
        palette=COLORBLIND_PALETTE,
        style="whitegrid",
        context="notebook"
    ),
    "minimal": ChartTheme(
        name="minimal",
        palette=DEFAULT_PALETTE,
        style="white",
        context="paper",
        font_scale=0.9
    ),
    "presentation": ChartTheme(
        name="presentation",
        palette=DEFAULT_PALETTE,
        style="whitegrid",
        context="talk",
        font_scale=1.2
    ),
}


def get_theme(name: str = "default") -> ChartTheme:
    """
    Get a pre-defined theme.
    
    Args:
        name: Theme name
        
    Returns:
        ChartTheme instance
    """
    return THEMES.get(name, THEMES["default"])


def apply_theme(name: str = "default") -> None:
    """
    Apply a pre-defined theme.
    
    Args:
        name: Theme name
    """
    theme = get_theme(name)
    theme.apply()
