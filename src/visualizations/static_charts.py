"""
Static chart generation using matplotlib and seaborn.

Provides high-quality publication-ready static visualizations for
financial data analysis.
"""

from typing import Optional, List, Union, Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as mpl_figure
import seaborn as sns
from scipy import stats

from .themes import ChartTheme, get_theme


class StaticCharts:
    """Static chart generator using matplotlib and seaborn."""
    
    def __init__(self, theme: Union[str, ChartTheme] = "default"):
        """
        Initialize static charts generator.
        
        Args:
            theme: Theme name or ChartTheme instance
        """
        if isinstance(theme, str):
            self.theme = get_theme(theme)
        else:
            self.theme = theme
        self.theme.apply()
    
    def plot_distribution(
        self,
        data: Union[pd.DataFrame, pd.Series, np.ndarray],
        field: Optional[str] = None,
        bins: Union[int, str] = 'auto',
        kde: bool = True,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: str = "Frequency",
        figsize: Tuple[int, int] = (10, 6),
        color: Optional[str] = None
    ) -> mpl_figure.Figure:
        """
        Create histogram with optional KDE overlay.
        
        Args:
            data: Data to plot (DataFrame, Series, or array)
            field: Field name if data is DataFrame
            bins: Number of bins or binning strategy
            kde: Whether to overlay KDE
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size (width, height)
            color: Bar color (defaults to theme color)
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract data
        if isinstance(data, pd.DataFrame):
            if field is None:
                raise ValueError("field required when data is DataFrame")
            values = data[field].dropna()
            if xlabel is None:
                xlabel = field
        else:
            values = pd.Series(data).dropna()
        
        # Set color
        if color is None:
            color = self.theme.get_color(0)
        
        # Plot histogram
        ax.hist(values, bins=bins, alpha=0.7, color=color, edgecolor='black', density=kde)
        
        # Add KDE if requested
        if kde:
            kde_x = np.linspace(values.min(), values.max(), 200)
            kde_obj = stats.gaussian_kde(values)
            kde_y = kde_obj(kde_x)
            ax.plot(kde_x, kde_y, color='red', linewidth=2, label='KDE')
            ax.legend()
        
        # Labels and title
        ax.set_xlabel(xlabel or "Value")
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_boxplot(
        self,
        data: pd.DataFrame,
        y: Optional[str] = None,
        x: Optional[str] = None,
        hue: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        orient: str = 'v'
    ) -> mpl_figure.Figure:
        """
        Create box plot for outlier visualization.
        
        Args:
            data: DataFrame containing data
            y: Column for y-axis (values)
            x: Column for x-axis (categories)
            hue: Column for color grouping
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            orient: Orientation ('v' for vertical, 'h' for horizontal)
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.boxplot(data=data, x=x, y=y, hue=hue, orient=orient, ax=ax)
        
        ax.set_xlabel(xlabel or x or "")
        ax.set_ylabel(ylabel or y or "Value")
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_violin(
        self,
        data: pd.DataFrame,
        y: Optional[str] = None,
        x: Optional[str] = None,
        hue: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        split: bool = False
    ) -> mpl_figure.Figure:
        """
        Create violin plot for distribution comparison.
        
        Args:
            data: DataFrame containing data
            y: Column for y-axis (values)
            x: Column for x-axis (categories)
            hue: Column for color grouping
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            split: Whether to split violins by hue
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.violinplot(data=data, x=x, y=y, hue=hue, split=split, ax=ax)
        
        ax.set_xlabel(xlabel or x or "")
        ax.set_ylabel(ylabel or y or "Value")
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(
        self,
        data: pd.DataFrame,
        method: str = 'pearson',
        annot: bool = True,
        fmt: str = '.2f',
        cmap: str = 'RdBu_r',
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        vmin: float = -1.0,
        vmax: float = 1.0
    ) -> mpl_figure.Figure:
        """
        Create correlation matrix heatmap.
        
        Args:
            data: DataFrame with numeric columns
            method: Correlation method (pearson, spearman, kendall)
            annot: Whether to annotate cells with values
            fmt: Format string for annotations
            cmap: Colormap name
            title: Plot title
            figsize: Figure size
            vmin: Minimum value for colormap
            vmax: Maximum value for colormap
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Compute correlation matrix
        corr = data.corr(method=method)
        
        # Create heatmap
        sns.heatmap(
            corr,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'{method.capitalize()} Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_scatter(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        hue: Optional[str] = None,
        size: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        add_regression: bool = False,
        alpha: float = 0.6
    ) -> mpl_figure.Figure:
        """
        Create scatter plot with optional regression line.
        
        Args:
            data: DataFrame containing data
            x: Column for x-axis
            y: Column for y-axis
            hue: Column for color grouping
            size: Column for point size
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            add_regression: Whether to add regression line
            alpha: Point transparency
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create scatter plot
        sns.scatterplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            size=size,
            alpha=alpha,
            ax=ax
        )
        
        # Add regression line if requested
        if add_regression:
            # Remove NaN values
            valid_data = data[[x, y]].dropna()
            if len(valid_data) > 1:
                z = np.polyfit(valid_data[x], valid_data[y], 1)
                p = np.poly1d(z)
                x_line = np.linspace(valid_data[x].min(), valid_data[x].max(), 100)
                ax.plot(x_line, p(x_line), "r--", linewidth=2, label='Regression')
                ax.legend()
        
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_time_series(
        self,
        data: pd.DataFrame,
        x: str,
        y: Union[str, List[str]],
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
        add_trend: bool = False
    ) -> mpl_figure.Figure:
        """
        Create time series line chart.
        
        Args:
            data: DataFrame containing data
            x: Column for x-axis (time)
            y: Column(s) for y-axis (values)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            add_trend: Whether to add trend line
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot single or multiple series
        if isinstance(y, str):
            y_cols = [y]
        else:
            y_cols = y
        
        for i, col in enumerate(y_cols):
            color = self.theme.get_color(i)
            ax.plot(data[x], data[col], label=col, color=color, linewidth=2)
        
        # Add trend line if requested
        if add_trend and len(y_cols) == 1:
            # Convert time to numeric for trend
            x_numeric = np.arange(len(data))
            valid_mask = data[y_cols[0]].notna()
            if valid_mask.sum() > 1:
                z = np.polyfit(x_numeric[valid_mask], data[y_cols[0]][valid_mask], 1)
                p = np.poly1d(z)
                ax.plot(data[x], p(x_numeric), "r--", linewidth=2, alpha=0.7, label='Trend')
        
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or "Value")
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        if len(y_cols) > 1 or add_trend:
            ax.legend()
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def plot_bar_chart(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        hue: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        orient: str = 'v',
        ci: Optional[Union[int, str]] = None
    ) -> mpl_figure.Figure:
        """
        Create bar chart for categorical comparisons.
        
        Args:
            data: DataFrame containing data
            x: Column for x-axis
            y: Column for y-axis
            hue: Column for color grouping
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            orient: Orientation ('v' or 'h')
            ci: Confidence interval (None, int, or 'sd')
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use errorbar parameter instead of deprecated ci
        errorbar = None if ci is None else ('sd' if ci == 'sd' else ci)
        
        sns.barplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            orient=orient,
            errorbar=errorbar,
            ax=ax
        )
        
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        if orient == 'h':
            plt.yticks(rotation=0)
        else:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def plot_stacked_bar(
        self,
        data: pd.DataFrame,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> mpl_figure.Figure:
        """
        Create stacked bar chart.
        
        Args:
            data: DataFrame with columns as categories and index as x-axis
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        data.plot(kind='bar', stacked=True, ax=ax, color=self.theme.get_colors(len(data.columns)))
        
        ax.set_xlabel(xlabel or "")
        ax.set_ylabel(ylabel or "Value")
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.legend(title="", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    
    def plot_pie_chart(
        self,
        data: Union[pd.Series, Dict[str, float]],
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 8),
        autopct: str = '%1.1f%%',
        startangle: int = 90
    ) -> mpl_figure.Figure:
        """
        Create pie chart for proportion visualization.
        
        Args:
            data: Series or dict with labels and values
            title: Plot title
            figsize: Figure size
            autopct: Format string for percentages
            startangle: Starting angle for first slice
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if isinstance(data, dict):
            labels = list(data.keys())
            values = list(data.values())
        else:
            labels = data.index.tolist()
            values = data.values
        
        colors = self.theme.get_colors(len(labels))
        
        ax.pie(
            values,
            labels=labels,
            autopct=autopct,
            startangle=startangle,
            colors=colors
        )
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_qq_plot(
        self,
        data: Union[pd.Series, np.ndarray],
        dist: str = 'norm',
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 8)
    ) -> mpl_figure.Figure:
        """
        Create Q-Q plot for distribution comparison.
        
        Args:
            data: Data to plot
            dist: Distribution to compare against
            title: Plot title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        values = pd.Series(data).dropna()
        
        stats.probplot(values, dist=dist, plot=ax)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'Q-Q Plot ({dist} distribution)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_pairplot(
        self,
        data: pd.DataFrame,
        vars: Optional[List[str]] = None,
        hue: Optional[str] = None,
        title: Optional[str] = None,
        diag_kind: str = 'hist',
        plot_kws: Optional[Dict[str, Any]] = None
    ) -> sns.PairGrid:
        """
        Create pair plot matrix for multivariate analysis.
        
        Args:
            data: DataFrame containing data
            vars: Columns to include
            hue: Column for color grouping
            title: Overall title
            diag_kind: Diagonal plot type ('hist' or 'kde')
            plot_kws: Additional plotting arguments
            
        Returns:
            seaborn PairGrid object
        """
        if plot_kws is None:
            plot_kws = {'alpha': 0.6}
        
        g = sns.pairplot(
            data,
            vars=vars,
            hue=hue,
            diag_kind=diag_kind,
            plot_kws=plot_kws
        )
        
        if title:
            g.fig.suptitle(title, y=1.01, fontsize=14, fontweight='bold')
        
        return g
