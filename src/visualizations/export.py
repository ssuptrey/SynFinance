"""
Export utilities for saving visualizations in various formats.

Provides functions to save charts as PNG, SVG, PDF, and HTML with
optimization and batch processing capabilities.
"""

from typing import Optional, List, Union
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.figure as mpl_figure
import plotly.graph_objects as go


class ExportManager:
    """Manager for exporting visualizations to various formats."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize export manager.
        
        Args:
            output_dir: Default output directory
        """
        self.output_dir = output_dir
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def save_figure(
        self,
        fig: Union[mpl_figure.Figure, go.Figure],
        filename: str,
        output_dir: Optional[str] = None,
        formats: List[str] = ['png'],
        **kwargs
    ) -> List[str]:
        """
        Save figure to file(s) in specified format(s).
        
        Args:
            fig: Matplotlib or Plotly figure
            filename: Base filename (without extension)
            output_dir: Output directory (uses default if None)
            formats: List of formats to save ('png', 'svg', 'pdf', 'html')
            **kwargs: Additional arguments for save functions
            
        Returns:
            List of saved file paths
        """
        output_dir = output_dir or self.output_dir or '.'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        # Handle matplotlib figures
        if isinstance(fig, mpl_figure.Figure):
            for fmt in formats:
                if fmt == 'html':
                    continue  # Skip HTML for matplotlib
                filepath = os.path.join(output_dir, f"{filename}.{fmt}")
                fig.savefig(filepath, format=fmt, **kwargs)
                saved_files.append(filepath)
        
        # Handle plotly figures
        elif isinstance(fig, go.Figure):
            for fmt in formats:
                filepath = os.path.join(output_dir, f"{filename}.{fmt}")
                if fmt == 'html':
                    fig.write_html(filepath, **kwargs)
                else:
                    fig.write_image(filepath, format=fmt, **kwargs)
                saved_files.append(filepath)
        
        return saved_files
    
    def export_to_png(
        self,
        fig: Union[mpl_figure.Figure, go.Figure],
        filepath: str,
        dpi: int = 300,
        width: Optional[int] = None,
        height: Optional[int] = None,
        transparent: bool = False
    ) -> str:
        """
        Export figure to high-resolution PNG.
        
        Args:
            fig: Matplotlib or Plotly figure
            filepath: Output file path
            dpi: Resolution in dots per inch
            width: Image width in pixels (Plotly only)
            height: Image height in pixels (Plotly only)
            transparent: Whether to use transparent background
            
        Returns:
            Path to saved file
        """
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(fig, mpl_figure.Figure):
            # Use the Agg backend's native PNG writer to avoid PIL issues
            canvas = fig.canvas
            if canvas is None:
                import matplotlib.pyplot as plt
                canvas = plt.figure().canvas
            
            # Save using canvas directly
            fig.savefig(
                filepath,
                format='png',
                dpi=dpi,
                bbox_inches='tight',
                transparent=transparent,
                pil_kwargs=None  # Disable PIL kwargs that might cause issues
            )
        elif isinstance(fig, go.Figure):
            fig.write_image(
                filepath,
                format='png',
                width=width or 1200,
                height=height or 800,
                scale=dpi/100
            )
        
        return filepath
    
    def export_to_svg(
        self,
        fig: Union[mpl_figure.Figure, go.Figure],
        filepath: str,
        transparent: bool = False
    ) -> str:
        """
        Export figure to vector graphics (SVG).
        
        Args:
            fig: Matplotlib or Plotly figure
            filepath: Output file path
            transparent: Whether to use transparent background
            
        Returns:
            Path to saved file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(fig, mpl_figure.Figure):
            fig.savefig(
                filepath,
                format='svg',
                bbox_inches='tight',
                transparent=transparent
            )
        elif isinstance(fig, go.Figure):
            fig.write_image(filepath, format='svg')
        
        return filepath
    
    def export_to_pdf(
        self,
        fig: Union[mpl_figure.Figure, go.Figure],
        filepath: str
    ) -> str:
        """
        Export figure to PDF.
        
        Args:
            fig: Matplotlib or Plotly figure
            filepath: Output file path
            
        Returns:
            Path to saved file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(fig, mpl_figure.Figure):
            fig.savefig(filepath, format='pdf', bbox_inches='tight')
        elif isinstance(fig, go.Figure):
            fig.write_image(filepath, format='pdf')
        
        return filepath
    
    def export_to_html(
        self,
        fig: go.Figure,
        filepath: str,
        include_plotlyjs: Union[bool, str] = 'cdn',
        auto_open: bool = False
    ) -> str:
        """
        Export Plotly figure to interactive HTML.
        
        Args:
            fig: Plotly figure
            filepath: Output file path
            include_plotlyjs: How to include plotly.js ('cdn', True, False)
            auto_open: Whether to open in browser
            
        Returns:
            Path to saved file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        fig.write_html(
            filepath,
            include_plotlyjs=include_plotlyjs,
            auto_open=auto_open
        )
        
        return filepath
    
    def batch_export(
        self,
        figures: List[Union[mpl_figure.Figure, go.Figure]],
        filenames: List[str],
        output_dir: Optional[str] = None,
        formats: List[str] = ['png'],
        **kwargs
    ) -> List[List[str]]:
        """
        Export multiple figures at once.
        
        Args:
            figures: List of figures to export
            filenames: List of base filenames
            output_dir: Output directory
            formats: List of formats to save
            **kwargs: Additional arguments for save functions
            
        Returns:
            List of lists of saved file paths
        """
        if len(figures) != len(filenames):
            raise ValueError("Number of figures must match number of filenames")
        
        all_saved_files = []
        for fig, filename in zip(figures, filenames):
            saved_files = self.save_figure(fig, filename, output_dir, formats, **kwargs)
            all_saved_files.append(saved_files)
        
        return all_saved_files
    
    def optimize_for_web(
        self,
        fig: Union[mpl_figure.Figure, go.Figure],
        filepath: str,
        max_width: int = 1200,
        quality: int = 85
    ) -> str:
        """
        Export figure optimized for web display.
        
        Args:
            fig: Figure to export
            filepath: Output file path
            max_width: Maximum width in pixels
            quality: JPEG quality (1-100)
            
        Returns:
            Path to saved file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(fig, mpl_figure.Figure):
            # Calculate DPI based on max width
            fig_width_inches = fig.get_figwidth()
            dpi = max_width / fig_width_inches
            fig.savefig(filepath, format='png', dpi=dpi, bbox_inches='tight')
        elif isinstance(fig, go.Figure):
            # Plotly export with optimized size
            fig.write_html(filepath, include_plotlyjs='cdn', config={'responsive': True})
        
        return filepath
    
    def close_all(self) -> None:
        """Close all matplotlib figures to free memory."""
        plt.close('all')
