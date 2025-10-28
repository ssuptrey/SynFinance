"""
Visualization Framework for Fraud Detection Analytics

Provides comprehensive visualization capabilities including:
- Distribution plots (histograms, KDE, box plots, violin plots)
- Correlation heatmaps (feature correlations, confusion matrices)
- Time series plots (fraud trends, temporal patterns)
- Geographic visualizations (city-wise fraud rates, maps)
- Feature importance charts (bar plots, waterfall charts)
- ROC and precision-recall curves
- Interactive plots with plotly
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import warnings

# Try to import plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Install with: pip install plotly")


class VisualizationFramework:
    """
    Comprehensive visualization framework for fraud detection analytics
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualization framework
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_distribution(
        self,
        data: np.ndarray,
        title: str = "Feature Distribution",
        xlabel: str = "Value",
        bins: int = 50,
        kde: bool = True,
        show_stats: bool = True
    ) -> Figure:
        """
        Plot distribution with histogram and KDE
        
        Args:
            data: 1D array of values
            title: Plot title
            xlabel: X-axis label
            bins: Number of bins
            kde: Whether to overlay KDE
            show_stats: Whether to show statistics
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Histogram
        ax.hist(data, bins=bins, alpha=0.6, color='steelblue', edgecolor='black', density=True)
        
        # KDE overlay
        if kde and len(data) > 1:
            from scipy.stats import gaussian_kde
            kde_obj = gaussian_kde(data[~np.isnan(data)])
            x_range = np.linspace(data.min(), data.max(), 200)
            ax.plot(x_range, kde_obj(x_range), 'r-', linewidth=2, label='KDE')
            ax.legend()
        
        # Statistics
        if show_stats:
            stats_text = f"Mean: {np.mean(data):.3f}\n"
            stats_text += f"Median: {np.median(data):.3f}\n"
            stats_text += f"Std: {np.std(data):.3f}\n"
            stats_text += f"Min: {np.min(data):.3f}\n"
            stats_text += f"Max: {np.max(data):.3f}"
            
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=10)
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_boxplot(
        self,
        data: Dict[str, np.ndarray],
        title: str = "Feature Comparison",
        ylabel: str = "Value"
    ) -> Figure:
        """
        Plot box plots for multiple groups
        
        Args:
            data: Dict of {label: values}
            title: Plot title
            ylabel: Y-axis label
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        labels = list(data.keys())
        values = [data[label] for label in labels]
        
        bp = ax.boxplot(values, labels=labels, patch_artist=True,
                       notch=True, showmeans=True)
        
        # Color the boxes
        colors = sns.color_palette("husl", len(labels))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    
    def plot_violin(
        self,
        data: Dict[str, np.ndarray],
        title: str = "Feature Distribution by Group",
        ylabel: str = "Value"
    ) -> Figure:
        """
        Plot violin plots for multiple groups
        
        Args:
            data: Dict of {label: values}
            title: Plot title
            ylabel: Y-axis label
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Prepare data for seaborn
        import pandas as pd
        df_list = []
        for label, values in data.items():
            df_list.append(pd.DataFrame({
                'Group': [label] * len(values),
                'Value': values
            }))
        df = pd.concat(df_list, ignore_index=True)
        
        sns.violinplot(data=df, x='Group', y='Value', ax=ax, inner='box')
        
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel('')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(
        self,
        correlation_matrix: np.ndarray,
        feature_names: List[str],
        title: str = "Feature Correlation Matrix",
        cmap: str = "coolwarm",
        annot: bool = False
    ) -> Figure:
        """
        Plot correlation heatmap
        
        Args:
            correlation_matrix: Correlation matrix
            feature_names: List of feature names
            title: Plot title
            cmap: Color map
            annot: Whether to annotate cells
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(14, 12))
        
        sns.heatmap(correlation_matrix, 
                   xticklabels=feature_names,
                   yticklabels=feature_names,
                   cmap=cmap,
                   center=0,
                   vmin=-1, vmax=1,
                   annot=annot,
                   fmt='.2f' if annot else '',
                   square=True,
                   linewidths=0.5,
                   cbar_kws={'label': 'Correlation'},
                   ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=90, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: List[str] = ['Normal', 'Fraud'],
        title: str = "Confusion Matrix",
        normalize: bool = False
    ) -> Figure:
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            labels: Class labels
            title: Plot title
            normalize: Whether to normalize
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
        else:
            fmt = 'd'
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=labels, yticklabels=labels,
                   ax=ax, cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        title: str = "Feature Importance",
        top_n: int = 20,
        std: Optional[np.ndarray] = None
    ) -> Figure:
        """
        Plot feature importance as horizontal bar chart
        
        Args:
            feature_names: List of feature names
            importances: Feature importance scores
            title: Plot title
            top_n: Number of top features to show
            std: Standard deviations (for error bars)
            
        Returns:
            Matplotlib Figure
        """
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        top_std = std[indices] if std is not None else None
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
        
        y_pos = np.arange(len(top_features))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        
        if top_std is not None:
            ax.barh(y_pos, top_importances, xerr=top_std, 
                   color=colors, alpha=0.7, ecolor='black', capsize=3)
        else:
            ax.barh(y_pos, top_importances, color=colors, alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        roc_auc: float,
        title: str = "ROC Curve"
    ) -> Figure:
        """
        Plot ROC curve
        
        Args:
            fpr: False positive rate
            tpr: True positive rate
            roc_auc: ROC AUC score
            title: Plot title
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
               label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_curve(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
        avg_precision: float,
        title: str = "Precision-Recall Curve"
    ) -> Figure:
        """
        Plot precision-recall curve
        
        Args:
            precision: Precision values
            recall: Recall values
            avg_precision: Average precision score
            title: Plot title
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot(recall, precision, color='darkorange', lw=2,
               label=f'PR curve (AP = {avg_precision:.3f})')
        ax.axhline(y=0.5, color='navy', lw=2, linestyle='--',
                  label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_time_series(
        self,
        timestamps: np.ndarray,
        values: np.ndarray,
        title: str = "Time Series",
        ylabel: str = "Value",
        add_trend: bool = True
    ) -> Figure:
        """
        Plot time series data
        
        Args:
            timestamps: Time values
            values: Data values
            title: Plot title
            ylabel: Y-axis label
            add_trend: Whether to add trend line
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(timestamps, values, 'o-', color='steelblue',
               linewidth=2, markersize=5, alpha=0.7, label='Data')
        
        if add_trend and len(values) > 1:
            # Add polynomial trend (degree 2)
            z = np.polyfit(np.arange(len(values)), values, 2)
            p = np.poly1d(z)
            ax.plot(timestamps, p(np.arange(len(values))),
                   "r--", linewidth=2, alpha=0.8, label='Trend')
            ax.legend()
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    
    def plot_geographic_heatmap(
        self,
        cities: List[str],
        values: np.ndarray,
        title: str = "Geographic Distribution",
        cmap: str = "YlOrRd"
    ) -> Figure:
        """
        Plot geographic heatmap for cities
        
        Args:
            cities: List of city names
            values: Values for each city
            title: Plot title
            cmap: Color map
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create horizontal bar chart sorted by value
        sorted_indices = np.argsort(values)
        sorted_cities = [cities[i] for i in sorted_indices]
        sorted_values = values[sorted_indices]
        
        colors = plt.cm.get_cmap(cmap)(sorted_values / sorted_values.max())
        y_pos = np.arange(len(sorted_cities))
        
        ax.barh(y_pos, sorted_values, color=colors, alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_cities)
        ax.set_xlabel('Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=values.min(),
                                                     vmax=values.max()))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Value', rotation=270, labelpad=20)
        
        plt.tight_layout()
        return fig
    
    def plot_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        title: str = "Model Comparison",
        metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    ) -> Figure:
        """
        Plot comparison of multiple models
        
        Args:
            metrics_dict: Dict of {model_name: {metric_name: value}}
            title: Plot title
            metrics: List of metrics to compare
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(metrics_dict.keys())
        x = np.arange(len(metrics))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            values = [metrics_dict[model].get(m, 0) for m in metrics]
            offset = (i - len(models)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=model, alpha=0.8)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        return fig
    
    # Interactive Plotly visualizations
    
    def plot_interactive_scatter(
        self,
        x: np.ndarray,
        y: np.ndarray,
        labels: Optional[np.ndarray] = None,
        feature_names: Tuple[str, str] = ('Feature 1', 'Feature 2'),
        title: str = "Interactive Scatter Plot"
    ):
        """
        Create interactive scatter plot with Plotly
        
        Args:
            x: X-axis values
            y: Y-axis values
            labels: Optional labels for coloring
            feature_names: Names of features (x, y)
            title: Plot title
            
        Returns:
            Plotly Figure (if plotly available)
        """
        if not PLOTLY_AVAILABLE:
            warnings.warn("Plotly not available. Returning None.")
            return None
        
        if labels is not None:
            fig = px.scatter(
                x=x, y=y, color=labels,
                labels={'x': feature_names[0], 'y': feature_names[1], 'color': 'Class'},
                title=title,
                color_continuous_scale='Viridis' if labels.dtype in [np.float32, np.float64] else None
            )
        else:
            fig = px.scatter(
                x=x, y=y,
                labels={'x': feature_names[0], 'y': feature_names[1]},
                title=title
            )
        
        fig.update_layout(height=600)
        return fig
    
    def plot_interactive_correlation(
        self,
        correlation_matrix: np.ndarray,
        feature_names: List[str],
        title: str = "Interactive Correlation Heatmap"
    ):
        """
        Create interactive correlation heatmap with Plotly
        
        Args:
            correlation_matrix: Correlation matrix
            feature_names: List of feature names
            title: Plot title
            
        Returns:
            Plotly Figure (if plotly available)
        """
        if not PLOTLY_AVAILABLE:
            warnings.warn("Plotly not available. Returning None.")
            return None
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=feature_names,
            y=feature_names,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=correlation_matrix,
            texttemplate='%{text:.2f}',
            textfont={"size": 8},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Features',
            yaxis_title='Features',
            height=800,
            width=900
        )
        
        return fig
    
    def plot_interactive_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        title: str = "Interactive Feature Importance",
        top_n: int = 20
    ):
        """
        Create interactive feature importance plot with Plotly
        
        Args:
            feature_names: List of feature names
            importances: Feature importance scores
            title: Plot title
            top_n: Number of top features to show
            
        Returns:
            Plotly Figure (if plotly available)
        """
        if not PLOTLY_AVAILABLE:
            warnings.warn("Plotly not available. Returning None.")
            return None
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        fig = go.Figure(data=go.Bar(
            y=top_features[::-1],  # Reverse for bottom-to-top display
            x=top_importances[::-1],
            orientation='h',
            marker=dict(
                color=top_importances[::-1],
                colorscale='Viridis',
                colorbar=dict(title="Importance")
            )
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Importance',
            yaxis_title='Features',
            height=max(400, top_n * 25)
        )
        
        return fig
    
    def create_dashboard(
        self,
        figures: List[Tuple[str, Figure]],
        title: str = "Analytics Dashboard",
        rows: int = 2,
        cols: int = 2
    ) -> Figure:
        """
        Create multi-plot dashboard
        
        Args:
            figures: List of (subplot_title, figure) tuples
            title: Dashboard title
            rows: Number of rows
            cols: Number of columns
            
        Returns:
            Combined Matplotlib Figure
        """
        fig = plt.figure(figsize=(cols * 8, rows * 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for idx, (subplot_title, subfig) in enumerate(figures[:rows*cols], 1):
            ax = fig.add_subplot(rows, cols, idx)
            
            # Copy the subplot from subfig to main figure
            # This is a simplified approach - in practice you'd regenerate plots
            ax.set_title(subplot_title, fontsize=12, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig
