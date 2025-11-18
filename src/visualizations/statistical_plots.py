"""
Statistical plots for model diagnostics and performance visualization.

Provides specialized visualizations for machine learning models, hypothesis testing,
and statistical analysis.
"""

from typing import Optional, List, Union, Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as mpl_figure
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc, confusion_matrix

from .themes import ChartTheme, get_theme


class StatisticalPlots:
    """Statistical plot generator for model diagnostics."""
    
    def __init__(self, theme: Union[str, ChartTheme] = "default"):
        """
        Initialize statistical plots generator.
        
        Args:
            theme: Theme name or ChartTheme instance
        """
        if isinstance(theme, str):
            self.theme = get_theme(theme)
        else:
            self.theme = theme
        self.theme.apply()
    
    def plot_regression(
        self,
        x: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        ci: int = 95,
        scatter_kws: Optional[Dict] = None,
        line_kws: Optional[Dict] = None
    ) -> mpl_figure.Figure:
        """
        Create regression plot with confidence interval.
        
        Args:
            x: Independent variable
            y: Dependent variable
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            ci: Confidence interval percentage
            scatter_kws: Scatter plot keywords
            line_kws: Line plot keywords
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        scatter_kws = scatter_kws or {'alpha': 0.6}
        line_kws = line_kws or {'color': 'red', 'linewidth': 2}
        
        sns.regplot(x=x, y=y, ax=ax, ci=ci, scatter_kws=scatter_kws, line_kws=line_kws)
        
        # Calculate R-squared
        if len(x) > 1:
            correlation = np.corrcoef(x, y)[0, 1]
            r_squared = correlation ** 2
            ax.text(
                0.05, 0.95, f'R² = {r_squared:.4f}',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        ax.set_xlabel(xlabel or "X")
        ax.set_ylabel(ylabel or "Y")
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_residuals(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 5)
    ) -> mpl_figure.Figure:
        """
        Create residual plots for regression diagnostics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Overall title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        residuals = np.array(y_true) - np.array(y_pred)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Residuals vs Fitted
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Fitted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Fitted')
        
        # Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title('Normal Q-Q Plot')
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(
        self,
        y_true: Union[pd.Series, np.ndarray, List],
        y_score: Union[pd.Series, np.ndarray, List],
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 8)
    ) -> mpl_figure.Figure:
        """
        Create ROC curve for binary classification.
        
        Args:
            y_true: True binary labels
            y_score: Predicted scores or probabilities
            title: Plot title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(
        self,
        y_true: Union[pd.Series, np.ndarray, List],
        y_pred: Union[pd.Series, np.ndarray, List],
        labels: Optional[List[str]] = None,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 8),
        normalize: bool = False,
        cmap: str = 'Blues'
    ) -> mpl_figure.Figure:
        """
        Create confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            title: Plot title
            figsize: Figure size
            normalize: Whether to normalize values
            cmap: Colormap name
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            square=True,
            xticklabels=labels or ['Negative', 'Positive'],
            yticklabels=labels or ['Negative', 'Positive'],
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
            ax=ax
        )
        
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: Union[pd.Series, np.ndarray, List],
        title: Optional[str] = None,
        top_n: Optional[int] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> mpl_figure.Figure:
        """
        Create feature importance bar chart.
        
        Args:
            feature_names: List of feature names
            importances: Feature importance values
            title: Plot title
            top_n: Number of top features to show
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create DataFrame and sort
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        # Select top N if specified
        if top_n:
            df = df.tail(top_n)
        
        # Create horizontal bar chart
        ax.barh(df['feature'], df['importance'], color=self.theme.get_color(0))
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_learning_curve(
        self,
        train_sizes: Union[np.ndarray, List],
        train_scores: Union[np.ndarray, List],
        val_scores: Union[np.ndarray, List],
        title: Optional[str] = None,
        xlabel: str = 'Training Examples',
        ylabel: str = 'Score',
        figsize: Tuple[int, int] = (10, 6)
    ) -> mpl_figure.Figure:
        """
        Create learning curve plot.
        
        Args:
            train_sizes: Number of training examples
            train_scores: Training scores
            val_scores: Validation scores
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert to arrays
        train_sizes = np.array(train_sizes)
        train_scores = np.array(train_scores)
        val_scores = np.array(val_scores)
        
        # Calculate mean and std
        if train_scores.ndim == 2:
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
        else:
            train_mean = train_scores
            train_std = np.zeros_like(train_scores)
        
        if val_scores.ndim == 2:
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
        else:
            val_mean = val_scores
            val_std = np.zeros_like(val_scores)
        
        # Plot training scores
        ax.plot(train_sizes, train_mean, 'o-', color=self.theme.get_color(0), label='Training score')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color=self.theme.get_color(0))
        
        # Plot validation scores
        ax.plot(train_sizes, val_mean, 'o-', color=self.theme.get_color(1), label='Validation score')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color=self.theme.get_color(1))
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title('Learning Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_validation_curve(
        self,
        param_range: Union[np.ndarray, List],
        train_scores: Union[np.ndarray, List],
        val_scores: Union[np.ndarray, List],
        param_name: str = 'Parameter',
        title: Optional[str] = None,
        ylabel: str = 'Score',
        figsize: Tuple[int, int] = (10, 6),
        xscale: str = 'linear'
    ) -> mpl_figure.Figure:
        """
        Create validation curve for hyperparameter tuning.
        
        Args:
            param_range: Parameter values
            train_scores: Training scores
            val_scores: Validation scores
            param_name: Parameter name for x-axis
            title: Plot title
            ylabel: Y-axis label
            figsize: Figure size
            xscale: X-axis scale ('linear' or 'log')
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert to arrays
        param_range = np.array(param_range)
        train_scores = np.array(train_scores)
        val_scores = np.array(val_scores)
        
        # Calculate mean and std
        if train_scores.ndim == 2:
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
        else:
            train_mean = train_scores
            train_std = np.zeros_like(train_scores)
        
        if val_scores.ndim == 2:
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
        else:
            val_mean = val_scores
            val_std = np.zeros_like(val_scores)
        
        # Plot training scores
        ax.plot(param_range, train_mean, 'o-', color=self.theme.get_color(0), label='Training score')
        ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color=self.theme.get_color(0))
        
        # Plot validation scores
        ax.plot(param_range, val_mean, 'o-', color=self.theme.get_color(1), label='Validation score')
        ax.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color=self.theme.get_color(1))
        
        ax.set_xlabel(param_name)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'Validation Curve ({param_name})', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_lift_curve(
        self,
        y_true: Union[pd.Series, np.ndarray, List],
        y_score: Union[pd.Series, np.ndarray, List],
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> mpl_figure.Figure:
        """
        Create lift curve for model performance analysis.
        
        Args:
            y_true: True binary labels
            y_score: Predicted scores
            title: Plot title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create DataFrame and sort by score
        df = pd.DataFrame({'true': y_true, 'score': y_score})
        df = df.sort_values('score', ascending=False).reset_index(drop=True)
        
        # Calculate cumulative gains
        total_positives = df['true'].sum()
        df['cumulative_positives'] = df['true'].cumsum()
        df['percentage_data'] = (np.arange(len(df)) + 1) / len(df) * 100
        df['lift'] = (df['cumulative_positives'] / (np.arange(len(df)) + 1)) / (total_positives / len(df))
        
        # Plot lift curve
        ax.plot(df['percentage_data'], df['lift'], color=self.theme.get_color(0), linewidth=2, label='Model')
        ax.axhline(y=1, color='r', linestyle='--', label='Baseline')
        ax.set_xlabel('Percentage of Data')
        ax.set_ylabel('Lift')
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title('Lift Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_acf_pacf(
        self,
        data: Union[pd.Series, np.ndarray],
        lags: int = 40,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 5)
    ) -> mpl_figure.Figure:
        """
        Create ACF and PACF plots for time series analysis.
        
        Args:
            data: Time series data
            lags: Number of lags to display
            title: Overall title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # ACF plot
        plot_acf(data, lags=lags, ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)')
        
        # PACF plot
        plot_pacf(data, lags=lags, ax=axes[1])
        axes[1].set_title('Partial Autocorrelation Function (PACF)')
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        return fig
