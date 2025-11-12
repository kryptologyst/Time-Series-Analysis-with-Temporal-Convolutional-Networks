"""
Comprehensive visualization module for time series analysis.

This module provides various visualization functions for:
- Time series plotting
- Forecasting results
- Anomaly detection results
- Model comparison
- Statistical analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict, Any, Union
import warnings
from dataclasses import dataclass

# Plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive plots will be disabled.")

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@dataclass
class PlotConfig:
    """Configuration for plotting."""
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 100
    style: str = "seaborn-v0_8"
    color_palette: str = "husl"
    interactive: bool = False


class TimeSeriesVisualizer:
    """Comprehensive time series visualizer."""
    
    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize visualizer.
        
        Args:
            config: Plotting configuration
        """
        self.config = config or PlotConfig()
        self._setup_style()
    
    def _setup_style(self) -> None:
        """Setup plotting style."""
        plt.style.use(self.config.style)
        sns.set_palette(self.config.color_palette)
    
    def plot_time_series(
        self,
        data: np.ndarray,
        title: str = "Time Series",
        xlabel: str = "Time",
        ylabel: str = "Value",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot a single time series.
        
        Args:
            data: Time series data
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            save_path: Path to save the plot
        """
        figsize = figsize or self.config.figsize
        
        if self.config.interactive and PLOTLY_AVAILABLE:
            self._plot_time_series_interactive(data, title, xlabel, ylabel)
        else:
            self._plot_time_series_static(data, title, xlabel, ylabel, figsize, save_path)
    
    def _plot_time_series_static(
        self,
        data: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
        figsize: Tuple[int, int],
        save_path: Optional[str]
    ) -> None:
        """Plot time series using matplotlib."""
        plt.figure(figsize=figsize, dpi=self.config.dpi)
        plt.plot(data, linewidth=1.5)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.show()
    
    def _plot_time_series_interactive(
        self,
        data: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str
    ) -> None:
        """Plot time series using plotly."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=data,
            mode='lines',
            name='Time Series',
            line=dict(width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            hovermode='x unified'
        )
        
        fig.show()
    
    def plot_components(
        self,
        components: Dict[str, np.ndarray],
        title: str = "Time Series Components",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot time series components (trend, seasonal, noise, etc.).
        
        Args:
            components: Dictionary of component names and data
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
        """
        figsize = figsize or (self.config.figsize[0], self.config.figsize[1] * len(components))
        
        if self.config.interactive and PLOTLY_AVAILABLE:
            self._plot_components_interactive(components, title)
        else:
            self._plot_components_static(components, title, figsize, save_path)
    
    def _plot_components_static(
        self,
        components: Dict[str, np.ndarray],
        title: str,
        figsize: Tuple[int, int],
        save_path: Optional[str]
    ) -> None:
        """Plot components using matplotlib."""
        n_components = len(components)
        fig, axes = plt.subplots(n_components, 1, figsize=figsize, dpi=self.config.dpi)
        
        if n_components == 1:
            axes = [axes]
        
        for i, (name, data) in enumerate(components.items()):
            axes[i].plot(data, linewidth=1.5)
            axes[i].set_title(f"{name.title()} Component", fontsize=14, fontweight='bold')
            axes[i].set_ylabel("Value", fontsize=12)
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel("Time", fontsize=12)
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.show()
    
    def _plot_components_interactive(
        self,
        components: Dict[str, np.ndarray],
        title: str
    ) -> None:
        """Plot components using plotly."""
        fig = make_subplots(
            rows=len(components),
            cols=1,
            subplot_titles=list(components.keys()),
            vertical_spacing=0.05
        )
        
        for i, (name, data) in enumerate(components.items()):
            fig.add_trace(
                go.Scatter(y=data, mode='lines', name=name, line=dict(width=2)),
                row=i+1, col=1
            )
        
        fig.update_layout(
            title=title,
            height=300 * len(components),
            showlegend=False
        )
        
        fig.show()
    
    def plot_forecasting_results(
        self,
        actual: np.ndarray,
        predictions: np.ndarray,
        title: str = "Forecasting Results",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot forecasting results comparing actual vs predicted values.
        
        Args:
            actual: Actual values
            predictions: Predicted values
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
        """
        figsize = figsize or self.config.figsize
        
        if self.config.interactive and PLOTLY_AVAILABLE:
            self._plot_forecasting_results_interactive(actual, predictions, title)
        else:
            self._plot_forecasting_results_static(actual, predictions, title, figsize, save_path)
    
    def _plot_forecasting_results_static(
        self,
        actual: np.ndarray,
        predictions: np.ndarray,
        title: str,
        figsize: Tuple[int, int],
        save_path: Optional[str]
    ) -> None:
        """Plot forecasting results using matplotlib."""
        plt.figure(figsize=figsize, dpi=self.config.dpi)
        
        plt.plot(actual, label='Actual', linewidth=2, alpha=0.8)
        plt.plot(predictions, label='Predicted', linewidth=2, alpha=0.8)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.show()
    
    def _plot_forecasting_results_interactive(
        self,
        actual: np.ndarray,
        predictions: np.ndarray,
        title: str
    ) -> None:
        """Plot forecasting results using plotly."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=actual,
            mode='lines',
            name='Actual',
            line=dict(width=2, color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            y=predictions,
            mode='lines',
            name='Predicted',
            line=dict(width=2, color='red')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified'
        )
        
        fig.show()
    
    def plot_anomaly_detection(
        self,
        data: np.ndarray,
        anomalies: np.ndarray,
        title: str = "Anomaly Detection Results",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot anomaly detection results.
        
        Args:
            data: Time series data
            anomalies: Anomaly labels (0: normal, 1: anomaly)
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
        """
        figsize = figsize or self.config.figsize
        
        if self.config.interactive and PLOTLY_AVAILABLE:
            self._plot_anomaly_detection_interactive(data, anomalies, title)
        else:
            self._plot_anomaly_detection_static(data, anomalies, title, figsize, save_path)
    
    def _plot_anomaly_detection_static(
        self,
        data: np.ndarray,
        anomalies: np.ndarray,
        title: str,
        figsize: Tuple[int, int],
        save_path: Optional[str]
    ) -> None:
        """Plot anomaly detection using matplotlib."""
        plt.figure(figsize=figsize, dpi=self.config.dpi)
        
        # Plot normal points
        normal_mask = anomalies == 0
        plt.plot(np.where(normal_mask)[0], data[normal_mask], 
                'o', color='blue', alpha=0.6, markersize=3, label='Normal')
        
        # Plot anomalies
        anomaly_mask = anomalies == 1
        if np.any(anomaly_mask):
            plt.plot(np.where(anomaly_mask)[0], data[anomaly_mask], 
                    'o', color='red', alpha=0.8, markersize=5, label='Anomaly')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.show()
    
    def _plot_anomaly_detection_interactive(
        self,
        data: np.ndarray,
        anomalies: np.ndarray,
        title: str
    ) -> None:
        """Plot anomaly detection using plotly."""
        fig = go.Figure()
        
        # Plot normal points
        normal_mask = anomalies == 0
        fig.add_trace(go.Scatter(
            x=np.where(normal_mask)[0],
            y=data[normal_mask],
            mode='markers',
            name='Normal',
            marker=dict(color='blue', size=3, opacity=0.6)
        ))
        
        # Plot anomalies
        anomaly_mask = anomalies == 1
        if np.any(anomaly_mask):
            fig.add_trace(go.Scatter(
                x=np.where(anomaly_mask)[0],
                y=data[anomaly_mask],
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=5, opacity=0.8)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='closest'
        )
        
        fig.show()
    
    def plot_model_comparison(
        self,
        actual: np.ndarray,
        predictions: Dict[str, np.ndarray],
        title: str = "Model Comparison",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot comparison of multiple models.
        
        Args:
            actual: Actual values
            predictions: Dictionary of model names and predictions
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
        """
        figsize = figsize or self.config.figsize
        
        if self.config.interactive and PLOTLY_AVAILABLE:
            self._plot_model_comparison_interactive(actual, predictions, title)
        else:
            self._plot_model_comparison_static(actual, predictions, title, figsize, save_path)
    
    def _plot_model_comparison_static(
        self,
        actual: np.ndarray,
        predictions: Dict[str, np.ndarray],
        title: str,
        figsize: Tuple[int, int],
        save_path: Optional[str]
    ) -> None:
        """Plot model comparison using matplotlib."""
        plt.figure(figsize=figsize, dpi=self.config.dpi)
        
        # Plot actual values
        plt.plot(actual, label='Actual', linewidth=2, alpha=0.8, color='black')
        
        # Plot predictions
        colors = plt.cm.Set1(np.linspace(0, 1, len(predictions)))
        for i, (model_name, pred) in enumerate(predictions.items()):
            plt.plot(pred, label=model_name, linewidth=2, alpha=0.8, color=colors[i])
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.show()
    
    def _plot_model_comparison_interactive(
        self,
        actual: np.ndarray,
        predictions: Dict[str, np.ndarray],
        title: str
    ) -> None:
        """Plot model comparison using plotly."""
        fig = go.Figure()
        
        # Plot actual values
        fig.add_trace(go.Scatter(
            y=actual,
            mode='lines',
            name='Actual',
            line=dict(width=2, color='black')
        ))
        
        # Plot predictions
        colors = px.colors.qualitative.Set1
        for i, (model_name, pred) in enumerate(predictions.items()):
            fig.add_trace(go.Scatter(
                y=pred,
                mode='lines',
                name=model_name,
                line=dict(width=2, color=colors[i % len(colors)])
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified'
        )
        
        fig.show()
    
    def plot_residuals(
        self,
        actual: np.ndarray,
        predictions: np.ndarray,
        title: str = "Residuals Analysis",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot residuals analysis.
        
        Args:
            actual: Actual values
            predictions: Predicted values
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
        """
        residuals = actual - predictions
        
        figsize = figsize or (15, 10)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=self.config.dpi)
        
        # Residuals over time
        axes[0, 0].plot(residuals, alpha=0.7)
        axes[0, 0].set_title("Residuals Over Time")
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("Residuals")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title("Residuals Distribution")
        axes[0, 1].set_xlabel("Residuals")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals vs Fitted
        axes[1, 0].scatter(predictions, residuals, alpha=0.7)
        axes[1, 0].set_title("Residuals vs Fitted")
        axes[1, 0].set_xlabel("Fitted Values")
        axes[1, 0].set_ylabel("Residuals")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("Q-Q Plot")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_matrix(
        self,
        data: np.ndarray,
        title: str = "Correlation Matrix",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot correlation matrix for multivariate time series.
        
        Args:
            data: Multivariate time series data
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
        """
        if data.ndim == 1:
            warnings.warn("Data is univariate. Correlation matrix not applicable.")
            return
        
        figsize = figsize or self.config.figsize
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(data.T)
        
        plt.figure(figsize=figsize, dpi=self.config.dpi)
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.2f'
        )
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.show()


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    from data_generator import SyntheticTimeSeriesGenerator, TimeSeriesConfig
    
    config = TimeSeriesConfig(length=500, random_seed=42)
    generator = SyntheticTimeSeriesGenerator(config)
    ts, components = generator.generate_time_series("full")
    
    # Create visualizer
    visualizer = TimeSeriesVisualizer()
    
    # Plot time series
    visualizer.plot_time_series(ts, "Synthetic Time Series")
    
    # Plot components
    visualizer.plot_components(components, "Time Series Components")
    
    # Plot anomaly detection
    anomalies = (components["anomalies"] != 0).astype(int)
    visualizer.plot_anomaly_detection(ts, anomalies, "Anomaly Detection Results")
