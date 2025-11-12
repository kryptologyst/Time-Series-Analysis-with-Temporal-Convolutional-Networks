"""
Synthetic time series data generation module.

This module provides functions to generate realistic synthetic time series
with various patterns including trends, seasonality, noise, and anomalies.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class TimeSeriesConfig:
    """Configuration for synthetic time series generation."""
    length: int = 1000
    trend_strength: float = 0.1
    seasonal_period: int = 24
    seasonal_strength: float = 0.5
    noise_level: float = 0.1
    anomaly_probability: float = 0.05
    anomaly_strength: float = 2.0
    random_seed: Optional[int] = None


class SyntheticTimeSeriesGenerator:
    """
    Generator for synthetic time series with various patterns.
    """
    
    def __init__(self, config: TimeSeriesConfig):
        """
        Initialize the generator with configuration.
        
        Args:
            config: Configuration for time series generation
        """
        self.config = config
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
    
    def generate_trend(self, length: int) -> np.ndarray:
        """
        Generate a linear trend component.
        
        Args:
            length: Length of the time series
            
        Returns:
            Trend component
        """
        trend_slope = self.config.trend_strength * np.random.uniform(-1, 1)
        return np.linspace(0, trend_slope * length, length)
    
    def generate_seasonality(self, length: int) -> np.ndarray:
        """
        Generate seasonal component with multiple harmonics.
        
        Args:
            length: Length of the time series
            
        Returns:
            Seasonal component
        """
        t = np.arange(length)
        seasonal = np.zeros(length)
        
        # Primary seasonal component
        seasonal += self.config.seasonal_strength * np.sin(2 * np.pi * t / self.config.seasonal_period)
        
        # Add harmonics for more complex seasonality
        for harmonic in [2, 3]:
            amplitude = self.config.seasonal_strength / harmonic
            seasonal += amplitude * np.sin(2 * np.pi * harmonic * t / self.config.seasonal_period)
        
        return seasonal
    
    def generate_noise(self, length: int) -> np.ndarray:
        """
        Generate Gaussian noise component.
        
        Args:
            length: Length of the time series
            
        Returns:
            Noise component
        """
        return np.random.normal(0, self.config.noise_level, length)
    
    def generate_anomalies(self, length: int) -> np.ndarray:
        """
        Generate anomaly component with random spikes.
        
        Args:
            length: Length of the time series
            
        Returns:
            Anomaly component
        """
        anomalies = np.zeros(length)
        num_anomalies = int(length * self.config.anomaly_probability)
        
        if num_anomalies > 0:
            anomaly_indices = np.random.choice(length, num_anomalies, replace=False)
            anomaly_values = np.random.normal(0, self.config.anomaly_strength, num_anomalies)
            anomalies[anomaly_indices] = anomaly_values
        
        return anomalies
    
    def generate_ar_process(self, length: int, ar_coeffs: List[float]) -> np.ndarray:
        """
        Generate AR(p) process.
        
        Args:
            length: Length of the time series
            ar_coeffs: AR coefficients
            
        Returns:
            AR process
        """
        p = len(ar_coeffs)
        ar_process = np.zeros(length)
        
        # Initialize with random values
        ar_process[:p] = np.random.normal(0, 1, p)
        
        # Generate AR process
        for t in range(p, length):
            ar_process[t] = np.sum(ar_coeffs * ar_process[t-p:t]) + np.random.normal(0, 0.1)
        
        return ar_process
    
    def generate_complex_pattern(self, length: int) -> np.ndarray:
        """
        Generate complex pattern with multiple components.
        
        Args:
            length: Length of the time series
            
        Returns:
            Complex pattern component
        """
        # Generate multiple sine waves with different frequencies
        t = np.arange(length)
        pattern = np.zeros(length)
        
        frequencies = [0.1, 0.05, 0.02]
        amplitudes = [0.3, 0.2, 0.1]
        
        for freq, amp in zip(frequencies, amplitudes):
            pattern += amp * np.sin(2 * np.pi * freq * t)
        
        return pattern
    
    def generate_time_series(self, pattern_type: str = "full") -> Tuple[np.ndarray, dict]:
        """
        Generate synthetic time series based on specified pattern.
        
        Args:
            pattern_type: Type of pattern to generate
                - "full": All components (trend + seasonality + noise + anomalies)
                - "trend": Only trend component
                - "seasonal": Only seasonal component
                - "ar": AR process
                - "complex": Complex pattern
                
        Returns:
            Tuple of (time_series, components_dict)
        """
        length = self.config.length
        
        if pattern_type == "full":
            trend = self.generate_trend(length)
            seasonal = self.generate_seasonality(length)
            noise = self.generate_noise(length)
            anomalies = self.generate_anomalies(length)
            
            time_series = trend + seasonal + noise + anomalies
            
            components = {
                "trend": trend,
                "seasonal": seasonal,
                "noise": noise,
                "anomalies": anomalies,
                "total": time_series
            }
            
        elif pattern_type == "trend":
            time_series = self.generate_trend(length)
            components = {"trend": time_series}
            
        elif pattern_type == "seasonal":
            time_series = self.generate_seasonality(length)
            components = {"seasonal": time_series}
            
        elif pattern_type == "ar":
            ar_coeffs = [0.6, -0.3, 0.1]  # AR(3) process
            time_series = self.generate_ar_process(length, ar_coeffs)
            components = {"ar_process": time_series}
            
        elif pattern_type == "complex":
            time_series = self.generate_complex_pattern(length)
            components = {"complex_pattern": time_series}
            
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        return time_series, components


def create_forecasting_dataset(
    time_series: np.ndarray,
    sequence_length: int,
    forecast_horizon: int = 1,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create dataset for time series forecasting.
    
    Args:
        time_series: Input time series
        sequence_length: Length of input sequences
        forecast_horizon: Number of steps to forecast
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    n_samples = len(time_series) - sequence_length - forecast_horizon + 1
    
    X = []
    y = []
    
    for i in range(n_samples):
        X.append(time_series[i:i+sequence_length])
        y.append(time_series[i+sequence_length:i+sequence_length+forecast_horizon])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data
    train_size = int(len(X) * train_ratio)
    val_size = int(len(X) * val_ratio)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def add_missing_values(
    time_series: np.ndarray,
    missing_ratio: float = 0.05,
    missing_pattern: str = "random"
) -> np.ndarray:
    """
    Add missing values to time series.
    
    Args:
        time_series: Input time series
        missing_ratio: Ratio of missing values
        missing_pattern: Pattern of missing values ("random", "consecutive")
        
    Returns:
        Time series with missing values (NaN)
    """
    ts_copy = time_series.copy()
    n_missing = int(len(time_series) * missing_ratio)
    
    if missing_pattern == "random":
        missing_indices = np.random.choice(len(time_series), n_missing, replace=False)
        ts_copy[missing_indices] = np.nan
        
    elif missing_pattern == "consecutive":
        start_idx = np.random.randint(0, len(time_series) - n_missing)
        ts_copy[start_idx:start_idx+n_missing] = np.nan
        
    else:
        raise ValueError(f"Unknown missing pattern: {missing_pattern}")
    
    return ts_copy


def generate_multiple_series(
    n_series: int,
    config: TimeSeriesConfig,
    pattern_types: Optional[List[str]] = None
) -> List[Tuple[np.ndarray, dict]]:
    """
    Generate multiple time series with different patterns.
    
    Args:
        n_series: Number of series to generate
        config: Configuration for time series generation
        pattern_types: List of pattern types to use
        
    Returns:
        List of tuples (time_series, components)
    """
    if pattern_types is None:
        pattern_types = ["full", "trend", "seasonal", "ar", "complex"]
    
    series_list = []
    generator = SyntheticTimeSeriesGenerator(config)
    
    for i in range(n_series):
        pattern_type = pattern_types[i % len(pattern_types)]
        ts, components = generator.generate_time_series(pattern_type)
        series_list.append((ts, components))
    
    return series_list


# Example usage and testing
if __name__ == "__main__":
    # Create configuration
    config = TimeSeriesConfig(
        length=1000,
        trend_strength=0.1,
        seasonal_period=24,
        seasonal_strength=0.5,
        noise_level=0.1,
        anomaly_probability=0.05,
        random_seed=42
    )
    
    # Generate time series
    generator = SyntheticTimeSeriesGenerator(config)
    ts, components = generator.generate_time_series("full")
    
    print(f"Generated time series of length: {len(ts)}")
    print(f"Components: {list(components.keys())}")
    print(f"Time series statistics:")
    print(f"  Mean: {np.mean(ts):.4f}")
    print(f"  Std: {np.std(ts):.4f}")
    print(f"  Min: {np.min(ts):.4f}")
    print(f"  Max: {np.max(ts):.4f}")
