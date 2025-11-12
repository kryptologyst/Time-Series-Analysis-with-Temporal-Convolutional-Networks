"""
Unit tests for time series analysis modules.

This module contains comprehensive tests for:
- Data generation
- TCN models
- Forecasting methods
- Anomaly detection
- Visualization utilities
"""

import pytest
import numpy as np
import torch
import pandas as pd
from unittest.mock import patch, MagicMock

# Import modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data_generator import (
    SyntheticTimeSeriesGenerator, 
    TimeSeriesConfig, 
    create_forecasting_dataset,
    add_missing_values
)
from src.tcn import TCN, TCNRegressor, TemporalBlock, calculate_receptive_field
from src.forecasting import LSTMForecaster, ARIMAForecaster, evaluate_forecaster
from src.anomaly_detection import (
    StatisticalAnomalyDetector, 
    IsolationForestDetector,
    AutoencoderAnomalyDetector
)
from src.visualization import TimeSeriesVisualizer, PlotConfig
from src.config import ConfigManager, Logger, CheckpointManager


class TestDataGenerator:
    """Test cases for data generation module."""
    
    def test_time_series_config_defaults(self):
        """Test TimeSeriesConfig default values."""
        config = TimeSeriesConfig()
        assert config.length == 1000
        assert config.trend_strength == 0.1
        assert config.seasonal_period == 24
        assert config.seasonal_strength == 0.5
        assert config.noise_level == 0.1
        assert config.anomaly_probability == 0.05
        assert config.anomaly_strength == 2.0
        assert config.random_seed is None
    
    def test_synthetic_generator_initialization(self):
        """Test SyntheticTimeSeriesGenerator initialization."""
        config = TimeSeriesConfig(length=500, random_seed=42)
        generator = SyntheticTimeSeriesGenerator(config)
        assert generator.config == config
    
    def test_generate_trend(self):
        """Test trend generation."""
        config = TimeSeriesConfig(length=100, trend_strength=0.5, random_seed=42)
        generator = SyntheticTimeSeriesGenerator(config)
        trend = generator.generate_trend(100)
        
        assert len(trend) == 100
        assert isinstance(trend, np.ndarray)
        # Trend should be monotonic
        assert np.all(np.diff(trend) >= 0) or np.all(np.diff(trend) <= 0)
    
    def test_generate_seasonality(self):
        """Test seasonality generation."""
        config = TimeSeriesConfig(length=100, seasonal_period=10, seasonal_strength=0.5, random_seed=42)
        generator = SyntheticTimeSeriesGenerator(config)
        seasonal = generator.generate_seasonality(100)
        
        assert len(seasonal) == 100
        assert isinstance(seasonal, np.ndarray)
        # Should be periodic
        assert np.isclose(seasonal[0], seasonal[10], atol=0.1)
    
    def test_generate_noise(self):
        """Test noise generation."""
        config = TimeSeriesConfig(length=100, noise_level=0.1, random_seed=42)
        generator = SyntheticTimeSeriesGenerator(config)
        noise = generator.generate_noise(100)
        
        assert len(noise) == 100
        assert isinstance(noise, np.ndarray)
        assert np.isclose(np.mean(noise), 0, atol=0.1)
        assert np.isclose(np.std(noise), 0.1, atol=0.05)
    
    def test_generate_anomalies(self):
        """Test anomaly generation."""
        config = TimeSeriesConfig(length=100, anomaly_probability=0.1, random_seed=42)
        generator = SyntheticTimeSeriesGenerator(config)
        anomalies = generator.generate_anomalies(100)
        
        assert len(anomalies) == 100
        assert isinstance(anomalies, np.ndarray)
        assert np.sum(anomalies != 0) <= 20  # Should have around 10 anomalies
    
    def test_generate_time_series_full(self):
        """Test full time series generation."""
        config = TimeSeriesConfig(length=100, random_seed=42)
        generator = SyntheticTimeSeriesGenerator(config)
        ts, components = generator.generate_time_series("full")
        
        assert len(ts) == 100
        assert isinstance(ts, np.ndarray)
        assert isinstance(components, dict)
        assert "trend" in components
        assert "seasonal" in components
        assert "noise" in components
        assert "anomalies" in components
        assert "total" in components
    
    def test_create_forecasting_dataset(self):
        """Test forecasting dataset creation."""
        ts = np.random.randn(100)
        X_train, y_train, X_val, y_val, X_test, y_test = create_forecasting_dataset(
            ts, sequence_length=10, forecast_horizon=1
        )
        
        assert len(X_train) > 0
        assert len(y_train) > 0
        assert len(X_val) > 0
        assert len(y_val) > 0
        assert len(X_test) > 0
        assert len(y_test) > 0
        
        # Check shapes
        assert X_train.shape[1] == 10  # sequence_length
        assert y_train.shape[1] == 1   # forecast_horizon
    
    def test_add_missing_values(self):
        """Test adding missing values."""
        ts = np.random.randn(100)
        ts_with_missing = add_missing_values(ts, missing_ratio=0.1, missing_pattern="random")
        
        assert len(ts_with_missing) == 100
        assert np.sum(np.isnan(ts_with_missing)) > 0
        assert np.sum(np.isnan(ts_with_missing)) <= 20


class TestTCN:
    """Test cases for TCN module."""
    
    def test_temporal_block_initialization(self):
        """Test TemporalBlock initialization."""
        block = TemporalBlock(1, 16, 3, dilation=1)
        assert block.conv1.in_channels == 1
        assert block.conv1.out_channels == 16
        assert block.conv2.in_channels == 16
        assert block.conv2.out_channels == 16
    
    def test_temporal_block_forward(self):
        """Test TemporalBlock forward pass."""
        block = TemporalBlock(1, 16, 3, dilation=1)
        x = torch.randn(2, 1, 100)
        output = block(x)
        
        assert output.shape == (2, 16, 100)
    
    def test_tcn_initialization(self):
        """Test TCN initialization."""
        tcn = TCN(1, [16, 32, 64])
        assert len(tcn.network) == 3
        assert tcn.linear.out_features == 1
    
    def test_tcn_forward(self):
        """Test TCN forward pass."""
        tcn = TCN(1, [16, 32, 64])
        x = torch.randn(2, 1, 100)
        output = tcn(x)
        
        assert output.shape == (2, 1)
    
    def test_tcn_regressor_initialization(self):
        """Test TCNRegressor initialization."""
        regressor = TCNRegressor(1, [16, 32, 64], 1)
        assert regressor.tcn is not None
        assert regressor.output_layer.out_features == 1
    
    def test_tcn_regressor_forward(self):
        """Test TCNRegressor forward pass."""
        regressor = TCNRegressor(1, [16, 32, 64], 1)
        x = torch.randn(2, 1, 100)
        output = regressor(x)
        
        assert output.shape == (2, 1)
    
    def test_calculate_receptive_field(self):
        """Test receptive field calculation."""
        rf = calculate_receptive_field(kernel_size=2, num_layers=3, dilation_base=2)
        assert rf > 0
        assert isinstance(rf, int)


class TestForecasting:
    """Test cases for forecasting module."""
    
    def test_lstm_forecaster_initialization(self):
        """Test LSTMForecaster initialization."""
        forecaster = LSTMForecaster(hidden_size=64, num_layers=2)
        assert forecaster.hidden_size == 64
        assert forecaster.num_layers == 2
    
    @patch('src.forecasting.STATSMODELS_AVAILABLE', True)
    def test_arima_forecaster_initialization(self):
        """Test ARIMAForecaster initialization."""
        forecaster = ARIMAForecaster(order=(1, 1, 1))
        assert forecaster.order == (1, 1, 1)
    
    def test_evaluate_forecaster(self):
        """Test forecaster evaluation."""
        # Mock forecaster
        class MockForecaster:
            def predict(self, X):
                return np.random.randn(len(X))
        
        forecaster = MockForecaster()
        X_test = np.random.randn(50)
        y_test = np.random.randn(50)
        
        metrics = evaluate_forecaster(forecaster, X_test, y_test)
        
        assert "mse" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mape" in metrics
        assert all(isinstance(v, float) for v in metrics.values())


class TestAnomalyDetection:
    """Test cases for anomaly detection module."""
    
    def test_statistical_detector_initialization(self):
        """Test StatisticalAnomalyDetector initialization."""
        detector = StatisticalAnomalyDetector(method="zscore", threshold=2.0)
        assert detector.method == "zscore"
        assert detector.threshold == 2.0
    
    def test_statistical_detector_fit_predict(self):
        """Test StatisticalAnomalyDetector fit and predict."""
        detector = StatisticalAnomalyDetector(method="zscore", threshold=2.0)
        data = np.random.randn(100)
        detector.fit(data)
        
        assert detector.mean is not None
        assert detector.std is not None
        
        predictions = detector.predict(data)
        assert len(predictions) == 100
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_statistical_detector_anomaly_scores(self):
        """Test StatisticalAnomalyDetector anomaly scores."""
        detector = StatisticalAnomalyDetector(method="zscore", threshold=2.0)
        data = np.random.randn(100)
        detector.fit(data)
        
        scores = detector.get_anomaly_scores(data)
        assert len(scores) == 100
        assert isinstance(scores, np.ndarray)
    
    @patch('src.anomaly_detection.SKLEARN_AVAILABLE', True)
    def test_isolation_forest_detector(self):
        """Test IsolationForestDetector."""
        detector = IsolationForestDetector(contamination=0.1)
        data = np.random.randn(100)
        detector.fit(data)
        
        predictions = detector.predict(data)
        assert len(predictions) == 100
        assert all(pred in [0, 1] for pred in predictions)


class TestVisualization:
    """Test cases for visualization module."""
    
    def test_plot_config_initialization(self):
        """Test PlotConfig initialization."""
        config = PlotConfig()
        assert config.figsize == (12, 8)
        assert config.dpi == 100
        assert config.style == "seaborn-v0_8"
    
    def test_visualizer_initialization(self):
        """Test TimeSeriesVisualizer initialization."""
        config = PlotConfig()
        visualizer = TimeSeriesVisualizer(config)
        assert visualizer.config == config


class TestConfig:
    """Test cases for configuration module."""
    
    def test_config_manager_initialization(self):
        """Test ConfigManager initialization."""
        manager = ConfigManager()
        assert isinstance(manager.config, dict)
    
    def test_config_manager_get_model_config(self):
        """Test ConfigManager get_model_config."""
        manager = ConfigManager()
        model_config = manager.get_model_config()
        assert hasattr(model_config, 'tcn_hidden_sizes')
        assert hasattr(model_config, 'learning_rate')
    
    def test_config_manager_get_data_config(self):
        """Test ConfigManager get_data_config."""
        manager = ConfigManager()
        data_config = manager.get_data_config()
        assert hasattr(data_config, 'length')
        assert hasattr(data_config, 'trend_strength')
    
    def test_logger_initialization(self):
        """Test Logger initialization."""
        from src.config import LoggingConfig
        config = LoggingConfig()
        logger = Logger(config)
        assert logger.config == config
    
    def test_checkpoint_manager_initialization(self):
        """Test CheckpointManager initialization."""
        manager = CheckpointManager("test_output")
        assert manager.output_dir.exists()


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from data generation to evaluation."""
        # Generate data
        config = TimeSeriesConfig(length=100, random_seed=42)
        generator = SyntheticTimeSeriesGenerator(config)
        ts, components = generator.generate_time_series("full")
        
        # Create forecasting dataset
        X_train, y_train, X_val, y_val, X_test, y_test = create_forecasting_dataset(
            ts, sequence_length=10, forecast_horizon=1
        )
        
        # Train TCN model
        tcn_model = TCNRegressor(1, [16, 32], 1)
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(-1)
        y_train_tensor = torch.FloatTensor(y_train)
        
        # Simple training step
        optimizer = torch.optim.Adam(tcn_model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        tcn_model.train()
        optimizer.zero_grad()
        outputs = tcn_model(X_train_tensor).squeeze()
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Test prediction
        tcn_model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).unsqueeze(-1)
            predictions = tcn_model(X_test_tensor).squeeze()
        
        assert len(predictions) == len(y_test)
        
        # Test anomaly detection
        detector = StatisticalAnomalyDetector(method="zscore", threshold=2.0)
        detector.fit(ts)
        anomalies = detector.predict(ts)
        
        assert len(anomalies) == len(ts)
        assert all(anomaly in [0, 1] for anomaly in anomalies)


if __name__ == "__main__":
    pytest.main([__file__])
