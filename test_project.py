#!/usr/bin/env python3
"""
Test script to verify the time series analysis project works correctly.

This script performs basic functionality tests to ensure all modules
are working properly.
"""

import sys
import os
import numpy as np
import torch

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.data_generator import SyntheticTimeSeriesGenerator, TimeSeriesConfig
        from src.tcn import TCNRegressor, TCN
        from src.forecasting import LSTMForecaster
        from src.anomaly_detection import StatisticalAnomalyDetector
        from src.visualization import TimeSeriesVisualizer
        from src.config import ConfigManager
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_data_generation():
    """Test data generation functionality."""
    print("Testing data generation...")
    
    try:
        from src.data_generator import SyntheticTimeSeriesGenerator, TimeSeriesConfig
        
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
        
        print("✓ Data generation successful")
        return True
    except Exception as e:
        print(f"✗ Data generation error: {e}")
        return False

def test_tcn_model():
    """Test TCN model functionality."""
    print("Testing TCN model...")
    
    try:
        from src.tcn import TCNRegressor
        
        model = TCNRegressor(
            input_size=1,
            hidden_sizes=[16, 32],
            output_size=1
        )
        
        # Test forward pass
        x = torch.randn(2, 1, 50)
        output = model(x)
        
        assert output.shape == (2, 1)
        assert isinstance(output, torch.Tensor)
        
        print("✓ TCN model successful")
        return True
    except Exception as e:
        print(f"✗ TCN model error: {e}")
        return False

def test_anomaly_detection():
    """Test anomaly detection functionality."""
    print("Testing anomaly detection...")
    
    try:
        from src.anomaly_detection import StatisticalAnomalyDetector
        
        detector = StatisticalAnomalyDetector(method="zscore", threshold=2.0)
        data = np.random.randn(100)
        
        detector.fit(data)
        anomalies = detector.predict(data)
        
        assert len(anomalies) == 100
        assert all(anomaly in [0, 1] for anomaly in anomalies)
        
        print("✓ Anomaly detection successful")
        return True
    except Exception as e:
        print(f"✗ Anomaly detection error: {e}")
        return False

def test_config_management():
    """Test configuration management."""
    print("Testing configuration management...")
    
    try:
        from src.config import ConfigManager
        
        manager = ConfigManager()
        model_config = manager.get_model_config()
        data_config = manager.get_data_config()
        
        assert hasattr(model_config, 'tcn_hidden_sizes')
        assert hasattr(data_config, 'length')
        
        print("✓ Configuration management successful")
        return True
    except Exception as e:
        print(f"✗ Configuration management error: {e}")
        return False

def test_visualization():
    """Test visualization functionality."""
    print("Testing visualization...")
    
    try:
        from src.visualization import TimeSeriesVisualizer, PlotConfig
        
        config = PlotConfig()
        visualizer = TimeSeriesVisualizer(config)
        
        data = np.random.randn(100)
        components = {"trend": data, "seasonal": data, "noise": data}
        
        # Test that visualization methods exist
        assert hasattr(visualizer, 'plot_time_series')
        assert hasattr(visualizer, 'plot_components')
        assert hasattr(visualizer, 'plot_forecasting_results')
        
        print("✓ Visualization successful")
        return True
    except Exception as e:
        print(f"✗ Visualization error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("TIME SERIES ANALYSIS PROJECT - FUNCTIONALITY TEST")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_data_generation,
        test_tcn_model,
        test_anomaly_detection,
        test_config_management,
        test_visualization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The project is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
