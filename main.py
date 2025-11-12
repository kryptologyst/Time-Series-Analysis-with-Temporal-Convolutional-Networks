"""
Main script for comprehensive time series analysis.

This script demonstrates:
- Synthetic data generation
- Multiple forecasting methods (TCN, LSTM, ARIMA, Prophet)
- Anomaly detection
- Model comparison and evaluation
- Visualization of results
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import warnings
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_generator import SyntheticTimeSeriesGenerator, TimeSeriesConfig, create_forecasting_dataset
from src.tcn import TCNRegressor
from src.forecasting import LSTMForecaster, ARIMAForecaster, ProphetForecaster, EnsembleForecaster, evaluate_forecaster
from src.anomaly_detection import StatisticalAnomalyDetector, IsolationForestDetector, AutoencoderAnomalyDetector, EnsembleAnomalyDetector
from src.visualization import TimeSeriesVisualizer, PlotConfig
from src.config import ConfigManager, ExperimentManager, ExperimentConfig

# Suppress warnings
warnings.filterwarnings('ignore')


def generate_synthetic_data(config: TimeSeriesConfig) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate synthetic time series data.
    
    Args:
        config: Data configuration
        
    Returns:
        Tuple of (time_series, components)
    """
    print("Generating synthetic time series data...")
    
    generator = SyntheticTimeSeriesGenerator(config)
    ts, components = generator.generate_time_series("full")
    
    print(f"Generated time series of length: {len(ts)}")
    print(f"Components: {list(components.keys())}")
    print(f"Statistics:")
    print(f"  Mean: {np.mean(ts):.4f}")
    print(f"  Std: {np.std(ts):.4f}")
    print(f"  Min: {np.min(ts):.4f}")
    print(f"  Max: {np.max(ts):.4f}")
    
    return ts, components


def prepare_forecasting_data(ts: np.ndarray, sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for forecasting.
    
    Args:
        ts: Time series data
        sequence_length: Length of input sequences
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    print(f"Preparing forecasting data with sequence length: {sequence_length}")
    
    X_train, y_train, X_val, y_val, X_test, y_test = create_forecasting_dataset(
        ts, sequence_length=sequence_length, forecast_horizon=1
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_forecasting_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> Dict[str, Any]:
    """
    Train multiple forecasting models.
    
    Args:
        X_train: Training input data
        y_train: Training target data
        X_val: Validation input data
        y_val: Validation target data
        
    Returns:
        Dictionary of trained models
    """
    print("Training forecasting models...")
    
    models = {}
    
    # TCN Model
    print("Training TCN model...")
    try:
        tcn_model = TCNRegressor(
            input_size=1,
            hidden_sizes=[16, 32, 64],
            output_size=1,
            kernel_size=2,
            dropout=0.2
        )
        
        # Convert data to tensors
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(-1)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val).unsqueeze(-1)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # Train TCN
        optimizer = torch.optim.Adam(tcn_model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(50):
            # Training
            tcn_model.train()
            optimizer.zero_grad()
            outputs = tcn_model(X_train_tensor).squeeze()
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # Validation
            tcn_model.eval()
            with torch.no_grad():
                val_outputs = tcn_model(X_val_tensor).squeeze()
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}")
        
        models["TCN"] = tcn_model
        print("TCN training completed")
        
    except Exception as e:
        print(f"TCN training failed: {e}")
    
    # LSTM Model
    print("Training LSTM model...")
    try:
        lstm_model = LSTMForecaster(
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            epochs=50
        )
        lstm_model.fit(X_train, y_train)
        models["LSTM"] = lstm_model
        print("LSTM training completed")
        
    except Exception as e:
        print(f"LSTM training failed: {e}")
    
    # ARIMA Model
    print("Training ARIMA model...")
    try:
        arima_model = ARIMAForecaster(order=(1, 1, 1))
        arima_model.fit(X_train, y_train)
        models["ARIMA"] = arima_model
        print("ARIMA training completed")
        
    except Exception as e:
        print(f"ARIMA training failed: {e}")
    
    # Prophet Model
    print("Training Prophet model...")
    try:
        prophet_model = ProphetForecaster()
        prophet_model.fit(X_train, y_train)
        models["Prophet"] = prophet_model
        print("Prophet training completed")
        
    except Exception as e:
        print(f"Prophet training failed: {e}")
    
    print(f"Successfully trained {len(models)} models")
    return models


def evaluate_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate trained models on test data.
    
    Args:
        models: Dictionary of trained models
        X_test: Test input data
        y_test: Test target data
        
    Returns:
        Dictionary of evaluation metrics for each model
    """
    print("Evaluating models on test data...")
    
    results = {}
    
    for name, model in models.items():
        try:
            print(f"Evaluating {name}...")
            metrics = evaluate_forecaster(model, X_test, y_test)
            results[name] = metrics
            
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  MAPE: {metrics['mape']:.4f}%")
            
        except Exception as e:
            print(f"Evaluation failed for {name}: {e}")
    
    return results


def detect_anomalies(ts: np.ndarray, anomaly_labels: np.ndarray) -> Dict[str, Any]:
    """
    Detect anomalies using multiple methods.
    
    Args:
        ts: Time series data
        anomaly_labels: True anomaly labels
        
    Returns:
        Dictionary of anomaly detection results
    """
    print("Detecting anomalies...")
    
    detectors = {}
    results = {}
    
    # Statistical Detector
    print("Training Statistical detector...")
    try:
        stat_detector = StatisticalAnomalyDetector(method="zscore", threshold=2.0)
        stat_detector.fit(ts)
        detectors["Statistical"] = stat_detector
        print("Statistical detector trained")
    except Exception as e:
        print(f"Statistical detector failed: {e}")
    
    # Isolation Forest Detector
    print("Training Isolation Forest detector...")
    try:
        iso_detector = IsolationForestDetector(contamination=0.1)
        iso_detector.fit(ts)
        detectors["IsolationForest"] = iso_detector
        print("Isolation Forest detector trained")
    except Exception as e:
        print(f"Isolation Forest detector failed: {e}")
    
    # Autoencoder Detector
    print("Training Autoencoder detector...")
    try:
        autoencoder_detector = AutoencoderAnomalyDetector(
            encoding_dim=32,
            hidden_dims=[64, 32],
            epochs=50
        )
        autoencoder_detector.fit(ts)
        detectors["Autoencoder"] = autoencoder_detector
        print("Autoencoder detector trained")
    except Exception as e:
        print(f"Autoencoder detector failed: {e}")
    
    # Evaluate detectors
    for name, detector in detectors.items():
        try:
            predictions = detector.predict(ts)
            accuracy = np.mean(predictions == anomaly_labels)
            results[name] = {
                "accuracy": accuracy,
                "predictions": predictions
            }
            print(f"{name} accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"Evaluation failed for {name}: {e}")
    
    return results


def visualize_results(
    ts: np.ndarray,
    components: Dict[str, np.ndarray],
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    anomaly_results: Dict[str, Any],
    output_dir: str = "outputs"
) -> None:
    """
    Visualize all results.
    
    Args:
        ts: Time series data
        components: Time series components
        models: Trained models
        X_test: Test input data
        y_test: Test target data
        anomaly_results: Anomaly detection results
        output_dir: Output directory for plots
    """
    print("Creating visualizations...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = TimeSeriesVisualizer()
    
    # Plot time series
    visualizer.plot_time_series(ts, "Synthetic Time Series")
    plt.savefig(os.path.join(output_dir, "time_series.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot components
    visualizer.plot_components(components, "Time Series Components")
    plt.savefig(os.path.join(output_dir, "components.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot forecasting results
    predictions = {}
    for name, model in models.items():
        try:
            pred = model.predict(X_test)
            predictions[name] = pred
        except Exception as e:
            print(f"Prediction failed for {name}: {e}")
    
    if predictions:
        visualizer.plot_model_comparison(y_test, predictions, "Model Comparison")
        plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot anomaly detection results
    if anomaly_results:
        for name, result in anomaly_results.items():
            if "predictions" in result:
                visualizer.plot_anomaly_detection(
                    ts, result["predictions"], f"Anomaly Detection - {name}"
                )
                plt.savefig(os.path.join(output_dir, f"anomaly_detection_{name.lower()}.png"), 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def main():
    """Main function to run the complete analysis."""
    print("=" * 60)
    print("COMPREHENSIVE TIME SERIES ANALYSIS")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Configuration
    data_config = TimeSeriesConfig(
        length=1000,
        trend_strength=0.1,
        seasonal_period=24,
        seasonal_strength=0.5,
        noise_level=0.1,
        anomaly_probability=0.05,
        anomaly_strength=2.0,
        random_seed=42
    )
    
    # Generate synthetic data
    ts, components = generate_synthetic_data(data_config)
    
    # Prepare forecasting data
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_forecasting_data(ts)
    
    # Train models
    models = train_forecasting_models(X_train, y_train, X_val, y_val)
    
    # Evaluate models
    if models:
        evaluation_results = evaluate_models(models, X_test, y_test)
        
        # Print evaluation summary
        print("\n" + "=" * 40)
        print("EVALUATION SUMMARY")
        print("=" * 40)
        for model_name, metrics in evaluation_results.items():
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                print(f"  {metric.upper()}: {value:.4f}")
    
    # Anomaly detection
    anomaly_labels = (components["anomalies"] != 0).astype(int)
    anomaly_results = detect_anomalies(ts, anomaly_labels)
    
    # Visualize results
    visualize_results(ts, components, models, X_test, y_test, anomaly_results)
    
    # Save results
    results = {
        "data_config": data_config.__dict__,
        "evaluation_results": evaluation_results if models else {},
        "anomaly_results": {k: {"accuracy": v["accuracy"]} for k, v in anomaly_results.items()}
    }
    
    import json
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("Results saved to:")
    print("  - results.json")
    print("  - outputs/ directory (plots)")
    print("=" * 60)


if __name__ == "__main__":
    main()
