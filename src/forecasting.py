"""
Comprehensive time series forecasting module.

This module provides implementations of various forecasting methods including:
- ARIMA models
- Prophet
- LSTM/GRU networks
- Transformer models
- TCN (Temporal Convolutional Networks)
- Ensemble methods
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Optional, Dict, Any, Union
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Statistical models
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. ARIMA models will be disabled.")

# Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not available. Prophet models will be disabled.")

# Sklearn models
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Some models will be disabled.")


@dataclass
class ForecastConfig:
    """Configuration for forecasting models."""
    sequence_length: int = 30
    forecast_horizon: int = 1
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    patience: int = 10
    random_seed: Optional[int] = None


class BaseForecaster(ABC):
    """Abstract base class for all forecasters."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BaseForecaster':
        """Fit the model to the training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass


class ARIMAForecaster(BaseForecaster):
    """ARIMA-based forecaster."""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        """
        Initialize ARIMA forecaster.
        
        Args:
            order: ARIMA order (p, d, q)
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA forecasting")
        
        self.order = order
        self.model = None
        self.fitted_model = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'ARIMAForecaster':
        """Fit ARIMA model."""
        if y is not None:
            # Use y as the target if provided
            data = y
        else:
            # Use X as univariate time series
            data = X.flatten() if X.ndim > 1 else X
        
        self.model = ARIMA(data, order=self.order)
        self.fitted_model = self.model.fit()
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # For ARIMA, we predict the next values after the training data
        n_predictions = len(X) if X.ndim == 1 else X.shape[0]
        predictions = self.fitted_model.forecast(steps=n_predictions)
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = {
            "model_type": "ARIMA",
            "order": self.order,
            "fitted": self.fitted_model is not None
        }
        
        if self.fitted_model is not None:
            info.update({
                "aic": self.fitted_model.aic,
                "bic": self.fitted_model.bic,
                "log_likelihood": self.fitted_model.llf
            })
        
        return info


class ProphetForecaster(BaseForecaster):
    """Prophet-based forecaster."""
    
    def __init__(self, **prophet_params):
        """
        Initialize Prophet forecaster.
        
        Args:
            **prophet_params: Parameters for Prophet model
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is required for Prophet forecasting")
        
        self.prophet_params = prophet_params
        self.model = None
        self.fitted_model = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'ProphetForecaster':
        """Fit Prophet model."""
        if y is not None:
            data = y
        else:
            data = X.flatten() if X.ndim > 1 else X
        
        # Create DataFrame for Prophet
        df = pd.DataFrame({
            'ds': pd.date_range(start='2020-01-01', periods=len(data), freq='H'),
            'y': data
        })
        
        self.model = Prophet(**self.prophet_params)
        self.fitted_model = self.model.fit(df)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        n_predictions = len(X) if X.ndim == 1 else X.shape[0]
        
        # Create future DataFrame
        future = self.fitted_model.make_future_dataframe(periods=n_predictions, freq='H')
        
        # Make predictions
        forecast = self.fitted_model.predict(future)
        
        # Return only the forecasted values
        return forecast['yhat'].tail(n_predictions).values
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": "Prophet",
            "parameters": self.prophet_params,
            "fitted": self.fitted_model is not None
        }


class LSTMForecaster(BaseForecaster):
    """LSTM-based forecaster."""
    
    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        config: Optional[ForecastConfig] = None
    ):
        """
        Initialize LSTM forecaster.
        
        Args:
            hidden_size: Size of hidden layers
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            config: Forecasting configuration
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.config = config or ForecastConfig()
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'LSTMForecaster':
        """Fit LSTM model."""
        if y is not None:
            # Use y as target
            target = y
        else:
            # Use X as univariate time series
            target = X.flatten() if X.ndim > 1 else X
        
        # Prepare sequences
        X_seq, y_seq = self._prepare_sequences(target)
        
        # Normalize data
        if self.scaler is not None:
            X_seq = self.scaler.fit_transform(X_seq.reshape(-1, 1)).reshape(X_seq.shape)
            y_seq = self.scaler.transform(y_seq.reshape(-1, 1)).flatten()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).unsqueeze(-1)
        y_tensor = torch.FloatTensor(y_seq)
        
        # Create model
        self.model = LSTMModel(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=1,
            dropout=self.dropout
        )
        
        # Train model
        self._train_model(X_tensor, y_tensor)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        
        # Prepare input sequences
        if X.ndim == 1:
            X_seq = self._create_sequences(X, self.config.sequence_length)
        else:
            X_seq = X
        
        # Normalize if scaler is available
        if self.scaler is not None:
            X_seq = self.scaler.transform(X_seq.reshape(-1, 1)).reshape(X_seq.shape)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq).unsqueeze(-1)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(X_tensor).squeeze()
        
        # Denormalize if scaler is available
        if self.scaler is not None:
            predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        return predictions.numpy()
    
    def _prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for training."""
        X, y = [], []
        for i in range(len(data) - self.config.sequence_length):
            X.append(data[i:i+self.config.sequence_length])
            y.append(data[i+self.config.sequence_length])
        return np.array(X), np.array(y)
    
    def _create_sequences(self, data: np.ndarray, seq_len: int) -> np.ndarray:
        """Create sequences from data."""
        sequences = []
        for i in range(len(data) - seq_len + 1):
            sequences.append(data[i:i+seq_len])
        return np.array(sequences)
    
    def _train_model(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Train the LSTM model."""
        # Split data
        train_size = int(len(X) * self.config.train_ratio)
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:], y[train_size:]
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val).squeeze()
                val_loss = criterion(val_outputs, y_val).item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": "LSTM",
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "fitted": self.model is not None
        }


class LSTMModel(nn.Module):
    """LSTM model for time series forecasting."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2
    ):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = out[:, -1, :]
        
        # Apply dropout and fully connected layer
        out = self.dropout(out)
        out = self.fc(out)
        
        return out


class EnsembleForecaster(BaseForecaster):
    """Ensemble forecaster combining multiple models."""
    
    def __init__(self, forecasters: List[BaseForecaster], weights: Optional[List[float]] = None):
        """
        Initialize ensemble forecaster.
        
        Args:
            forecasters: List of forecasters to combine
            weights: Weights for each forecaster (if None, equal weights)
        """
        self.forecasters = forecasters
        self.weights = weights or [1.0 / len(forecasters)] * len(forecasters)
        
        if len(self.weights) != len(forecasters):
            raise ValueError("Number of weights must match number of forecasters")
        
        # Normalize weights
        self.weights = np.array(self.weights) / np.sum(self.weights)
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'EnsembleForecaster':
        """Fit all forecasters."""
        for forecaster in self.forecasters:
            forecaster.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        predictions = []
        for forecaster in self.forecasters:
            pred = forecaster.predict(X)
            predictions.append(pred)
        
        # Weighted average
        predictions = np.array(predictions)
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return ensemble_pred
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": "Ensemble",
            "num_models": len(self.forecasters),
            "weights": self.weights.tolist(),
            "forecasters": [f.get_model_info() for f in self.forecasters]
        }


def evaluate_forecaster(
    forecaster: BaseForecaster,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate a forecaster on test data.
    
    Args:
        forecaster: Trained forecaster
        X_test: Test input data
        y_test: Test target data
        
    Returns:
        Dictionary of evaluation metrics
    """
    predictions = forecaster.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    from data_generator import SyntheticTimeSeriesGenerator, TimeSeriesConfig
    
    config = TimeSeriesConfig(length=1000, random_seed=42)
    generator = SyntheticTimeSeriesGenerator(config)
    ts, _ = generator.generate_time_series("full")
    
    # Create forecasting dataset
    from data_generator import create_forecasting_dataset
    X_train, y_train, X_val, y_val, X_test, y_test = create_forecasting_dataset(
        ts, sequence_length=30, forecast_horizon=1
    )
    
    # Test different forecasters
    forecasters = []
    
    # ARIMA
    if STATSMODELS_AVAILABLE:
        arima_forecaster = ARIMAForecaster(order=(1, 1, 1))
        arima_forecaster.fit(X_train, y_train)
        forecasters.append(arima_forecaster)
    
    # LSTM
    lstm_forecaster = LSTMForecaster(hidden_size=64, num_layers=2)
    lstm_forecaster.fit(X_train, y_train)
    forecasters.append(lstm_forecaster)
    
    # Evaluate forecasters
    for forecaster in forecasters:
        metrics = evaluate_forecaster(forecaster, X_test, y_test)
        print(f"{forecaster.get_model_info()['model_type']} Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        print()
