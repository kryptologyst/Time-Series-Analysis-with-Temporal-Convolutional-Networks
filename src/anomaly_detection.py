"""
Anomaly detection module for time series data.

This module provides various anomaly detection methods including:
- Statistical methods (Z-score, IQR)
- Machine learning methods (Isolation Forest, One-Class SVM)
- Deep learning methods (Autoencoders, LSTM-based)
- Hybrid methods combining multiple approaches
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

# Sklearn models
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Some anomaly detection methods will be disabled.")

# Statistical methods
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Statistical methods will be disabled.")


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection."""
    threshold_method: str = "iqr"  # "iqr", "zscore", "percentile"
    threshold_value: float = 1.5
    window_size: int = 10
    contamination: float = 0.1
    random_seed: Optional[int] = None


class BaseAnomalyDetector(ABC):
    """Abstract base class for all anomaly detectors."""
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseAnomalyDetector':
        """Fit the detector to the data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies in the data."""
        pass
    
    @abstractmethod
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores for the data."""
        pass


class StatisticalAnomalyDetector(BaseAnomalyDetector):
    """Statistical-based anomaly detector."""
    
    def __init__(self, method: str = "zscore", threshold: float = 3.0):
        """
        Initialize statistical anomaly detector.
        
        Args:
            method: Detection method ("zscore", "iqr", "modified_zscore")
            threshold: Threshold for anomaly detection
        """
        self.method = method
        self.threshold = threshold
        self.mean = None
        self.std = None
        self.median = None
        self.mad = None  # Median Absolute Deviation
        
    def fit(self, X: np.ndarray) -> 'StatisticalAnomalyDetector':
        """Fit the detector."""
        if X.ndim > 1:
            X = X.flatten()
        
        self.mean = np.mean(X)
        self.std = np.std(X)
        self.median = np.median(X)
        
        if SCIPY_AVAILABLE:
            self.mad = stats.median_abs_deviation(X)
        else:
            # Manual MAD calculation
            self.mad = np.median(np.abs(X - self.median))
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies."""
        scores = self.get_anomaly_scores(X)
        
        if self.method == "zscore":
            anomalies = np.abs(scores) > self.threshold
        elif self.method == "iqr":
            anomalies = scores > self.threshold
        elif self.method == "modified_zscore":
            anomalies = np.abs(scores) > self.threshold
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return anomalies.astype(int)
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        if X.ndim > 1:
            X = X.flatten()
        
        if self.method == "zscore":
            scores = (X - self.mean) / self.std
        elif self.method == "iqr":
            q75, q25 = np.percentile(X, [75, 25])
            iqr = q75 - q25
            scores = (X - self.median) / iqr
        elif self.method == "modified_zscore":
            scores = 0.6745 * (X - self.median) / self.mad
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return scores


class IsolationForestDetector(BaseAnomalyDetector):
    """Isolation Forest-based anomaly detector."""
    
    def __init__(self, contamination: float = 0.1, random_state: Optional[int] = None):
        """
        Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of anomalies
            random_state: Random state for reproducibility
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Isolation Forest")
        
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        """Fit the detector."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Normalize data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Isolation Forest
        self.model.fit(X_scaled)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies."""
        scores = self.get_anomaly_scores(X)
        return (scores < 0).astype(int)
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Normalize data
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly scores
        scores = self.model.decision_function(X_scaled)
        
        return scores


class AutoencoderAnomalyDetector(BaseAnomalyDetector):
    """Autoencoder-based anomaly detector."""
    
    def __init__(
        self,
        encoding_dim: int = 32,
        hidden_dims: List[int] = [64, 32],
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32
    ):
        """
        Initialize autoencoder detector.
        
        Args:
            encoding_dim: Dimension of the encoding layer
            hidden_dims: Dimensions of hidden layers
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.threshold = None
        
    def fit(self, X: np.ndarray) -> 'AutoencoderAnomalyDetector':
        """Fit the autoencoder."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Normalize data
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Create autoencoder model
        input_dim = X_scaled.shape[1]
        self.model = Autoencoder(
            input_dim=input_dim,
            encoding_dim=self.encoding_dim,
            hidden_dims=self.hidden_dims
        )
        
        # Train autoencoder
        self._train_model(X_scaled)
        
        # Calculate reconstruction threshold
        self._calculate_threshold(X_scaled)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies."""
        scores = self.get_anomaly_scores(X)
        return (scores > self.threshold).astype(int)
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores (reconstruction errors)."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Normalize data
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Get reconstructions
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)
            reconstructions = self.model(X_tensor)
            
            # Calculate reconstruction errors
            errors = torch.mean((X_tensor - reconstructions) ** 2, dim=1)
            
        return errors.numpy()
    
    def _train_model(self, X: np.ndarray) -> None:
        """Train the autoencoder model."""
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                reconstructions = self.model(batch_X)
                loss = criterion(reconstructions, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.6f}")
    
    def _calculate_threshold(self, X: np.ndarray) -> None:
        """Calculate reconstruction threshold."""
        scores = self.get_anomaly_scores(X)
        self.threshold = np.percentile(scores, 95)  # 95th percentile as threshold


class Autoencoder(nn.Module):
    """Autoencoder model for anomaly detection."""
    
    def __init__(self, input_dim: int, encoding_dim: int, hidden_dims: List[int]):
        super(Autoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = encoding_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class LSTMANomalyDetector(BaseAnomalyDetector):
    """LSTM-based anomaly detector."""
    
    def __init__(
        self,
        sequence_length: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32
    ):
        """
        Initialize LSTM anomaly detector.
        
        Args:
            sequence_length: Length of input sequences
            hidden_size: Size of LSTM hidden layers
            num_layers: Number of LSTM layers
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.threshold = None
        
    def fit(self, X: np.ndarray) -> 'LSTMANomalyDetector':
        """Fit the LSTM detector."""
        if X.ndim > 1:
            X = X.flatten()
        
        # Normalize data
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X.reshape(-1, 1)).flatten()
        else:
            X_scaled = X
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled)
        
        # Create LSTM model
        self.model = LSTMPredictor(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=1
        )
        
        # Train model
        self._train_model(X_seq, y_seq)
        
        # Calculate threshold
        self._calculate_threshold(X_seq, y_seq)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies."""
        scores = self.get_anomaly_scores(X)
        return (scores > self.threshold).astype(int)
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores (prediction errors)."""
        if X.ndim > 1:
            X = X.flatten()
        
        # Normalize data
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X.reshape(-1, 1)).flatten()
        else:
            X_scaled = X
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).unsqueeze(-1)
            predictions = self.model(X_tensor).squeeze()
            
            # Calculate prediction errors
            errors = torch.abs(predictions - torch.FloatTensor(y_seq))
            
        return errors.numpy()
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i+self.sequence_length])
            y.append(data[i+self.sequence_length])
        return np.array(X), np.array(y)
    
    def _train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the LSTM model."""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).unsqueeze(-1)
        y_tensor = torch.FloatTensor(y)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = self.model(batch_X).squeeze()
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.6f}")
    
    def _calculate_threshold(self, X: np.ndarray, y: np.ndarray) -> None:
        """Calculate prediction error threshold."""
        scores = self.get_anomaly_scores(X.flatten())
        self.threshold = np.percentile(scores, 95)  # 95th percentile as threshold


class LSTMPredictor(nn.Module):
    """LSTM model for time series prediction."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
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
        
        # Apply fully connected layer
        out = self.fc(out)
        
        return out


class EnsembleAnomalyDetector(BaseAnomalyDetector):
    """Ensemble anomaly detector combining multiple methods."""
    
    def __init__(self, detectors: List[BaseAnomalyDetector], weights: Optional[List[float]] = None):
        """
        Initialize ensemble detector.
        
        Args:
            detectors: List of anomaly detectors
            weights: Weights for each detector (if None, equal weights)
        """
        self.detectors = detectors
        self.weights = weights or [1.0 / len(detectors)] * len(detectors)
        
        if len(self.weights) != len(detectors):
            raise ValueError("Number of weights must match number of detectors")
        
        # Normalize weights
        self.weights = np.array(self.weights) / np.sum(self.weights)
        
    def fit(self, X: np.ndarray) -> 'EnsembleAnomalyDetector':
        """Fit all detectors."""
        for detector in self.detectors:
            detector.fit(X)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies using ensemble."""
        predictions = []
        for detector in self.detectors:
            pred = detector.predict(X)
            predictions.append(pred)
        
        # Weighted voting
        predictions = np.array(predictions)
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        
        # Convert to binary predictions
        return (ensemble_pred > 0.5).astype(int)
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble anomaly scores."""
        scores = []
        for detector in self.detectors:
            score = detector.get_anomaly_scores(X)
            scores.append(score)
        
        # Weighted average of scores
        scores = np.array(scores)
        ensemble_scores = np.average(scores, axis=0, weights=self.weights)
        
        return ensemble_scores


def evaluate_anomaly_detector(
    detector: BaseAnomalyDetector,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluate an anomaly detector on test data.
    
    Args:
        detector: Trained anomaly detector
        X_test: Test input data
        y_test: Test true labels (0: normal, 1: anomaly)
        
    Returns:
        Dictionary of evaluation metrics
    """
    predictions = detector.predict(X_test)
    
    # Calculate metrics
    if SKLEARN_AVAILABLE:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    else:
        # Basic metrics without sklearn
        accuracy = np.mean(predictions == y_test)
        return {"accuracy": accuracy}


# Example usage
if __name__ == "__main__":
    # Generate synthetic data with anomalies
    from data_generator import SyntheticTimeSeriesGenerator, TimeSeriesConfig
    
    config = TimeSeriesConfig(
        length=1000,
        anomaly_probability=0.1,
        anomaly_strength=3.0,
        random_seed=42
    )
    generator = SyntheticTimeSeriesGenerator(config)
    ts, components = generator.generate_time_series("full")
    
    # Create anomaly labels
    anomaly_labels = (components["anomalies"] != 0).astype(int)
    
    # Test different detectors
    detectors = []
    
    # Statistical detector
    stat_detector = StatisticalAnomalyDetector(method="zscore", threshold=2.0)
    stat_detector.fit(ts)
    detectors.append(stat_detector)
    
    # Isolation Forest
    if SKLEARN_AVAILABLE:
        iso_detector = IsolationForestDetector(contamination=0.1)
        iso_detector.fit(ts)
        detectors.append(iso_detector)
    
    # Evaluate detectors
    for detector in detectors:
        metrics = evaluate_anomaly_detector(detector, ts, anomaly_labels)
        print(f"{detector.__class__.__name__} Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()
