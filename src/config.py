"""
Configuration management module for time series analysis.

This module provides configuration management using YAML files,
logging setup, and checkpoint saving/loading functionality.
"""

import yaml
import json
import logging
import os
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import torch
from datetime import datetime


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    # TCN parameters
    tcn_hidden_sizes: list = None
    tcn_kernel_size: int = 2
    tcn_dropout: float = 0.2
    
    # LSTM parameters
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    patience: int = 10
    
    # Data parameters
    sequence_length: int = 30
    forecast_horizon: int = 1
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    
    def __post_init__(self):
        if self.tcn_hidden_sizes is None:
            self.tcn_hidden_sizes = [16, 32, 64]


@dataclass
class DataConfig:
    """Configuration for data generation."""
    length: int = 1000
    trend_strength: float = 0.1
    seasonal_period: int = 24
    seasonal_strength: float = 0.5
    noise_level: float = 0.1
    anomaly_probability: float = 0.05
    anomaly_strength: float = 2.0
    random_seed: Optional[int] = None


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    console_output: bool = True


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    experiment_name: str = "time_series_experiment"
    output_dir: str = "outputs"
    save_models: bool = True
    save_predictions: bool = True
    save_plots: bool = True
    random_seed: Optional[int] = None


class ConfigManager:
    """Configuration manager for time series analysis."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                elif self.config_path.endswith('.json'):
                    return json.load(f)
                else:
                    raise ValueError("Unsupported config file format. Use .yaml or .json")
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "model": asdict(ModelConfig()),
            "data": asdict(DataConfig()),
            "logging": asdict(LoggingConfig()),
            "experiment": asdict(ExperimentConfig())
        }
    
    def save_config(self, path: str) -> None:
        """Save configuration to file."""
        with open(path, 'w') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                yaml.dump(self.config, f, default_flow_style=False)
            elif path.endswith('.json'):
                json.dump(self.config, f, indent=2)
            else:
                raise ValueError("Unsupported config file format. Use .yaml or .json")
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        return ModelConfig(**self.config.get("model", {}))
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration."""
        return DataConfig(**self.config.get("data", {}))
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return LoggingConfig(**self.config.get("logging", {}))
    
    def get_experiment_config(self) -> ExperimentConfig:
        """Get experiment configuration."""
        return ExperimentConfig(**self.config.get("experiment", {}))
    
    def update_config(self, section: str, updates: Dict[str, Any]) -> None:
        """Update configuration section."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section].update(updates)


class Logger:
    """Logger for time series analysis."""
    
    def __init__(self, config: LoggingConfig):
        """
        Initialize logger.
        
        Args:
            config: Logging configuration
        """
        self.config = config
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with configuration."""
        logger = logging.getLogger("time_series_analysis")
        logger.setLevel(getattr(logging, self.config.level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(self.config.format)
        
        # Console handler
        if self.config.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if self.config.file_path:
            os.makedirs(os.path.dirname(self.config.file_path), exist_ok=True)
            file_handler = logging.FileHandler(self.config.file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)


class CheckpointManager:
    """Checkpoint manager for saving and loading models."""
    
    def __init__(self, output_dir: str):
        """
        Initialize checkpoint manager.
        
        Args:
            output_dir: Output directory for checkpoints
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(
        self,
        model: Any,
        epoch: int,
        loss: float,
        metrics: Dict[str, float],
        model_name: str = "model"
    ) -> str:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            epoch: Current epoch
            loss: Current loss
            metrics: Current metrics
            model_name: Name of the model
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save PyTorch models
        if isinstance(model, torch.nn.Module):
            checkpoint["model_state_dict"] = model.state_dict()
            checkpoint_path = self.output_dir / f"{model_name}_epoch_{epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
        
        # Save other models using pickle
        else:
            checkpoint["model"] = model
            checkpoint_path = self.output_dir / f"{model_name}_epoch_{epoch}.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
        
        return str(checkpoint_path)
    
    def load_model(
        self,
        checkpoint_path: str,
        model: Optional[torch.nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load state into (for PyTorch models)
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.suffix == '.pth':
            checkpoint = torch.load(checkpoint_path)
            if model is not None:
                model.load_state_dict(checkpoint["model_state_dict"])
        else:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
        
        return checkpoint
    
    def get_best_checkpoint(self, model_name: str, metric: str = "loss") -> Optional[str]:
        """
        Get path to best checkpoint based on metric.
        
        Args:
            model_name: Name of the model
            metric: Metric to use for comparison
            
        Returns:
            Path to best checkpoint or None
        """
        checkpoints = list(self.output_dir.glob(f"{model_name}_epoch_*.pth")) + \
                     list(self.output_dir.glob(f"{model_name}_epoch_*.pkl"))
        
        if not checkpoints:
            return None
        
        best_checkpoint = None
        best_value = float('inf') if metric == "loss" else float('-inf')
        
        for checkpoint_path in checkpoints:
            try:
                checkpoint = self.load_model(str(checkpoint_path))
                value = checkpoint["metrics"].get(metric, checkpoint.get("loss", float('inf')))
                
                if metric == "loss":
                    if value < best_value:
                        best_value = value
                        best_checkpoint = str(checkpoint_path)
                else:
                    if value > best_value:
                        best_value = value
                        best_checkpoint = str(checkpoint_path)
            except Exception as e:
                print(f"Error loading checkpoint {checkpoint_path}: {e}")
                continue
        
        return best_checkpoint
    
    def save_predictions(
        self,
        predictions: Dict[str, Any],
        filename: str = "predictions.json"
    ) -> str:
        """
        Save predictions to file.
        
        Args:
            predictions: Predictions dictionary
            filename: Filename to save to
            
        Returns:
            Path to saved file
        """
        file_path = self.output_dir / filename
        
        with open(file_path, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
        
        return str(file_path)
    
    def save_experiment_results(
        self,
        results: Dict[str, Any],
        filename: str = "experiment_results.json"
    ) -> str:
        """
        Save experiment results to file.
        
        Args:
            results: Results dictionary
            filename: Filename to save to
            
        Returns:
            Path to saved file
        """
        file_path = self.output_dir / filename
        
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return str(file_path)


class ExperimentManager:
    """Experiment manager for organizing experiments."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment manager.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.experiment_dir = self.output_dir / config.experiment_name
        
        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(str(self.experiment_dir))
        
        # Initialize logger
        log_config = LoggingConfig(
            file_path=str(self.experiment_dir / "experiment.log")
        )
        self.logger = Logger(log_config)
    
    def get_model_dir(self, model_name: str) -> Path:
        """Get directory for specific model."""
        model_dir = self.experiment_dir / "models" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    def get_plots_dir(self) -> Path:
        """Get directory for plots."""
        plots_dir = self.experiment_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        return plots_dir
    
    def get_data_dir(self) -> Path:
        """Get directory for data."""
        data_dir = self.experiment_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    
    def save_config(self, config: Dict[str, Any]) -> str:
        """Save experiment configuration."""
        config_path = self.experiment_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return str(config_path)
    
    def log_experiment_start(self) -> None:
        """Log experiment start."""
        self.logger.info(f"Starting experiment: {self.config.experiment_name}")
        self.logger.info(f"Output directory: {self.experiment_dir}")
    
    def log_experiment_end(self) -> None:
        """Log experiment end."""
        self.logger.info(f"Experiment completed: {self.config.experiment_name}")


# Example usage
if __name__ == "__main__":
    # Create configuration
    config_manager = ConfigManager()
    
    # Get configurations
    model_config = config_manager.get_model_config()
    data_config = config_manager.get_data_config()
    logging_config = config_manager.get_logging_config()
    experiment_config = config_manager.get_experiment_config()
    
    # Initialize experiment manager
    experiment_manager = ExperimentManager(experiment_config)
    experiment_manager.log_experiment_start()
    
    # Save configuration
    experiment_manager.save_config(config_manager.config)
    
    # Initialize logger
    logger = experiment_manager.logger
    logger.info("Configuration loaded successfully")
    
    # Example checkpoint saving
    checkpoint_manager = experiment_manager.checkpoint_manager
    logger.info("Checkpoint manager initialized")
    
    experiment_manager.log_experiment_end()
