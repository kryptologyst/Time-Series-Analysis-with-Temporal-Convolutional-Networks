# Time Series Analysis with Temporal Convolutional Networks

A comprehensive Python project for time series analysis featuring multiple forecasting methods, anomaly detection, and interactive visualization.

## Features

- **Temporal Convolutional Networks (TCN)**: Modern alternative to RNNs for sequence modeling
- **Multiple Forecasting Methods**: ARIMA, Prophet, LSTM, GRU, and ensemble methods
- **Anomaly Detection**: Statistical methods, Isolation Forest, Autoencoders, and LSTM-based detection
- **Synthetic Data Generation**: Realistic time series with trends, seasonality, and anomalies
- **Interactive Dashboard**: Streamlit-based web interface for exploration
- **Comprehensive Visualization**: Static and interactive plots using Matplotlib and Plotly
- **Model Comparison**: Side-by-side evaluation of different approaches
- **Configuration Management**: YAML-based configuration with logging and checkpointing

## Project Structure

```
├── src/                          # Source code modules
│   ├── tcn.py                   # TCN implementation
│   ├── data_generator.py         # Synthetic data generation
│   ├── forecasting.py           # Forecasting methods
│   ├── anomaly_detection.py     # Anomaly detection methods
│   ├── visualization.py        # Visualization utilities
│   └── config.py               # Configuration management
├── data/                        # Data storage
├── models/                      # Saved models
├── notebooks/                   # Jupyter notebooks
├── tests/                       # Unit tests
├── config/                      # Configuration files
├── logs/                        # Log files
├── main.py                      # Main analysis script
├── dashboard.py                 # Streamlit dashboard
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Time-Series-Analysis-with-Temporal-Convolutional-Networks.git
cd Time-Series-Analysis-with-Temporal-Convolutional-Networks
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Interface

Run the complete analysis pipeline:

```bash
python main.py
```

This will:
- Generate synthetic time series data
- Train multiple forecasting models (TCN, LSTM, ARIMA, Prophet)
- Perform anomaly detection
- Create visualizations
- Save results to `outputs/` directory

### Interactive Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

The dashboard provides:
- Interactive data generation with customizable parameters
- Real-time model training and evaluation
- Anomaly detection with multiple methods
- Interactive visualizations
- Model comparison and performance metrics

## Usage Examples

### Basic Time Series Analysis

```python
from src.data_generator import SyntheticTimeSeriesGenerator, TimeSeriesConfig
from src.tcn import TCNRegressor
from src.forecasting import LSTMForecaster
from src.anomaly_detection import StatisticalAnomalyDetector

# Generate synthetic data
config = TimeSeriesConfig(length=1000, random_seed=42)
generator = SyntheticTimeSeriesGenerator(config)
ts, components = generator.generate_time_series("full")

# Train TCN model
tcn_model = TCNRegressor(
    input_size=1,
    hidden_sizes=[16, 32, 64],
    output_size=1
)

# Train LSTM model
lstm_model = LSTMForecaster(hidden_size=64, num_layers=2)
lstm_model.fit(X_train, y_train)

# Detect anomalies
detector = StatisticalAnomalyDetector(method="zscore", threshold=2.0)
detector.fit(ts)
anomalies = detector.predict(ts)
```

### Custom Configuration

```python
from src.config import ConfigManager, ExperimentManager

# Load configuration
config_manager = ConfigManager("config/experiment.yaml")

# Get specific configurations
model_config = config_manager.get_model_config()
data_config = config_manager.get_data_config()

# Initialize experiment
experiment_config = config_manager.get_experiment_config()
experiment_manager = ExperimentManager(experiment_config)
```

## Model Architecture

### Temporal Convolutional Network (TCN)

The TCN implementation features:
- **Causal Convolutions**: Prevents future leakage
- **Dilated Convolutions**: Captures long-term dependencies
- **Residual Connections**: Enables stable training
- **Dropout Regularization**: Prevents overfitting

```python
class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        # Implementation details...
```

### Forecasting Methods

- **ARIMA**: AutoRegressive Integrated Moving Average
- **Prophet**: Facebook's forecasting tool
- **LSTM**: Long Short-Term Memory networks
- **Ensemble**: Combines multiple models

### Anomaly Detection

- **Statistical Methods**: Z-score, IQR, Modified Z-score
- **Machine Learning**: Isolation Forest, One-Class SVM
- **Deep Learning**: Autoencoders, LSTM-based detection
- **Ensemble Methods**: Combines multiple detectors

## Configuration

The project uses YAML-based configuration:

```yaml
model:
  tcn_hidden_sizes: [16, 32, 64]
  tcn_kernel_size: 2
  tcn_dropout: 0.2
  learning_rate: 0.001
  batch_size: 32
  epochs: 100

data:
  length: 1000
  trend_strength: 0.1
  seasonal_period: 24
  seasonal_strength: 0.5
  noise_level: 0.1
  anomaly_probability: 0.05

logging:
  level: "INFO"
  file_path: "logs/experiment.log"
  console_output: true

experiment:
  experiment_name: "time_series_experiment"
  output_dir: "outputs"
  save_models: true
  save_predictions: true
```

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## Performance Metrics

The project evaluates models using:
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **Accuracy**: For anomaly detection

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Dependencies

### Core Dependencies
- `numpy>=1.21.0`
- `pandas>=1.3.0`
- `matplotlib>=3.5.0`
- `seaborn>=0.11.0`
- `scipy>=1.7.0`

### Machine Learning
- `scikit-learn>=1.0.0`
- `torch>=1.12.0`
- `statsmodels>=0.13.0`
- `prophet>=1.1.0`

### Visualization
- `plotly>=5.0.0`
- `streamlit>=1.20.0`

### Development
- `pytest>=7.0.0`
- `black>=22.0.0`
- `flake8>=4.0.0`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original TCN paper: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
- Prophet: Facebook's forecasting tool
- PyTorch: Deep learning framework
- Streamlit: Web app framework

## Citation

If you use this project in your research, please cite:

```bibtex
@software{time_series_tcn,
  title={Time Series Analysis with Temporal Convolutional Networks},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Time-Series-Analysis-with-Temporal-Convolutional-Networks}
}
```

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the examples in the `notebooks/` directory

## Roadmap

- [ ] Add Transformer models
- [ ] Implement probabilistic forecasting
- [ ] Add more anomaly detection methods
- [ ] Support for multivariate time series
- [ ] Real-time streaming analysis
- [ ] Model deployment with FastAPI
- [ ] Integration with cloud platforms
# Time-Series-Analysis-with-Temporal-Convolutional-Networks
