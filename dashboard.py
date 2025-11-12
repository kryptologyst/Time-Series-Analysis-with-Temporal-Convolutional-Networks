"""
Streamlit dashboard for interactive time series analysis.

This dashboard provides:
- Interactive data generation
- Model training and comparison
- Anomaly detection
- Visualization of results
- Model performance metrics
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import sys
import os
from typing import Dict, List, Tuple, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_generator import SyntheticTimeSeriesGenerator, TimeSeriesConfig, create_forecasting_dataset
from src.tcn import TCNRegressor
from src.forecasting import LSTMForecaster, ARIMAForecaster, ProphetForecaster, evaluate_forecaster
from src.anomaly_detection import StatisticalAnomalyDetector, IsolationForestDetector, AutoencoderAnomalyDetector
from src.visualization import TimeSeriesVisualizer

# Page configuration
st.set_page_config(
    page_title="Time Series Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
</style>
""", unsafe_allow_html=True)

def generate_data(config: TimeSeriesConfig) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Generate synthetic time series data."""
    generator = SyntheticTimeSeriesGenerator(config)
    ts, components = generator.generate_time_series("full")
    return ts, components

def train_tcn_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> TCNRegressor:
    """Train TCN model."""
    model = TCNRegressor(
        input_size=1,
        hidden_sizes=[16, 32, 64],
        output_size=1,
        kernel_size=2,
        dropout=0.2
    )
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(-1)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val).unsqueeze(-1)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor).squeeze()
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        progress_bar.progress((epoch + 1) / 50)
        status_text.text(f"Epoch {epoch + 1}/50 - Loss: {loss.item():.6f}")
    
    return model

def train_lstm_model(X_train: np.ndarray, y_train: np.ndarray) -> LSTMForecaster:
    """Train LSTM model."""
    model = LSTMForecaster(hidden_size=64, num_layers=2, dropout=0.2, epochs=50)
    model.fit(X_train, y_train)
    return model

def plot_time_series_interactive(ts: np.ndarray, title: str = "Time Series"):
    """Plot time series using Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=ts,
        mode='lines',
        name='Time Series',
        line=dict(width=2, color='#1f77b4')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode='x unified',
        height=400
    )
    
    return fig

def plot_components_interactive(components: Dict[str, np.ndarray]):
    """Plot components using Plotly."""
    fig = make_subplots(
        rows=len(components),
        cols=1,
        subplot_titles=list(components.keys()),
        vertical_spacing=0.05
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (name, data) in enumerate(components.items()):
        fig.add_trace(
            go.Scatter(
                y=data, 
                mode='lines', 
                name=name, 
                line=dict(width=2, color=colors[i % len(colors)])
            ),
            row=i+1, col=1
        )
    
    fig.update_layout(
        title="Time Series Components",
        height=200 * len(components),
        showlegend=False
    )
    
    return fig

def plot_forecasting_results(actual: np.ndarray, predictions: Dict[str, np.ndarray]):
    """Plot forecasting results."""
    fig = go.Figure()
    
    # Plot actual values
    fig.add_trace(go.Scatter(
        y=actual,
        mode='lines',
        name='Actual',
        line=dict(width=2, color='black')
    ))
    
    # Plot predictions
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, (model_name, pred) in enumerate(predictions.items()):
        fig.add_trace(go.Scatter(
            y=pred,
            mode='lines',
            name=model_name,
            line=dict(width=2, color=colors[i % len(colors)])
        ))
    
    fig.update_layout(
        title="Forecasting Results",
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode='x unified',
        height=500
    )
    
    return fig

def plot_anomaly_detection(ts: np.ndarray, anomalies: np.ndarray, title: str):
    """Plot anomaly detection results."""
    fig = go.Figure()
    
    # Plot normal points
    normal_mask = anomalies == 0
    fig.add_trace(go.Scatter(
        x=np.where(normal_mask)[0],
        y=ts[normal_mask],
        mode='markers',
        name='Normal',
        marker=dict(color='blue', size=3, opacity=0.6)
    ))
    
    # Plot anomalies
    anomaly_mask = anomalies == 1
    if np.any(anomaly_mask):
        fig.add_trace(go.Scatter(
            x=np.where(anomaly_mask)[0],
            y=ts[anomaly_mask],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=5, opacity=0.8)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode='closest',
        height=400
    )
    
    return fig

def main():
    """Main dashboard function."""
    # Header
    st.markdown('<h1 class="main-header">📈 Time Series Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Data generation parameters
    st.sidebar.subheader("Data Generation")
    length = st.sidebar.slider("Time Series Length", 100, 2000, 1000)
    trend_strength = st.sidebar.slider("Trend Strength", 0.0, 1.0, 0.1)
    seasonal_period = st.sidebar.slider("Seasonal Period", 5, 50, 24)
    seasonal_strength = st.sidebar.slider("Seasonal Strength", 0.0, 1.0, 0.5)
    noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1)
    anomaly_probability = st.sidebar.slider("Anomaly Probability", 0.0, 0.2, 0.05)
    anomaly_strength = st.sidebar.slider("Anomaly Strength", 1.0, 5.0, 2.0)
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    sequence_length = st.sidebar.slider("Sequence Length", 10, 100, 30)
    learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001)
    epochs = st.sidebar.slider("Epochs", 10, 100, 50)
    
    # Generate data button
    if st.sidebar.button("Generate Data", key="generate_data"):
        st.session_state.generate_data = True
    
    # Main content
    if st.session_state.get("generate_data", False):
        # Generate data
        config = TimeSeriesConfig(
            length=length,
            trend_strength=trend_strength,
            seasonal_period=seasonal_period,
            seasonal_strength=seasonal_strength,
            noise_level=noise_level,
            anomaly_probability=anomaly_probability,
            anomaly_strength=anomaly_strength,
            random_seed=42
        )
        
        ts, components = generate_data(config)
        st.session_state.ts = ts
        st.session_state.components = components
        
        # Prepare forecasting data
        X_train, y_train, X_val, y_val, X_test, y_test = create_forecasting_dataset(
            ts, sequence_length=sequence_length, forecast_horizon=1
        )
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.session_state.X_val = X_val
        st.session_state.y_val = y_val
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        
        st.success("Data generated successfully!")
    
    # Display data if available
    if "ts" in st.session_state:
        ts = st.session_state.ts
        components = st.session_state.components
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Overview", "Components", "Forecasting", "Anomaly Detection", "Model Comparison"])
        
        with tab1:
            st.subheader("Time Series Overview")
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Length", len(ts))
            with col2:
                st.metric("Mean", f"{np.mean(ts):.4f}")
            with col3:
                st.metric("Std", f"{np.std(ts):.4f}")
            with col4:
                st.metric("Min/Max", f"{np.min(ts):.4f}/{np.max(ts):.4f}")
            
            # Plot
            fig = plot_time_series_interactive(ts, "Generated Time Series")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Time Series Components")
            
            # Plot components
            fig = plot_components_interactive(components)
            st.plotly_chart(fig, use_container_width=True)
            
            # Component statistics
            st.subheader("Component Statistics")
            component_stats = pd.DataFrame({
                "Component": list(components.keys()),
                "Mean": [np.mean(comp) for comp in components.values()],
                "Std": [np.std(comp) for comp in components.values()],
                "Min": [np.min(comp) for comp in components.values()],
                "Max": [np.max(comp) for comp in components.values()]
            })
            st.dataframe(component_stats, use_container_width=True)
        
        with tab3:
            st.subheader("Forecasting Models")
            
            # Model selection
            col1, col2 = st.columns(2)
            with col1:
                train_tcn = st.button("Train TCN Model", key="train_tcn")
            with col2:
                train_lstm = st.button("Train LSTM Model", key="train_lstm")
            
            # Train models
            if train_tcn:
                with st.spinner("Training TCN model..."):
                    tcn_model = train_tcn_model(
                        st.session_state.X_train, 
                        st.session_state.y_train,
                        st.session_state.X_val,
                        st.session_state.y_val
                    )
                    st.session_state.tcn_model = tcn_model
                st.success("TCN model trained successfully!")
            
            if train_lstm:
                with st.spinner("Training LSTM model..."):
                    lstm_model = train_lstm_model(
                        st.session_state.X_train, 
                        st.session_state.y_train
                    )
                    st.session_state.lstm_model = lstm_model
                st.success("LSTM model trained successfully!")
            
            # Display results
            if "tcn_model" in st.session_state or "lstm_model" in st.session_state:
                st.subheader("Forecasting Results")
                
                predictions = {}
                
                if "tcn_model" in st.session_state:
                    tcn_model = st.session_state.tcn_model
                    tcn_model.eval()
                    with torch.no_grad():
                        X_test_tensor = torch.FloatTensor(st.session_state.X_test).unsqueeze(-1)
                        tcn_pred = tcn_model(X_test_tensor).squeeze().numpy()
                        predictions["TCN"] = tcn_pred
                
                if "lstm_model" in st.session_state:
                    lstm_model = st.session_state.lstm_model
                    lstm_pred = lstm_model.predict(st.session_state.X_test)
                    predictions["LSTM"] = lstm_pred
                
                if predictions:
                    # Plot results
                    fig = plot_forecasting_results(st.session_state.y_test, predictions)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Metrics
                    st.subheader("Model Performance")
                    metrics_data = []
                    for model_name, pred in predictions.items():
                        mse = np.mean((st.session_state.y_test - pred) ** 2)
                        mae = np.mean(np.abs(st.session_state.y_test - pred))
                        rmse = np.sqrt(mse)
                        mape = np.mean(np.abs((st.session_state.y_test - pred) / st.session_state.y_test)) * 100
                        
                        metrics_data.append({
                            "Model": model_name,
                            "RMSE": f"{rmse:.4f}",
                            "MAE": f"{mae:.4f}",
                            "MAPE": f"{mape:.2f}%"
                        })
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True)
        
        with tab4:
            st.subheader("Anomaly Detection")
            
            # Anomaly detection methods
            col1, col2, col3 = st.columns(3)
            with col1:
                detect_statistical = st.button("Statistical Detection", key="detect_statistical")
            with col2:
                detect_isolation = st.button("Isolation Forest", key="detect_isolation")
            with col3:
                detect_autoencoder = st.button("Autoencoder", key="detect_autoencoder")
            
            # Perform detection
            if detect_statistical:
                with st.spinner("Running statistical anomaly detection..."):
                    detector = StatisticalAnomalyDetector(method="zscore", threshold=2.0)
                    detector.fit(ts)
                    anomalies = detector.predict(ts)
                    st.session_state.statistical_anomalies = anomalies
                st.success("Statistical detection completed!")
            
            if detect_isolation:
                with st.spinner("Running Isolation Forest detection..."):
                    detector = IsolationForestDetector(contamination=0.1)
                    detector.fit(ts)
                    anomalies = detector.predict(ts)
                    st.session_state.isolation_anomalies = anomalies
                st.success("Isolation Forest detection completed!")
            
            if detect_autoencoder:
                with st.spinner("Running Autoencoder detection..."):
                    detector = AutoencoderAnomalyDetector(encoding_dim=32, hidden_dims=[64, 32], epochs=30)
                    detector.fit(ts)
                    anomalies = detector.predict(ts)
                    st.session_state.autoencoder_anomalies = anomalies
                st.success("Autoencoder detection completed!")
            
            # Display results
            anomaly_methods = {
                "Statistical": "statistical_anomalies",
                "Isolation Forest": "isolation_anomalies",
                "Autoencoder": "autoencoder_anomalies"
            }
            
            for method_name, session_key in anomaly_methods.items():
                if session_key in st.session_state:
                    anomalies = st.session_state[session_key]
                    
                    st.subheader(f"{method_name} Results")
                    
                    # Plot
                    fig = plot_anomaly_detection(ts, anomalies, f"Anomaly Detection - {method_name}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Anomalies", np.sum(anomalies))
                    with col2:
                        st.metric("Anomaly Rate", f"{np.mean(anomalies) * 100:.2f}%")
                    with col3:
                        st.metric("Accuracy", f"{np.mean(anomalies == (components['anomalies'] != 0).astype(int)) * 100:.2f}%")
        
        with tab5:
            st.subheader("Model Comparison")
            
            if "tcn_model" in st.session_state or "lstm_model" in st.session_state:
                st.info("Model comparison will be displayed here once multiple models are trained.")
                
                # Comparison metrics
                if "tcn_model" in st.session_state and "lstm_model" in st.session_state:
                    st.subheader("Performance Comparison")
                    
                    # Calculate metrics for both models
                    tcn_model = st.session_state.tcn_model
                    lstm_model = st.session_state.lstm_model
                    
                    tcn_model.eval()
                    with torch.no_grad():
                        X_test_tensor = torch.FloatTensor(st.session_state.X_test).unsqueeze(-1)
                        tcn_pred = tcn_model(X_test_tensor).squeeze().numpy()
                    
                    lstm_pred = lstm_model.predict(st.session_state.X_test)
                    
                    # Create comparison plot
                    fig = plot_forecasting_results(st.session_state.y_test, {"TCN": tcn_pred, "LSTM": lstm_pred})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Comparison table
                    comparison_data = []
                    for model_name, pred in [("TCN", tcn_pred), ("LSTM", lstm_pred)]:
                        mse = np.mean((st.session_state.y_test - pred) ** 2)
                        mae = np.mean(np.abs(st.session_state.y_test - pred))
                        rmse = np.sqrt(mse)
                        mape = np.mean(np.abs((st.session_state.y_test - pred) / st.session_state.y_test)) * 100
                        
                        comparison_data.append({
                            "Model": model_name,
                            "RMSE": f"{rmse:.4f}",
                            "MAE": f"{mae:.4f}",
                            "MAPE": f"{mape:.2f}%"
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
            else:
                st.info("Train models first to see comparison results.")

if __name__ == "__main__":
    main()
