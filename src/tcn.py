"""
Temporal Convolutional Network (TCN) implementation with modern best practices.

This module provides a comprehensive TCN implementation with:
- Causal convolutions to prevent future leakage
- Dilated convolutions for long-term dependencies
- Residual connections for stable training
- Dropout for regularization
- Batch normalization for improved convergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import math


class TemporalBlock(nn.Module):
    """
    A temporal block with causal convolution, batch normalization, and residual connection.
    
    Args:
        n_inputs: Number of input channels
        n_outputs: Number of output channels
        kernel_size: Size of the convolution kernel
        stride: Stride of the convolution
        dilation: Dilation rate for the convolution
        padding: Padding for the convolution
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        dropout: float = 0.2
    ):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
    
    def init_weights(self) -> None:
        """Initialize weights using Xavier initialization."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the temporal block.
        
        Args:
            x: Input tensor of shape (batch_size, n_inputs, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, n_outputs, sequence_length)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """
    Remove the last element of the temporal dimension to ensure causality.
    """
    
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Remove the last chomp_size elements from the temporal dimension.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with last chomp_size elements removed
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TCN(nn.Module):
    """
    Temporal Convolutional Network for sequence modeling.
    
    Args:
        num_inputs: Number of input features
        num_channels: List of hidden channel sizes for each layer
        kernel_size: Size of the convolution kernel
        dropout: Dropout probability
        num_outputs: Number of output features (default: 1 for regression)
    """
    
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
        num_outputs: int = 1
    ):
        super(TCN, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            padding = (kernel_size - 1) * dilation_size
            
            layers += [
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=padding, dropout=dropout
                )
            ]
        
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], num_outputs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TCN.
        
        Args:
            x: Input tensor of shape (batch_size, num_inputs, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, num_outputs)
        """
        # Pass through temporal blocks
        y = self.network(x)
        
        # Take the last time step for prediction
        y = y[:, :, -1]
        
        # Apply final linear layer
        return self.linear(y)


class TCNRegressor(nn.Module):
    """
    TCN-based regressor with additional features for time series forecasting.
    
    Args:
        input_size: Number of input features
        hidden_sizes: List of hidden layer sizes
        output_size: Number of output features
        kernel_size: Size of the convolution kernel
        dropout: Dropout probability
        use_batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int = 1,
        kernel_size: int = 2,
        dropout: float = 0.2,
        use_batch_norm: bool = True
    ):
        super(TCNRegressor, self).__init__()
        
        self.tcn = TCN(
            num_inputs=input_size,
            num_channels=hidden_sizes,
            kernel_size=kernel_size,
            dropout=dropout,
            num_outputs=hidden_sizes[-1]
        )
        
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_sizes[-1])
        
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TCN regressor.
        
        Args:
            x: Input tensor of shape (batch_size, input_size, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Get TCN features
        features = self.tcn.network(x)
        features = features[:, :, -1]  # Take last time step
        
        # Apply batch normalization if enabled
        if self.use_batch_norm:
            features = self.batch_norm(features)
        
        # Apply output layer
        output = self.output_layer(features)
        
        return output


def calculate_receptive_field(
    kernel_size: int,
    num_layers: int,
    dilation_base: int = 2
) -> int:
    """
    Calculate the receptive field of a TCN.
    
    Args:
        kernel_size: Size of the convolution kernel
        num_layers: Number of layers in the TCN
        dilation_base: Base for exponential dilation
        
    Returns:
        Receptive field size
    """
    receptive_field = 1
    for i in range(num_layers):
        dilation = dilation_base ** i
        receptive_field += (kernel_size - 1) * dilation
    
    return receptive_field


def get_tcn_model_size(model: nn.Module) -> int:
    """
    Calculate the total number of parameters in a TCN model.
    
    Args:
        model: TCN model
        
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
