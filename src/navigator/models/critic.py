"""
Critic network for the Navigator RL agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class ConvBlock(nn.Module):
    """
    A simple convolutional block with Conv3D, GroupNorm, and GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, num_groups=8):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        return F.gelu(self.norm(self.conv(x)))

class CriticNetwork(nn.Module):
    """
    Critic network for PPO that estimates state values.

    The network processes 3D patches (CNN), a sequence of past positions (LSTM),
    and combines these features to output value estimates.
    """

    def __init__(self, input_channels=4, lstm_hidden_size=128, num_lstm_layers=1): # Patches input_channels=4
        super().__init__()

        # CNN for patches
        self.conv1 = ConvBlock(input_channels, 16, kernel_size=3, padding=1, num_groups=8)
        self.pool1 = nn.Conv3d(16, 16, kernel_size=2, stride=2, padding=0, bias=False)
        self.conv2 = ConvBlock(16, 32, kernel_size=3, padding=1, num_groups=16)
        self.pool2 = nn.Conv3d(32, 32, kernel_size=2, stride=2, padding=0, bias=False)
        self.conv3 = ConvBlock(32, 64, kernel_size=3, padding=1, num_groups=32)
        self.pool3 = nn.Conv3d(64, 64, kernel_size=2, stride=2, padding=0, bias=False)
        self.cnn_flatten = nn.Flatten()

        self.embed = nn.Linear(3, 128)
        encoder_layer = nn.TransformerEncoderLayer(128, 8, 512, dropout=0.05, batch_first=True)
        self.attention = nn.TransformerEncoder(encoder_layer, 2)

        # Combined feature processing
        # LazyLinear adapts to: flattened_cnn_features + lstm_hidden_size
        self.feature_extractor_fc = nn.LazyLinear(256)
        self.value_head = nn.Linear(256, 1)

    def forward(self, patches: torch.Tensor, position_sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Process patches with CNN
        x_conv = self.conv1(patches)
        x_conv = self.pool1(x_conv)
        x_conv = self.conv2(x_conv)
        x_conv = self.pool2(x_conv)
        x_conv = self.conv3(x_conv)
        x_conv = self.pool3(x_conv)
        cnn_features = self.cnn_flatten(x_conv)

        # We take the output of the last time step
        lstm_features = self.attention(self.embed(position_sequence), src_key_padding_mask=mask).mean(dim=1)

        # Concatenate CNN features and LSTM features
        combined_features = torch.cat((cnn_features, lstm_features), dim=1)
        
        # Pass combined features through the head
        features = F.gelu(self.feature_extractor_fc(combined_features)) # Added GELU activation
        out = self.value_head(features)
        return out

class StateActionValueNetwork(nn.Module):
    """
    Critic network for TD3 that estimates state-action values.

    Processes 3D patches (CNN), a sequence of past positions (LSTM),
    combines these with actions, and outputs Q-value estimates.
    """

    def __init__(self, input_channels=4, action_dim=3, lstm_hidden_size=128, num_lstm_layers=1): # Patches input_channels=4
        super().__init__()

        # CNN for patches
        self.conv1 = ConvBlock(input_channels, 16, kernel_size=3, padding=1, num_groups=8)
        self.pool1 = nn.Conv3d(16, 16, kernel_size=2, stride=2, padding=0, bias=False)
        self.conv2 = ConvBlock(16, 32, kernel_size=3, padding=1, num_groups=16)
        self.pool2 = nn.Conv3d(32, 32, kernel_size=2, stride=2, padding=0, bias=False)
        self.conv3 = ConvBlock(32, 64, kernel_size=3, padding=1, num_groups=32)
        self.pool3 = nn.Conv3d(64, 64, kernel_size=2, stride=2, padding=0, bias=False)
        self.cnn_flatten = nn.Flatten()

        # LSTM for position sequence
        self.lstm_input_size = 3 # Z, Y, X coordinates
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.num_lstm_layers,
            batch_first=True
        )
        
        # Observation feature processing (after CNN and LSTM)
        # LazyLinear adapts to: flattened_cnn_features + lstm_hidden_size
        self.obs_feature_fc1 = nn.LazyLinear(512)
        self.obs_feature_norm1 = nn.GroupNorm(32, 512)  # Assuming 512 is divisible by 32
        self.obs_feature_act1 = nn.GELU()
        self.obs_feature_fc2 = nn.Linear(512, 256)
        self.obs_feature_norm2 = nn.GroupNorm(16, 256)  # Assuming 256 is divisible by 16
        self.obs_feature_act2 = nn.GELU()
        
        # Prediction head (combines processed observation features and action)
        self.predict = nn.Linear(256 + action_dim, 1)

    def forward(self, patches: torch.Tensor, position_sequence: torch.Tensor, action: torch.Tensor):
        # Process patches with CNN
        x_conv = self.conv1(patches)
        x_conv = self.pool1(x_conv)
        x_conv = self.conv2(x_conv)
        x_conv = self.pool2(x_conv)
        x_conv = self.conv3(x_conv)
        x_conv = self.pool3(x_conv)
        cnn_features = self.cnn_flatten(x_conv)

        # Process position sequence with LSTM
        lstm_out, _ = self.lstm(position_sequence)
        lstm_features = lstm_out[:, -1, :] # Shape (B, lstm_hidden_size)
        
        # Concatenate CNN features and LSTM features
        combined_obs_features = torch.cat((cnn_features, lstm_features), dim=1)
        
        # Process combined observation features
        obs_f = self.obs_feature_fc1(combined_obs_features)
        obs_f = self.obs_feature_act1(self.obs_feature_norm1(obs_f))
        obs_f = self.obs_feature_fc2(obs_f)
        extracted_obs_features = self.obs_feature_act2(self.obs_feature_norm2(obs_f))
        
        # Concatenate with action and predict Q-value
        out = self.predict(torch.cat([extracted_obs_features, action], dim=1))
        return out
