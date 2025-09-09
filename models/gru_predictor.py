"""
gru_predictor.py

Implements a Seq2Seq GRU model for trajectory prediction.
"""

import torch
from torch import nn


# Define Seq2Seq GRU Model
class TrajPredictor(nn.Module):
    """
    Sequence-to-sequence GRU model for trajectory prediction.

    Args:
        input_size (int): Number of input features per timestep (default=2 for x,y).
        hidden_size (int): Number of hidden units in the GRU.
        output_size (int): Number of output features per timestep (default=2 for x,y).
        num_layers (int): Number of stacked GRU layers.

    Forward Pass:
        - Encodes past LOOK_BACK timesteps using a GRU encoder.
        - Decodes step by step autoregressively to generate FORWARD_LEN future coordinates.
    """

    def __init__(self, input_size=3, hidden_size=128, output_size=3, num_layers=1):
        super(TrajPredictor, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Apply Xavier initialization
        self._init_weights()

    def _init_weights(self):
        # Initialize GRU weights
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Initialize Linear layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input sequence of shape (batch, LOOK_BACK, input_size).
            pred_len (int): Number of future steps to predict.

        Returns:
            torch.Tensor: Predicted future sequence of shape (batch, future_len, output_size).
        """
        # Encode past trajectory
        _, h = self.gru(x)

        # Squeeze the first dimension to get (batch_size, hidden_size)
        final_hidden_state = h.squeeze(0)

        # Pass the final hidden state through a linear layer to get the single output
        output = self.fc(final_hidden_state)

        return output
