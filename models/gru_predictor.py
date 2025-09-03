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
        self.encoder = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.GRU(output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x, future_len=10):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input sequence of shape (batch, LOOK_BACK, input_size).
            future_len (int): Number of future steps to predict.

        Returns:
            torch.Tensor: Predicted future sequence of shape (batch, future_len, output_size).
        """
        # Encode past trajectory
        _, h = self.encoder(x)

        # First decoder input = last input point
        decoder_input = x[:, -1:, :]  # shape (batch, 1, input_size)
        outputs = []

        # Autoregressive decoding
        for _ in range(future_len):
            out, h = self.decoder(decoder_input, h)
            pred = self.fc(out)  # (batch, 1, output_size)
            outputs.append(pred)
            decoder_input = pred  # feed prediction back

        outputs = torch.cat(outputs, dim=1)  # (batch, future_len, output_size)
        return outputs
