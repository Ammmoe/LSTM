"""
lstm_model.py

This module implements a sequence-to-sequence LSTM model in PyTorch
for 2D trajectory prediction using sine and cosine generated trajectories.

- Generates training and test data using the `generate_sine_cosine_trajectories` function.
- Splits sequences into training and test sets.
- Defines a Seq2Seq LSTM model with an encoder-decoder architecture.
- Trains the model using MSE loss and visualizes predicted trajectories.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from trajectory_generator import generate_sine_cosine_trajectories

# pylint: disable=all
# Parameters
LOOK_BACK = 50  # past frames
FORWARD_LEN = 10  # future frames
FEATURES = 2  # x,y coords
EPOCHS = 10
BATCH_SIZE = 64

# Generate Dummy Data
n_samples = 100
traj_len = 200

data = generate_sine_cosine_trajectories(
    n_samples=n_samples, traj_len=traj_len, noise_scale=0.05
)

X, y = [], []
for traj in data:
    for i in range(len(traj) - LOOK_BACK - FORWARD_LEN):
        X.append(traj[i : i + LOOK_BACK])
        y.append(traj[i + LOOK_BACK : i + LOOK_BACK + FORWARD_LEN])

X = np.array(X)  # (num_sequences, LOOK_BACK, 2)
y = np.array(y)  # (num_sequences, FORWARD_LEN, 2)

# Split sequences into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Train sequences:", X_train_tensor.shape)
print("Test sequences:", X_test_tensor.shape)


# Define Seq2Seq LSTM Model
class TrajPredictor(nn.Module):
    """
    Sequence-to-sequence LSTM model for trajectory prediction.

    Args:
        input_size (int): Number of input features per timestep (default=2 for x,y).
        hidden_size (int): Number of hidden units in the LSTM.
        output_size (int): Number of output features per timestep (default=2 for x,y).
        num_layers (int): Number of stacked LSTM layers.

    Forward Pass:
        - Encodes past LOOK_BACK timesteps using an LSTM encoder.
        - Decodes step by step autoregressively to generate FORWARD_LEN future coordinates.
    """

    def __init__(self, input_size=2, hidden_size=128, output_size=2, num_layers=1):
        super(TrajPredictor, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
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
        _, (h, c) = self.encoder(x)

        # First decoder input = last input point
        decoder_input = x[:, -1:, :]  # shape (batch, 1, input_size)
        outputs = []

        # Autoregressive decoding
        for _ in range(future_len):
            out, (h, c) = self.decoder(decoder_input, (h, c))
            pred = self.fc(out)  # (batch, 1, output_size)
            outputs.append(pred)
            decoder_input = pred  # feed prediction back

        outputs = torch.cat(outputs, dim=1)  # (batch, future_len, output_size)
        return outputs


# Train Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrajPredictor().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        predictions = model(batch_x, future_len=FORWARD_LEN)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(train_loader):.6f}")

# Take the first test sequence
test_input = X_test_tensor[0:1].to(device)
true_future = y_test_tensor[0:1].cpu().numpy()

# Model prediction
model.eval()
test_loss = 0
pred_future = None
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x, future_len=FORWARD_LEN)
        loss = criterion(outputs, batch_y)
        test_loss += loss.item()

        # Save prediction for the very first test input
        if pred_future is None:
            pred_future = outputs.cpu().numpy()
            print("Saved pred_future shape:", pred_future.shape)

avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.6f}")

# Convert tensors to numpy
past = test_input[0].cpu().numpy()  # shape (LOOK_BACK, 2)
true_fut = y_test_tensor[0:1].cpu().numpy()[0]  # shape (FORWARD_LEN, 2)
if pred_future is None:
    with torch.no_grad():
        pred_future = model(test_input, future_len=FORWARD_LEN).cpu().numpy()
pred_fut = pred_future[0]  # shape (FORWARD_LEN, 2)

# Concatenate last past point with future to make continuous lines
true_line = np.vstack([past[-1:], true_fut])
pred_line = np.vstack([past[-1:], pred_fut])

# Plot actual vs predicted test trajectory
plt.figure(figsize=(6, 6))
plt.plot(past[:, 0], past[:, 1], "b.-", label="Past (50)")
plt.plot(true_line[:, 0], true_line[:, 1], "g.-", label="True Future (10)")
plt.plot(pred_line[:, 0], pred_line[:, 1], "r.-", label="Predicted Future (10)")
plt.legend()
plt.title("2D Trajectory Prediction with PyTorch LSTM")
plt.show()
