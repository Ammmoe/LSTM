"""
train.py

This script trains and evaluates a sequence-to-sequence LSTM model for trajectory prediction.
It uses synthetic sine/cosine-based trajectories to simulate smooth 3D motion data.

Main steps:
- Generate synthetic 3D trajectory data with optional noise.
- Transform trajectories into past (LOOK_BACK) and future (FORWARD_LEN) sequences.
- Split data into training and testing sets.
- Train the Seq2Seq LSTM model defined in `models/lstm_predictor.py` using MSE loss.
- Evaluate model performance on the test set.
- Visualize sample predicted trajectories against ground truth using 3D plots.
"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from data.trajectory_generator import generate_sine_cosine_trajectories_3d
from utils.visualization import plot_3d_trajectories_subplots
from models.lstm_predictor import TrajPredictor
from utils.logger import get_logger

# pylint: disable=all
# Data parameters
LOOK_BACK = 50  # past frames
FORWARD_LEN = 10  # future frames
FEATURES = 2  # x,y coords
N_SAMPLES = 100
TRAJ_LEN = 200
NOISE_SCALE = 0.05

# Model parameters
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MODEL_INPUT_SIZE = 3
MODEL_HIDDEN_SIZE = 128
MODEL_OUTPUT_SIZE = 3
MODEL_NUM_LAYERS = 1

# Plotting parameters
NUM_PLOTS = 3

# Setup logger and experiment folder
logger, exp_dir = get_logger()

logger.info("Experiment started")
logger.info(f"Experiment folder: {exp_dir}")

# (n_samples, traj_len, 2)
data_3d = generate_sine_cosine_trajectories_3d(
    n_samples=N_SAMPLES, traj_len=TRAJ_LEN, noise_scale=NOISE_SCALE
)

X, y = [], []
for traj in data_3d:
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

# Log dataset sizes
logger.info(f"Total sequences: {X.shape[0]}")
logger.info(f"Train sequences: {X_train_tensor.shape}")
logger.info(f"Test sequences: {X_test_tensor.shape}")


# Train Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrajPredictor(
    input_size=MODEL_INPUT_SIZE,
    hidden_size=MODEL_HIDDEN_SIZE,
    output_size=MODEL_OUTPUT_SIZE,
    num_layers=MODEL_NUM_LAYERS,
).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

    avg_train_loss = total_loss / len(train_loader)

    # Log per-epoch training metrics
    logger.info(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {avg_train_loss:.6f}")

# Evaluate Model
model.eval()
test_loss = 0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x, future_len=FORWARD_LEN)
        loss = criterion(outputs, batch_y)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)

# Log final test metrics
logger.info(f"Test Loss: {avg_test_loss:.6f}")

# Save trained model
torch.save(model.state_dict(), os.path.join(exp_dir, "model.pt"))
logger.info("Model saved")

# Save config / hyperparameters
config = {
    "device": str(device),
    "model_module": model.__class__.__module__,
    "model_class": model.__class__.__name__,
    "model_input_size": MODEL_INPUT_SIZE,
    "model_hidden_size": MODEL_HIDDEN_SIZE,
    "model_output_size": MODEL_OUTPUT_SIZE,
    "model_num_layers": MODEL_NUM_LAYERS,
    "LOOK_BACK": LOOK_BACK,
    "FORWARD_LEN": FORWARD_LEN,
    "FEATURES": FEATURES,
    "EPOCHS": EPOCHS,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": LEARNING_RATE,
    "N_SAMPLES": N_SAMPLES,
    "TRAJ_LEN": TRAJ_LEN,
    "NOISE_SCALE": NOISE_SCALE,
    "NUM_PLOTS": NUM_PLOTS,
}

config_path = os.path.join(exp_dir, "config.txt")
with open(config_path, "w") as f:
    for k, v in config.items():
        f.write(f"{k}: {v}\n")

logger.info("Config saved")

# Visualize prediction for three random test sequences
random_test_indices = np.random.choice(len(X_test_tensor), NUM_PLOTS, replace=False)

trajectory_sets = []
for idx in random_test_indices:
    test_input = X_test_tensor[idx : idx + 1].to(device)  # shape (1, LOOK_BACK, 2)
    true_future = y_test_tensor[idx].numpy()  # shape (FORWARD_LEN, 2)

    with torch.no_grad():
        pred_future = model(test_input, future_len=FORWARD_LEN).cpu().numpy()

    past = test_input[0].cpu().numpy()  # shape (LOOK_BACK, 2)
    pred_future = pred_future[0]  # shape (FORWARD_LEN, 2)

    # Concatenate last past point with future to make continuous lines
    true_line = np.vstack([past[-1:], true_future])
    pred_line = np.vstack([past[-1:], pred_future])

    trajectory_sets.append((past, true_line, pred_line))

# Plot actual vs predicted test trajectory
plot_path = os.path.join(exp_dir, "trajectory_plot.png")
plot_3d_trajectories_subplots(trajectory_sets, save_path=plot_path)
