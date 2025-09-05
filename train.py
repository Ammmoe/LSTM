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
import time
import json
import torch
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from data.artificial_trajectory_generator import generate_sine_cosine_trajectories_3d
from models.gru_predictor import TrajPredictor
from utils.logger import get_logger
from utils.plot_generator import plot_3d_trajectories_subplots
from utils.model_evaluator import evaluate_metrics
from data.trajectory_loader import load_quadcopter_trajectories_in_meters

# pylint: disable=all
# Settings
DATA_TYPE = "quadcopter"  # "artificial" or "quadcopter"

# Data parameters
LOOK_BACK = 50  # past frames
FORWARD_LEN = 10  # future frames
FEATURES = 3  # x, y, z

# Initialize variables
N_SAMPLES = 0
data_3d = None

# Training parameters
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# Plotting parameters
NUM_PLOTS = 6

# Setup logger and experiment folder
logger, exp_dir = get_logger()

# Ensure experiment folder exists
os.makedirs(exp_dir, exist_ok=True)

logger.info("Experiment started")
logger.info("Experiment folder: %s", exp_dir)

if DATA_TYPE == "quadcopter":
    logger.info("Using quadcopter trajectory data")

    # Load trajectories
    data_3d, N_SAMPLES = load_quadcopter_trajectories_in_meters(
        "data/quadcopter_flights.csv"
    )

else:
    logger.info("Using artificial sine/cosine trajectory data")

    # Generate synthetic 3D trajectory data
    N_SAMPLES = 100
    TRAJ_LEN = 200
    NOISE_SCALE = 0.05

    # (n_samples, traj_len, 2)
    data_3d = generate_sine_cosine_trajectories_3d(
        n_samples=N_SAMPLES, traj_len=TRAJ_LEN, noise_scale=NOISE_SCALE
    )

# Prepare DataLoaders
X, y = [], []
for trajectory in data_3d:
    for i in range(len(trajectory) - LOOK_BACK - FORWARD_LEN + 1):
        X.append(trajectory[i : i + LOOK_BACK])
        y.append(trajectory[i + LOOK_BACK : i + LOOK_BACK + FORWARD_LEN])

X = np.array(X, dtype=np.float32)  # shape (num_sequences, LOOK_BACK, 3)
y = np.array(y, dtype=np.float32)  # shape (num_sequences, FORWARD_LEN, 3)

# Split sequences into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Scale data to [0, 1]
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

# Fit on TRAIN ONLY (flatten sequences)
scaler_X.fit(X_train.reshape(-1, 3))
scaler_y.fit(y_train.reshape(-1, 3))

X_train_scaled = scaler_X.transform(X_train.reshape(-1, 3)).reshape(X_train.shape)
X_test_scaled = scaler_X.transform(X_test.reshape(-1, 3)).reshape(X_test.shape)

y_train_scaled = scaler_y.transform(y_train.reshape(-1, 3)).reshape(y_train.shape)
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 3)).reshape(y_test.shape)

# Convert to tensors
X_train_tensor = torch.tensor(
    X_train_scaled, dtype=torch.float32
)  # shape (num_samples, LOOK_BACK, 3)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# Now create DataLoaders
train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(X_test_tensor, y_test_tensor), batch_size=BATCH_SIZE, shuffle=False
)

# Log dataset sizes
total_sequences = X_train_tensor.shape[0] + X_test_tensor.shape[0]
logger.info("Total sequences: %d", total_sequences)
logger.info("Train sequences: %s", X_train_tensor.shape)
logger.info("Test sequences: %s", X_test_tensor.shape)

# Train Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_params = {
    "input_size": 3,
    "hidden_size": 64,
    "output_size": 3,
    "num_layers": 2,
}
model = TrajPredictor(**model_params).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Log model info
logger.info("Model module: %s", model.__class__.__module__)
logger.info("Model architecture:\n%s", model)

# Log time taken for training
training_start_time = time.time()

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
    logger.info("Epoch %d/%d - Train Loss: %.7f", epoch + 1, EPOCHS, avg_train_loss)

# Log total training time
training_end_time = time.time()
elapsed_time = training_end_time - training_start_time
logger.info("Total training time: %.2f seconds", elapsed_time)

# Evaluate Model
model.eval()
test_loss = 0
inference_times = []
all_preds = []
all_trues = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        start_inf = time.time()
        outputs = model(batch_x, future_len=FORWARD_LEN)
        end_inf = time.time()

        # Record inference time per batch
        inference_times.append(end_inf - start_inf)

        # Compute loss
        loss = criterion(outputs, batch_y)
        test_loss += loss.item()

        # Collect predictions and ground truth
        all_preds.append(outputs.cpu())
        all_trues.append(batch_y.cpu())

# Compute average test loss and average inference time per sequence
avg_test_loss = test_loss / len(test_loader)
total_inf_time = sum(inference_times)
avg_inf_time_per_sequence = total_inf_time / len(X_test_tensor)

# Log final test metrics
logger.info("Test Loss (scaled): %.6f", avg_test_loss)
logger.info(
    "Average inference time per sequence: %.6f seconds", avg_inf_time_per_sequence
)

# Concatenate all batches
y_pred = torch.cat(all_preds, dim=0)
y_true = torch.cat(all_trues, dim=0)

# Compute evaluation metrics (inverse scaling applied)
mse, rmse, mae = evaluate_metrics(y_true, y_pred, scaler_y)

# Log all metrics
logger.info("Test MSE: %.6f meters^2", mse)
logger.info("Test RMSE: %.6f meters", rmse)
logger.info("Test MAE: %.6f meters", mae)

# Save trained model
torch.save(model.state_dict(), os.path.join(exp_dir, "model.pt"))
logger.info("Model saved")

# Save config / hyperparameters
config = {
    "device": str(device),
    "model_module": model.__class__.__module__,
    "model_class": model.__class__.__name__,
    "model_params": model_params,
    "LOOK_BACK": LOOK_BACK,
    "FORWARD_LEN": FORWARD_LEN,
    "FEATURES": FEATURES,
    "EPOCHS": EPOCHS,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": LEARNING_RATE,
    "N_SAMPLES": N_SAMPLES,
    "NUM_PLOTS": NUM_PLOTS,
}

config_path = os.path.join(exp_dir, "config.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)

logger.info("Config saved")

# Make sure NUM_PLOTS does not exceed available test samples
NUM_PLOTS = min(NUM_PLOTS, len(X_test_tensor))

# Visualize prediction for three random test sequences
random_test_indices = np.random.choice(len(X_test_tensor), NUM_PLOTS, replace=False)

trajectory_sets = []
for idx in random_test_indices:
    test_input = X_test_tensor[idx : idx + 1].to(device)  # shape (1, LOOK_BACK, 3)
    true_future = y_test_tensor[idx].numpy()  # shape (FORWARD_LEN, 3)

    with torch.no_grad():
        pred_future = model(test_input, future_len=FORWARD_LEN).cpu().numpy()

    past = test_input[0].cpu().numpy()  # shape (LOOK_BACK, 3)
    pred_future = pred_future[0]  # shape (FORWARD_LEN, 3)

    # Inverse transform to original scale
    past_orig = scaler_X.inverse_transform(past)
    true_future_orig = scaler_y.inverse_transform(true_future)
    pred_future_orig = scaler_y.inverse_transform(pred_future)

    # Concatenate last past point with future to make continuous lines
    true_line = np.vstack([past_orig[-1:], true_future_orig])
    pred_line = np.vstack([past_orig[-1:], pred_future_orig])

    trajectory_sets.append((past_orig, true_line, pred_line))

# Plot actual vs predicted test trajectory
plot_path = os.path.join(exp_dir, "trajectory_plot.png")
plot_3d_trajectories_subplots(trajectory_sets, save_path=plot_path)
