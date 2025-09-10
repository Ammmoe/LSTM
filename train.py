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
from utils.plot_generator import plot_3d_pred_vs_true, plot_3d_trajectories_subplots
from utils.model_evaluator import evaluate_metrics
from data.trajectory_loader import (
    load_quadcopter_trajectories_in_meters,
    load_zurich_single_utm_trajectory,
)

# pylint: disable=all
# Settings
DATA_TYPE = "zurich"  # "artificial" or "quadcopter" or "zurich"

# Data parameters
LOOK_BACK = 50  # past frames
FORWARD_LEN = 2  # future frames

# Initialize variables
N_SAMPLES = 0
data_3d = None

# Training parameters
EPOCHS = 500
BATCH_SIZE = 70
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

elif DATA_TYPE == "zurich":
    logger.info("Using Zurich flight dataset")

    # Load trajectories
    data_3d, N_SAMPLES = load_zurich_single_utm_trajectory(
        "data/zurich_flights_downsampled_2.csv"
    )
    print("Loaded Zurich dataset with %d samples" % N_SAMPLES)

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
    seq_count = len(trajectory) - LOOK_BACK - FORWARD_LEN + 1
    for i in range(seq_count):
        X.append(trajectory[i : i + LOOK_BACK])
        y.append(trajectory[i + LOOK_BACK + FORWARD_LEN - 1])

X = np.array(X, dtype=np.float32)  # shape (num_sequences, LOOK_BACK, 3)
y = np.array(y, dtype=np.float32)  # shape (num_sequences, 3)

# Split sequences into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# Scale data to [0, 1]
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

# Fit on TRAIN ONLY (flatten sequences)
scaler_X.fit(X_train.reshape(-1, 3))
scaler_y.fit(y_train)

X_train_scaled = scaler_X.transform(X_train.reshape(-1, 3)).reshape(X_train.shape)
X_test_scaled = scaler_X.transform(X_test.reshape(-1, 3)).reshape(X_test.shape)

y_train_scaled = scaler_y.transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

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
    "input_size": 3,  # x, y, z
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

# Early stopping parameters
patience = 15
best_loss = float("inf")
epochs_no_improve = 0
early_stop = False

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Log per-epoch training metrics
    logger.info("Epoch %d/%d - Train Loss: %.7f", epoch + 1, EPOCHS, avg_train_loss)

    # Check for early stopping
    if avg_train_loss < best_loss:
        best_loss = avg_train_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pt"))
    else:
        epochs_no_improve += 1

    # If no improvement for 'patience' epochs, stop training
    if epochs_no_improve >= patience:
        logger.info("Early stopping triggered after %d epochs", epoch + 1)
        early_stop = True
        break

# If training completed without early stopping
if not early_stop:
    logger.info("Training finished without early stopping.")

# Save last-epoch model
torch.save(model.state_dict(), os.path.join(exp_dir, "last_model.pt"))

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
total_sequences = 0

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_size = batch_x.size(0)
        total_sequences += batch_size  # accumulate total sequences

        start_inf = time.time()
        outputs = model(batch_x)
        end_inf = time.time()

        # Record inference time per batch
        inference_times.append(end_inf - start_inf)

        # Compute loss
        loss = criterion(outputs, batch_y)
        test_loss += loss.item()

        # Collect predictions and ground truth
        all_preds.append(outputs.cpu())
        all_trues.append(batch_y.cpu())

# Compute average test loss
avg_test_loss = test_loss / len(test_loader)

# Calculate total sequences and total inference time
total_inf_time = sum(inference_times)

# Average inference time per sequence
avg_inf_time_per_sequence = total_inf_time / total_sequences

# Average inference time per batch
avg_inf_time_per_batch = total_inf_time / len(test_loader)

# Log final test metrics
logger.info("Test Loss (scaled): %.6f", avg_test_loss)
logger.info(
    "Average inference time per sequence: %.6f seconds", avg_inf_time_per_sequence
)
logger.info("Average inference time per batch: %.6f seconds", avg_inf_time_per_batch)

# Concatenate all batches
y_pred = torch.cat(all_preds, dim=0)
y_true = torch.cat(all_trues, dim=0)

# Compute evaluation metrics (inverse scaling applied)
mse, rmse, mae = evaluate_metrics(y_true, y_pred, scaler_y)

# Log all metrics
logger.info("Test MSE: %.6f meters^2", mse)
logger.info("Test RMSE: %.6f meters", rmse)
logger.info("Test MAE: %.6f meters", mae)

# Save config / hyperparameters
config = {
    "device": str(device),
    "model_module": model.__class__.__module__,
    "model_class": model.__class__.__name__,
    "model_params": model_params,
    "LOOK_BACK": LOOK_BACK,
    "FORWARD_LEN": FORWARD_LEN,
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

trajectory_sets = []

# Prepare data for plotting
y_pred_np = y_pred.numpy()
y_true_np = y_true.numpy()

# Inverse scales to original coordinates for plotting
true_future_orig = scaler_y.inverse_transform(y_true_np)
pred_future_orig = scaler_y.inverse_transform(y_pred_np)

# Prepare trajectory set (no past points)
trajectory_sets = [(true_future_orig, pred_future_orig)]

# Plot actual vs predicted test trajectory
plot_path = os.path.join(exp_dir, "trajectory_plot_overall.png")

# Define labels based on model type
labels = []
if config["model_module"] == "models.lstm_predictor":
    labels = ["Actual Trajectory", "LSTM"]
elif config["model_module"] == "models.gru_predictor":
    labels = ["Actual Trajectory", "GRU"]
elif config["model_module"] == "models.rnn_predictor":
    labels = ["Actual Trajectory", "RNN"]
else:
    labels = ["Actual Trajectory", "Predicted Trajectory"]

# Generate and save plot
plot_3d_pred_vs_true(
    true_future_orig, pred_future_orig, labels=labels, save_path=plot_path
)

# Plot individual trajectories with past, true future, and predicted future
# Make sure NUM_PLOTS does not exceed available test samples
NUM_PLOTS = min(NUM_PLOTS, len(X_test_tensor))

# Select evenly spaced indices from test set for non-overlapping sequences
step = LOOK_BACK + FORWARD_LEN
indices = np.arange(0, len(X_test_tensor), step)

# Visualize prediction for three random test sequences
random_test_indices = np.random.choice(indices, NUM_PLOTS, replace=False)

trajectory_sets_2 = []
for idx in random_test_indices:
    test_input = X_test_tensor[idx : idx + 1].to(device)  # shape (1, LOOK_BACK, 3)
    true_future = y_test_tensor[idx].numpy()  # shape (3,)

    with torch.no_grad():
        pred_future = model(test_input, pred_len=1).cpu().numpy()  # shape (1, 3)

    past = test_input[0].cpu().numpy()  # shape (LOOK_BACK, 3)
    pred_future = pred_future[0]  # shape (3,)

    # Inverse transform to original scale
    past_orig = scaler_X.inverse_transform(past)
    true_future_orig_2 = scaler_y.inverse_transform(true_future.reshape(1, -1))
    pred_future_orig_2 = scaler_y.inverse_transform(pred_future.reshape(1, -1))

    # Concatenate last past point with future to make continuous lines
    true_line = np.vstack([past_orig[-1:], true_future_orig_2])
    pred_line = np.vstack([past_orig[-1:], pred_future_orig_2])

    trajectory_sets_2.append((past_orig, true_line, pred_line))

# Plot actual vs predicted test trajectory
plot_path = os.path.join(exp_dir, "trajectory_plot_subplots.png")
plot_3d_trajectories_subplots(trajectory_sets_2, save_path=plot_path)
