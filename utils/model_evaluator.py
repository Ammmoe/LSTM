"""
metrics.py

This module provides evaluation utilities for trajectory prediction models.
It includes functions to compute error metrics (MSE, RMSE, MAE) between predicted
and ground-truth trajectories, with support for inverse scaling to restore values
to their original units (e.g., meters).
"""

from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
import torch


def evaluate_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor, scaler: MinMaxScaler
) -> Tuple[
    float,
    float,
    float,
    float,
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
]:
    """
    Compute regression evaluation metrics (MSE, RMSE, MAE) for trajectory predictions.

    Args:
        y_true (torch.Tensor): Ground-truth values with shape (batch_size, seq_len, 3).
        y_pred (torch.Tensor): Predicted values with shape (batch_size, seq_len, 3).
        scaler (MinMaxScaler): Fitted scaler used to inverse-transform data
                                back to original units (e.g., meters).

    Returns:
        Tuple[float, float, float]:
            - mse: Mean Squared Error (in original units²).
            - rmse: Root Mean Squared Error (in original units).
            - mae: Mean Absolute Error (in original units).

    Notes:
        - The function automatically reshapes tensors, performs inverse scaling,
            and converts results back into torch for metric computation.
        - All returned metrics are scalar floats.
    """
    # Get feature dimensions dynamically
    num_features_y = y_true.shape[-1]  # 3 or 4 depending on USE_TIME_FEATURE

    # reshape for inverse transform
    y_true_np = y_true.reshape(-1, num_features_y).cpu().numpy()
    y_pred_np = y_pred.reshape(-1, num_features_y).cpu().numpy()

    # inverse transform
    y_true_inv = scaler.inverse_transform(y_true_np)[..., :3]  # only (x,y,z)
    y_pred_inv = scaler.inverse_transform(y_pred_np)[..., :3]

    # back to torch tensors for metric computation
    y_true_t = torch.tensor(y_true_inv, dtype=torch.float32)
    y_pred_t = torch.tensor(y_pred_inv, dtype=torch.float32)

    # Per-axis metrics
    axis_errors = y_true_t - y_pred_t
    axis_mse = torch.mean(axis_errors**2, dim=0)  # shape (3,)
    axis_rmse = torch.sqrt(axis_mse)
    axis_mae = torch.mean(torch.abs(axis_errors), dim=0)

    mse_x, mse_y, mse_z = axis_mse.tolist()
    rmse_x, rmse_y, rmse_z = axis_rmse.tolist()
    mae_x, mae_y, mae_z = axis_mae.tolist()

    # Overall metrics
    mse = torch.mean((y_true_t - y_pred_t) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(y_true_t - y_pred_t))

    # Euclidean distance error (mean over all points)
    ede = torch.mean(torch.sqrt(torch.sum((y_true_t - y_pred_t) ** 2, dim=1)))

    return (
        mse.item(),
        rmse.item(),
        mae.item(),
        ede.item(),
        (mse_x, mse_y, mse_z),
        (rmse_x, rmse_y, rmse_z),
        (mae_x, mae_y, mae_z),
    )
