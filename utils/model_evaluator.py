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
) -> Tuple[float, float, float]:
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
    # reshape for inverse transform
    y_true_np = y_true.reshape(-1, 4).cpu().numpy()
    y_pred_np = y_pred.reshape(-1, 4).cpu().numpy()

    # inverse transform
    y_true_inv = scaler.inverse_transform(y_true_np)[..., :3]  # only (x,y,z)
    y_pred_inv = scaler.inverse_transform(y_pred_np)[..., :3]

    # back to torch tensors for metric computation
    y_true_t = torch.tensor(y_true_inv, dtype=torch.float32)
    y_pred_t = torch.tensor(y_pred_inv, dtype=torch.float32)

    mse = torch.mean((y_true_t - y_pred_t) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(y_true_t - y_pred_t))
    return mse.item(), rmse.item(), mae.item()
