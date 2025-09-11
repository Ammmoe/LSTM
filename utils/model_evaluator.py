"""
metrics.py

Provides evaluation utilities for trajectory prediction models.

This module includes functions to compute common regression metrics such as
Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute
Error (MAE) between predicted and ground-truth trajectories. It supports
inverse scaling to restore values to their original units (e.g., meters),
and computes both overall and per-axis metrics.
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
    Compute regression metrics for predicted 3D trajectories.

    Args:
        y_true (torch.Tensor): Ground-truth trajectory of shape (batch_size, seq_len, 3).
        y_pred (torch.Tensor): Predicted trajectory of shape (batch_size, seq_len, 3).
        scaler (MinMaxScaler): Fitted scaler used to inverse-transform predictions
                            back to original units (e.g., meters).

    Returns:
        tuple:
            - mse (float): Mean Squared Error over all points.
            - rmse (float): Root Mean Squared Error over all points.
            - mae (float): Mean Absolute Error over all points.
            - ede (float): Mean Euclidean distance error over all points.
            - axis_mse (tuple[float, float, float]): MSE per axis (x, y, z).
            - axis_rmse (tuple[float, float, float]): RMSE per axis (x, y, z).
            - axis_mae (tuple[float, float, float]): MAE per axis (x, y, z).

    Notes:
        - The function reshapes tensors, applies inverse scaling, and converts
        results back to torch tensors for metric computation.
        - Supports computing both overall metrics and per-axis metrics for 3D trajectories.
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
