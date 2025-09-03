"""
trajectory_loader.py

This module provides utilities to load and preprocess 3D quadcopter trajectory data
from CSV files for use in trajectory prediction models.

Functions:
- load_quadcopter_trajectories(csv_path: str) -> tuple[np.ndarray, int]:
    Loads 3D trajectories for each unique flight in the CSV. Trajectories are truncated
    to the minimum length across all flights to ensure uniform sequence length.
    Returns both the trajectories array and the number of flights.

Usage Example:
    from quadcopter_data_loader import load_quadcopter_trajectories

    trajectories, n_samples = load_quadcopter_trajectories("quadcopter.csv")
"""

import pandas as pd
import numpy as np


def load_quadcopter_trajectories(csv_path: str) -> tuple[np.ndarray, int]:
    """
    Load 3D quadcopter trajectories from a CSV and return a uniform array of trajectories.

    Each trajectory corresponds to a unique flight. Trajectories are truncated to the
    minimum length among all flights to ensure consistent sequence length.

    Args:
        csv_path (str): Path to CSV file containing drone data. Must contain columns:
                        ['flight', 'position_x', 'position_y', 'position_z'].

    Returns:
        trajectories (np.ndarray): Array of shape (n_flights, traj_len, 3) containing 3D positions.
        n_samples (int): Number of flights (trajectories).
    """
    df = pd.read_csv(csv_path)
    flight_ids = df["flight"].unique()
    n_samples = len(flight_ids)

    # Determine minimum trajectory length across all flights
    min_len = df.groupby("flight").size().min()

    # Build list of trajectories, truncated to min_len
    trajectories = [
        df[df["flight"] == fid][["position_x", "position_y", "position_z"]].values[
            :min_len
        ]
        for fid in flight_ids
    ]

    return np.stack(trajectories), n_samples
