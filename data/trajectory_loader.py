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
from utils.coordinate_converter import latlon_to_meters


def load_quadcopter_trajectories_in_meters(csv_path: str):
    """
    Load 3D quadcopter trajectories from CSV and convert positions to meters.
    Returns a list of full-length trajectories (no truncation).

    Args:
        csv_path (str): Must contain ['flight', 'position_x', 'position_y', 'position_z']

    Returns:
        trajectories (list of np.ndarray): Each trajectory is (traj_len, 3) in meters
        n_samples (int): Number of flights
    """
    df = pd.read_csv(csv_path)
    flight_ids = df["flight"].unique()
    n_samples = len(flight_ids)

    trajectories = []
    for fid in flight_ids:
        traj_df = df[df["flight"] == fid][["position_x", "position_y", "position_z"]]

        # Reference = first point of THIS flight
        ref_lat = traj_df["position_y"].iloc[0]
        ref_lon = traj_df["position_x"].iloc[0]

        x, y = latlon_to_meters(
            traj_df["position_y"].values,
            traj_df["position_x"].values,
            ref_lat=ref_lat,
            ref_lon=ref_lon,
        )
        z = traj_df["position_z"].values
        traj_meters = np.stack([x, y, z], axis=1)
        trajectories.append(traj_meters)

    return trajectories, n_samples
