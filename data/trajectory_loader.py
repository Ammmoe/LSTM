"""
trajectory_loader.py

Utilities for loading and preprocessing 3D trajectory data from CSV files
for use in trajectory prediction models.

Supports multiple datasets, including:
- Quadcopter flights (multiple trajectories per file)
- Zurich flight dataset (single trajectory)

Provides conversion from latitude/longitude to relative meters.
"""

import pandas as pd
import numpy as np
from utils.coordinate_converter import latlon_to_meters


def load_quadcopter_trajectories_in_meters(csv_path: str):
    """
    Load 3D quadcopter trajectories from CSV and convert positions to meters.

    Each trajectory corresponds to a unique flight and retains its full length.
    The first point of each flight is used as the reference origin for coordinate conversion.

    Args:
        csv_path (str): Path to CSV file containing columns
                        ['flight', 'position_x', 'position_y', 'position_z'].

    Returns:
        trajectories (list of np.ndarray): List of trajectories, each of shape (traj_len, 3) in meters.
        n_samples (int): Number of unique flights in the CSV.
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


def load_zurich_single_utm_trajectory(csv_path: str):
    """
    Load a single 3D trajectory from the Zurich flight dataset.

    Converts latitude/longitude to relative meters using the first point as reference
    and adjusts altitude so that the trajectory starts at z=0.

    Args:
        csv_path (str): Path to CSV file containing columns ['lat', 'lon', 'alt'].

    Returns:
        trajectory (list of np.ndarray): List containing a single trajectory of shape (traj_len, 3) in meters.
        n_samples (int): Always 1, since this dataset contains only one trajectory.
    """

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # remove leading/trailing whitespace
    df = df[["lat", "lon", "alt"]]

    # Use the first row as reference
    ref_lat = df["lat"].iloc[0]
    ref_lon = df["lon"].iloc[0]
    # ref_alt = df["alt"].iloc[0]

    # Convert lat/lon → meters relative to first point
    x, y = latlon_to_meters(
        df["lat"].values,
        df["lon"].values,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
    )

    # Altitude relative to first points
    # z = df["alt"].values - ref_alt
    z = df["alt"].values

    trajectory = np.stack([np.asarray(x), np.asarray(y), np.asarray(z)], axis=1)

    return [trajectory], 1
