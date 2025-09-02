"""
trajectory_generator.py

This module generates smooth 2D trajectories using sine and cosine patterns 
with random amplitude and optional noise.
"""
import numpy as np


def generate_sine_cosine_trajectories(
    n_samples: int = 100, traj_len: int = 200, noise_scale: float = 0.02
) -> np.ndarray:
    """
    Generate smooth 2D trajectories using sine and cosine patterns with random amplitude.

    Args:
        n_samples (int): Number of trajectories to generate.
        traj_len (int): Number of timesteps per trajectory.
        noise_scale (float): Standard deviation of Gaussian noise to add.

    Returns:
        np.ndarray: Array of shape (n_samples, traj_len, 2) containing 2D trajectories.
    """
    data = np.zeros((n_samples, traj_len, 2), dtype=np.float32)

    for i in range(n_samples):
        # Random amplitude for x and y
        amplitude_x = np.random.uniform(0.5, 2.0)
        amplitude_y = np.random.uniform(0.5, 2.0)

        # Random frequency and phase
        freq_x = np.random.uniform(0.01, 0.05)
        freq_y = np.random.uniform(0.01, 0.05)
        phase_x = np.random.uniform(0, 2 * np.pi)
        phase_y = np.random.uniform(0, 2 * np.pi)

        t = np.arange(traj_len)
        x = amplitude_x * np.sin(2 * np.pi * freq_x * t + phase_x)
        y = amplitude_y * np.cos(2 * np.pi * freq_y * t + phase_y)

        # Add small Gaussian noise
        x += np.random.randn(traj_len) * noise_scale
        y += np.random.randn(traj_len) * noise_scale

        data[i, :, 0] = x
        data[i, :, 1] = y

    return data
