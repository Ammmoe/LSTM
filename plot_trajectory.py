"""
trajectory_generator.py

This module generates smooth 2D and 3D trajectories using sine and cosine patterns
with random amplitude and optional noise.
"""

from typing import cast
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_2d_trajectory(
    *trajectories, labels=None, colors=None, title="2D Trajectory Plot", figsize=(6, 6)
):
    """
    Plot 2D trajectories.

    Args:
        *trajectories: One or more np.ndarray of shape (traj_len, 2)
        labels (list): Labels for each trajectory
        colors (list): Colors for each trajectory
        title (str): Plot title
        figsize (tuple): Figure size
    """
    _, ax = plt.subplots(figsize=figsize)

    for i, traj in enumerate(trajectories):
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            f"{colors[i] if colors else '.'}-",
            label=labels[i] if labels else None,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    if labels:
        ax.legend()
    plt.show()


def plot_3d_trajectory(
    *trajectories, labels=None, colors=None, title="3D Trajectory Plot", figsize=(8, 6)
):
    """
    Plot 3D trajectories.

    Args:
        *trajectories: One or more np.ndarray of shape (traj_len, 3)
        labels (list): Labels for each trajectory
        colors (list): Colors for each trajectory
        title (str): Plot title
        figsize (tuple): Figure size
    """
    fig = plt.figure(figsize=figsize)
    ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))
    for i, traj in enumerate(trajectories):
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            traj[:, 2],
            f"{colors[i] if colors else '.'}-",
            label=labels[i] if labels else None,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    if labels:
        ax.legend()
    plt.show()


# Example usage:

# plot_2d_trajectory(past, true_line, pred_line,
#                    labels=["Past", "True", "Predicted"],
#                    colors=["b", "g", "r"],
#                    title="2D Trajectory Prediction")

# plot_3d_trajectory(traj_3d_sample1, traj_3d_sample2,
#                    labels=["Trajectory 1", "Trajectory 2"],
#                    colors=["b", "r"],
#                    title="3D Trajectories")
