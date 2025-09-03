"""
trajectory_generator.py

This module generates smooth 2D and 3D trajectories using sine and cosine patterns
with random amplitude, frequency and optional noise.
"""

from typing import cast, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_2d_trajectory(
    *trajectories,
    labels=None,
    colors=None,
    title="2D Trajectory Plot",
    figsize=(6, 6),
    save_path: str | None = None,
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

    # save figure if path provided
    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_3d_trajectory(
    *trajectories,
    labels=None,
    colors=None,
    title="3D Trajectory Plot",
    figsize=(8, 6),
    save_path: str | None = None,
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

    # save figure if path provided
    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_3d_trajectories_subplots(
    trajectory_sets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    labels: List[str] | None = None,
    colors: List[str] | None = None,
    title: str = "3D Trajectory Predictions (Random Examples)",
    figsize: Tuple[int, int] = (15, 5),
    save_path: str | None = None,
) -> None:
    """
    Plot multiple 3D trajectory sets as subplots in one figure.

    Args:
        trajectory_sets (list): List of tuples, each containing (past, true_line, pred_line),
                                where each is np.ndarray of shape (traj_len, 3).
        labels (list): Labels for each line in a subplot.
        colors (list): Colors for each line in a subplot.
        title (str): Overall figure title.
        figsize (tuple): Figure size.
    """
    num_plots = len(trajectory_sets)
    fig = plt.figure(figsize=figsize)

    for i, (past, true_line, pred_line) in enumerate(trajectory_sets, 1):
        ax = cast(Axes3D, fig.add_subplot(1, num_plots, i, projection="3d"))
        past_color = colors[0] if colors else "b"
        true_color = colors[1] if colors else "g"
        pred_color = colors[2] if colors else "r"
        past_label = labels[0] if labels else "Past"
        true_label = labels[1] if labels else "True"
        pred_label = labels[2] if labels else "Predicted"
        ax.plot(past[:, 0], past[:, 1], past[:, 2], f"{past_color}.-", label=past_label)
        ax.plot(
            true_line[:, 0],
            true_line[:, 1],
            true_line[:, 2],
            f"{true_color}.-",
            label=true_label,
        )
        ax.plot(
            pred_line[:, 0],
            pred_line[:, 1],
            pred_line[:, 2],
            f"{pred_color}.-",
            label=pred_label,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Trajectory {i}")
        ax.legend()

    plt.suptitle(title)
    plt.tight_layout()

    # save figure if path provided
    if save_path:
        plt.savefig(save_path)

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
