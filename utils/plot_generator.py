"""
trajectory_generator.py

This module generates smooth 2D and 3D trajectories using sine and cosine patterns
with random amplitude, frequency and optional noise.
"""

from typing import cast, List, Tuple
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator


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
    figsize: Tuple[int, int] = (15, 10),
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
    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    for i, (past, true_line, pred_line) in enumerate(trajectory_sets, 1):
        ax = cast(Axes3D, fig.add_subplot(rows, cols, i, projection="3d"))
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


def plot_3d_geo_trajectories(
    trajectory_sets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    labels: List[str] | None = None,
    colors: List[str] | None = None,
    title: str = "3D Trajectory Predictions (Lat/Lon/Alt)",
    figsize: Tuple[int, int] = (15, 10),
    lat_grid: float = 0.005,
    lon_grid: float = 0.005,
    alt_grid: float = 1.0,
    save_path: str | None = None,
) -> None:
    """
    Plot multiple 3D trajectory sets as subplots with custom scaling for
    latitude, longitude, and altitude axes.
    """
    num_plots = len(trajectory_sets)
    fig = plt.figure(figsize=figsize)
    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    for i, (past, true_line, pred_line) in enumerate(trajectory_sets, 1):
        ax = cast(Axes3D, fig.add_subplot(rows, cols, i, projection="3d"))

        # Colors and labels
        past_color, true_color, pred_color = colors or ["b", "g", "r"]
        past_label, true_label, pred_label = labels or ["Past", "True", "Predicted"]

        # Plot lines
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

        # Axis labels
        ax.set_xlabel("Latitude / degree")
        ax.set_ylabel("Longitude / degree")
        ax.set_zlabel("Altitude / meter")
        ax.set_title(f"Trajectory {i}")
        ax.legend()

        # Compute limits
        lat_min = min(past[:, 0].min(), true_line[:, 0].min(), pred_line[:, 0].min())
        lat_max = max(past[:, 0].max(), true_line[:, 0].max(), pred_line[:, 0].max())
        lon_min = min(past[:, 1].min(), true_line[:, 1].min(), pred_line[:, 1].min())
        lon_max = max(past[:, 1].max(), true_line[:, 1].max(), pred_line[:, 1].max())
        alt_min = float(
            min(past[:, 2].min(), true_line[:, 2].min(), pred_line[:, 2].min())
        )
        alt_max = float(
            max(past[:, 2].max(), true_line[:, 2].max(), pred_line[:, 2].max())
        )

        # Set axis limits
        ax.set_xlim(lat_min - lat_grid, lat_max + lat_grid)
        ax.set_ylim(lon_min - lon_grid, lon_max + lon_grid)
        ax.set_zlim(alt_min - alt_grid, alt_max + alt_grid)

        # Set custom grid ticks
        ax.set_xticks(np.arange(lat_min, lat_max + lat_grid, lat_grid))
        ax.set_yticks(np.arange(lon_min, lon_max + lon_grid, lon_grid))
        ax.zaxis.set_ticks(np.arange(alt_min, alt_max + alt_grid, alt_grid))

        # Limit number of ticks to avoid clutter
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))  # max 10 ticks
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=10))

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_3d_pred_vs_true(
    true_coords: np.ndarray,
    pred_coords: np.ndarray,
    labels: List[str] | None = None,
    colors: List[str] | None = None,
    title: str = "3D Trajectory: True vs Predicted",
    figsize: Tuple[int, int] = (10, 8),
    save_path: str | None = None,
) -> None:
    """
    Plot a 3D trajectory comparing true vs predicted coordinates.

    Args:
        true_coords (np.ndarray): Array of shape (num_points, 3) with true coordinates.
        pred_coords (np.ndarray): Array of shape (num_points, 3) with predicted coordinates.
        labels (list): Labels for the lines (default ["True", "Predicted"]).
        colors (list): Colors for the lines (default ["g", "r"]).
        title (str): Plot title.
        figsize (tuple): Figure size.
        save_path (str): If provided, saves the plot to this path.
    """
    labels = labels or ["Actual Trajectory", "GRU"]
    colors = colors or ["r"]

    fig = plt.figure(figsize=figsize)
    ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))

    ax.plot(
        true_coords[:, 0],
        true_coords[:, 1],
        true_coords[:, 2],
        color='k',        # black
        linestyle='--',   # dashed
        linewidth=1.5,   # thinner line
        marker='o',
        markersize=2,    # smaller markers
        label=labels[0],
    )
    ax.plot(
        pred_coords[:, 0],
        pred_coords[:, 1],
        pred_coords[:, 2],
        color=colors[0],
        linewidth=1.5,   # thinner line
        marker='o',
        markersize=2,    # smaller markers
        label=labels[1],
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

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
