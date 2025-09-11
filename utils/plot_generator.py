"""
plot_generator.py

Provides utilities for plotting 2D and 3D trajectories, including multiple
trajectory sets with optional subplots, labels, colors, and axis scaling.

Functions include:
- plot_2d_trajectory: Plot one or more 2D trajectories.
- plot_3d_trajectory: Plot one or more 3D trajectories.
- plot_3d_trajectories_subplots: Plot multiple sets of 3D trajectories in subplots.
- plot_3d_geo_trajectories: Plot 3D trajectories with latitude/longitude/altitude axes.
- plot_3d_pred_vs_true: Compare predicted vs true 3D trajectories in a single plot.

All plotting functions support optional figure saving.
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
    Plot one or more 2D trajectories.

    Args:
        *trajectories: One or more np.ndarray of shape (traj_len, 2).
        labels (list, optional): Labels for each trajectory.
        colors (list, optional): Colors for each trajectory.
        title (str): Plot title.
        figsize (tuple): Figure size.
        save_path (str, optional): Path to save the figure.
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
    Plot one or more 3D trajectories.

    Args:
        *trajectories: One or more np.ndarray of shape (traj_len, 3).
        labels (list, optional): Labels for each trajectory.
        colors (list, optional): Colors for each trajectory.
        title (str): Plot title.
        figsize (tuple): Figure size.
        save_path (str, optional): Path to save the figure.
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
    Plot multiple 3D trajectory sets as subplots in a single figure.

    Args:
        trajectory_sets (list): List of tuples, each containing (past, true_line, pred_line),
                                each np.ndarray of shape (traj_len, 3).
        labels (list, optional): Labels for lines within each subplot.
        colors (list, optional): Colors for lines within each subplot.
        title (str): Overall figure title.
        figsize (tuple): Figure size.
        save_path (str, optional): Path to save the figure.
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
    Plot multiple 3D trajectory sets with latitude, longitude, and altitude axes.

    Args:
        trajectory_sets (list): List of tuples (past, true_line, pred_line) of shape (traj_len, 3).
        labels (list, optional): Labels for the lines within each subplot.
        colors (list, optional): Colors for the lines within each subplot.
        title (str): Overall figure title.
        figsize (tuple): Figure size.
        lat_grid (float): Grid spacing for latitude axis.
        lon_grid (float): Grid spacing for longitude axis.
        alt_grid (float): Grid spacing for altitude axis.
        save_path (str, optional): Path to save the figure.
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
    Plot a 3D trajectory comparing predicted vs true coordinates.

    Args:
        true_coords (np.ndarray): Array of shape (num_points, 3) with ground-truth coordinates.
        pred_coords (np.ndarray): Array of shape (num_points, 3) with predicted coordinates.
        labels (list, optional): Labels for true and predicted lines (default ["Actual Trajectory", "Predicted Trajectory"]).
        colors (list, optional): Colors for predicted line (default ["r"]).
        title (str): Plot title.
        figsize (tuple): Figure size.
        save_path (str, optional): Path to save the figure.
    """

    labels = labels or ["Actual Trajectory", "Predicted Trajectory"]
    colors = colors or ["r"]

    fig = plt.figure(figsize=figsize)
    ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))

    ax.plot(
        true_coords[:, 0],
        true_coords[:, 1],
        true_coords[:, 2],
        color="k",  # black
        linestyle="--",  # dashed
        linewidth=1,  # thinner line
        label=labels[0],
    )
    ax.plot(
        pred_coords[:, 0],
        pred_coords[:, 1],
        pred_coords[:, 2],
        color=colors[0],
        linewidth=1,  # thinner line
        label=labels[1],
    )

    ax.set_xlabel("Y")
    ax.set_ylabel("X")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
