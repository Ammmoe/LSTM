"""
logger.py

Provides a utility function to create a timestamped experiment logger.

This module allows users to easily set up logging for machine learning experiments
or other scripts. It automatically creates an experiment folder with a timestamp
inside a specified root directory and configures a logger to write logs both
to a file and optionally to the console.

Functions:
- get_logger(exp_root="experiments", log_name="train.log")
    Creates a logger and an experiment folder, returning both for use in scripts.
"""

import logging
import os
from datetime import datetime


def get_logger(exp_root="experiments", log_name="train.log"):
    """
    Sets up a logger that writes to a timestamped experiment folder.

    Args:
        exp_root (str): Root folder for all experiments.
        log_name (str): Name of the log file.

    Returns:
        logger (logging.Logger): Configured logger.
        exp_dir (str): Path to the experiment folder.
    """
    # Ensure root experiments folder exists
    os.makedirs(exp_root, exist_ok=True)

    # Create a timestamped experiment folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(exp_root, timestamp)
    os.makedirs(exp_dir, exist_ok=True)

    # Setup logger
    logger = logging.getLogger(timestamp)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if this function is called multiple times
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(exp_dir, log_name))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Optional: also log to console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger, exp_dir
