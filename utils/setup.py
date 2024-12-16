"""
Experiment setup utilities for DETR.
"""

import logging
import os
from typing import Tuple

import torch.distributed as dist


class RankFormatter(logging.Formatter):
    """Custom formatter that includes rank information."""

    def format(self, record):
        # Add rank to the record if in distributed mode
        if dist.is_initialized():
            record.rank = dist.get_rank()
            # Check if the message already contains rank information
            if not record.msg.startswith("[Rank"):
                fmt = "[%(asctime)s][Rank %(rank)d] %(message)s"
            else:
                fmt = "[%(asctime)s]%(message)s"
        else:
            fmt = "[%(asctime)s] %(message)s"

        # Create a base formatter without milliseconds
        base_formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

        # Format the base message
        formatted = base_formatter.format(record)

        # Add milliseconds manually
        if hasattr(record, "created"):
            # Get milliseconds from the record's created timestamp
            msecs = int((record.created - int(record.created)) * 1000)
            # Find the timestamp in the formatted string and add milliseconds
            timestamp_end = formatted.find("]")
            if timestamp_end != -1:
                formatted = (
                    formatted[: timestamp_end - 1]
                    + f".{msecs:03d}"
                    + formatted[timestamp_end - 1 :]
                )

        return formatted


def setup_logging(
    log_dir: str, log_name: str = "train.log", log_level: int = logging.DEBUG
) -> None:
    """Set up logging configuration.

    Args:
        log_dir: Directory for log files
        log_name: Name of the log file (default: train.log)
        log_level: Logging level (default: logging.INFO)
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create rank-specific log file name
    rank = dist.get_rank() if dist.is_initialized() else 0
    rank_log_name = (
        f"{os.path.splitext(log_name)[0]}.rank{rank}{os.path.splitext(log_name)[1]}"
    )
    log_file = os.path.join(log_dir, rank_log_name)

    # Create handlers
    file_handler = logging.FileHandler(log_file, mode="w")
    console_handler = logging.StreamHandler()

    # Set levels
    file_handler.setLevel(log_level)
    console_handler.setLevel(log_level)

    # Create and set formatter
    formatter = RankFormatter()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Log initial setup message
    logging.info(f"Logging initialized for rank {rank} to {log_file}")

    # Prevent logging messages from being propagated to the root logger
    root_logger.propagate = False


def setup_experiment_dir(
    base_dir: str, experiment_name: str
) -> Tuple[str, str, str, str]:
    """Set up experiment directories.

    Args:
        base_dir: Base directory for all experiments
        experiment_name: Name of the experiment

    Returns:
        Tuple of (experiment_dir, log_dir, checkpoint_dir, visualization_dir)
    """
    experiment_dir = os.path.join(base_dir, experiment_name)
    log_dir = os.path.join(experiment_dir, "logs")
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    visualization_dir = os.path.join(experiment_dir, "visualizations")

    # Create directories
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    return experiment_dir, log_dir, checkpoint_dir, visualization_dir
