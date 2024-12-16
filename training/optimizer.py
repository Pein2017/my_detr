"""
Optimization utilities for DETR.
"""

from typing import Dict, List

import torch
from omegaconf import DictConfig


def create_optimizer(
    model_parameters: List[Dict],
    config: DictConfig,
) -> torch.optim.Optimizer:
    """
    Create optimizer for model parameters.

    Args:
        model_parameters: List of parameter groups
        config: Configuration dictionary

    Returns:
        PyTorch optimizer
    """
    return torch.optim.AdamW(
        model_parameters,
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: DictConfig,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary

    Returns:
        PyTorch learning rate scheduler
    """
    # Initialize learning rate scheduler
    if config.optimizer.lr_scheduler.name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.optimizer.lr_scheduler.step_size,
            gamma=config.optimizer.lr_scheduler.gamma,
        )
    elif config.optimizer.lr_scheduler.name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.max_epochs,
            eta_min=config.optimizer.lr_scheduler.min_lr,
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config.optimizer.lr_scheduler.name}")

    # Apply warmup if configured
    if config.optimizer.lr_scheduler.warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=config.optimizer.lr_scheduler.warmup_factor,
                    total_iters=config.optimizer.lr_scheduler.warmup_epochs,
                ),
                scheduler,
            ],
            milestones=[config.optimizer.lr_scheduler.warmup_epochs],
        )

    return scheduler
