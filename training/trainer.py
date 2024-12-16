"""
Training utilities for DETR.
"""

import datetime
import logging
import os
import time
from abc import ABC
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import CocoDataset
from evaluation.coco_eval import CocoEvaluator
from evaluation.metrics import evaluate_predictions, prepare_predictions
from models.detr import DETR
from models.matcher import HungarianMatcher
from training.distributed import (
    create_distributed_dataloader,
    create_distributed_model,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    reduce_dict,
    setup_distributed,
    synchronize,
)
from training.losses import compute_loss
from training.optimizer import create_optimizer, create_scheduler
from utils.metric_logger import MetricLogger, SmoothedValue


class MetricHandler:
    """Handles metric logging and evaluation logic."""

    def __init__(self, writer: Optional[SummaryWriter], config: DictConfig):
        self.writer = writer
        self.config = config
        self.metric_logger = MetricLogger(writer=writer)
        self.debug_mode = config.debug.enabled
        # Default metrics that should always be present
        self.default_metrics = {
            "map": 0.0,
            "map_50": 0.0,
            "map_75": 0.0,
            "map_small": 0.0,
            "map_medium": 0.0,
            "map_large": 0.0,
        }
        # Initialize metric buffers for interval logging
        self.metric_buffers = {}
        self.last_log_step = 0

        # Store the initial global step for resuming
        self.initial_global_step = 0
        if writer is not None and config.training.resume_from:
            try:
                from tensorboard.backend.event_processing.event_accumulator import (
                    EventAccumulator,
                )

                log_dir = str(
                    Path(config.output.base_dir) / config.logging.tensorboard.log_dir
                )
                event_acc = EventAccumulator(log_dir)
                event_acc.Reload()

                # Find the last step from any scalar event
                for tag in event_acc.Tags()["scalars"]:
                    events = event_acc.Scalars(tag)
                    if events:
                        self.initial_global_step = max(
                            self.initial_global_step, events[-1].step
                        )

                if is_main_process() and self.initial_global_step > 0:
                    logging.info(
                        f"Resuming metrics from global step: {self.initial_global_step}"
                    )
            except Exception as e:
                if is_main_process():
                    logging.warning(
                        f"Could not determine last tensorboard step: {str(e)}"
                    )

    def _get_prefixed_metrics(
        self, metrics: Dict[str, float], prefix: str
    ) -> Dict[str, float]:
        """Add proper prefix to metrics based on train/val split."""
        prefixed = {}
        for name, value in metrics.items():
            # Remove any existing metrics/ prefix
            clean_name = name.replace("metrics/", "")
            # Group under train_metrics or val_metrics
            prefixed[f"{prefix}_metrics/{clean_name}"] = value
        return prefixed

    def _should_log(self, batch_idx: int, effective_batches: int) -> bool:
        """Determine if we should log metrics at this step based on log_step_ratio."""
        log_every_n_steps = max(
            1, int(effective_batches * self.config.logging.log_step_ratio)
        )
        is_log_step = batch_idx % log_every_n_steps == 0
        is_last_step = batch_idx == effective_batches - 1
        return is_log_step or is_last_step

    def _update_metric_buffers(self, metrics_dict: Dict[str, float], batch_size: int):
        """Update metric buffers with new values."""
        for name, value in metrics_dict.items():
            if name not in self.metric_buffers:
                self.metric_buffers[name] = {"sum": 0.0, "count": 0, "last": 0.0}

            if torch.is_tensor(value):
                value = value.item()

            # For map metrics, we want to keep the last value as they are already averaged
            if "map" in name.lower():
                self.metric_buffers[name]["sum"] = value * batch_size
                self.metric_buffers[name]["count"] = batch_size
                self.metric_buffers[name]["last"] = value
            else:
                self.metric_buffers[name]["sum"] += value * batch_size
                self.metric_buffers[name]["count"] += batch_size
                self.metric_buffers[name]["last"] = value

    def _get_aggregated_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics from buffers."""
        aggregated = {}
        for name, buffer in self.metric_buffers.items():
            # Use last value for optimizer metrics and map metrics
            if "optimizer/" in name or "map" in name.lower():
                aggregated[name] = buffer["last"]
            else:
                if buffer["count"] > 0:
                    aggregated[name] = buffer["sum"] / buffer["count"]
                else:
                    aggregated[name] = 0.0
        return aggregated

    def _log_to_tensorboard(self, metrics_dict: Dict[str, float], global_step: int):
        """Log metrics to tensorboard if writer is available."""
        if not self.writer:
            return

        # Adjust global step if resuming
        adjusted_step = global_step + self.initial_global_step

        for name, value in metrics_dict.items():
            # Skip None or non-numeric values
            if value is None or not isinstance(value, (int, float)):
                continue

            try:
                self.writer.add_scalar(name, value, adjusted_step)
                if is_main_process():
                    logging.debug(
                        f"Logged to tensorboard: {name}={value:.4f} at step {adjusted_step}"
                    )
            except Exception as e:
                if is_main_process():
                    logging.error(f"Failed to log {name} to tensorboard: {str(e)}")

    def update_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        targets: list,
        evaluator: Optional[CocoEvaluator],
        prefix: str,
        batch_idx: int,
        effective_batches: int,
        current_epoch: int,
        lr: float,
    ):
        """Update metrics for the current step."""
        global_step = current_epoch * effective_batches + batch_idx

        # Update loss metrics
        if loss_dict:
            if is_distributed():
                loss_dict_reduced = reduce_dict(loss_dict)
            else:
                loss_dict_reduced = {k: v.detach() for k, v in loss_dict.items()}

            prefixed_losses = {
                f"{prefix}_loss/{name}": loss.item() if torch.is_tensor(loss) else loss
                for name, loss in loss_dict_reduced.items()
            }
            self._update_metric_buffers(prefixed_losses, len(targets))

        # Update evaluator and compute metrics
        if evaluator is not None:
            try:
                predictions = prepare_predictions(
                    outputs, targets, score_threshold=0.001
                )
                if predictions:
                    evaluator.update(predictions)

                    # Compute metrics at logging intervals or last batch
                    if (
                        self._should_log(batch_idx, effective_batches)
                        or batch_idx == effective_batches - 1
                    ):
                        evaluator.synchronize_between_processes()
                        metrics = evaluate_predictions(
                            evaluator, is_main_process(), current_epoch
                        )
                        metrics = metrics or self.default_metrics.copy()
                        prefixed_metrics = self._get_prefixed_metrics(metrics, prefix)
                        self._update_metric_buffers(prefixed_metrics, len(targets))
                        evaluator.reset()
            except Exception as e:
                logging.error(f"Error during evaluation: {str(e)}")
                if is_main_process():
                    logging.info(
                        f"[Epoch {current_epoch}] All metrics are zero. This is normal during early training."
                    )
                prefixed_metrics = self._get_prefixed_metrics(
                    self.default_metrics.copy(), prefix
                )
                self._update_metric_buffers(prefixed_metrics, len(targets))

        # Update optimizer metrics
        optimizer_metrics = {
            "optimizer/learning_rate": lr,
            "optimizer/epoch": float(current_epoch),
        }
        self._update_metric_buffers(optimizer_metrics, len(targets))

        # Log metrics at appropriate intervals
        if self._should_log(batch_idx, effective_batches):
            # Get aggregated metrics
            aggregated_metrics = self._get_aggregated_metrics()

            # Update metric logger
            self.metric_logger.update(
                metrics_dict=aggregated_metrics,
                batch_size=len(targets),
                current_batch=batch_idx,
            )

            # Log to tensorboard and console
            if is_main_process():
                self._log_to_tensorboard(aggregated_metrics, global_step)

                # Log to console
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                metrics_str = " ".join(
                    [f"{k}: {v:.4f}" for k, v in aggregated_metrics.items()]
                )
                log_msg = (
                    f"[{current_time}] "
                    f"[Rank {get_rank()}] "
                    f"{prefix.capitalize()} Epoch [{current_epoch}/{self.config.training.max_epochs}] "
                    f"[{batch_idx}/{effective_batches}] "
                    f"{metrics_str}"
                )
                logging.info(log_msg)

            # Reset buffers after logging
            self.metric_buffers.clear()
            self.last_log_step = batch_idx

    def get_epoch_metrics(self) -> Dict[str, float]:
        """Get metrics for the entire epoch."""
        if is_distributed():
            self.metric_logger.synchronize_between_processes()
        metrics = self.metric_logger.get_epoch_metrics()
        # Ensure we have all required metrics for both train and val
        result = {}
        # Add default metrics for both train and val
        for k, v in self.default_metrics.items():
            result[f"train_metrics/{k}"] = v
            result[f"val_metrics/{k}"] = v
        if metrics:
            result.update(metrics)
        return result

    def reset(self):
        """Reset metric logger for new epoch."""
        self.metric_logger = MetricLogger(writer=self.writer)
        # Initialize with default metrics for both train and val
        train_metrics = {
            f"train_metrics/{k}": v for k, v in self.default_metrics.items()
        }
        val_metrics = {f"val_metrics/{k}": v for k, v in self.default_metrics.items()}
        self.metric_logger.update(metrics_dict={**train_metrics, **val_metrics})


class CheckpointManager:
    """Handles checkpoint saving and loading operations."""

    def __init__(self, output_dir: Path, config: DictConfig):
        self.output_dir = output_dir
        self.config = config
        self.checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = config.logging.get(
            "save_top_k", 3
        )  # Default to keeping top 3 models
        self.best_metrics = []  # List of (metric_value, checkpoint_path) tuples

    def _cleanup_old_checkpoints(self):
        """Remove checkpoints that are not in top-k."""
        if not self.best_metrics:
            return

        # Keep only top-k checkpoints
        checkpoints_to_remove = [path for _, path in self.best_metrics[self.top_k :]]
        for checkpoint_path in checkpoints_to_remove:
            try:
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                    logging.info(f"Removed checkpoint: {checkpoint_path}")
            except Exception as e:
                logging.error(
                    f"Failed to remove checkpoint {checkpoint_path}: {str(e)}"
                )

        # Update best_metrics list to keep only top-k
        self.best_metrics = self.best_metrics[: self.top_k]

    def save(
        self,
        epoch: int,
        global_step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        metric_value: float,
        is_best: bool = False,
    ):
        """Save training checkpoint."""
        if not is_main_process():
            return

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "config": self.config,
            "metric_value": metric_value,
        }

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Update best_metrics list
        self.best_metrics.append((metric_value, str(checkpoint_path)))
        self.best_metrics.sort(
            key=lambda x: x[0]
        )  # Sort by metric value (ascending for loss)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        # Save best checkpoint if specified
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logging.info(f"Saved best model with metric value: {metric_value:.4f}")

        logging.info(
            f"Saved checkpoint for epoch {epoch} with metric value: {metric_value:.4f}"
        )

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the latest checkpoint in the directory."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        if not checkpoints:
            return None

        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.stem.split("_")[-1]))
        return str(checkpoints[-1])

    def get_best_checkpoint(self) -> Optional[str]:
        """Get the path to the best checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pth"
        return str(best_path) if best_path.exists() else None

    def load(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
    ) -> tuple[int, int, float]:
        """Load training checkpoint.

        Returns:
            Tuple of (epoch, global_step, metric_value)
        """
        if is_main_process():
            logging.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle DDP model state dict
        state_dict = checkpoint["model_state_dict"]
        if isinstance(model, DistributedDataParallel):
            # If current model is DDP but checkpoint is not
            if not any(k.startswith("module.") for k in state_dict.keys()):
                if is_main_process():
                    logging.info(
                        "Converting state dict for DDP model (adding 'module.' prefix)"
                    )
                state_dict = {f"module.{k}": v for k, v in state_dict.items()}
        else:
            # If current model is not DDP but checkpoint is
            if any(k.startswith("module.") for k in state_dict.keys()):
                if is_main_process():
                    logging.info(
                        "Converting DDP state dict for non-DDP model (removing 'module.' prefix)"
                    )
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }

        # Load state dict and log any unexpected keys
        try:
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )
            if is_main_process():
                if missing_keys:
                    logging.warning("Missing keys in state dict:")
                    for key in missing_keys:
                        logging.warning(f"  - {key}")
                if unexpected_keys:
                    logging.warning("Unexpected keys in state dict:")
                    for key in unexpected_keys:
                        logging.warning(f"  - {key}")
        except Exception as e:
            if is_main_process():
                logging.error(f"Error loading state dict: {str(e)}")
            raise

        # Load optimizer and scheduler states
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        except Exception as e:
            if is_main_process():
                logging.error(f"Error loading optimizer or scheduler state: {str(e)}")
            raise

        metric_value = checkpoint.get("metric_value", float("inf"))

        if is_main_process():
            logging.info("Successfully loaded checkpoint state dict")

        return checkpoint["epoch"], checkpoint["global_step"], metric_value


class TrainingLoopManager:
    """Manages the main training loop and epoch-level operations."""

    def __init__(
        self,
        config: DictConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        checkpoint_manager: CheckpointManager,
        metric_handler: MetricHandler,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.checkpoint_manager = checkpoint_manager
        self.metric_handler = metric_handler
        self.current_epoch = 0
        self.best_metric = float("-inf")

    def run_training(
        self,
        trainer: "BaseTrainer",
        start_epoch: int,
        max_epochs: int,
    ):
        """Run the main training loop."""
        self.current_epoch = start_epoch

        for epoch in range(start_epoch, max_epochs):
            self.current_epoch = epoch

            # Train for one epoch
            trainer.train_epoch()

            # Validate
            val_metrics = trainer.validate()

            # Update learning rate
            self.lr_scheduler.step()

            # Save checkpoint (only on main process)
            if is_main_process():
                is_best = False
                if val_metrics is not None and trainer.is_better_checkpoint(
                    val_metrics, self.best_metric
                ):
                    self.best_metric = trainer.get_checkpoint_metric(val_metrics)
                    is_best = True

                self.checkpoint_manager.save(
                    epoch=epoch,
                    global_step=trainer.global_step,
                    model=self.model,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    metric_value=trainer.get_checkpoint_metric(val_metrics),
                    is_best=is_best,
                )

            if dist.is_initialized():
                dist.barrier()

        if self.metric_handler.writer is not None:
            self.metric_handler.writer.close()


class BatchProcessor:
    """Handles batch-level operations during training."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: DictConfig,
        compute_loss_fn=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.compute_loss_fn = compute_loss_fn

    def process_batch(
        self, images: torch.Tensor, targets: list, training: bool = True
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Process a single batch of data."""
        mode = "train" if training else "val"
        logging.debug(
            f"[Rank {get_rank()}] Processing {mode} batch: images={images.shape}"
        )

        # Move data to device
        images = images.to(self.device, non_blocking=True)
        targets = [
            {k: v.to(self.device, non_blocking=True) for k, v in t.items()}
            for t in targets
        ]

        # Set model mode
        self.model.train(training)

        # Forward pass
        logging.debug(f"[Rank {get_rank()}] Starting forward pass")
        if not training:
            with torch.no_grad():
                outputs = self.model(images)
                loss_dict = self.compute_loss_fn(outputs, targets)
        else:
            # Ensure input requires grad in training mode
            if not images.requires_grad:
                images.requires_grad_(True)

            outputs = self.model(images)
            loss_dict = self.compute_loss_fn(outputs, targets)

            # Verify loss requires grad
            if not loss_dict["loss"].requires_grad:
                raise ValueError(
                    "Loss tensor doesn't require gradients. Check model parameters and loss computation."
                )

        logging.debug(
            f"[Rank {get_rank()}] Loss computation completed: {loss_dict['loss'].item():.4f}"
        )

        return outputs, loss_dict

    def update_step(self, loss: torch.Tensor):
        """Perform backward pass and parameter update."""
        try:
            # Zero gradients
            self.optimizer.zero_grad()

            # Check if loss requires grad
            if not loss.requires_grad:
                raise ValueError(
                    "Loss tensor doesn't require gradients. Check model parameters and loss computation."
                )

            loss.backward()

            # Clip gradients
            if self.config.training.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clip_val
                )

            # Update parameters
            self.optimizer.step()

        except Exception as e:
            logging.error("[Rank %d] Error in update step: %s", get_rank(), str(e))
            raise


class BaseTrainer(ABC):
    """Base trainer class with common functionality."""

    def __init__(
        self,
        config: DictConfig,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        output_dir: str,
    ):
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.current_epoch = 0
        self.global_step = 0
        self.training_loop_manager = None  # Will be set by set_training_loop_manager

        # Setup model and ensure parameters require gradients
        for param in model.parameters():
            param.requires_grad_(True)

        self.model = create_distributed_model(
            model,
            device,
            sync_bn=config.distributed.sync_batchnorm,
            find_unused_parameters=config.distributed.find_unused_parameters,
        )

        # Verify model parameters require gradients
        requires_grad = any(p.requires_grad for p in self.model.parameters())
        if not requires_grad:
            raise ValueError("No model parameters require gradients")

        # Setup data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Setup training components
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Setup logging and metrics
        if is_main_process():
            # Create tensorboard writer with purge_step to handle resuming
            log_dir = self.output_dir / "tensorboard"
            if config.training.resume_from:
                # If resuming, try to find the last global step from existing events
                try:
                    from tensorboard.backend.event_processing.event_accumulator import (
                        EventAccumulator,
                    )

                    event_acc = EventAccumulator(str(log_dir))
                    event_acc.Reload()

                    # Find the last step from any scalar event
                    last_step = 0
                    for tag in event_acc.Tags()["scalars"]:
                        events = event_acc.Scalars(tag)
                        if events:
                            last_step = max(last_step, events[-1].step)

                    if last_step > 0:
                        if is_main_process():
                            logging.info(
                                f"Resuming tensorboard logging from step {last_step}"
                            )
                        self.writer = SummaryWriter(log_dir, purge_step=last_step)
                    else:
                        self.writer = SummaryWriter(log_dir)
                except Exception as e:
                    if is_main_process():
                        logging.warning(
                            f"Failed to determine last tensorboard step: {str(e)}"
                        )
                    self.writer = SummaryWriter(log_dir)
            else:
                self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None

        # Initialize metric handler and checkpoint manager
        self.metric_handler = MetricHandler(self.writer, config)
        self.checkpoint_manager = CheckpointManager(self.output_dir, config)

        # Setup debug mode
        self.debug = config.debug.enabled
        self.debug_batches = config.debug.num_batches
        if self.debug and is_main_process():
            logging.warning(
                f"Running in debug mode with {self.debug_batches} batches per epoch"
            )

        # Batch processor will be initialized by child classes
        self.batch_processor = None

    def set_training_loop_manager(self, manager: TrainingLoopManager):
        """Set the training loop manager instance."""
        self.training_loop_manager = manager
        # Update current epoch from manager
        self.current_epoch = manager.current_epoch

    def train(self):
        """Main training loop."""
        try:
            self.training_loop_manager.run_training(
                trainer=self,
                start_epoch=self.current_epoch,
                max_epochs=self.config.training.max_epochs,
            )
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            if dist.is_initialized():
                dist.barrier()
            if self.writer is not None:
                self.writer.close()
            raise

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()

        try:
            metrics = self._run_epoch(
                self.train_loader,
                self.train_evaluator if hasattr(self, "train_evaluator") else None,
                training=True,
            )
            if metrics is None:
                metrics = {"map": 0.0}
            return metrics
        except Exception as e:
            logging.error(f"Training epoch failed: {str(e)}")
            raise

    @torch.no_grad()
    def validate(self) -> Optional[Dict[str, float]]:
        """Run validation and return metrics."""
        try:
            metrics = self._run_epoch(
                self.val_loader, self.val_evaluator, training=False
            )
            if metrics is None:
                metrics = {"map": 0.0}

            if dist.is_initialized():
                dist.barrier()

            return metrics

        except Exception as e:
            logging.error(f"Validation failed: {str(e)}")
            if dist.is_initialized():
                dist.barrier()
            raise

    def is_better_checkpoint(
        self, metrics: Dict[str, float], best_so_far: float
    ) -> bool:
        """Determine if current checkpoint is better based on mAP."""
        # Get the map value safely, defaulting to 0 if not present
        current_map = metrics.get("map", 0.0)
        return current_map > best_so_far

    def get_checkpoint_metric(self, metrics: Dict[str, float]) -> float:
        """Get mAP as the checkpoint comparison metric."""
        # Get the map value safely, defaulting to 0 if not present
        return metrics.get("map", 0.0)

    def get_effective_batches(self, num_batches: int) -> int:
        """Get the effective number of batches to process.

        Args:
            num_batches: Total number of batches available

        Returns:
            Number of batches to process (limited in debug mode)
        """
        if self.debug:
            return min(self.debug_batches, num_batches)
        return num_batches

    def _run_epoch(
        self,
        data_loader: DataLoader,
        evaluator: Optional[CocoEvaluator],
        training: bool = True,
    ) -> Optional[Dict[str, float]]:
        """Run a single epoch of training or validation."""
        mode = "train" if training else "val"

        # Reset metric logger for new epoch
        self.metric_handler.reset()

        # Get effective number of batches
        num_batches = len(data_loader)
        effective_batches = self.get_effective_batches(num_batches)

        # Initialize timing trackers
        iter_time = SmoothedValue(fmt="{median:.4f} ({global_avg:.4f})")
        data_time = SmoothedValue(fmt="{median:.4f} ({global_avg:.4f})")
        start_time = time.time()

        # Get current epoch from training loop manager
        current_epoch = (
            self.training_loop_manager.current_epoch
            if hasattr(self, "training_loop_manager")
            else self.current_epoch
        )

        # Process batches with metric logging
        for batch_idx, (images, targets) in enumerate(data_loader):
            data_end = time.time()
            data_time.update(data_end - start_time)

            try:
                # Process batch
                outputs, loss_dict = self.batch_processor.process_batch(
                    images, targets, training
                )

                # Update step if training
                if training:
                    self.batch_processor.update_step(loss_dict["loss"])

                # Log metrics
                self.metric_handler.update_metrics(
                    loss_dict,
                    outputs,
                    targets,
                    evaluator,
                    mode,
                    batch_idx,
                    effective_batches,
                    current_epoch,
                    self.optimizer.param_groups[0]["lr"],
                )

                # Calculate log frequency based on ratio
                log_every_n_steps = max(
                    1, int(effective_batches * self.config.logging.log_step_ratio)
                )

                # Log progress at appropriate intervals
                if (
                    batch_idx % log_every_n_steps == 0
                    or batch_idx == effective_batches - 1
                ):
                    eta_seconds = iter_time.global_avg * (effective_batches - batch_idx)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    if is_main_process():
                        timing_str = f"eta: {eta_string:>8} iter_t: {iter_time} data_t: {data_time}"
                        metrics_str = str(self.metric_handler.metric_logger)
                        current_batch = self.metric_handler.metric_logger.current_batch
                        log_msg = (
                            f"[{current_time}] "
                            f"[Rank {get_rank()}] "
                            f"{mode.capitalize()} Epoch [{current_epoch}/{self.config.training.max_epochs}] "
                            f"[{current_batch}/{effective_batches}] "
                            f"{timing_str} "
                            f"{metrics_str}"
                        )
                        logging.info(log_msg)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.error(f"GPU out of memory in batch {batch_idx}: {str(e)}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise RuntimeError(f"GPU out of memory: {str(e)}")
                else:
                    logging.error(f"Runtime error in batch {batch_idx}: {str(e)}")
                    raise

            except Exception as e:
                logging.error(
                    f"Critical error in batch {batch_idx}: {str(e)}", exc_info=True
                )
                # Ensure all processes know about the failure
                if dist.is_initialized():
                    dist.barrier()
                raise RuntimeError(f"Training failed: {str(e)}") from e

            # Update timing
            iter_time.update(time.time() - data_end)
            start_time = time.time()

            # Break if we've processed enough batches
            if batch_idx >= effective_batches - 1:
                break

        # Return epoch metrics
        return self.metric_handler.get_epoch_metrics()


class DETRTrainer(BaseTrainer):
    """DETR-specific trainer implementation."""

    def __init__(
        self,
        config: DictConfig,
        model: DETR,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        output_dir: str,
        resume_from: Optional[str] = None,
    ):
        super().__init__(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
            output_dir=output_dir,
        )

        # Setup DETR-specific components
        self.matcher = HungarianMatcher(
            cost_class=config.loss.cost_class,
            cost_bbox=config.loss.cost_bbox,
            cost_giou=config.loss.cost_giou,
        )

        # Setup evaluators
        self.setup_evaluators()

        # Setup batch processor with loss computation function
        self.batch_processor = BatchProcessor(
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
            config=config,
            compute_loss_fn=self.compute_loss,
        )

    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a checkpoint file or directory.

        Args:
            checkpoint_path: Path to checkpoint file or directory containing checkpoints
        """
        if (
            not hasattr(self, "training_loop_manager")
            or self.training_loop_manager is None
        ):
            raise RuntimeError(
                "Training loop manager must be set before resuming from checkpoint"
            )

        try:
            if is_main_process():
                logging.info("=" * 80)
                logging.info("Resuming Training from Checkpoint")
                logging.info("=" * 80)
                logging.info(f"Original checkpoint path: {checkpoint_path}")
                logging.info(f"Resume mode: {self.config.training.resume_mode}")

            # If directory is provided, try to find best or latest checkpoint based on resume_mode
            if os.path.isdir(checkpoint_path):
                checkpoint_dir = Path(checkpoint_path) / "checkpoints"
                if not checkpoint_dir.exists():
                    raise ValueError(
                        f"Checkpoint directory not found: {checkpoint_dir}"
                    )

                if is_main_process():
                    logging.info(f"Searching for checkpoints in: {checkpoint_dir}")

                # Choose checkpoint based on resume_mode
                if self.config.training.resume_mode.lower() == "best":
                    checkpoint_path = self.checkpoint_manager.get_best_checkpoint()
                    if checkpoint_path:
                        if is_main_process():
                            logging.info(f"Found best checkpoint: {checkpoint_path}")
                            logging.info("Resuming from best checkpoint")
                    else:
                        if is_main_process():
                            logging.warning(
                                "Best checkpoint not found, falling back to latest"
                            )
                        checkpoint_path = (
                            self.checkpoint_manager.get_latest_checkpoint()
                        )
                else:  # "latest" mode
                    checkpoint_path = self.checkpoint_manager.get_latest_checkpoint()
                    if checkpoint_path:
                        if is_main_process():
                            logging.info(f"Found latest checkpoint: {checkpoint_path}")
                            logging.info("Resuming from latest checkpoint")

                if not checkpoint_path:
                    raise ValueError(f"No checkpoints found in {checkpoint_dir}")

            # Load the checkpoint
            if is_main_process():
                logging.info(f"Loading checkpoint state from: {checkpoint_path}")

            epoch, global_step, metric_value = self.checkpoint_manager.load(
                checkpoint_path,
                self.model,
                self.optimizer,
                self.lr_scheduler,
                self.device,
            )

            self.current_epoch = epoch + 1  # Start from next epoch
            self.global_step = global_step
            if hasattr(self, "training_loop_manager"):
                self.training_loop_manager.current_epoch = self.current_epoch
                self.training_loop_manager.best_metric = metric_value

            if is_main_process():
                logging.info("-" * 40)
                logging.info("Successfully restored checkpoint state:")
                logging.info(f"  - Previous epoch: {epoch}")
                logging.info(f"  - Global step: {global_step}")
                logging.info(f"  - Best metric value: {metric_value:.4f}")
                logging.info(f"  - Resuming from epoch: {self.current_epoch}")
                logging.info("-" * 40)

            # Ensure all processes are synchronized
            if dist.is_initialized():
                dist.barrier()

        except Exception as e:
            logging.error("=" * 80)
            logging.error("Failed to resume from checkpoint")
            logging.error("-" * 40)
            logging.error(f"Error details: {str(e)}")
            logging.error("=" * 80)
            raise

    def _create_evaluator(self, ann_file: str, split: str) -> CocoEvaluator:
        """Create a COCO evaluator for a specific data split.

        Args:
            ann_file: Path to annotation file
            split: Data split name ('train' or 'val')

        Returns:
            Initialized CocoEvaluator
        """
        logging.info(f"[Rank {get_rank()}] Loading {split} annotations...")
        evaluator = CocoEvaluator.from_annotations_file(
            ann_file,
            debug_mode=self.debug,
            debug_root=os.path.join(self.config.data.data_root_dir, "debug_data")
            if self.debug
            else None,
            debug_samples=self.debug_batches,
        )
        logging.info(f"[Rank {get_rank()}] {split.capitalize()} evaluator ready")
        return evaluator

    def setup_evaluators(self):
        """Initialize COCO evaluators for training and validation."""
        # Get annotation file paths
        train_ann_file = os.path.join(
            self.config.data.data_root_dir, self.config.data.train_ann
        )
        val_ann_file = os.path.join(
            self.config.data.data_root_dir, self.config.data.val_ann
        )

        # Log initialization start
        logging.info(
            f"[Rank {get_rank()}] Setting up evaluators (is_main_process={is_main_process()})"
        )
        logging.info(f"[Rank {get_rank()}] Train annotations: {train_ann_file}")
        logging.info(f"[Rank {get_rank()}] Val annotations: {val_ann_file}")

        try:
            # Create evaluators
            self.train_evaluator = self._create_evaluator(train_ann_file, "train")
            self.val_evaluator = self._create_evaluator(val_ann_file, "val")

            # Synchronize across processes
            logging.info(
                f"[Rank {get_rank()}] Waiting at evaluator initialization barrier"
            )
            synchronize()
            logging.info(f"[Rank {get_rank()}] Passed evaluator initialization barrier")
            logging.info(f"[Rank {get_rank()}] All evaluators initialized successfully")

        except Exception as e:
            logging.error(
                f"[Rank {get_rank()}] Failed to initialize evaluators: {str(e)}"
            )
            self.train_evaluator = None
            self.val_evaluator = None
            # Ensure all processes know about the failure
            logging.info(f"[Rank {get_rank()}] Waiting at evaluator failure barrier")
            synchronize()
            logging.info(f"[Rank {get_rank()}] Passed evaluator failure barrier")
            if is_main_process():
                logging.warning(
                    "Training will continue without COCO evaluation. "
                    "Please check if the annotation files exist and are valid."
                )

    def compute_loss(
        self, outputs: Dict[str, torch.Tensor], targets: Any
    ) -> Dict[str, torch.Tensor]:
        """Compute DETR loss."""
        # Compute loss
        loss_dict = compute_loss(
            outputs=outputs,
            targets=targets,
            matcher=self.matcher,
            num_classes=self.config.model.num_classes,
            loss_config=self.config.loss,
        )

        # Ensure loss requires gradients before reduction
        if self.model.training:
            for k, v in loss_dict.items():
                if not v.requires_grad:
                    v.requires_grad_(True)

        # Synchronize loss components across ranks while preserving gradients
        reduced_dict = {}
        world_size = get_world_size()
        if world_size > 1:
            for k, v in loss_dict.items():
                if torch.is_tensor(v):
                    v = v.clone()
                    dist.all_reduce(v)
                    v = v / world_size
                reduced_dict[k] = v
        else:
            reduced_dict = loss_dict

        return reduced_dict


def create_trainer(
    config: DictConfig,
    output_dir: str,
    resume_from: Optional[str] = None,
) -> DETRTrainer:
    """
    Create and initialize a trainer instance.

    Args:
        config: Configuration object
        output_dir: Directory for saving outputs
        resume_from: Optional path to checkpoint file or directory to resume from

    Returns:
        Initialized trainer instance
    """
    # Setup distributed training
    device, world_size, rank = setup_distributed()

    # Create model
    model = DETR(config)

    # Create datasets and dataloaders
    dataset = CocoDataset(config)
    train_loader = create_distributed_dataloader(
        dataset.train_dataset, config, world_size, rank, shuffle=True
    )
    val_loader = create_distributed_dataloader(
        dataset.val_dataset, config, world_size, rank, shuffle=False
    )

    # Create optimizer and scheduler
    optimizer = create_optimizer(model.parameters(), config)
    lr_scheduler = create_scheduler(optimizer, config)

    # Create trainer without resuming yet
    trainer = DETRTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        output_dir=output_dir,
        resume_from=None,  # Don't resume yet
    )

    # Create and set training loop manager first
    training_loop_manager = TrainingLoopManager(
        config=config,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint_manager=trainer.checkpoint_manager,
        metric_handler=trainer.metric_handler,
    )
    trainer.set_training_loop_manager(training_loop_manager)

    # Now resume from checkpoint if provided
    if resume_from:
        trainer.resume_from_checkpoint(resume_from)

    return trainer
