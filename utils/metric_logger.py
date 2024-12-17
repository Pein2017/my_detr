"""
Metric logging utilities for DETR.
"""

import datetime
import logging
import time
from collections import defaultdict, deque
from typing import Dict, Optional, Union

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from training.distributed import (
    get_rank,
    is_distributed,
    is_main_process,
    reduce_dict,
)


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a window."""

    def __init__(self, window_size: int = 20, fmt: Optional[str] = None):
        """
        Args:
            window_size: Window size for smoothing
            fmt: Format string for printing
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"

    def update(self, value: Union[float, torch.Tensor], n: int = 1):
        """Update with new value."""
        if torch.is_tensor(value):
            value = value.item()
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self) -> float:
        """Get median value."""
        d = torch.tensor(list(self.deque))
        return d.median().item() if len(d) > 0 else 0.0

    @property
    def avg(self) -> float:
        """Get average of window."""
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item() if len(d) > 0 else 0.0

    @property
    def global_avg(self) -> float:
        """Get global average."""
        return self.total / self.count if self.count > 0 else 0.0

    def __str__(self) -> str:
        """String representation."""
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
        )


class MetricLogger:
    """Log and synchronize metrics across processes."""

    def __init__(
        self,
        delimiter: str = "\t",
        writer: Optional[SummaryWriter] = None,
    ):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.writer = writer
        self.epoch = 0
        self.global_step = 0
        self.start_time = time.time()
        self.current_batch = 0
        self.epoch_start_time = time.time()
        self.epoch_history = []

    def start_epoch(self):
        """Mark the start of a new epoch for timing purposes."""
        self.epoch_start_time = time.time()

    def end_epoch(self):
        """Mark the end of an epoch and record timing."""
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_history.append(epoch_time)
        if self.writer is not None:
            self.writer.add_scalar("time/epoch", epoch_time, self.epoch)
        return epoch_time

    def estimate_training_eta(self, current_epoch: int, max_epochs: int) -> str:
        """Estimate time remaining for entire training based on epoch history."""
        if not self.epoch_history:
            return "N/A"

        avg_epoch_time = sum(self.epoch_history) / len(self.epoch_history)
        remaining_epochs = max_epochs - current_epoch
        eta_seconds = avg_epoch_time * remaining_epochs

        # Format ETA string
        days = int(eta_seconds // (24 * 3600))
        hours = int((eta_seconds % (24 * 3600)) // 3600)
        minutes = int((eta_seconds % 3600) // 60)

        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"

    def update(
        self,
        loss_dict: Optional[Dict[str, torch.Tensor]] = None,
        metrics_dict: Optional[Dict[str, float]] = None,
        batch_size: Optional[int] = None,
        current_batch: Optional[int] = None,
        **kwargs,
    ):
        """
        Update metrics.

        Args:
            loss_dict: Dictionary of loss tensors
            metrics_dict: Dictionary of metric values
            batch_size: Batch size for proper averaging
            current_batch: Current batch index
            **kwargs: Additional key-value pairs to log
        """
        if current_batch is not None:
            self.current_batch = current_batch

        if loss_dict:
            # Synchronize loss values in distributed training
            if is_distributed():
                loss_dict_reduced = reduce_dict(loss_dict)
            else:
                loss_dict_reduced = {k: v.detach() for k, v in loss_dict.items()}

            # Update loss meters
            for name, loss in loss_dict_reduced.items():
                if torch.is_tensor(loss):
                    loss = loss.item()
                self.meters[f"loss/{name}"].update(loss, batch_size or 1)

        if metrics_dict:
            # Update metric meters
            for name, value in metrics_dict.items():
                self.meters[f"metrics/{name}"].update(value, batch_size or 1)

        # Update additional metrics
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def synchronize_between_processes(self):
        """
        Synchronize metrics between processes in distributed training.
        """
        if not is_distributed():
            return

        for meter in self.meters.values():
            tensor = torch.tensor(
                [meter.count, meter.total],
                dtype=torch.float64,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            dist.all_reduce(tensor)
            meter.count = int(tensor[0].item())
            meter.total = tensor[1].item()

    def log_every(
        self,
        iterable,
        print_freq: int,
        epoch: Optional[int] = None,
        prefix: str = "",
        total_length: Optional[int] = None,
        max_epochs: Optional[int] = None,
    ):
        """
        Log metrics for every print_freq iterations.

        Args:
            iterable: Iterable to log metrics for
            print_freq: How often to log
            epoch: Current epoch number
            prefix: String to prefix to the log message
            total_length: Total number of iterations (for progress bar)
            max_epochs: Total number of epochs for ETA calculation
        """
        if total_length is None:
            total_length = (
                len(iterable) if hasattr(iterable, "__len__") else float("inf")
            )

        space_fmt = len(str(total_length))

        if epoch is not None:
            if "[" not in prefix:
                prefix = f"[{epoch}] {prefix}"

        rank = get_rank()

        iter_time = SmoothedValue(fmt="{median:.4f} ({global_avg:.4f})")
        data_time = SmoothedValue(fmt="{median:.4f} ({global_avg:.4f})")
        self.start_time = time.time()

        i = 0
        for obj in iterable:
            data_end = time.time()
            data_time.update(data_end - self.start_time)
            yield obj
            iter_time.update(time.time() - data_end)

            if i % print_freq == 0 or i == total_length - 1:
                # Calculate ETAs
                iter_eta_seconds = iter_time.global_avg * (total_length - i)
                iter_eta_string = str(datetime.timedelta(seconds=int(iter_eta_seconds)))

                # Calculate training ETA if we have epoch history
                training_eta = "N/A"
                if max_epochs is not None and epoch is not None:
                    training_eta = self.estimate_training_eta(epoch, max_epochs)

                if is_main_process():
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    header = f"[{current_time}][Rank {rank}] {prefix}"
                    progress_str = f"[{i:{space_fmt}d}/{total_length:{space_fmt}d}]"
                    timing_str = (
                        f"epoch_eta: {iter_eta_string:>8} "
                        f"training_eta: {training_eta:>8} "
                        f"iter_time: {iter_time.median:.2f}s ({iter_time.global_avg:.2f}s) "
                        f"data_time: {data_time.median:.2f}s ({data_time.global_avg:.2f}s)"
                    )
                    metrics_str = str(self)
                    log_msg = f"{header} {progress_str} {timing_str} {metrics_str}"
                    logging.info(log_msg)

                    if self.writer is not None:
                        step = self.global_step + i
                        for name, meter in self.meters.items():
                            self.writer.add_scalar(name, meter.global_avg, step)
                        self.writer.add_scalar("time/iter", iter_time.avg, step)
                        self.writer.add_scalar("time/data", data_time.avg, step)

            i += 1
            self.start_time = time.time()

        self.global_step += i

    def __str__(self) -> str:
        """Get string representation of current metrics."""
        # Order losses in a consistent way
        loss_order = ["loss_ce", "loss_bbox", "loss_giou"]
        loss_str = []

        # First add main losses in consistent order
        for loss_name in loss_order:
            full_name = f"loss/{loss_name}"
            if full_name in self.meters:
                loss_str.append(f"{loss_name}: {self.meters[full_name]}")

        # Then add any auxiliary losses in the same order
        aux_losses = []
        i = 0
        while True:
            aux_found = False
            for loss_name in loss_order:
                aux_name = f"loss/{loss_name}_{i}"
                if aux_name in self.meters:
                    aux_found = True
                    aux_losses.append(f"{loss_name}_{i}: {self.meters[aux_name]}")
            if not aux_found:
                break
            i += 1

        # Add total loss if present
        if "loss/loss" in self.meters:
            loss_str.append(f"total_loss: {self.meters['loss/loss']}")

        # Add auxiliary losses after main losses
        loss_str.extend(aux_losses)

        # Add other metrics
        metric_str = []
        for name, meter in self.meters.items():
            if "loss" not in name.lower():  # Other metrics
                metric_str.append(f"{name}: {meter}")

        return self.delimiter.join(loss_str + metric_str)

    def get_epoch_metrics(self) -> Dict[str, float]:
        """Get dictionary of metrics for the epoch."""
        return {name: meter.global_avg for name, meter in self.meters.items()}
