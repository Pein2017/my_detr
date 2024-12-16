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
    ):
        """
        Log metrics for every print_freq iterations.

        Args:
            iterable: Iterable to log metrics for
            print_freq: How often to log
            epoch: Current epoch number
            prefix: String to prefix to the log message
            total_length: Total number of iterations (for progress bar)
        """
        if total_length is None:
            # Try to get length, otherwise use infinity
            total_length = (
                len(iterable) if hasattr(iterable, "__len__") else float("inf")
            )

        # Calculate padding for consistent formatting
        space_fmt = len(str(total_length))

        if epoch is not None:
            if "[" not in prefix:
                prefix = f"[{epoch}] {prefix}"

        # Store rank for reuse
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
                eta_seconds = iter_time.global_avg * (total_length - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if is_main_process():
                    # Get current timestamp
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Create header with current timestamp and batch progress
                    header = f"[{current_time}][Rank {rank}] {prefix}"

                    # Format progress with consistent spacing
                    progress_str = f"[{i:{space_fmt}d}/{total_length:{space_fmt}d}]"

                    # Format timing info with fixed width
                    timing_str = (
                        f"eta: {eta_string:>8} "
                        f"iter_t: {iter_time} "
                        f"data_t: {data_time}"
                    )

                    # Format metrics with consistent delimiter
                    metrics_str = str(self)

                    # Combine all parts with proper spacing
                    log_msg = f"{header} {progress_str} {timing_str} {metrics_str}"
                    logging.info(log_msg)

                    # Log to tensorboard
                    if self.writer is not None:
                        step = self.global_step + i
                        # Log all metrics
                        for name, meter in self.meters.items():
                            self.writer.add_scalar(name, meter.global_avg, step)
                        # Log times
                        self.writer.add_scalar("time/iter", iter_time.avg, step)
                        self.writer.add_scalar("time/data", data_time.avg, step)

            i += 1
            self.start_time = time.time()

        self.global_step += i

    def __str__(self) -> str:
        """Get string representation of current metrics."""
        loss_str = []
        for name, meter in self.meters.items():
            if "loss" in name.lower():  # Group losses together
                loss_str.append(f"{name}: {meter}")

        metric_str = []
        for name, meter in self.meters.items():
            if "loss" not in name.lower():  # Other metrics
                metric_str.append(f"{name}: {meter}")

        return self.delimiter.join(loss_str + metric_str)

    def get_epoch_metrics(self) -> Dict[str, float]:
        """Get dictionary of metrics for the epoch."""
        return {name: meter.global_avg for name, meter in self.meters.items()}
