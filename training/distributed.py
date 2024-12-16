"""
Distributed training utilities for DETR.
"""

import datetime
import logging
import os
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from data.dataset import collate_fn


def setup_distributed() -> Tuple[torch.device, int, int]:
    """
    Setup distributed training.

    Returns:
        Tuple of (device, world_size, rank)
    """
    if not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        return torch.device("cpu"), 1, 0

    # Check if running in distributed mode
    if "LOCAL_RANK" not in os.environ:
        # Single GPU mode
        device = torch.device("cuda:0")
        logging.info("Running in single GPU mode")
        return device, 1, 0

    # Distributed mode setup
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    device = torch.device(f"cuda:{local_rank}")

    # Initialize process group
    if not dist.is_initialized():
        try:
            timeout = datetime.timedelta(minutes=10)
            master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
            master_port = os.environ.get("MASTER_PORT", "29500")
            init_method = f"tcp://{master_addr}:{master_port}"
            backend = "nccl" if torch.cuda.is_available() else "gloo"

            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=world_size,
                rank=rank,
                timeout=timeout,
            )

            # Set device after initialization
            torch.cuda.set_device(device)
            logging.info(
                f"Initialized distributed process group: rank={rank}, world_size={world_size}"
            )

        except Exception as e:
            logging.error(f"Failed to initialize process group: {str(e)}")
            raise

    return device, world_size, rank


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as e:
            logging.error(f"Error destroying process group: {str(e)}")


def create_distributed_model(
    model: torch.nn.Module,
    device: torch.device,
    sync_bn: bool = True,
    find_unused_parameters: bool = True,
) -> torch.nn.Module:
    """
    Create distributed model.

    Args:
        model: PyTorch model
        device: Device to place model on
        sync_bn: Whether to use synchronized batch normalization
        find_unused_parameters: Whether to find unused parameters in DDP

    Returns:
        Distributed model
    """
    model = model.to(device)

    # Only wrap with DDP if running in distributed mode with multiple processes
    if dist.is_initialized() and dist.get_world_size() > 1:
        # Convert batch norm layers to sync batch norm if requested
        if sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        try:
            # Enable DDP debug mode for better error messages
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

            # Log parameter information before DDP wrapping
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            logging.info(
                f"Model parameters before DDP wrapping: "
                f"total={total_params}, trainable={trainable_params}"
            )

            # Use more flexible DDP settings
            model = DDP(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                output_device=device.index if device.type == "cuda" else None,
                find_unused_parameters=True,  # Always enable for debugging
                broadcast_buffers=True,
                gradient_as_bucket_view=True,  # More memory efficient
                static_graph=False,
            )
            logging.info(
                "Model wrapped with DistributedDataParallel "
                "(find_unused_parameters=True, static_graph=False)"
            )

            # Verify DDP initialization
            if not isinstance(model, DDP):
                raise RuntimeError("Failed to initialize DDP")

            # Log DDP configuration and parameter information
            ddp_params = sum(p.numel() for p in model.parameters())
            logging.info(
                f"DDP Configuration: device={device}, "
                f"find_unused_parameters=True, "
                f"gradient_as_bucket_view=True, "
                f"static_graph=False, "
                f"total_parameters={ddp_params}"
            )

        except Exception as e:
            logging.error(f"Failed to initialize DDP: {str(e)}")
            raise
    else:
        logging.info("Running in single GPU mode, skipping DDP wrapper")

    return model


def create_distributed_sampler(
    dataset,
    world_size: int,
    rank: int,
    shuffle: bool = True,
) -> Optional[DistributedSampler]:
    """
    Create distributed sampler for dataset.

    Args:
        dataset: PyTorch dataset
        world_size: Number of distributed processes
        rank: Rank of current process
        shuffle: Whether to shuffle the data

    Returns:
        Distributed sampler if using distributed training, else None
    """
    # Only use DistributedSampler in distributed mode with multiple processes
    if dist.is_initialized() and world_size > 1:
        return DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
    return None


def create_distributed_dataloader(
    dataset,
    config: DictConfig,
    world_size: int,
    rank: int,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create dataloader with optional distributed sampler.

    Args:
        dataset: PyTorch dataset
        config: Configuration dictionary
        world_size: Number of distributed processes
        rank: Rank of current process
        shuffle: Whether to shuffle the data

    Returns:
        DataLoader with optional distributed sampler
    """
    sampler = create_distributed_sampler(dataset, world_size, rank, shuffle)

    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,  # Don't shuffle if using sampler
        num_workers=config.training.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=config.data.persistent_workers,
    )


def is_distributed() -> bool:
    """Check if distributed training is enabled and initialized."""
    return dist.is_initialized() and dist.get_world_size() > 1


def get_world_size() -> int:
    """Get the number of processes in the distributed training."""
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank() -> int:
    """Get the rank of current process."""
    return dist.get_rank() if dist.is_initialized() else 0


def is_main_process() -> bool:
    """Check if this is the main process in distributed training."""
    return get_rank() == 0


def synchronize():
    """Synchronize all processes in distributed mode."""
    if is_distributed():
        dist.barrier()


def reduce_dict(
    input_dict: Dict[str, torch.Tensor], average: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results.

    Args:
        input_dict: Dictionary of tensors to reduce
        average: Whether to average or sum the values

    Returns:
        Dictionary with reduced values
    """
    if not is_distributed():
        return input_dict

    with torch.no_grad():
        reduced_dict = {}
        for k, v in input_dict.items():
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
            if average:
                v = v / get_world_size()
            reduced_dict[k] = v
        return reduced_dict
