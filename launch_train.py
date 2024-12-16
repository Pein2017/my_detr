"""
DETR training script.
Supports distributed training and mixed precision.
"""

import logging
import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from training.trainer import create_trainer
from utils.setup import setup_experiment_dir, setup_logging
from training.distributed import setup_distributed, cleanup_distributed

# Set environment variables
os.environ["TORCH_HOME"] = "/data/training_code/Pein/DETR/my_detr/pretrained"


def set_seed(seed: int, rank: int = 0) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Base random seed
        rank: Process rank for distributed training
    """
    # Different seed for different processes
    seed = seed + rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # In distributed training, we want deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@hydra.main(version_base=None, config_path="configs", config_name="default")
def hydra_entry(config: DictConfig) -> None:
    """Hydra entry point.

    Args:
        config: Configuration dictionary from Hydra
    """
    try:
        # Set up distributed environment first
        device, world_size, rank = setup_distributed()
        logging.info(
            f"Distributed setup complete: rank={rank}, world_size={world_size}, device={device}"
        )

        # Set random seed with rank
        set_seed(config.training.seed, rank)

        # Setup experiment directories (only on rank 0)
        if rank == 0:
            experiment_dir, log_dir, _, _ = setup_experiment_dir(
                base_dir=os.path.join(os.getcwd(), config.output.base_dir),
                experiment_name=config.logging.experiment_name,
            )
            # Setup logging after distributed initialization
            setup_logging(
                log_dir=log_dir,
                log_level=logging.DEBUG if config.debug.enabled else logging.INFO,
            )
            logging.info(f"Starting training on {config.distributed.num_gpus} GPUs")
        else:
            experiment_dir = os.path.join(
                os.getcwd(), config.output.base_dir, config.logging.experiment_name
            )
            # Setup logging for non-rank-0 processes
            log_dir = os.path.join(experiment_dir, "logs")
            setup_logging(
                log_dir=log_dir,
                log_level=logging.DEBUG if config.debug.enabled else logging.INFO,
            )

        # Create trainer
        trainer = create_trainer(
            config=config,
            output_dir=experiment_dir,
            resume_from=config.training.get("resume_from", None),
        )

        try:
            # Start training
            trainer.train()
        except Exception as e:
            if rank == 0:
                logging.error(f"Training failed with error: {str(e)}")
            raise

    except Exception as e:
        logging.error(f"Setup failed with error: {str(e)}")
        raise

    finally:
        # Cleanup distributed training
        cleanup_distributed()


if __name__ == "__main__":
    hydra_entry()
