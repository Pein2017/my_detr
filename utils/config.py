# Configuration utilities for DETR
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from omegaconf import MISSING, OmegaConf


@dataclass
class DataConfig:
    """Configuration for dataset paths and loading settings."""

    data_root_dir: str = MISSING
    train_dir: str = MISSING
    val_dir: str = MISSING
    train_ann: str = MISSING
    val_ann: str = MISSING
    pin_memory: bool = MISSING
    shuffle_train: bool = MISSING
    shuffle_val: bool = MISSING
    persistent_workers: bool = MISSING


@dataclass
class OutputConfig:
    """Configuration for output directories."""

    base_dir: str = MISSING
    log_dir: str = MISSING
    checkpoint_dir: str = MISSING
    visualization_dir: str = MISSING


@dataclass
class PositionEmbeddingConfig:
    """Configuration for position embedding."""

    type: str = MISSING
    normalize: bool = MISSING


@dataclass
class BBoxPredictorConfig:
    """Configuration for bounding box predictor."""

    num_layers: int = MISSING
    hidden_dim: int = MISSING


@dataclass
class InitConfig:
    """Configuration for model initialization."""

    xavier_uniform: bool = MISSING
    prior_prob: float = MISSING


@dataclass
class ModelConfig:
    """Configuration for DETR model architecture."""

    backbone_name: str = MISSING
    pretrained_backbone: bool = MISSING
    num_classes: int = MISSING
    hidden_dim: int = MISSING
    nheads: int = MISSING
    num_encoder_layers: int = MISSING
    num_decoder_layers: int = MISSING
    dim_feedforward: int = MISSING
    dropout: float = MISSING
    num_queries: int = MISSING
    learnable_tgt: bool = MISSING
    position_embedding: PositionEmbeddingConfig = field(
        default_factory=PositionEmbeddingConfig
    )
    bbox_predictor: BBoxPredictorConfig = field(default_factory=BBoxPredictorConfig)
    init: InitConfig = field(default_factory=InitConfig)


@dataclass
class TrainingConfig:
    """Configuration for training process."""

    max_epochs: int = MISSING
    batch_size: int = MISSING
    num_workers: int = MISSING
    seed: int = MISSING
    deterministic: bool = MISSING
    gradient_clip_val: float = MISSING
    accumulate_grad_batches: int = MISSING
    check_val_every_n_epoch: int = MISSING
    precision: int = MISSING
    detect_anomaly: bool = MISSING
    resume_from: Optional[str] = None
    resume_mode: str = MISSING


@dataclass
class LRSchedulerConfig:
    """Configuration for learning rate scheduler."""

    name: str = MISSING
    step_size: int = MISSING
    gamma: float = MISSING
    warmup_epochs: int = MISSING
    warmup_factor: float = MISSING
    min_lr: float = MISSING


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""

    name: str = MISSING
    lr: float = MISSING
    weight_decay: float = MISSING
    backbone_lr_factor: float = MISSING
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)


@dataclass
class LossConfig:
    """Configuration for loss computation."""

    cost_class: float = MISSING
    cost_bbox: float = MISSING
    cost_giou: float = MISSING
    class_loss_coef: float = MISSING
    bbox_loss_coef: float = MISSING
    giou_loss_coef: float = MISSING
    empty_weight: float = MISSING


@dataclass
class NormalizeConfig:
    """Configuration for input normalization."""

    mean: List[float] = MISSING
    std: List[float] = MISSING


@dataclass
class HorizontalFlipConfig:
    """Configuration for horizontal flip augmentation."""

    enabled: bool = MISSING
    prob: float = MISSING


@dataclass
class RandomResizeConfig:
    """Configuration for random resize augmentation."""

    enabled: bool = MISSING
    scales: List[int] = MISSING
    crop_size: List[int] = MISSING


@dataclass
class ColorJitterConfig:
    """Configuration for color jitter augmentation."""

    enabled: bool = MISSING


@dataclass
class TrainAugmentationConfig:
    """Configuration for training augmentations."""

    horizontal_flip: HorizontalFlipConfig = field(default_factory=HorizontalFlipConfig)
    random_resize: RandomResizeConfig = field(default_factory=RandomResizeConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)


@dataclass
class ValAugmentationConfig:
    """Configuration for validation augmentations."""

    scales: List[int] = MISSING
    max_size: int = MISSING
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation pipeline."""

    scales: List[int] = MISSING
    max_size: int = MISSING
    min_size: int = MISSING
    color_jitter: ColorJitterConfig = field(default_factory=ColorJitterConfig)
    train: TrainAugmentationConfig = field(default_factory=TrainAugmentationConfig)
    val: ValAugmentationConfig = field(default_factory=ValAugmentationConfig)


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    strategy: str = MISSING
    sync_batchnorm: bool = MISSING
    find_unused_parameters: bool = MISSING
    num_nodes: int = MISSING
    num_gpus: int = MISSING


@dataclass
class TensorBoardConfig:
    """Configuration for TensorBoard logging."""

    default_hp_metric: bool = MISSING
    log_dir: str = MISSING


@dataclass
class VisualizationConfig:
    """Configuration for visualization during training."""

    enabled: bool = MISSING
    num_images: int = MISSING
    score_threshold: float = MISSING


@dataclass
class DebugConfig:
    """Configuration for debug mode.

    Attributes:
        enabled: Whether debug mode is enabled
        num_batches: Number of batches to process in debug mode (limits dataset size)
        samples: Number of samples to visualize in debug mode
    """

    enabled: bool = False
    num_batches: int = 2  # Default to 2 batches in debug mode
    samples: int = MISSING


@dataclass
class ProcessLogsConfig:
    """Configuration for process logs."""

    enabled: bool = True
    output_dir: str = "tensorboard/process_logs"


@dataclass
class LoggingConfig:
    """Configuration for logging and visualization.

    The logging frequency is controlled by log_step_ratio, which determines what fraction
    of steps within an epoch should trigger logging. For example, if log_step_ratio is 0.05,
    logging will occur every 5% of the total steps in an epoch. The actual logging frequency
    in steps will be calculated as: max(1, int(total_steps_per_epoch * log_step_ratio)).
    """

    save_top_k: int = MISSING
    monitor_metric: str = MISSING
    monitor_mode: str = MISSING
    experiment_name: str = MISSING
    log_step_ratio: float = 0.05  # Default to logging every 5% of steps in an epoch
    save_checkpoint_every_n_epochs: int = 1
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    process_logs: ProcessLogsConfig = field(default_factory=ProcessLogsConfig)


@dataclass
class CalculatedConfig:
    """Configuration for calculated values."""

    total_batch_size: int = MISSING
    experiment_name: str = MISSING


@dataclass
class DETRConfig:
    """Root configuration class for DETR."""

    defaults: List[Union[str, dict]] = field(default_factory=list)
    hydra: dict = field(default_factory=dict)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    calculated: CalculatedConfig = field(default_factory=CalculatedConfig)


def load_config(
    config_name: str = "default", config_dir: str = "configs"
) -> DETRConfig:
    """Load and process configuration file.

    Args:
        config_name: Name of config file without .yaml extension (e.g. 'default', 'experiment/small')
        config_dir: Directory containing config files
    """
    # Load base structured config
    base_conf = OmegaConf.structured(DETRConfig)

    # Determine config path
    config_path = os.path.join(config_dir, f"{config_name}.yaml")
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found: {config_path}")

    # Load user config
    user_conf = OmegaConf.load(config_path)

    # If this is a variant, load its parent first
    if "defaults" in user_conf:
        for default in user_conf.defaults:
            if isinstance(default, str):
                parent_path = os.path.join(config_dir, f"{default}.yaml")
                if os.path.exists(parent_path):
                    parent_conf = OmegaConf.load(parent_path)
                    base_conf = OmegaConf.merge(base_conf, parent_conf)
            elif isinstance(default, dict):
                for _, path in default.items():
                    if path != "_self_":
                        parent_path = os.path.join(config_dir, f"{path}.yaml")
                        if os.path.exists(parent_path):
                            parent_conf = OmegaConf.load(parent_path)
                            base_conf = OmegaConf.merge(base_conf, parent_conf)

    # Merge with user config (overrides parent settings)
    config = OmegaConf.merge(base_conf, user_conf)

    # Process environment variables
    config = _process_env_vars(config)

    # Create directories
    _create_output_dirs(config)

    return config


def _process_env_vars(config: DETRConfig) -> DETRConfig:
    """Replace environment variables in configuration."""
    config_dict = OmegaConf.to_container(config, resolve=True)
    processed_dict = {}

    def process_value(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            if env_var not in os.environ:
                raise ValueError(f"Environment variable {env_var} not set")
            return os.environ[env_var]
        elif isinstance(value, dict):
            return {k: process_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [process_value(v) for v in value]
        return value

    processed_dict = process_value(config_dict)
    return OmegaConf.create(processed_dict)


def _create_output_dirs(config: DETRConfig) -> None:
    """Create output directories if they don't exist."""
    dirs = [
        config.output.log_dir,
        config.output.checkpoint_dir,
        config.output.visualization_dir,
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
