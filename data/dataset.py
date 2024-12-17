import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .box_ops import box_cxcywh_to_xyxy
from .coco import build_dataset
from .transforms import make_coco_transforms

# Constants for common field names
FIELD_ORIG_SIZE = "orig_size"
FIELD_RESIZE_SCALE = "resize_scale"
FIELD_PADDING = "padding"
FIELD_UNPADDED_SIZE = "unpadded_size"
FIELD_PADDED_SIZE = "padded_size"
FIELD_PADDING_MASK = "padding_mask"
FIELD_BOXES = "boxes"
FIELD_LABELS = "labels"

# Constants for data configuration
CONFIG_DATA = "data"
CONFIG_TRAINING = "training"
CONFIG_DEBUG = "debug"
CONFIG_AUGMENTATION = "augmentation"

# Constants for normalization
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
SEED = 43

# Constants for augmentation
DEFAULT_TRAIN_SCALES = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
DEFAULT_VAL_SCALES = [800]
DEFAULT_MAX_SIZE = 1333
DEFAULT_MIN_SIZE = 384
DEFAULT_CROP_SIZE = 600


def create_padding_mask(batch_images: List[torch.Tensor]) -> torch.Tensor:
    """Create padding mask for a batch of images of different sizes."""
    max_size = tuple(max(s) for s in zip(*[img.shape for img in batch_images]))

    # Create padding mask (True indicates padding)
    padding_masks = []
    for img in batch_images:
        # Create mask of shape [H, W] where True indicates padding
        padding_mask = torch.ones((max_size[-2], max_size[-1]), dtype=torch.bool)
        padding_mask[: img.shape[-2], : img.shape[-1]] = False
        padding_masks.append(padding_mask)

    return torch.stack(padding_masks)


def collate_fn(
    batch: List[Tuple[torch.Tensor, Dict]],
) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Custom collate function for DETR batches.
    Handles variable-sized images by padding to the largest size in the batch.
    Properly adjusts box coordinates through resizing and padding.

    Args:
        batch: List of tuples (image, target)
            - image: [3, H, W]
            - target: Dict containing annotations

    Returns:
        images: Tensor of shape [B, 3, max_H, max_W]
        targets: List of target dictionaries
    """
    batch = list(zip(*batch))
    images, targets = batch[0], batch[1]

    # Get maximum size in the batch
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))

    # Create padded image tensor
    padded_images = []
    for img, target in zip(images, targets):
        # Store original size before any transformations
        if FIELD_ORIG_SIZE not in target:
            target[FIELD_ORIG_SIZE] = torch.tensor(list(img.shape[-2:]))

        # Calculate resize scale
        orig_h, orig_w = target[FIELD_ORIG_SIZE]
        curr_h, curr_w = img.shape[-2:]
        scale_h = curr_h / orig_h
        scale_w = curr_w / orig_w

        # Store resize scale for box coordinate adjustment
        target[FIELD_RESIZE_SCALE] = torch.tensor([scale_w, scale_h])

        # Calculate padding sizes (only right and bottom)
        pad_h = max_size[-2] - curr_h
        pad_w = max_size[-1] - curr_w

        # Store padding info
        target[FIELD_PADDING] = torch.tensor(
            [0, pad_w, 0, pad_h]
        )  # left, right, top, bottom
        target[FIELD_UNPADDED_SIZE] = torch.tensor([curr_h, curr_w])
        target[FIELD_PADDED_SIZE] = torch.tensor([max_size[-2], max_size[-1]])

        # Apply padding (left=0, right=pad_w, top=0, bottom=pad_h)
        padded_img = F.pad(img, (0, pad_w, 0, pad_h), value=0)
        padded_images.append(padded_img)

        # Create padding mask (True indicates padding)
        padding_mask = torch.ones((max_size[-2], max_size[-1]), dtype=torch.bool)
        padding_mask[:curr_h, :curr_w] = False
        target[FIELD_PADDING_MASK] = padding_mask

        # Adjust box coordinates if they exist
        if FIELD_BOXES in target and len(target[FIELD_BOXES]):
            boxes = target[FIELD_BOXES]  # boxes are in cxcywh format and normalized

            # First denormalize using original size
            boxes_denorm = boxes * torch.tensor(
                [orig_w, orig_h, orig_w, orig_h], dtype=torch.float32
            )

            # Apply resize scale
            boxes_resized = boxes_denorm * torch.tensor(
                [scale_w, scale_h, scale_w, scale_h], dtype=torch.float32
            )

            # Normalize using padded size
            boxes_norm = boxes_resized / torch.tensor(
                [max_size[-1], max_size[-2], max_size[-1], max_size[-2]],
                dtype=torch.float32,
            )

            target[FIELD_BOXES] = boxes_norm

    # Stack images
    batched_images = torch.stack(padded_images)

    return batched_images, targets


class CocoDataset:
    """
    Dataset class for DETR object detection model.
    Handles COCO dataset loading and preparation.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the dataset.

        Args:
            config: Configuration dictionary containing data settings
                   Should include:
                   - data: Dict with paths and settings
                   - training: Dict with batch size and workers
                   - augmentation: Dict with augmentation settings
                   - debug: Dict with debug settings (optional)
                     - enabled: bool, whether to use debug mode
                     - samples: int, number of samples in debug mode
        """
        self.config = config

        # Data paths
        data_config = config[CONFIG_DATA]
        self.data_root_dir = data_config.data_root_dir
        self.train_dir = data_config.train_dir
        self.val_dir = data_config.val_dir
        self.train_ann = data_config.train_ann
        self.val_ann = data_config.val_ann

        # DataLoader settings
        training_config = config[CONFIG_TRAINING]
        self.batch_size = training_config.batch_size
        self.num_workers = training_config.num_workers
        self.pin_memory = data_config.pin_memory
        self.shuffle_train = data_config.shuffle_train
        self.shuffle_val = data_config.shuffle_val

        # Debug settings
        debug_config = getattr(config, CONFIG_DEBUG, None)
        self.debug_mode = debug_config.enabled if debug_config else False
        self.debug_samples = debug_config.num_batches if debug_config else 16

        # Augmentation settings
        aug_config = config[CONFIG_AUGMENTATION]
        self.train_transforms = make_coco_transforms("train", aug_config)
        self.val_transforms = make_coco_transforms("val", aug_config)

        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self._setup_datasets()

    def _setup_datasets(self):
        """Initialize train and validation datasets."""
        common_args = {
            "data_root_dir": self.data_root_dir,
            "debug_mode": self.debug_mode,
            "debug_samples": self.debug_samples,
        }

        self.train_dataset = build_dataset(
            image_set="train",
            transforms=self.train_transforms,
            train_dir=self.train_dir,
            train_ann=self.train_ann,
            **common_args,
        )

        self.val_dataset = build_dataset(
            image_set="val",
            transforms=self.val_transforms,
            val_dir=self.val_dir,
            val_ann=self.val_ann,
            **common_args,
        )

    def _get_dataloader(self, dataset, shuffle: bool) -> DataLoader:
        """Create and return a dataloader with common settings."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
        )

    def get_train_dataloader(self) -> DataLoader:
        """Create and return training dataloader."""
        return self._get_dataloader(self.train_dataset, self.shuffle_train)

    def get_val_dataloader(self) -> DataLoader:
        """Create and return validation dataloader."""
        return self._get_dataloader(self.val_dataset, self.shuffle_val)

    def get_test_dataloader(self) -> DataLoader:
        """Create and return test dataloader (same as validation for COCO)."""
        return self.get_val_dataloader()


def plot_batch(
    images,
    targets,
    output_dir: str,
    batch_idx: int = 0,
    prefix: str = "aug",
):
    """Plot an image with its bounding boxes after augmentation."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert tensor to numpy and transpose to [H, W, C]
    img = images[batch_idx].permute(1, 2, 0).cpu().numpy()

    # Denormalize image
    img = NORMALIZE_STD * img + NORMALIZE_MEAN
    img = np.clip(img, 0, 1)

    # Create figure and axes
    plt.figure(figsize=(12, 12))
    plt.imshow(img)

    # Get target boxes and plot them
    target = targets[batch_idx]
    boxes = target[FIELD_BOXES].cpu()  # boxes are in cxcywh format and normalized
    labels = target[FIELD_LABELS].cpu().numpy()

    # Get image size for denormalization
    h, w = img.shape[:2]

    # Convert from cxcywh to xyxy format and denormalize
    boxes = box_cxcywh_to_xyxy(boxes)
    boxes = boxes * torch.tensor([w, h, w, h], dtype=torch.float32)
    boxes = boxes.numpy()

    # Plot each box with different colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(boxes)))
    for box, label, color in zip(boxes, labels, colors):
        x0, y0, x1, y1 = box
        w_box = x1 - x0
        h_box = y1 - y0

        rect = plt.Rectangle(
            (x0, y0),
            w_box,
            h_box,
            fill=False,
            color=color,
            linewidth=2,
            linestyle="-",
        )
        plt.gca().add_patch(rect)
        plt.text(
            x0,
            y0 - 5,
            f"Class {label}",
            color=color,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor=color),
            fontsize=10,
        )

    # Add image information
    info_text = [
        f"Image Size: {h}x{w}",
        f"Num Boxes: {len(boxes)}",
        f"Original Size: {target.get(FIELD_ORIG_SIZE, 'N/A').tolist()}",
    ]
    plt.text(
        10,
        30,
        "\n".join(info_text),
        color="white",
        bbox=dict(facecolor="black", alpha=0.7),
        fontsize=10,
        verticalalignment="top",
    )

    plt.axis("off")
    plt.savefig(
        os.path.join(output_dir, f"{prefix}_batch_{batch_idx}.png"),
        bbox_inches="tight",
        dpi=150,
    )
    plt.close()


def test_demo():
    """Demo to visualize augmentations with bounding boxes."""

    from omegaconf import DictConfig
    from utils.config import (
        AugmentationConfig,
        CenterCropConfig,
        DataConfig,
        DebugConfig,
        HorizontalFlipConfig,
        NormalizeConfig,
        RandomErasingConfig,
        RandomPadConfig,
        RandomResizeConfig,
        TrainAugmentationConfig,
        TrainingConfig,
        ValAugmentationConfig,
    )

    # Create basic config for demo
    config = {
        "data": DataConfig(
            data_root_dir="coco",
            train_dir="train2017",
            val_dir="val2017",
            train_ann="annotations/instances_train2017.json",
            val_ann="annotations/instances_val2017.json",
            pin_memory=True,
            shuffle_train=True,
            shuffle_val=False,
            persistent_workers=True,
        ),
        "training": TrainingConfig(
            batch_size=2,
            num_workers=0,
            seed=SEED,
            deterministic=True,
            gradient_clip_val=0.1,
            accumulate_grad_batches=1,
            check_val_every_n_epoch=1,
            precision=32,
            detect_anomaly=False,
            resume_from=None,
            resume_mode="latest",
        ),
        "debug": DebugConfig(
            enabled=False,
            num_batches=2,
        ),
    }
    config = DictConfig(config)

    # Create configs for each augmentation step
    aug_configs = {
        "1_horizontal_flip": AugmentationConfig(
            scales=[480, 512, 544, 576, 608],
            max_size=800,
            train=TrainAugmentationConfig(
                horizontal_flip=HorizontalFlipConfig(prob=0.5),
                normalize=NormalizeConfig(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ),
            val=ValAugmentationConfig(
                scales=[800],
                max_size=1333,
                normalize=NormalizeConfig(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ),
        ),
        "2_random_resize": AugmentationConfig(
            scales=[480, 512, 544, 576, 608],
            max_size=800,
            train=TrainAugmentationConfig(
                random_resize=RandomResizeConfig(
                    scales=[400, 500, 600], crop_size=[384, 600]
                ),
                normalize=NormalizeConfig(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ),
            val=ValAugmentationConfig(
                scales=[800],
                max_size=1333,
                normalize=NormalizeConfig(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ),
        ),
        "3_random_size_crop": AugmentationConfig(
            scales=[480, 512, 544, 576, 608],
            max_size=800,
            train=TrainAugmentationConfig(
                random_resize=RandomResizeConfig(
                    scales=[400, 500, 600], crop_size=[384, 600]
                ),
                normalize=NormalizeConfig(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ),
            val=ValAugmentationConfig(
                scales=[800],
                max_size=1333,
                normalize=NormalizeConfig(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ),
        ),
        "4_random_pad": AugmentationConfig(
            scales=[480, 512, 544, 576, 608],
            max_size=800,
            train=TrainAugmentationConfig(
                random_pad=RandomPadConfig(max_pad=100),
                normalize=NormalizeConfig(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ),
            val=ValAugmentationConfig(
                scales=[800],
                max_size=1333,
                normalize=NormalizeConfig(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ),
        ),
        "5_center_crop": AugmentationConfig(
            scales=[480, 512, 544, 576, 608],
            max_size=800,
            train=TrainAugmentationConfig(
                center_crop=CenterCropConfig(size=(384, 600)),
                normalize=NormalizeConfig(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ),
            val=ValAugmentationConfig(
                scales=[800],
                max_size=1333,
                center_crop=CenterCropConfig(size=(384, 600)),
                normalize=NormalizeConfig(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ),
        ),
        "6_random_erase": AugmentationConfig(
            scales=[480, 512, 544, 576, 608],
            max_size=800,
            train=TrainAugmentationConfig(
                random_erasing=RandomErasingConfig(
                    prob=0.8,
                    scale=(0.15, 0.5),
                    ratio=(0.3, 2.0),
                    value=0.5,
                ),
                normalize=NormalizeConfig(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ),
            val=ValAugmentationConfig(
                scales=[800],
                max_size=1333,
                normalize=NormalizeConfig(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ),
        ),
        "7_random_select": AugmentationConfig(
            scales=[480, 512, 544, 576, 608],
            max_size=800,
            train=TrainAugmentationConfig(
                random_resize=RandomResizeConfig(
                    scales=[400, 500, 600], crop_size=[384, 600]
                ),
                normalize=NormalizeConfig(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ),
            val=ValAugmentationConfig(
                scales=[800],
                max_size=1333,
                normalize=NormalizeConfig(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ),
        ),
        "8_full_pipeline": AugmentationConfig(
            scales=DEFAULT_TRAIN_SCALES,
            max_size=DEFAULT_MAX_SIZE,
            train=TrainAugmentationConfig(
                horizontal_flip=HorizontalFlipConfig(prob=0.5),
                random_pad=RandomPadConfig(max_pad=100),
                random_resize=RandomResizeConfig(
                    scales=[400, 500, 600], crop_size=[384, 600]
                ),
                center_crop=CenterCropConfig(size=(384, 600)),
                random_erasing=RandomErasingConfig(
                    prob=0.5,
                    scale=(0.02, 0.33),
                    ratio=(0.3, 3.3),
                    value=0.0,
                ),
                normalize=NormalizeConfig(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ),
            val=ValAugmentationConfig(
                scales=[800],
                max_size=1333,
                center_crop=CenterCropConfig(size=(384, 600)),
                normalize=NormalizeConfig(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ),
        ),
    }

    # Process each augmentation configuration
    for config_idx, (config_name, aug_config) in enumerate(aug_configs.items()):
        print(f"\nTesting {config_name} configuration:")

        try:
            # Update config with current augmentation config
            config.augmentation = aug_config

            # Test both train and val transforms
            for image_set in ["train", "val"]:
                # Build dataset with current augmentation config
                dataset = build_dataset(
                    image_set=image_set,
                    data_root_dir=config.data.data_root_dir,
                    transforms=make_coco_transforms(image_set, config.augmentation),
                    train_dir=config.data.train_dir,
                    val_dir=config.data.val_dir,
                    train_ann=config.data.train_ann,
                    val_ann=config.data.val_ann,
                    debug_mode=config.debug.enabled,
                    debug_samples=config.debug.num_batches,
                )

                # Test multiple samples and multiple runs for each sample
                num_samples = 3  # Number of different images to test
                num_runs = (
                    5 if image_set == "train" else 1
                )  # Only one run for val since it's deterministic

                # Use different base indices for each configuration and validation
                base_idx = config_idx * num_samples
                if image_set == "val":
                    base_idx += (
                        len(aug_configs) * num_samples
                    )  # Use later samples for validation

                for sample_idx in range(num_samples):
                    # Calculate actual dataset index
                    dataset_idx = (base_idx + sample_idx) % len(dataset)

                    # Get a sample image
                    base_img, base_target = dataset[dataset_idx]

                    # Run augmentation multiple times on the same image
                    for run_idx in range(num_runs):
                        # Get a new augmented version
                        img, target = dataset[dataset_idx]

                        # Create a batch of one image
                        images = img.unsqueeze(0)
                        targets = [target]

                        # Store original size
                        if FIELD_ORIG_SIZE not in target:
                            target[FIELD_ORIG_SIZE] = torch.tensor(list(img.shape[-2:]))

                        # Create output directory for this configuration and sample
                        config_output_dir = os.path.join(
                            "augmentation_demo",
                            f"{config_idx+1:02d}_{config_name}",
                            image_set,
                        )
                        os.makedirs(config_output_dir, exist_ok=True)

                        # Plot the sample
                        plot_batch(
                            images,
                            targets,
                            output_dir=config_output_dir,
                            batch_idx=0,
                            prefix=f"sample{sample_idx}_run{run_idx}",
                        )

                print(
                    f"Saved {num_samples} samples with {num_runs} runs each for {image_set} set"
                )

        except Exception as e:
            print(f"Error testing {config_name} configuration: {str(e)}")
            continue

    print("\nAugmentation demonstration complete!")


if __name__ == "__main__":
    test_demo()
