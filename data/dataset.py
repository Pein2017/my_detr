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
        if "orig_size" not in target:
            target["orig_size"] = torch.tensor(list(img.shape[-2:]))

        # Calculate resize scale
        orig_h, orig_w = target["orig_size"]
        curr_h, curr_w = img.shape[-2:]
        scale_h = curr_h / orig_h
        scale_w = curr_w / orig_w

        # Store resize scale for box coordinate adjustment
        target["resize_scale"] = torch.tensor([scale_w, scale_h])

        # Calculate padding sizes (only right and bottom)
        pad_h = max_size[-2] - curr_h
        pad_w = max_size[-1] - curr_w

        # Store padding info
        target["padding"] = torch.tensor(
            [0, pad_w, 0, pad_h]
        )  # left, right, top, bottom
        target["unpadded_size"] = torch.tensor([curr_h, curr_w])
        target["padded_size"] = torch.tensor([max_size[-2], max_size[-1]])

        # Apply padding (left=0, right=pad_w, top=0, bottom=pad_h)
        padded_img = F.pad(img, (0, pad_w, 0, pad_h), value=0)
        padded_images.append(padded_img)

        # Create padding mask (True indicates padding)
        padding_mask = torch.ones((max_size[-2], max_size[-1]), dtype=torch.bool)
        padding_mask[:curr_h, :curr_w] = False
        target["padding_mask"] = padding_mask

        # Adjust box coordinates if they exist
        if "boxes" in target and len(target["boxes"]):
            boxes = target["boxes"]  # boxes are in cxcywh format and normalized

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

            target["boxes"] = boxes_norm

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
        self.data_root_dir = config.data.data_root_dir
        self.train_dir = config.data.train_dir
        self.val_dir = config.data.val_dir
        self.train_ann = config.data.train_ann
        self.val_ann = config.data.val_ann

        # DataLoader settings
        self.batch_size = config.training.batch_size
        self.num_workers = config.training.num_workers
        self.pin_memory = config.data.pin_memory
        self.shuffle_train = config.data.shuffle_train
        self.shuffle_val = config.data.shuffle_val

        # Debug settings
        self.debug_mode = config.debug.enabled if hasattr(config, "debug") else False
        self.debug_samples = (
            config.debug.num_batches if hasattr(config, "debug") else 16
        )

        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self._setup_datasets()

    def _setup_datasets(self):
        """Initialize train and validation datasets."""
        self.train_dataset = build_dataset(
            image_set="train",
            data_root_dir=self.data_root_dir,
            transforms=make_coco_transforms("train", self.config.augmentation),
            train_dir=self.train_dir,
            train_ann=self.train_ann,
            debug_mode=self.debug_mode,
            debug_samples=self.debug_samples,
        )

        self.val_dataset = build_dataset(
            image_set="val",
            data_root_dir=self.data_root_dir,
            transforms=make_coco_transforms("val", self.config.augmentation),
            val_dir=self.val_dir,
            val_ann=self.val_ann,
            debug_mode=self.debug_mode,
            debug_samples=self.debug_samples,
        )

    def get_train_dataloader(self) -> DataLoader:
        """Create and return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
        )

    def get_val_dataloader(self) -> DataLoader:
        """Create and return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_val,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
        )

    def get_test_dataloader(self) -> DataLoader:
        """Create and return test dataloader (same as validation for COCO)."""
        return self.get_val_dataloader()


def plot_batch(images, targets, prefix: str, batch_idx: int, output_dir="viz_output"):
    """
    Plot a batch of images with their bounding boxes and save to files.
    Visualizes boxes through resizing and padding transformations.

    Args:
        images: Tensor of shape [B, 3, H, W]
        targets: List of dictionaries containing target annotations
        prefix: Prefix for output filename
        batch_idx: Which batch to visualize
        output_dir: Directory to save visualization images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert tensor to numpy and transpose to [H, W, C]
    img = images[batch_idx].permute(1, 2, 0).cpu().numpy()

    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Get target boxes and plot them
    target = targets[batch_idx]
    boxes = target["boxes"].cpu()  # boxes are in cxcywh format and normalized
    labels = target["labels"].cpu().numpy()

    # Get padded size for denormalization
    pad_h, pad_w = target["padded_size"]

    # Convert from cxcywh to xyxy format
    boxes = box_cxcywh_to_xyxy(boxes)

    # Denormalize boxes using padded size
    boxes = boxes * torch.tensor([pad_w, pad_h, pad_w, pad_h], dtype=torch.float32)
    boxes = boxes.numpy()

    # Plot each box
    for box, label in zip(boxes, labels):
        x0, y0, x1, y1 = box
        w_box = x1 - x0
        h_box = y1 - y0

        rect = plt.Rectangle(
            (x0, y0), w_box, h_box, fill=False, color="red", linewidth=2
        )
        ax.add_patch(rect)
        ax.text(
            x0,
            y0,
            f"Class {label}",
            bbox=dict(facecolor="white", alpha=0.7),
        )

    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{prefix}_batch_{batch_idx}.png"))
    plt.close()


def main():
    """Test data loading and visualize multiple batches."""
    from utils.config import (
        AugmentationConfig,
        HorizontalFlipConfig,
        NormalizeConfig,
        RandomResizeConfig,
        TrainAugmentationConfig,
        ValAugmentationConfig,
    )

    # Create augmentation config
    train_normalize = NormalizeConfig(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    val_normalize = NormalizeConfig(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    horizontal_flip = HorizontalFlipConfig(
        enabled=True,
        prob=0.5,
    )

    random_resize = RandomResizeConfig(
        enabled=True,
        scales=[400, 500, 600],
        crop_size=[384, 600],
    )

    train_aug = TrainAugmentationConfig(
        horizontal_flip=horizontal_flip,
        random_resize=random_resize,
        normalize=train_normalize,
    )

    val_aug = ValAugmentationConfig(
        scales=[800],
        max_size=1333,
        normalize=val_normalize,
    )

    aug_config = AugmentationConfig(
        scales=[480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
        max_size=1333,
        train=train_aug,
        val=val_aug,
    )

    # Initialize dataset with your COCO path
    bs = 4
    dataset = CocoDataset(
        config={
            "data": {
                "data_root_dir": "/data/training_code/Pein/DETR/my_detr/coco",
                "train_dir": "train2017",
                "val_dir": "val2017",
                "train_ann": "annotations/instances_train2017.json",
                "val_ann": "annotations/instances_val2017.json",
                "pin_memory": True,
                "shuffle_train": True,
                "shuffle_val": False,
            },
            "training": {
                "batch_size": bs,
                "num_workers": 4,
            },
            "augmentation": aug_config,
        }
    )
    output_dir = "test_viz"

    # Get train and val dataloaders
    train_loader = dataset.get_train_dataloader()
    val_loader = dataset.get_val_dataloader()

    print("\nDataset sizes:")
    print(f"Training set: {len(dataset.train_dataset)} images")
    print(f"Validation set: {len(dataset.val_dataset)} images")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)

    # Visualize multiple training batches
    print("\nGenerating training set visualizations...")
    for batch_num in range(bs + 1):  # Generate 5 batches of samples
        try:
            train_batch = next(iter(train_loader))
            images, targets = train_batch

            print(f"\nTraining Batch {batch_num + 1}:")
            print(f"Images shape: {images.shape}")
            print(f"Number of targets: {len(targets)}")

            # Visualize each image in the batch
            for idx in range(images.shape[0]):
                plot_batch(
                    images,
                    targets,
                    "train",
                    batch_idx=idx,
                    output_dir=output_dir,
                )

                # Print target information for the first image of each batch
                if idx == 0:
                    print("\nFirst image targets:")
                    for k, v in targets[idx].items():
                        if isinstance(v, torch.Tensor):
                            print(f"{k}: shape {v.shape}, dtype {v.dtype}")
                        else:
                            print(f"{k}: {v}")
        except StopIteration:
            print("Reached end of training dataset")
            break

    # Visualize multiple validation batches
    print("\nGenerating validation set visualizations...")
    for batch_num in range(bs + 1):
        try:
            val_batch = next(iter(val_loader))
            val_images, val_targets = val_batch

            print(f"\nValidation Batch {batch_num + 1}:")
            print(f"Images shape: {val_images.shape}")
            print(f"Number of targets: {len(val_targets)}")

            # Visualize each image in the batch
            for idx in range(val_images.shape[0]):
                plot_batch(
                    val_images,
                    val_targets,
                    "val",
                    batch_idx=idx,
                    output_dir=output_dir,
                )

                # Print target information for the first image of each batch
                if idx == 0:
                    print("\nFirst image targets:")
                    for k, v in val_targets[idx].items():
                        if isinstance(v, torch.Tensor):
                            print(f"{k}: shape {v.shape}, dtype {v.dtype}")
                        else:
                            print(f"{k}: {v}")
        except StopIteration:
            print("Reached end of validation dataset")
            break

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
