import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torchvision
from PIL import Image

# Unified cache for both real and fake COCO datasets
_coco_cache = {}


class FakeCOCO:
    """A COCO-like interface for fake datasets."""

    def __init__(self, images, annotations):
        self.images = images
        self.annotations = annotations
        self.imgs = {img["id"]: img for img in images}
        self.anns = {ann["id"]: ann for ann in annotations}
        self.imgToAnns = self._create_image_to_anns_mapping()
        self.cats = self._create_category_mapping()

    def _create_image_to_anns_mapping(self):
        """Create mapping from image id to annotations."""
        mapping = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in mapping:
                mapping[img_id] = []
            mapping[img_id].append(ann)
        return mapping

    def _create_category_mapping(self):
        """Create fake COCO category mapping."""
        categories = set()
        for ann in self.annotations:
            categories.add(ann["category_id"])
        return {
            cat_id: {"id": cat_id, "name": f"class_{cat_id}"}
            for cat_id in sorted(categories)
        }

    def getCatIds(self, catNms=None, supNms=None, catIds=None):
        """Get category IDs matching the given filters.

        Args:
            catNms: Category names to filter by (not used in fake dataset)
            supNms: Supercategory names to filter by (not used in fake dataset)
            catIds: Category IDs to filter by

        Returns:
            List of category IDs
        """
        # If specific catIds are provided, filter by them
        if catIds is not None:
            return [cat_id for cat_id in catIds if cat_id in self.cats]
        # Otherwise return all category IDs
        return sorted(self.cats.keys())

    def getImgIds(self):
        """Get all image IDs."""
        return sorted(self.imgs.keys())

    def loadImgs(self, ids):
        if isinstance(ids, list):
            return [self.imgs[img_id] for img_id in ids if img_id in self.imgs]
        return [self.imgs[ids]] if ids in self.imgs else []

    def loadAnns(self, ids):
        if isinstance(ids, list):
            return [self.anns[ann_id] for ann_id in ids if ann_id in self.anns]
        return [self.anns[ids]] if ids in self.anns else []

    def getAnnIds(self, imgIds=None, catIds=None, areaRng=None, iscrowd=None):
        """Get annotation ids matching the given filters."""
        # Start with all annotations
        anns = self.annotations

        # Filter by image ids
        if imgIds is not None:
            if isinstance(imgIds, (int, float)):
                # Single image id
                anns = self.imgToAnns.get(imgIds, [])
            else:
                # List of image ids
                anns = []
                for imgId in imgIds:
                    anns.extend(self.imgToAnns.get(imgId, []))

        # Filter by category ids
        if catIds is not None:
            anns = [ann for ann in anns if ann["category_id"] in catIds]

        # Filter by area range
        if areaRng is not None:
            anns = [ann for ann in anns if areaRng[0] <= ann["area"] <= areaRng[1]]

        # Filter by crowd flag
        if iscrowd is not None:
            anns = [ann for ann in anns if ann["iscrowd"] == iscrowd]

        return [ann["id"] for ann in anns]

    def loadRes(self, predictions):
        """Create a new FakeCOCO object with prediction results."""
        if not predictions:
            return FakeCOCO(self.images, [])

        # Create new annotations from predictions
        new_anns = []
        for idx, pred in enumerate(predictions):
            new_anns.append(
                {
                    "id": idx,
                    "image_id": pred["image_id"],
                    "category_id": pred["category_id"],
                    "bbox": pred["bbox"],
                    "score": pred["score"],
                    "area": pred["bbox"][2] * pred["bbox"][3],
                    "iscrowd": 0,
                }
            )

        return FakeCOCO(self.images, new_anns)


def create_fake_coco_data(root_dir: str, num_samples: int = 16) -> tuple:
    """Create fake COCO dataset for debugging.

    Generates visually distinct images with random shapes and colors for easy debugging.
    Each image contains 1-3 objects with different shapes (rectangles, circles, triangles).
    Training and validation images are visually distinct to help identify them.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    # Create fake images and annotations
    images = []
    annotations = []
    image_ids = []

    # Fixed size for fake images (larger for better visibility)
    img_size = (320, 320)

    # Colors for better visibility
    COLORS = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]

    # Create directory if it doesn't exist
    os.makedirs(root_dir, exist_ok=True)

    # Determine if this is validation set based on directory name
    is_val = "val" in str(root_dir).lower()
    bg_color = (
        (30, 30, 50) if is_val else (50, 50, 50)
    )  # Darker blue tint for validation

    for img_id in range(num_samples):
        # Create background with different tint for train/val
        img = Image.new("RGB", img_size, color=bg_color)
        draw = ImageDraw.Draw(img)

        # Add a small text indicator in the corner
        draw.text((10, 10), "VAL" if is_val else "TRAIN", fill=(200, 200, 200))

        # Save the image
        img_filename = f"fake_img_{img_id:04d}.jpg"
        img_path = os.path.join(root_dir, img_filename)

        # Add image info
        images.append(
            {
                "id": img_id,
                "file_name": img_filename,
                "height": img_size[1],
                "width": img_size[0],
            }
        )
        image_ids.append(img_id)

        # Create 1-3 random shapes per image
        num_shapes = np.random.randint(1, 4)
        for ann_id in range(num_shapes):
            # Random color for this shape
            color = COLORS[np.random.randint(0, len(COLORS))]

            # Random box dimensions (larger for better visibility)
            x = np.random.randint(0, img_size[0] - 100)
            y = np.random.randint(0, img_size[1] - 100)
            w = np.random.randint(50, min(img_size[0] - x, 150))
            h = np.random.randint(50, min(img_size[1] - y, 150))

            # Draw shape with thick outline
            shape_type = np.random.randint(0, 3)
            if shape_type == 0:  # Rectangle
                draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
                # Add fill with transparency for validation
                if is_val:
                    r, g, b = color
                    fill_color = (r, g, b, 64)  # Semi-transparent
                    draw.rectangle(
                        [x + 3, y + 3, x + w - 3, y + h - 3], fill=fill_color
                    )
            elif shape_type == 1:  # Circle
                draw.ellipse([x, y, x + w, y + h], outline=color, width=3)
                if is_val:
                    r, g, b = color
                    fill_color = (r, g, b, 64)
                    draw.ellipse([x + 3, y + 3, x + w - 3, y + h - 3], fill=fill_color)
            else:  # Triangle
                points = [
                    (x + w // 2, y),  # top
                    (x, y + h),  # bottom left
                    (x + w, y + h),  # bottom right
                ]
                draw.polygon(points, outline=color, width=3)
                if is_val:
                    r, g, b = color
                    fill_color = (r, g, b, 64)
                    inner_points = [  # Slightly smaller triangle for fill
                        (x + w // 2, y + 3),
                        (x + 3, y + h - 3),
                        (x + w - 3, y + h - 3),
                    ]
                    draw.polygon(inner_points, fill=fill_color)

            # Add annotation
            annotations.append(
                {
                    "id": len(annotations),
                    "image_id": img_id,
                    "category_id": shape_type
                    + 1,  # 1: rectangle, 2: circle, 3: triangle
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )

        # Save the image
        img.save(img_path, quality=95)
        logging.debug(f"Saved debug image: {img_path}")

    # Create COCO-like interface
    fake_coco = FakeCOCO(images, annotations)

    logging.info(
        f"Created fake {'validation' if is_val else 'training'} dataset with {len(images)} images and {len(annotations)} annotations in {root_dir}"
    )

    return fake_coco, image_ids


class CocoDetection(torchvision.datasets.CocoDetection):
    """COCO dataset with custom transforms for object detection."""

    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        transforms: callable = None,
        debug_mode: bool = False,
        debug_samples: int = 16,
    ):
        self.root = img_folder
        self.transforms = transforms

        if debug_mode:
            # Get or create fake COCO dataset
            self.coco = get_coco_api_from_dataset(
                ann_file="fake_dataset",
                is_fake=True,
                root_dir=img_folder,
                num_samples=debug_samples,
            )
            self.ids = list(range(debug_samples))
        else:
            # Get or create real COCO dataset
            self.coco = get_coco_api_from_dataset(ann_file, is_fake=False)
            self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict]:
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample to fetch

        Returns:
            tuple: (image, target) where:
                - image is a PIL Image
                - target is a dict containing:
                    - boxes (Tensor[N, 4]): The ground-truth boxes in [x0, y0, x1, y1] format
                    - labels (Tensor[N]): The class labels for each ground-truth box
                    - image_id (Tensor[1]): The image ID
                    - area (Tensor[N]): The area of each box
                    - iscrowd (Tensor[N]): Whether the target is a crowd (1) or not (0)
                    - orig_size (Tensor[2]): Original image size [H, W]
                    - size (Tensor[2]): Processed image size [H, W]
        """
        # Load image and annotations using parent class
        coco = self.coco
        img_id = self.ids[idx]

        # Load image
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        # Load annotations
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        # Get image size
        w, h = img.size
        image_id = self.ids[idx]

        # Initialize lists for annotations
        boxes = []
        classes = []
        areas = []
        iscrowds = []

        # Process annotations
        for anno in target:
            # Skip crowd annotations
            if anno.get("iscrowd", 0) == 1:
                continue

            # Get bbox in XYXY format
            bbox = anno["bbox"]
            bbox = [
                bbox[0],  # x
                bbox[1],  # y
                bbox[0] + bbox[2],  # x + w
                bbox[1] + bbox[3],  # y + h
            ]

            # Clip boxes to image size
            bbox = [
                min(max(0, bbox[0]), w),
                min(max(0, bbox[1]), h),
                min(max(0, bbox[2]), w),
                min(max(0, bbox[3]), h),
            ]

            # Skip invalid boxes
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue

            boxes.append(bbox)
            classes.append(anno["category_id"])
            areas.append(anno["area"])
            iscrowds.append(anno.get("iscrowd", 0))

        # Create target dictionary
        target = {}

        # Convert to tensors
        if boxes:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(classes, dtype=torch.int64)
            target["area"] = torch.as_tensor(areas, dtype=torch.float32)
            target["iscrowd"] = torch.as_tensor(iscrowds, dtype=torch.uint8)
        else:
            # Handle empty annotations
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.uint8)

        # Add image information
        target["image_id"] = torch.tensor([image_id])
        target["orig_size"] = torch.as_tensor([h, w])
        target["size"] = torch.as_tensor([h, w])

        # Apply transforms if any
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def get_coco_api_from_dataset(
    ann_file: str,
    is_fake: bool = False,
    root_dir: Optional[str] = None,
    num_samples: int = 16,
) -> Union[CocoDetection, FakeCOCO]:
    """Get COCO API object with caching support.

    Args:
        ann_file: Path to annotation file or identifier for fake dataset
        is_fake: Whether to create a fake dataset
        root_dir: Directory for fake dataset images (required if is_fake=True)
        num_samples: Number of samples for fake dataset

    Returns:
        COCO or FakeCOCO object
    """
    # Include root_dir in cache key to differentiate between train and val
    cache_key = f"{ann_file}_{is_fake}_{num_samples}_{root_dir}"

    if cache_key in _coco_cache:
        logging.info(f"Using cached COCO dataset for {ann_file} in {root_dir}")
        return _coco_cache[cache_key]

    if is_fake:
        if root_dir is None:
            raise ValueError("root_dir must be provided for fake datasets")
        logging.info(f"Creating fake dataset in {root_dir}")
        coco_api, _ = create_fake_coco_data(root_dir, num_samples)
    else:
        from pycocotools.coco import COCO

        logging.info(f"Loading COCO annotations from {ann_file}")
        coco_api = COCO(ann_file)
        logging.info(
            f"Successfully loaded COCO annotations: {len(coco_api.imgs)} images, {len(coco_api.anns)} annotations"
        )

    _coco_cache[cache_key] = coco_api
    return coco_api


def build_dataset(
    image_set: str,
    data_root_dir: str,
    transforms: callable,
    train_dir: str = "train2017",
    train_ann: str = "annotations/instances_train2017.json",
    val_dir: str = "val2017",
    val_ann: str = "annotations/instances_val2017.json",
    debug_mode: bool = False,
    debug_samples: int = 16,
):
    """
    Build COCO detection dataset.

    Args:
        image_set (str): Either 'train' or 'val'
        data_root_dir (str): Path to COCO dataset
        transforms (callable): Transforms to apply to images and targets
        train_dir (str): Training images directory name
        train_ann (str): Training annotations file name
        val_dir (str): Validation images directory name
        val_ann (str): Validation annotations file name
        debug_mode (bool): If True, only load a small subset of the dataset
        debug_samples (int): Number of samples to load in debug mode

    Returns:
        dataset: A CocoDetection dataset instance
    """
    root = Path(data_root_dir)

    if debug_mode:
        # Create debug directories
        debug_dir = root / "debug_data"
        debug_train_dir = debug_dir / "train"
        debug_val_dir = debug_dir / "val"

        # Create directories if they don't exist
        debug_train_dir.mkdir(parents=True, exist_ok=True)
        debug_val_dir.mkdir(parents=True, exist_ok=True)

        # Use debug directories instead of real ones
        PATHS = {
            "train": (debug_train_dir, "fake_train_dataset"),
            "val": (debug_val_dir, "fake_val_dataset"),
        }

        logging.info(f"Using debug dataset with {debug_samples} samples in {debug_dir}")
    else:
        # Use real COCO paths
        PATHS = {
            "train": (root / train_dir, root / train_ann),
            "val": (root / val_dir, root / val_ann),
        }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(
        str(img_folder),
        str(ann_file),
        transforms=transforms,
        debug_mode=debug_mode,
        debug_samples=debug_samples,
    )

    if debug_mode:
        logging.info(f"Created debug dataset with {len(dataset)} samples")
        logging.info(f"Debug images will be saved to: {img_folder}")

    return dataset
