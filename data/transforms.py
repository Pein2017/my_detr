"""
Transforms and data augmentation for both image + bbox.
"""

import random
from typing import List, Optional, Tuple, Union

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image

from utils.config import AugmentationConfig

from .box_ops import box_xyxy_to_cxcywh


def crop(image, target, region):
    """Crop the image and adjust the bounding boxes."""
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # Update target size
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    # Remove elements for which the boxes have zero area
    if "boxes" in target:
        cropped_boxes = target["boxes"].reshape(-1, 2, 2)
        keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def resize(
    image: Image.Image,
    target: Optional[dict],
    size: Union[int, Tuple[int, int]],
    max_size: Optional[int] = None,
) -> Tuple[Image.Image, Optional[dict]]:
    """Resize image and bounding boxes."""
    w, h = image.size
    if isinstance(size, (list, tuple)):
        size = size[0]

    # Calculate new size maintaining aspect ratio
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    # Calculate scale factor
    scale_factor = size / min(h, w)
    if max_size is not None:
        scale_factor = min(scale_factor, max_size / max(h, w))

    # Calculate new dimensions
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    # Resize image
    image = F.resize(image, (new_h, new_w))

    if target is None:
        return image, None

    # Scale bounding boxes
    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor(
            [scale_factor, scale_factor, scale_factor, scale_factor]
        )
        target["boxes"] = scaled_boxes

    if "area" in target:
        target["area"] = target["area"] * (scale_factor * scale_factor)

    target["size"] = torch.tensor([new_h, new_w])

    return image, target


def hflip(image, target):
    """Horizontally flip the image and adjust the bounding boxes."""
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor(
            [-1, 1, -1, 1]
        ) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    return flipped_image, target


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, img: Image.Image, target: Optional[dict] = None
    ) -> Tuple[Image.Image, Optional[dict]]:
        if random.random() < self.p:
            img = F.hflip(img)
            if target is not None:
                target = target.copy()
                if "boxes" in target:
                    boxes = target["boxes"]
                    boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor(
                        [-1, 1, -1, 1]
                    ) + torch.as_tensor([1, 0, 1, 0])
                    target["boxes"] = boxes
        return img, target


class RandomResize:
    def __init__(self, scales: List[int], max_size: Optional[int] = None):
        self.scales = scales
        self.max_size = max_size

    def __call__(
        self, img: Image.Image, target: Optional[dict] = None
    ) -> Tuple[Image.Image, Optional[dict]]:
        size = random.choice(self.scales)
        return resize(img, target, size, self.max_size)


class RandomSizeCrop:
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(
        self, img: Image.Image, target: Optional[dict] = None
    ) -> Tuple[Image.Image, Optional[dict]]:
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class RandomSelect:
    """Randomly selects between transforms1 and transforms2."""

    def __init__(self, transforms1, transforms2, p: float = 0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(
        self, img: Image.Image, target: Optional[dict] = None
    ) -> Tuple[Image.Image, Optional[dict]]:
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class Normalize:
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std

    def __call__(
        self, image: torch.Tensor, target: Optional[dict] = None
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """Normalize image and convert boxes to cxcywh format."""
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None

        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def make_coco_transforms(image_set: str, config: AugmentationConfig) -> Compose:
    """Create transformation pipeline for COCO dataset."""
    if image_set == "train":
        normalize = Compose(
            [
                ToTensor(),
                Normalize(
                    mean=config.train.normalize.mean, std=config.train.normalize.std
                ),
            ]
        )

        return Compose(
            [
                RandomHorizontalFlip(p=config.train.horizontal_flip.prob),
                RandomSelect(
                    RandomResize(scales=config.scales, max_size=config.max_size),
                    Compose(
                        [
                            RandomResize(scales=config.train.random_resize.scales),
                            RandomSizeCrop(
                                min_size=config.train.random_resize.crop_size[0],
                                max_size=config.train.random_resize.crop_size[1],
                            ),
                            RandomResize(
                                scales=config.scales, max_size=config.max_size
                            ),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val":
        return Compose(
            [
                RandomResize(scales=config.val.scales, max_size=config.val.max_size),
                ToTensor(),
                Normalize(mean=config.val.normalize.mean, std=config.val.normalize.std),
            ]
        )

    raise ValueError(f"unknown {image_set}")
