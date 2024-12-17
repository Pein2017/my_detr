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


def crop(
    image: Image.Image, target: Optional[dict], region: Tuple[int, int, int, int]
) -> Tuple[Image.Image, Optional[dict]]:
    """Crop the image and adjust the bounding boxes."""
    cropped_image = F.crop(image, *region)
    if target is None:
        return cropped_image, None

    target = target.copy()
    i, j, h, w = region
    target["size"] = torch.tensor([h, w])

    if "boxes" in target:
        boxes = target["boxes"]  # boxes in xyxy format
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)

        # Remove boxes with zero area
        keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        for field in ["boxes", "labels", "area", "iscrowd"]:
            if field in target:
                target[field] = target[field][keep]

    return cropped_image, target


def resize(
    image: Image.Image,
    target: Optional[dict],
    size: int,
    max_size: Optional[int] = None,
) -> Tuple[Image.Image, Optional[dict]]:
    """Resize image and bounding boxes."""
    w, h = image.size

    scale = size / min(h, w)
    if max_size is not None:
        scale = min(scale, max_size / max(h, w))

    new_w = int(w * scale)
    new_h = int(h * scale)
    image = F.resize(image, (new_h, new_w))

    if target is None:
        return image, None

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]  # boxes in xyxy format
        scaled_boxes = boxes * scale
        target["boxes"] = scaled_boxes
        target["area"] = target["area"] * (scale * scale)
    target["size"] = torch.tensor([new_h, new_w])

    return image, target


def hflip(
    image: Image.Image, target: Optional[dict]
) -> Tuple[Image.Image, Optional[dict]]:
    """Horizontally flip the image and adjust the bounding boxes."""
    flipped_image = F.hflip(image)
    if target is None:
        return flipped_image, None

    w, h = image.size
    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]  # boxes in xyxy format
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor(
            [-1, 1, -1, 1]
        ) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    return flipped_image, target


def pad(
    image: Image.Image, target: Optional[dict], padding: Tuple[int, int]
) -> Tuple[Image.Image, Optional[dict]]:
    """Pad the image and adjust the bounding boxes."""
    # Pad only on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None

    target = target.copy()
    target["size"] = torch.tensor(padded_image.size[::-1])
    return padded_image, target


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, img: Image.Image, target: Optional[dict] = None
    ) -> Tuple[Image.Image, Optional[dict]]:
        if random.random() < self.p:
            return hflip(img, target)
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


class CenterCrop:
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(
        self, img: Image.Image, target: Optional[dict] = None
    ) -> Tuple[Image.Image, Optional[dict]]:
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomPad:
    def __init__(self, max_pad: int):
        self.max_pad = max_pad

    def __call__(
        self, img: Image.Image, target: Optional[dict] = None
    ) -> Tuple[Image.Image, Optional[dict]]:
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


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


class ToTensor:
    def __call__(
        self, img: Image.Image, target: Optional[dict] = None
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        return F.to_tensor(img), target


class RandomErasing:
    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(
        self, img: torch.Tensor, target: Optional[dict] = None
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        return self.eraser(img), target


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
        if "boxes" in target:
            h, w = image.shape[-2:]
            # Convert from xyxy to cxcywh format and normalize
            boxes = box_xyxy_to_cxcywh(target["boxes"])
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose:
    def __init__(
        self,
        transforms: List[
            Union[
                RandomHorizontalFlip,
                RandomSelect,
                RandomResize,
                RandomSizeCrop,
                ToTensor,
                Normalize,
            ]
        ],
    ):
        self.transforms = transforms

    def __call__(
        self, image: Union[Image.Image, torch.Tensor], target: Optional[dict] = None
    ) -> Tuple[Union[Image.Image, torch.Tensor], Optional[dict]]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def make_coco_transforms(image_set: str, config: AugmentationConfig) -> Compose:
    """Create transformation pipeline for COCO dataset."""
    normalize = Compose(
        [
            ToTensor(),
            Normalize(mean=config.train.normalize.mean, std=config.train.normalize.std),
        ]
    )

    if image_set == "train":
        return Compose(
            [
                # Horizontal flip
                RandomHorizontalFlip(p=config.train.horizontal_flip.prob),
                # Random resize and crop
                RandomSelect(
                    RandomResize(config.scales, max_size=config.max_size),
                    Compose(
                        [
                            RandomResize(config.train.random_resize.scales),
                            RandomSizeCrop(*config.train.random_resize.crop_size),
                            RandomResize(config.scales, max_size=config.max_size),
                        ]
                    ),
                ),
                # Random padding
                RandomPad(max_pad=config.train.random_pad.max_pad),
                # Center crop
                CenterCrop(size=config.train.center_crop.size),
                # Normalization
                normalize,
                # Random erasing
                RandomErasing(
                    p=config.train.random_erasing.prob,
                    scale=config.train.random_erasing.scale,
                    ratio=config.train.random_erasing.ratio,
                    value=config.train.random_erasing.value,
                ),
            ]
        )

    if image_set == "val":
        return Compose(
            [
                RandomResize([config.val.scales[0]], max_size=config.val.max_size),
                CenterCrop(size=config.val.center_crop.size),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")
