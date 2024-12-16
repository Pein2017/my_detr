import os
from typing import Dict, Tuple

import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50

os.environ["TORCH_HOME"] = "/data/training_code/Pein/DETR/my_detr/pretrained"


class BackboneBase(nn.Module):
    """
    CNN backbone that extracts features from images.
    Removes classification head and returns feature maps.
    """

    def __init__(self, name: str, pretrained: bool):
        super().__init__()

        backbone_models: Dict[str, Tuple[callable, int]] = {
            "resnet18": (resnet18, 512),
            "resnet34": (resnet34, 512),
            "resnet50": (resnet50, 2048),
        }

        if name not in backbone_models:
            raise ValueError(f"Unsupported backbone: {name}")

        backbone_fn, self.num_channels = backbone_models[name]
        print(f"Using {name} backbone")
        print(f"Using the pretrained model: {pretrained}")
        backbone = backbone_fn(pretrained=pretrained)

        # Remove classification head (last two layers)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

    def forward(self, x):
        """
        Shape:
        - Input: [B, 3, H, W]
        - Output: [B, C, H/32, W/32] where C depends on backbone
        """
        return self.backbone(x)
