"""
Core DETR model class.
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn
from omegaconf import DictConfig

from models.detr import DETR
from models.matcher import HungarianMatcher
from training.losses import compute_loss


class DETRModel(nn.Module):
    """
    Core DETR model class.
    Handles model initialization and loss computation.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        # Initialize model
        self.model = DETR(config)

        # Initialize matcher for loss computation
        self.matcher = HungarianMatcher(
            cost_class=config.loss.cost_class,
            cost_bbox=config.loss.cost_bbox,
            cost_giou=config.loss.cost_giou,
        )

        # Ensure all parameters are properly initialized for gradient computation
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                logging.warning(f"Parameter {name} does not require gradients")
            param.requires_grad_(True)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            images: Input images tensor. Must have requires_grad=True for training.

        Returns:
            Dictionary containing model outputs
        """
        # Ensure input tensor is set up for gradient computation during training
        if self.training and not images.requires_grad:
            images.requires_grad_(True)

        # Get model outputs
        outputs = self.model(images)

        # Ensure all outputs are used in training
        if self.training:
            # Add intermediate outputs to ensure gradient flow
            outputs["aux_outputs"] = []
            hs = self.model.transformer(
                self.model.conv(self.model.backbone(images))
                .flatten(2)
                .permute(2, 0, 1),
                self.model.query_embed.weight.unsqueeze(1).repeat(
                    1, images.shape[0], 1
                ),
                self.model.position_embedding(
                    self.model.conv(self.model.backbone(images))
                )
                .flatten(2)
                .permute(2, 0, 1),
            )[0]

            # Add predictions from each decoder layer
            for layer_idx in range(hs.shape[0]):
                layer_out = hs[layer_idx].transpose(0, 1)
                outputs["aux_outputs"].append(
                    {
                        "pred_logits": self.model.class_embed(layer_out),
                        "pred_boxes": self.model.bbox_embed(layer_out).sigmoid(),
                    }
                )

        return outputs

    def compute_loss(
        self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Compute loss.

        Args:
            outputs: Model outputs
            targets: Ground truth targets

        Returns:
            Dictionary containing loss values
        """
        return compute_loss(
            outputs=outputs,
            targets=targets,
            matcher=self.matcher,
            num_classes=self.config.model.num_classes,
            loss_config=self.config.loss,
        )

    def get_param_groups(self) -> List[Dict]:
        """Get parameter groups for optimizer.

        Returns:
            List of parameter groups with different learning rates
        """
        # Calculate backbone learning rate
        backbone_lr = (
            self.config.optimizer.lr * self.config.optimizer.backbone_lr_factor
        )

        # Parameters that will use the backbone learning rate
        backbone_params = [
            p
            for n, p in self.model.named_parameters()
            if "backbone" in n and p.requires_grad
        ]
        # All other parameters will use the main learning rate
        other_params = [
            p
            for n, p in self.model.named_parameters()
            if "backbone" not in n and p.requires_grad
        ]

        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": other_params},
        ]
