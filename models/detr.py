import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .backbone import BackboneBase
from .position_encoding import PositionEmbeddingSine
from .transformer import Transformer


class MLP(nn.Module):
    """Simple multi-layer perceptron"""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETR(nn.Module):
    """
    DETR model implementation.
    Combines CNN backbone, transformer encoder-decoder, and prediction heads
    for object detection.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        # Extract model parameters from config
        self.num_classes = config.model.num_classes
        self.hidden_dim = config.model.hidden_dim
        self.backbone_name = config.model.backbone_name
        self.pretrained_backbone = config.model.pretrained_backbone

        # Initialize backbone
        self.backbone = BackboneBase(
            name=self.backbone_name, pretrained=self.pretrained_backbone
        )

        # Project backbone features to transformer dimension
        self.conv = nn.Conv2d(self.backbone.num_channels, self.hidden_dim, 1)

        # Initialize position embedding
        self.position_embedding = PositionEmbeddingSine(
            self.hidden_dim, normalize=config.model.position_embedding.normalize
        )

        # Initialize transformer
        self.transformer = Transformer(
            d_model=self.hidden_dim,
            nhead=config.model.nheads,
            num_encoder_layers=config.model.num_encoder_layers,
            num_decoder_layers=config.model.num_decoder_layers,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout,
        )

        # Object queries and prediction heads
        self.num_queries = config.model.num_queries
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)

        self.class_embed = nn.Linear(self.hidden_dim, self.num_classes + 1)
        self.bbox_embed = MLP(
            input_dim=self.hidden_dim,
            hidden_dim=config.model.bbox_predictor.hidden_dim,
            output_dim=4,
            num_layers=config.model.bbox_predictor.num_layers,
        )

        # Layer normalization for features
        self.norm = nn.LayerNorm(self.hidden_dim)

        # Initialize parameters
        if config.model.init.xavier_uniform:
            self._reset_parameters(config.model.init.prior_prob)

        # Ensure all parameters are properly initialized for gradient computation
        for param in self.parameters():
            param.requires_grad_(True)

    def _reset_parameters(self, prior_prob: float = 0.01):
        """Initialize weights and biases"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Initialize class prediction bias for better convergence
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = (
            torch.ones(self.class_embed.bias.shape) * bias_value
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, H, W]

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model outputs
                - 'pred_logits': Class predictions [batch_size, num_queries, num_classes + 1]
                - 'pred_boxes': Box predictions [batch_size, num_queries, 4] in (cx, cy, w, h) format
        """
        # Extract features from the backbone
        features = self.backbone(x)

        # Project and flatten the features
        projected_features = self.conv(features)
        h, w = projected_features.shape[-2:]
        flattened_features = projected_features.flatten(2).permute(
            2, 0, 1
        )  # [HW, B, C]

        # Add positional embeddings
        pos_embed = (
            self.position_embedding(projected_features).flatten(2).permute(2, 0, 1)
        )
        flattened_features = flattened_features + pos_embed

        # Pass through transformer
        transformer_output = self.transformer(
            flattened_features,
            self.query_embed.weight.unsqueeze(1).repeat(1, x.shape[0], 1),
            pos_embed,
        )

        # Get outputs from transformer and reshape for prediction
        hs = transformer_output[0]  # Shape: [num_queries, batch_size, hidden_dim]
        hs = hs.transpose(0, 1)  # Shape: [batch_size, num_queries, hidden_dim]

        # Predict classes and boxes
        outputs_class = self.class_embed(
            hs
        )  # [batch_size, num_queries, num_classes + 1]
        outputs_coord = self.bbox_embed(hs).sigmoid()  # [batch_size, num_queries, 4]

        return {
            "pred_logits": outputs_class,
            "pred_boxes": outputs_coord,
        }

    def configure_optimizer(self, config):
        """Configure optimizer with settings from config."""
        param_dicts = [
            {
                "params": [
                    p for n, p in self.named_parameters() if "backbone" not in n
                ],
                "lr": config.optimizer.lr,
            },
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n],
                "lr": config.optimizer.lr * config.optimizer.backbone_lr_factor,
            },
        ]

        optimizer = torch.optim.AdamW(
            param_dicts, weight_decay=config.optimizer.weight_decay
        )

        return optimizer
