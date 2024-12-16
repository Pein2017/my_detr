import math

import torch
import torch.nn as nn


class PositionEmbeddingSine(nn.Module):
    """
    2D sinusoidal position embeddings for image features.
    Each position (x,y) is encoded using sin/cos functions at different frequencies.
    """

    def __init__(self, d_model: int, normalize: bool = True, scale: float = None):
        """
        Args:
            d_model: Output dimension (must be even)
            normalize: Whether to normalize coordinates to [0, scale]
            scale: Scale factor for normalized coordinates (default: 2π)
        """
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")

        self.d_model = d_model
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create 2D positional embeddings.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Positional encoding of shape [B, d_model, H, W]
        """
        bs, _, h, w = x.shape

        # Create position indices
        y_embed = torch.arange(h, dtype=torch.float32, device=x.device)
        x_embed = torch.arange(w, dtype=torch.float32, device=x.device)

        if self.normalize:
            y_embed = y_embed / (h - 1) * self.scale
            x_embed = x_embed / (w - 1) * self.scale

        # Create meshgrid
        y_embed, x_embed = torch.meshgrid(y_embed, x_embed, indexing="ij")

        # Create frequency bands for half the dimensions (since we'll use both sin and cos)
        dim_t = torch.arange(self.d_model // 2, dtype=torch.float32, device=x.device)
        dim_t = 10000 ** (2 * dim_t / self.d_model)

        # Compute positional encodings
        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t

        # Interleave sin and cos
        pos_x = torch.stack(
            [pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1
        ).flatten(-2)
        pos_y = torch.stack(
            [pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1
        ).flatten(-2)

        # Combine x and y encodings
        pos = torch.cat([pos_y, pos_x], dim=-1)  # [H, W, d_model]

        # Reshape to match expected output shape [B, d_model, H, W]
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(bs, 1, 1, 1)

        return pos
