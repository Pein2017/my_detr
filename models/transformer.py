from typing import Optional

import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with self-attention and feed-forward network.
    Follows the architecture of the DETR paper.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Two-layer feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        pos: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Shape:
        - src, pos: [HW, B, D]
        - padding_mask: [B, HW] boolean mask, True indicates padding
        """
        # Self-attention with positional encodings
        q = k = src + pos
        src2 = self.self_attn(
            query=q,
            key=k,
            value=src,
            key_padding_mask=padding_mask,  # B x HW
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed-forward network
        src2 = self.ffn(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class TransformerDecoderLayer(nn.Module):
    """
    Single transformer decoder layer with self-attention, cross-attention, and feed-forward network.
    Implements the cross-attention mechanism between object queries and encoded image features.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Two-layer feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor,
        pos: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Shape:
        - tgt, query_pos: [N, B, D] where N is number of queries
        - memory, pos: [HW, B, D] where HW is feature spatial dimensions
        - padding_mask: [B, HW] boolean mask, True indicates padding
        """
        # Self-attention with query embeddings
        q = k = tgt + query_pos
        tgt2 = self.self_attn(
            query=q,
            key=k,
            value=tgt,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention between queries and encoded memory
        tgt2 = self.multihead_attn(
            query=tgt + query_pos,
            key=memory + pos,
            value=memory,
            key_padding_mask=padding_mask,  # B x HW
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed-forward network
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""

    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(
        self,
        src: torch.Tensor,
        pos: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Shape:
        - src, pos: [HW, B, D]
        - padding_mask: [B, HW]
        """
        for layer in self.layers:
            src = layer(src, pos, padding_mask)
        return src


class TransformerDecoder(nn.Module):
    """Stack of transformer decoder layers."""

    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(decoder_layer.self_attn.embed_dim)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor,
        pos: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Shape:
        - tgt, query_pos: [N, B, D]
        - memory, pos: [HW, B, D]
        - padding_mask: [B, HW]
        Returns:
        - output: [L, B, N, D] where L is the number of decoder layers
        """
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, query_pos, pos, padding_mask)
            # Apply normalization and store intermediate output
            normed_output = self.norm(output)
            # Reshape from [N, B, D] to [B, N, D]
            normed_output = normed_output.permute(1, 0, 2)
            intermediate.append(normed_output)

        # Stack intermediate outputs to get [L, B, N, D]
        return torch.stack(intermediate)


class Transformer(nn.Module):
    """
    Complete transformer model with encoder and decoder.
    Processes image features and generates object detections.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

    def forward(
        self,
        src: torch.Tensor,
        query_embed: torch.Tensor,
        pos_embed: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Shape:
        - src, pos_embed: [HW, B, D] image features and position encodings
        - query_embed: [N, B, D] learnable object queries
        - padding_mask: [B, HW] True indicates padding
        Returns:
        - hs: [N, B, D] object detection features
        - memory: [HW, B, D] encoded image features
        """
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, pos_embed, padding_mask)
        hs = self.decoder(tgt, memory, query_embed, pos_embed, padding_mask)

        return hs, memory
