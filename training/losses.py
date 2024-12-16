"""
Loss computation utilities for DETR.
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    targets: List[Dict[str, torch.Tensor]],
    matcher: nn.Module,
    num_classes: int,
    loss_config: Dict,
) -> Dict[str, torch.Tensor]:
    """
    Compute DETR loss.

    Args:
        outputs: Model outputs containing 'pred_logits' and 'pred_boxes'
        targets: List of target dictionaries
        matcher: Hungarian matcher module
        num_classes: Number of classes (including background)
        loss_config: Loss configuration dictionary

    Returns:
        Dictionary containing loss components
    """
    # Get predictions
    pred_logits = outputs["pred_logits"]
    pred_boxes = outputs["pred_boxes"]

    # Match predictions to targets
    indices = matcher(outputs, targets)

    # Compute classification loss
    target_classes = torch.full(
        (pred_logits.shape[0], pred_logits.shape[1]),
        num_classes,
        dtype=torch.int64,
        device=pred_logits.device,
    )

    for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
        target_classes[batch_idx, pred_idx] = targets[batch_idx]["labels"][tgt_idx]

    empty_weight = torch.ones(num_classes + 1, device=pred_logits.device)
    empty_weight[-1] = loss_config.empty_weight

    # Compute classification loss
    loss_ce = F.cross_entropy(
        pred_logits.transpose(1, 2), target_classes, weight=empty_weight
    )

    # Initialize box losses
    total_bbox_loss = torch.tensor(0.0, device=pred_logits.device)
    total_giou_loss = torch.tensor(0.0, device=pred_logits.device)
    num_boxes = 0

    # Compute box losses
    for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
        if len(pred_idx) == 0:
            continue

        pred_boxes_matched = pred_boxes[batch_idx, pred_idx]
        target_boxes = targets[batch_idx]["boxes"][tgt_idx]

        # L1 loss
        l1_loss = F.l1_loss(pred_boxes_matched, target_boxes, reduction="sum")
        total_bbox_loss = total_bbox_loss + l1_loss

        # GIoU loss
        giou_values = generalized_box_iou(
            box_cxcywh_to_xyxy(pred_boxes_matched),
            box_cxcywh_to_xyxy(target_boxes),
        )
        giou_loss = (1 - torch.diag(giou_values)).sum()
        total_giou_loss = total_giou_loss + giou_loss

        num_boxes += len(pred_idx)

    # Normalize box losses
    num_boxes = max(num_boxes, 1)
    loss_bbox = total_bbox_loss / num_boxes
    loss_giou = total_giou_loss / num_boxes

    # Compute total loss with coefficients
    total_loss = (
        loss_config.class_loss_coef * loss_ce
        + loss_config.bbox_loss_coef * loss_bbox
        + loss_config.giou_loss_coef * loss_giou
    )

    # Create loss dictionary
    loss_dict = {
        "loss": total_loss,
        "loss_ce": loss_ce,
        "loss_bbox": loss_bbox,
        "loss_giou": loss_giou,
    }

    return loss_dict
