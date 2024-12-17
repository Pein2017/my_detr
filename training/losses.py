"""
Loss computation utilities for DETR.
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def compute_single_loss(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    targets: List[Dict[str, torch.Tensor]],
    indices: List[tuple],
    num_classes: int,
    loss_config: Dict,
    log: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Compute loss for a single set of predictions.

    Args:
        pred_logits: Class predictions [batch_size, num_queries, num_classes + 1]
        pred_boxes: Box predictions [batch_size, num_queries, 4]
        targets: List of target dictionaries
        indices: List of (pred_idx, tgt_idx) tuples from matcher
        num_classes: Number of classes (including background)
        loss_config: Loss configuration dictionary
        log: Whether to log additional metrics (e.g., class error)

    Returns:
        Dictionary containing loss components
    """
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

    # Create loss dictionary
    losses = {
        "loss_ce": loss_ce,
        "loss_bbox": loss_bbox,
        "loss_giou": loss_giou,
    }

    # Add class error if logging is enabled
    if log:
        # Calculate class error (percentage of incorrect classifications)
        pred_classes = pred_logits[..., :-1].argmax(-1)  # Remove background class
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        pred_classes_matched = torch.cat(
            [pred_classes[i, J] for i, (J, _) in enumerate(indices)]
        )
        class_error = (
            100 - (pred_classes_matched == target_classes_o).float().mean() * 100
        )
        losses["class_error"] = class_error

    return losses


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    targets: List[Dict[str, torch.Tensor]],
    matcher: nn.Module,
    num_classes: int,
    loss_config: Dict,
) -> Dict[str, torch.Tensor]:
    """
    Compute DETR loss including auxiliary losses.

    Args:
        outputs: Model outputs containing 'pred_logits' and 'pred_boxes'
        targets: List of target dictionaries
        matcher: Hungarian matcher module
        num_classes: Number of classes (including background)
        loss_config: Loss configuration dictionary

    Returns:
        Dictionary containing all loss components
    """
    # Match predictions to targets for the main output
    indices = matcher(outputs, targets)

    # Compute main losses
    losses = compute_single_loss(
        outputs["pred_logits"],
        outputs["pred_boxes"],
        targets,
        indices,
        num_classes,
        loss_config,
        log=True,  # Log metrics for main prediction
    )

    # Compute auxiliary losses if present
    if "aux_outputs" in outputs:
        for i, aux_outputs in enumerate(outputs["aux_outputs"]):
            # Match predictions to targets for this auxiliary output
            aux_indices = matcher(aux_outputs, targets)

            # Compute loss for this auxiliary output
            aux_losses = compute_single_loss(
                aux_outputs["pred_logits"],
                aux_outputs["pred_boxes"],
                targets,
                aux_indices,
                num_classes,
                loss_config,
                log=False,  # Don't log metrics for auxiliary predictions
            )

            # Add auxiliary losses to main losses with suffix
            losses.update({f"{k}_{i}": v for k, v in aux_losses.items()})

    # Compute total loss with coefficients
    total_loss = (
        loss_config.class_loss_coef * losses["loss_ce"]
        + loss_config.bbox_loss_coef * losses["loss_bbox"]
        + loss_config.giou_loss_coef * losses["loss_giou"]
    )

    # Add auxiliary losses to total loss
    if "aux_outputs" in outputs:
        for i in range(len(outputs["aux_outputs"])):
            total_loss = total_loss + (
                loss_config.class_loss_coef * losses[f"loss_ce_{i}"]
                + loss_config.bbox_loss_coef * losses[f"loss_bbox_{i}"]
                + loss_config.giou_loss_coef * losses[f"loss_giou_{i}"]
            )
        # Scale total loss by number of predictions (main + auxiliary)
        total_loss = total_loss / (len(outputs["aux_outputs"]) + 1)

    losses["loss"] = total_loss
    return losses
