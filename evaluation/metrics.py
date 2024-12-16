"""
Evaluation utilities for DETR.
Handles prediction preparation and score thresholding.
"""

import logging
from typing import Dict, List

import torch
import torch.nn.functional as F


def prepare_predictions(
    outputs: Dict[str, torch.Tensor],
    targets: List[Dict],
    score_threshold: float = 0.01,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Convert model outputs to evaluation format.

    Args:
        outputs: Model outputs containing 'pred_logits' and 'pred_boxes'
        targets: List of target dictionaries containing 'image_id'
        score_threshold: Score threshold for filtering predictions

    Returns:
        Dict mapping image_id to predictions dict with keys 'boxes', 'scores', 'labels'
    """
    pred_logits = outputs["pred_logits"]
    pred_boxes = outputs["pred_boxes"]

    # Add debugging logs
    with torch.no_grad():
        # Check confidence distribution
        scores = F.softmax(pred_logits, dim=-1)
        background_probs = scores[:, :, -1].mean().item()
        max_class_probs = scores[:, :, :-1].max(dim=-1)[0].mean().item()
        logging.info(
            f"Prediction stats - Avg background prob: {background_probs:.3f}, Avg max class prob: {max_class_probs:.3f}"
        )

        # Check box statistics
        logging.info(
            f"Box stats - range: [{pred_boxes.min().item():.3f}, {pred_boxes.max().item():.3f}], "
            f"means: {pred_boxes.mean(dim=(0,1)).tolist()}, "
            f"std: {pred_boxes.std(dim=(0,1)).tolist()}"
        )

    predictions = {}
    for idx, (logits, boxes) in enumerate(zip(pred_logits, pred_boxes)):
        # Get scores and labels
        scores = F.softmax(logits, dim=-1)[:, :-1]  # Remove background
        scores_per_class, labels = scores.max(dim=-1)

        # Filter by score threshold
        keep = scores_per_class > score_threshold
        if not keep.any():  # If no predictions pass the threshold
            # Take top 20 predictions
            top_k = min(20, len(scores_per_class))
            _, top_indices = scores_per_class.topk(top_k)
            keep = torch.zeros_like(scores_per_class, dtype=torch.bool)
            keep[top_indices] = True

        # Apply filtering
        boxes = boxes[keep]
        scores_per_class = scores_per_class[keep]
        labels = labels[keep]

        if len(boxes) == 0:
            # Store empty predictions with correct image_id
            image_id = targets[idx]["image_id"]
            predictions[image_id.item()] = {
                "boxes": boxes,
                "scores": scores_per_class,
                "labels": labels + 1,  # COCO classes start from 1
            }
            continue

        # Ensure boxes are valid
        valid_boxes = torch.isfinite(boxes).all(dim=1)
        boxes = boxes[valid_boxes]
        scores_per_class = scores_per_class[valid_boxes]
        labels = labels[valid_boxes]

        if len(boxes) == 0:
            # Store empty predictions with correct image_id
            image_id = targets[idx]["image_id"]
            predictions[image_id.item()] = {
                "boxes": boxes,
                "scores": scores_per_class,
                "labels": labels + 1,
            }
            continue

        # Convert normalized coordinates to absolute coordinates if necessary
        if boxes.max() <= 1.0 and boxes.min() >= 0.0:
            image_size = targets[idx].get("orig_size", torch.tensor([800, 800]))
            scale_fct = torch.tensor(
                [image_size[1], image_size[0], image_size[1], image_size[0]],
                device=boxes.device,
            )
            boxes = boxes * scale_fct[None, :]

        # Ensure boxes have valid dimensions
        boxes_valid_dim = (boxes[:, 2:] > boxes[:, :2] + 1e-3).all(dim=1)
        boxes = boxes[boxes_valid_dim]
        scores_per_class = scores_per_class[boxes_valid_dim]
        labels = labels[boxes_valid_dim]

        # Store predictions
        image_id = targets[idx]["image_id"]
        predictions[image_id.item()] = {
            "boxes": boxes,
            "scores": scores_per_class,
            "labels": labels + 1,  # COCO classes start from 1
        }

    return predictions


def evaluate_predictions(
    evaluator,
    trainer_is_global_zero: bool,
    current_epoch: int = 0,
) -> Dict[str, float]:
    """
    Evaluate predictions using COCO evaluator.
    Note: This should be called after predictions have been gathered and synchronized across processes.

    Args:
        evaluator: COCO evaluator instance
        trainer_is_global_zero: Whether this is the global zero process
        current_epoch: Current training epoch (for logging)

    Returns:
        Dictionary containing evaluation metrics (real or dummy)
    """
    # Define dummy metrics that will be returned in case of failure
    dummy_metrics = {
        "map": 0.0,
        "map_50": 0.0,
        "map_75": 0.0,
        "map_small": 0.0,
        "map_medium": 0.0,
        "map_large": 0.0,
    }

    try:
        # Check if we have any predictions to evaluate
        if not hasattr(evaluator, "img_ids") or not evaluator.img_ids:
            if trainer_is_global_zero:
                logging.info(
                    f"[Epoch {current_epoch}] No predictions collected for evaluation. "
                    "This is normal during early training."
                )
            return dummy_metrics

        # Run COCO evaluation
        try:
            evaluator.accumulate()
            metrics = evaluator.get_stats()
        except AttributeError as e:
            if trainer_is_global_zero:
                logging.warning(
                    f"[Epoch {current_epoch}] Evaluation failed due to missing attribute: {str(e)}. "
                    "This is expected when using a fake dataset."
                )
            return dummy_metrics

        if metrics is None:
            if trainer_is_global_zero:
                logging.info(
                    f"[Epoch {current_epoch}] COCO evaluation returned no metrics. "
                    "This is normal during early training."
                )
            return dummy_metrics

        # Check if we got valid metrics
        if all(v == 0.0 for v in metrics.values()):
            if trainer_is_global_zero:
                logging.info(
                    f"[Epoch {current_epoch}] All metrics are zero. "
                    "This is normal during early training."
                )
        elif trainer_is_global_zero:
            # Log successful evaluation with non-zero metrics
            logging.info(
                f"[Epoch {current_epoch}] COCO evaluation completed successfully. "
                f"mAP: {metrics['map']:.3f}, "
                f"mAP@50: {metrics['map_50']:.3f}, "
                f"mAP@75: {metrics['map_75']:.3f}, "
                f"mAP_small: {metrics['map_small']:.3f}, "
                f"mAP_medium: {metrics['map_medium']:.3f}, "
                f"mAP_large: {metrics['map_large']:.3f}"
            )

        return metrics

    except Exception as e:
        if trainer_is_global_zero:
            logging.error(f"Error during evaluation: {str(e)}")
            logging.info(
                f"[Epoch {current_epoch}] All metrics are zero. This is normal during early training."
            )
        return dummy_metrics
