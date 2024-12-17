"""
COCO evaluation tools for object detection.
Simplified and focused version that handles only bbox detection.
"""

import copy
import logging
import os
import time
from typing import Dict, List

import torch
import torch.distributed as dist
from pycocotools.cocoeval import COCOeval

from data.coco import get_coco_api_from_dataset


# TODO: Test the Evaluator since the map are 0 all the time
class CocoEvaluator:
    def __init__(self, coco_gt):
        """
        Initialize COCO evaluator for bbox detection.

        Args:
            coco_gt: Ground truth COCO API object
        """
        logging.info("Starting CocoEvaluator initialization")
        try:
            self.coco_gt = copy.deepcopy(coco_gt)
            logging.info("Successfully copied ground truth COCO object")
            self.coco_dt = None  # Store COCO results object
            self.predictions = []
            self.img_ids = []
            self._initialized_eval = False
            logging.info("CocoEvaluator initialization completed")
        except Exception as e:
            logging.error(f"Error during CocoEvaluator initialization: {str(e)}")
            raise

    @staticmethod
    def _gather_predictions(
        predictions: List[Dict], device: torch.device
    ) -> List[Dict]:
        """
        Gather predictions from all processes.

        Args:
            predictions: List of prediction dictionaries
            device: Device to use for gathering

        Returns:
            List of gathered predictions
        """
        if not dist.is_initialized():
            return predictions

        world_size = dist.get_world_size()
        if world_size == 1:
            return predictions

        # Convert predictions to tensor format for gathering
        num_preds = len(predictions)
        num_preds_tensor = torch.tensor([num_preds], device=device)
        num_preds_list = [torch.ones_like(num_preds_tensor) for _ in range(world_size)]
        dist.all_gather(num_preds_list, num_preds_tensor)

        # Gather predictions from all processes
        max_preds = max(num_preds_list).item()
        pred_tensors = []
        for p in predictions:
            tensor = torch.tensor(
                [p["image_id"], p["category_id"]] + p["bbox"] + [p["score"]],
                device=device,
            )
            pred_tensors.append(tensor)

        # Pad predictions if necessary
        if len(pred_tensors) < max_preds:
            padding = torch.zeros(
                (max_preds - len(pred_tensors), 7), device=device
            )  # 7 = image_id, category_id, bbox(4), score
            pred_tensors.extend([padding[i] for i in range(padding.shape[0])])
        pred_tensor = torch.stack(pred_tensors)

        # Gather all predictions
        gathered_tensors = [torch.zeros_like(pred_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, pred_tensor)

        # Convert back to list of dictionaries
        gathered_predictions = []
        for num_preds, tensors in zip(num_preds_list, gathered_tensors):
            for i in range(num_preds.item()):
                t = tensors[i]
                gathered_predictions.append(
                    {
                        "image_id": int(t[0].item()),
                        "category_id": int(t[1].item()),
                        "bbox": t[2:6].tolist(),
                        "score": t[6].item(),
                    }
                )

        return gathered_predictions

    @classmethod
    def from_annotations_file(
        cls,
        ann_file: str,
        debug_mode: bool = False,
        debug_root: str = None,
        debug_samples: int = 16,
    ) -> "CocoEvaluator":
        """
        Create evaluator from COCO annotation file.

        Args:
            ann_file: Path to COCO annotation file
            debug_mode: Whether to use fake dataset for debugging
            debug_root: Root directory for fake dataset images
            debug_samples: Number of samples for fake dataset

        Returns:
            CocoEvaluator instance
        """
        try:
            if debug_mode:
                if debug_root is None:
                    raise ValueError(
                        "debug_root must be provided when debug_mode is True"
                    )
                # Use the unified caching mechanism for fake dataset
                coco_gt = get_coco_api_from_dataset(
                    ann_file="fake_dataset",
                    is_fake=True,
                    root_dir=debug_root,
                    num_samples=debug_samples,
                )
            else:
                if not os.path.exists(ann_file):
                    raise FileNotFoundError(f"Annotation file not found: {ann_file}")

                if not os.access(ann_file, os.R_OK):
                    raise PermissionError(f"Cannot read annotation file: {ann_file}")

                # Use the unified caching mechanism for real dataset
                coco_gt = get_coco_api_from_dataset(ann_file, is_fake=False)

            if len(coco_gt.imgs) == 0:
                raise ValueError("No images found in dataset")

            return cls(coco_gt)

        except Exception as e:
            logging.error(f"Failed to load COCO annotations: {str(e)}")
            raise

    def update(self, predictions: Dict[int, Dict[str, torch.Tensor]]):
        """
        Update evaluator with new predictions.

        Args:
            predictions: Dictionary mapping image IDs to prediction dictionaries
                Each prediction dictionary contains:
                - boxes: [N, 4] tensor of predicted boxes
                - scores: [N] tensor of confidence scores
                - labels: [N] tensor of predicted class labels
        """
        try:
            if not predictions:
                logging.warning("Received empty predictions dictionary")
                return

            # Validate prediction shapes before conversion
            for img_id, pred in predictions.items():
                if not all(k in pred for k in ["boxes", "scores", "labels"]):
                    raise ValueError(
                        f"Missing required keys in predictions for image {img_id}"
                    )

                boxes, scores, labels = pred["boxes"], pred["scores"], pred["labels"]
                if len(boxes) != len(scores) or len(boxes) != len(labels):
                    raise ValueError(
                        f"Shape mismatch in predictions for image {img_id}: "
                        f"boxes: {boxes.shape}, scores: {scores.shape}, labels: {labels.shape}"
                    )

            coco_results = self.prepare_for_coco_detection(predictions)
            logging.debug(f"Converted {len(coco_results)} predictions to COCO format")

            self.predictions.extend(coco_results)
            self.img_ids.extend(list(predictions.keys()))

        except (IndexError, ValueError) as e:
            logging.error(
                f"Critical error during prediction update: {str(e)}", exc_info=True
            )
            raise RuntimeError(
                f"Evaluation failed due to invalid predictions: {str(e)}"
            ) from e
        except Exception as e:
            logging.error(
                f"Unexpected error during prediction update: {str(e)}", exc_info=True
            )
            raise RuntimeError(f"Evaluation failed: {str(e)}") from e

    def synchronize_between_processes(self, timeout_seconds: float = 5.0):
        """
        Synchronize evaluation state across processes in distributed training.

        Args:
            timeout_seconds: Maximum time to wait for synchronization (default: 5s)
        """
        # Only perform synchronization if we're in distributed mode
        if not dist.is_initialized() or dist.get_world_size() == 1:
            logging.debug("Skipping synchronization for single process")
            return

        try:
            # First, synchronize whether each process has predictions
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            has_predictions = torch.tensor([len(self.predictions) > 0], device=device)
            dist.all_reduce(has_predictions)

            if has_predictions.item() == 0:
                logging.info("No predictions to synchronize across any process")
                # Ensure all processes are synchronized
                dist.barrier()
                return

            if not self.predictions:
                logging.info("No local predictions to synchronize")
                # Still need to participate in the gathering process
                local_size = torch.tensor([0], device=device)
                sizes = [
                    torch.zeros_like(local_size) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(sizes, local_size)

                # Create empty tensor of the maximum size
                max_size = max(size.item() for size in sizes)
                if max_size > 0:
                    local_preds = torch.zeros(
                        (max_size, 7), device=device
                    )  # 7 = image_id, category_id, bbox(4), score
                    gathered_preds = [
                        torch.zeros_like(local_preds)
                        for _ in range(dist.get_world_size())
                    ]
                    dist.all_gather(gathered_preds, local_preds)

                dist.barrier()
                return

            # Rest of the existing synchronization code...
            logging.debug("Starting prediction synchronization between processes")
            start_time = time.time()

            # Convert predictions to tensors in a single batch
            pred_tensors = []
            for p in self.predictions:
                try:
                    # Ensure all values are properly converted to float
                    bbox = [float(x) for x in p["bbox"]]  # Convert to list of floats
                    image_id = int(p["image_id"])
                    category_id = int(p["category_id"])
                    score = float(p["score"])

                    tensor = torch.tensor(
                        [image_id, category_id] + bbox + [score],
                        device=device,
                        dtype=torch.float32,
                    )
                    pred_tensors.append(tensor)
                except (TypeError, ValueError) as e:
                    logging.warning(
                        f"Skipping invalid prediction during sync: {str(e)}"
                    )
                    continue

            # Stack predictions into a single tensor
            local_preds = torch.stack(pred_tensors)
            local_size = torch.tensor([local_preds.size(0)], device=device)

            # Check timeout before heavy operations
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(
                    "Prediction synchronization timed out during local processing"
                )

            # Gather sizes from all processes
            sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
            dist.all_gather(sizes, local_size)
            logging.debug(
                f"Gathered prediction sizes from all processes: {[s.item() for s in sizes]}"
            )

            max_size = max(size.item() for size in sizes)

            # Pad local predictions if necessary
            if local_preds.size(0) < max_size:
                padding = torch.zeros(
                    (max_size - local_preds.size(0), local_preds.size(1)),
                    device=device,
                    dtype=local_preds.dtype,
                )
                local_preds = torch.cat([local_preds, padding], dim=0)

            # Check timeout before final gather
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(
                    "Prediction synchronization timed out before final gather"
                )

            # Gather all predictions
            gathered_preds = [
                torch.zeros_like(local_preds) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_preds, local_preds)
            logging.debug("Successfully gathered predictions from all processes")

            # Convert gathered predictions back to list format
            gathered_predictions = []
            for size, preds in zip(sizes, gathered_preds):
                for i in range(size.item()):
                    t = preds[i]
                    try:
                        pred = {
                            "image_id": int(t[0].item()),
                            "category_id": int(t[1].item()),
                            "bbox": [float(x) for x in t[2:6].tolist()],
                            "score": float(t[6].item()),
                        }
                        gathered_predictions.append(pred)
                    except (IndexError, ValueError) as e:
                        logging.warning(
                            f"Skipping invalid gathered prediction: {str(e)}"
                        )
                        continue

            # Remove duplicates while preserving order
            seen = set()
            unique_predictions = []
            unique_img_ids = []

            for pred in gathered_predictions:
                try:
                    img_id = pred["image_id"]
                    bbox_tuple = tuple(pred["bbox"])
                    key = (img_id, bbox_tuple, pred["category_id"], pred["score"])

                    if key not in seen:
                        seen.add(key)
                        unique_predictions.append(pred)
                        unique_img_ids.append(img_id)
                except (TypeError, ValueError) as e:
                    logging.warning(
                        f"Skipping duplicate removal for invalid prediction: {str(e)}"
                    )
                    continue

            # Update state
            self.predictions = unique_predictions
            self.img_ids = unique_img_ids
            self._initialized_eval = False
            logging.debug(f"Synchronized {len(self.predictions)} unique predictions")

        except TimeoutError as e:
            logging.error(f"Timeout during prediction synchronization: {str(e)}")
            self.predictions = []
            self.img_ids = []
            self._initialized_eval = False
            # Ensure processes are synchronized even on timeout
            dist.barrier()
            return

        except Exception as e:
            logging.error(
                f"Error during prediction synchronization: {str(e)}", exc_info=True
            )
            self.predictions = []
            self.img_ids = []
            self._initialized_eval = False
            # Ensure processes are synchronized even on error
            dist.barrier()
            return

        # Final synchronization
        dist.barrier()

    def accumulate(self):
        """
        Accumulate evaluation results and compute metrics.
        Note: This should be called after synchronize_between_processes().
        """
        if not self.predictions:
            logging.info("No predictions to accumulate, skipping evaluation")
            return

        try:
            # Initialize evaluation only once
            if not self._initialized_eval:
                # Convert predictions to proper format
                formatted_predictions = []
                for pred in self.predictions:
                    try:
                        # Validate prediction structure
                        required_keys = ["image_id", "category_id", "bbox", "score"]
                        if not all(k in pred for k in required_keys):
                            raise ValueError(
                                f"Missing required keys in prediction: {pred}"
                            )

                        # Ensure bbox is a list
                        bbox = (
                            pred["bbox"]
                            if isinstance(pred["bbox"], list)
                            else list(pred["bbox"])
                        )

                        # Validate bbox format
                        if len(bbox) != 4:
                            raise ValueError(f"Invalid bbox format: {bbox}")

                        formatted_pred = {
                            "image_id": int(pred["image_id"]),
                            "category_id": int(pred["category_id"]),
                            "bbox": [float(x) for x in bbox],
                            "score": float(pred["score"]),
                        }
                        formatted_predictions.append(formatted_pred)
                    except (TypeError, ValueError, IndexError) as e:
                        logging.error(f"Failed to format prediction {pred}: {str(e)}")
                        continue

                if not formatted_predictions:
                    logging.error(
                        "No valid predictions after formatting, terminating training"
                    )
                    raise RuntimeError("Evaluation failed: No valid predictions")

                logging.debug(
                    f"Loading {len(formatted_predictions)} predictions into COCO evaluator"
                )
                self.coco_dt = self.coco_gt.loadRes(formatted_predictions)
                self.coco_eval = COCOeval(self.coco_gt, self.coco_dt, "bbox")

                if self.img_ids:
                    self.coco_eval.params.imgIds = sorted(
                        list(set([int(x) for x in self.img_ids]))
                    )
                self._initialized_eval = True
                logging.debug("COCO evaluator initialized successfully")

            # Run evaluation
            logging.debug("Starting COCO evaluation")
            self.coco_eval.evaluate()
            self.coco_eval.accumulate()
            logging.debug("COCO evaluation completed successfully")

        except (IndexError, ValueError) as e:
            logging.error(f"Critical error during evaluation: {str(e)}", exc_info=True)
            # Clean up state
            self._initialized_eval = False
            self.coco_dt = None
            self.coco_eval = None
            # Raise the error to stop training
            raise RuntimeError(
                f"Evaluation failed due to invalid data: {str(e)}"
            ) from e
        except Exception as e:
            logging.error(f"Critical error during evaluation: {str(e)}", exc_info=True)
            # Clean up state
            self._initialized_eval = False
            self.coco_dt = None
            self.coco_eval = None
            # Raise the error to stop training
            raise RuntimeError(f"Evaluation failed: {str(e)}") from e

    def summarize(self):
        """
        Compute and print evaluation metrics.
        """
        if hasattr(self, "coco_eval") and self.coco_eval is not None:
            try:
                self.coco_eval.summarize()
            except Exception as e:
                logging.error(f"Error during summarization: {str(e)}")

    def get_stats(self) -> Dict[str, float]:
        """
        Get evaluation statistics.

        Returns:
            Dictionary containing AP metrics:
            - map: AP @ IoU=0.50:0.95
            - map_50: AP @ IoU=0.50
            - map_75: AP @ IoU=0.75
            - map_small: AP for small objects
            - map_medium: AP for medium objects
            - map_large: AP for large objects
        """
        default_stats = {
            "map": 0.0,
            "map_50": 0.0,
            "map_75": 0.0,
            "map_small": 0.0,
            "map_medium": 0.0,
            "map_large": 0.0,
        }

        if not hasattr(self, "coco_eval") or self.coco_eval is None:
            return default_stats

        try:
            stats = self.coco_eval.stats
            if stats is None or len(stats) < 6:
                return default_stats

            return {
                "map": float(stats[0]),  # AP @ IoU=0.50:0.95
                "map_50": float(stats[1]),  # AP @ IoU=0.50
                "map_75": float(stats[2]),  # AP @ IoU=0.75
                "map_small": float(stats[3]),  # AP for small objects
                "map_medium": float(stats[4]),  # AP for medium objects
                "map_large": float(stats[5]),  # AP for large objects
            }
        except Exception as e:
            logging.error(f"Error getting stats: {str(e)}")
            return default_stats

    def prepare_for_coco_detection(
        self, predictions: Dict[int, Dict[str, torch.Tensor]]
    ) -> List[Dict]:
        """
        Convert predictions to COCO format.

        Args:
            predictions: Dict mapping image_id to dict with keys:
                - boxes: tensor [N, 4] in [x_min, y_min, x_max, y_max] format
                - scores: tensor [N]
                - labels: tensor [N]

        Returns:
            List of dicts in COCO result format
        """
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction["boxes"]) == 0:
                continue

            boxes = prediction["boxes"]
            scores = prediction["scores"]
            labels = prediction["labels"]

            # Convert boxes from [x_min, y_min, x_max, y_max] to [x_min, y_min, width, height]
            x1, y1, x2, y2 = boxes.unbind(1)
            boxes = torch.stack((x1, y1, x2 - x1, y2 - y1), dim=1)

            # Convert to Python lists
            boxes = boxes.tolist()
            scores = scores.tolist()
            labels = labels.tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )

        return coco_results

    def reset(self):
        """Reset evaluator state."""
        self.predictions = []
        self.img_ids = []
        self.coco_dt = None
        self.coco_eval = None
        self._initialized_eval = False

        self._initialized_eval = False
