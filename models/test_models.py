"""Test DETR model with various input configurations."""

import logging
from typing import Dict

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from .detr import DETR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./models/test_model.log', mode='w'),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)


def create_test_config() -> Dict:
    """Create a test configuration using OmegaConf."""
    config = {
        "model": {
            "num_classes": 91,
            "hidden_dim": 256,
            "nheads": 8,
            "num_encoder_layers": 3,
            "num_decoder_layers": 3,
            "dim_feedforward": 1024,
            "dropout": 0.1,
            "num_queries": 100,
            "backbone_name": "resnet18",
            "pretrained_backbone": True,
            "learnable_tgt": False,
            "use_aux_loss": True,
            "position_embedding": {"type": "sine", "normalize": True},
            "bbox_predictor": {"num_layers": 3, "hidden_dim": 256},
            "init": {"xavier_uniform": True, "prior_prob": 0.01},
        },
        "optimizer": {
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "backbone_lr_factor": 0.1,
        },
    }
    return OmegaConf.create(config)


def log_tensor_info(name: str, tensor: torch.Tensor):
    """Log tensor information."""
    info = [
        f"{name}:",
        f"  Shape: {tensor.shape}",
        f"  Type: {tensor.dtype}",
        f"  Device: {tensor.device}"
    ]
    if tensor.numel() > 0:
        info.append(f"  Range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    logger.info("\n".join(info))


def test_backbone_output(model: DETR, device: torch.device):
    """Test backbone feature extraction."""
    logger.info("\nTesting backbone output...")
    x = torch.randn(2, 3, 224, 224).to(device)
    features = model.backbone(x)
    log_tensor_info("Backbone features", features)


def test_position_embedding(model: DETR, device: torch.device):
    """Test position embedding generation."""
    logger.info("\nTesting position embedding...")
    x = torch.randn(2, 3, 224, 224).to(device)
    features = model.backbone(x)
    features = model.conv(features)
    pos_embed = model.position_embedding(features)
    log_tensor_info("Position embedding", pos_embed)


def test_transformer_input_shapes(model: DETR, device: torch.device):
    """Test transformer input preparation."""
    logger.info("\nTesting transformer input/output shapes...")

    # Create input tensor
    x = torch.randn(2, 3, 224, 224).to(device)

    # Forward pass through the model
    with torch.no_grad():
        output = model(x)

    # Verify output shapes
    batch_size = x.shape[0]
    expected_class_shape = (batch_size, model.num_queries, model.num_classes + 1)
    expected_box_shape = (batch_size, model.num_queries, 4)

    logger.info("\nOutput shapes:")
    logger.info(
        f"Pred logits shape: {output['pred_logits'].shape} (expected {expected_class_shape})"
    )
    logger.info(
        f"Pred boxes shape: {output['pred_boxes'].shape} (expected {expected_box_shape})"
    )

    # Verify shapes match expected dimensions
    assert (
        output["pred_logits"].shape == expected_class_shape
    ), f"Class predictions shape mismatch: got {output['pred_logits'].shape}, expected {expected_class_shape}"
    assert (
        output["pred_boxes"].shape == expected_box_shape
    ), f"Box predictions shape mismatch: got {output['pred_boxes'].shape}, expected {expected_box_shape}"

    logger.info("\nAll output shapes verified successfully!")


def test_variable_size(config: Dict):
    """Test DETR with variable-size images (padding mask required)."""
    logger.info("\nTesting variable-size images...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DETR(config).to(device)
    model.eval()

    # Create batch with different image sizes
    max_h, max_w = 224, 224

    # Image 1: Full size
    img1 = torch.randn(3, 224, 224)
    mask1 = torch.zeros(224, 224, dtype=torch.bool)  # No padding

    # Image 2: Smaller with padding
    img2 = torch.randn(3, 160, 200)
    mask2 = torch.zeros(224, 224, dtype=torch.bool)
    mask2[160:, :] = True  # Padding in height
    mask2[:, 200:] = True  # Padding in width

    # Pad image 2 to match max size
    pad_h = max_h - img2.shape[1]
    pad_w = max_w - img2.shape[2]
    img2 = F.pad(img2, (0, pad_w, 0, pad_h), value=0)

    # Create batch
    images = torch.stack([img1, img2]).to(device)
    masks = torch.stack([mask1, mask2]).to(device)

    with torch.no_grad():
        output = model(images, masks)

    log_tensor_info("Input images", images)
    log_tensor_info("Padding masks", masks)
    log_tensor_info("Pred logits", output["pred_logits"])
    log_tensor_info("Pred boxes", output["pred_boxes"])


def test_auxiliary_outputs(model: DETR, device: torch.device):
    """Test auxiliary outputs from intermediate decoder layers."""
    logger.info("\nTesting auxiliary outputs...")
    x = torch.randn(2, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(x)
    
    # Verify main outputs
    log_tensor_info("Main pred logits", output["pred_logits"])
    log_tensor_info("Main pred boxes", output["pred_boxes"])
    
    # Verify auxiliary outputs
    if "aux_outputs" in output:
        logger.info("\nFound auxiliary outputs:")
        for i, aux_out in enumerate(output["aux_outputs"]):
            logger.info(f"\nLayer {i}:")
            log_tensor_info(f"Aux pred logits", aux_out["pred_logits"])
            log_tensor_info(f"Aux pred boxes", aux_out["pred_boxes"])
            
            # Verify shapes match main output
            assert aux_out["pred_logits"].shape == output["pred_logits"].shape, \
                f"Auxiliary logits shape mismatch at layer {i}"
            assert aux_out["pred_boxes"].shape == output["pred_boxes"].shape, \
                f"Auxiliary boxes shape mismatch at layer {i}"
    else:
        logger.warning("\nNo auxiliary outputs found (use_aux_loss might be False)")


def test_end_to_end(config: DictConfig):
    """Test complete forward pass with shape verification."""
    logger.info("\nTesting end-to-end forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DETR(config).to(device)
    model.eval()

    # Test with different batch sizes
    batch_sizes = [1, 2, 4]
    for batch_size in batch_sizes:
        logger.info(f"\nTesting batch size: {batch_size}")
        x = torch.randn(batch_size, 3, 224, 224).to(device)

        with torch.no_grad():
            output = model(x)

        # Verify output shapes
        B = batch_size
        N = model.num_queries
        C = model.num_classes + 1

        assert (
            output["pred_logits"].shape == (B, N, C)
        ), f"Logits shape mismatch: expected {(B, N, C)}, got {output['pred_logits'].shape}"
        assert (
            output["pred_boxes"].shape == (B, N, 4)
        ), f"Boxes shape mismatch: expected {(B, N, 4)}, got {output['pred_boxes'].shape}"

        log_tensor_info("Pred logits", output["pred_logits"])
        log_tensor_info("Pred boxes", output["pred_boxes"])

        # Verify auxiliary outputs if enabled
        if model.use_aux_loss:
            assert "aux_outputs" in output, "Auxiliary outputs missing when use_aux_loss=True"
            num_aux = len(output["aux_outputs"])
            expected_aux = model.transformer.decoder.num_layers - 1
            assert num_aux == expected_aux, \
                f"Expected {expected_aux} auxiliary outputs, got {num_aux}"

    logger.info("\nEnd-to-end test completed successfully!")


def test_optimizer(config: DictConfig):
    """Test optimizer configuration."""
    logger.info("\nTesting optimizer configuration...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DETR(config).to(device)

    optimizer = model.configure_optimizer(config)
    logger.info(f"Optimizer type: {type(optimizer).__name__}")

    # Verify parameter groups
    for i, param_group in enumerate(optimizer.param_groups):
        logger.info(f"\nParameter group {i}:")
        logger.info(f"Learning rate: {param_group['lr']}")
        logger.info(f"Weight decay: {param_group['weight_decay']}")
        logger.info(f"Number of parameters: {len(param_group['params'])}")


def main():
    """Run comprehensive tests."""
    logger.info("Starting DETR tests...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create test config
    config = create_test_config()
    logger.info("\nTest configuration:")
    logger.info("\n" + OmegaConf.to_yaml(config))

    # Initialize model
    model = DETR(config).to(device)
    model.eval()

    try:
        # Run component tests
        test_transformer_input_shapes(model, device)
        test_auxiliary_outputs(model, device)
        test_optimizer(config)

        # Run end-to-end tests
        test_end_to_end(config)

        logger.info("\nAll tests passed successfully!")

    except Exception as e:
        logger.error(f"\nTest failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
