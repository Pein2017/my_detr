"""Test DETR model with various input configurations."""

from typing import Dict

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from .detr import DETR


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
            "pretrained_backbone": False,
            "learnable_tgt": False,
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


def print_tensor_info(name: str, tensor: torch.Tensor):
    """Print tensor information."""
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Type: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    if tensor.numel() > 0:
        print(f"  Range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    print()


def test_backbone_output(model: DETR, device: torch.device):
    """Test backbone feature extraction."""
    print("\nTesting backbone output...")
    x = torch.randn(2, 3, 224, 224).to(device)
    features = model.backbone(x)
    print_tensor_info("Backbone features", features)


def test_position_embedding(model: DETR, device: torch.device):
    """Test position embedding generation."""
    print("\nTesting position embedding...")
    x = torch.randn(2, 3, 224, 224).to(device)
    features = model.backbone(x)
    features = model.conv(features)
    pos_embed = model.position_embedding(features)
    print_tensor_info("Position embedding", pos_embed)


def test_transformer_input_shapes(model: DETR, device: torch.device):
    """Test transformer input preparation."""
    print("\nTesting transformer input/output shapes...")

    # Create input tensor
    x = torch.randn(2, 3, 224, 224).to(device)

    # Forward pass through the model
    with torch.no_grad():
        output = model(x)

    # Verify output shapes
    batch_size = x.shape[0]
    expected_class_shape = (batch_size, model.num_queries, model.num_classes + 1)
    expected_box_shape = (batch_size, model.num_queries, 4)

    print("\nOutput shapes:")
    print(
        f"Pred logits shape: {output['pred_logits'].shape} (expected {expected_class_shape})"
    )
    print(
        f"Pred boxes shape: {output['pred_boxes'].shape} (expected {expected_box_shape})"
    )

    # Verify shapes match expected dimensions
    assert (
        output["pred_logits"].shape == expected_class_shape
    ), f"Class predictions shape mismatch: got {output['pred_logits'].shape}, expected {expected_class_shape}"
    assert (
        output["pred_boxes"].shape == expected_box_shape
    ), f"Box predictions shape mismatch: got {output['pred_boxes'].shape}, expected {expected_box_shape}"

    print("\nAll output shapes verified successfully!")


def test_fixed_size(config: Dict):
    """Test DETR with fixed-size images (no padding mask needed)."""
    print("\nTesting fixed-size images...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DETR(config).to(device)
    model.eval()

    # Fixed-size input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    print_tensor_info("Input", input_tensor)
    print_tensor_info("Pred logits", output["pred_logits"])
    print_tensor_info("Pred boxes", output["pred_boxes"])


def test_variable_size(config: Dict):
    """Test DETR with variable-size images (padding mask required)."""
    print("\nTesting variable-size images...")
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

    print_tensor_info("Input images", images)
    print_tensor_info("Padding masks", masks)
    print_tensor_info("Pred logits", output["pred_logits"])
    print_tensor_info("Pred boxes", output["pred_boxes"])


def test_end_to_end(config: DictConfig):
    """Test complete forward pass with shape verification."""
    print("\nTesting end-to-end forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DETR(config).to(device)
    model.eval()

    # Test with different batch sizes
    batch_sizes = [1, 2, 4]
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
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

        print_tensor_info("Pred logits", output["pred_logits"])
        print_tensor_info("Pred boxes", output["pred_boxes"])

    print("\nEnd-to-end test completed successfully!")


def test_optimizer(config: DictConfig):
    """Test optimizer configuration."""
    print("\nTesting optimizer configuration...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DETR(config).to(device)

    optimizer = model.configure_optimizer(config)
    print(f"Optimizer type: {type(optimizer).__name__}")

    # Verify parameter groups
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"\nParameter group {i}:")
        print(f"Learning rate: {param_group['lr']}")
        print(f"Weight decay: {param_group['weight_decay']}")
        print(f"Number of parameters: {len(param_group['params'])}")


def main():
    """Run comprehensive tests."""
    print("Starting DETR tests...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create test config
    config = create_test_config()
    print("\nTest configuration:")
    print(OmegaConf.to_yaml(config))

    # Initialize model
    model = DETR(config).to(device)
    model.eval()

    try:
        # Run component tests
        test_transformer_input_shapes(model, device)
        test_optimizer(config)

        # Run end-to-end tests
        test_end_to_end(config)

        print("\nAll tests passed successfully!")

    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
