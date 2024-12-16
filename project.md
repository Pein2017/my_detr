# DETR Implementation Project

## Overview
This project implements DETR (DEtection TRansformer) for object detection using PyTorch. The implementation focuses on clean code, type safety, and modularity while maintaining high performance.

## Project Structure
```
my_detr/
├── models/
│   ├── backbone.py        # CNN backbone
│   ├── detr.py           # Main model
│   ├── transformer.py    # Transformer
│   └── matcher.py        # Hungarian matcher
├── data/
│   ├── dataset.py       # Dataset and data loading
│   ├── transforms.py    # Augmentations
│   └── coco_eval.py    # Evaluation
├── training/
│   ├── trainer.py       # Training loop and utilities
│   ├── model_wrapper.py # Model wrapper with loss computation
│   ├── optimizer.py     # Optimizer and scheduler setup
│   └── distributed.py   # Distributed training utilities
├── utils/
│   ├── setup.py        # Experiment setup, logging, and utilities
│   └── config.py       # Configuration dataclasses and utilities
├── config/
│   ├── default.yaml     # Base configuration
│   └── experiment/      # Experiment variants
│       ├── small.yaml   # Lightweight config
│       └── resnet_50_ddp.yaml  # Multi-GPU config
└── train.py            # Training script
```

## Core Components

### 1. DETR Model (`models/detr.py`)
- CNN backbone (ResNet) for feature extraction
- Transformer encoder-decoder for object detection
- Prediction heads for class and box coordinates
- Position embeddings for spatial information
- Object queries for detection slots

### 2. Data Pipeline (`data/dataset.py`)
- Efficient batch collation with padding
- Dynamic image size handling
- Box coordinate normalization
- Configurable augmentation pipeline
- Train/val/test split management

### 3. Training System (`training/trainer.py`)
- Custom PyTorch training loop
- Hungarian matching for loss computation
- Multi-component loss function
- Automatic mixed precision support
- Multi-GPU support with DDP
- Comprehensive logging with TensorBoard
- Checkpoint management
- Distributed evaluation

### 4. Evaluation (`data/coco_eval.py`)
- Standard COCO metrics
- Distributed evaluation support
- Prediction format conversion
- Per-class performance tracking

## Configuration System

### Structure
The project uses a hierarchical configuration system based on Hydra/OmegaConf:

1. `config/default.yaml`: Base configuration with all default settings
2. `config/experiment/*.yaml`: Specialized configurations
   - `small.yaml`: Lightweight model for fast experimentation
   - `resnet_50_ddp.yaml`: Multi-GPU training with ResNet-50

### Key Configuration Sections
```yaml
data:
  data_root_dir: ${data_root_dir}  # Set via environment variable
  train_dir: train2017
  val_dir: val2017
  train_ann: annotations/instances_train2017.json
  val_ann: annotations/instances_val2017.json
  pin_memory: true
  shuffle_train: true
  shuffle_val: false
  persistent_workers: false

output:
  base_dir: output  # Base directory for all experiment outputs
  log_dir: tensorboard  # Relative to experiment directory
  checkpoint_dir: checkpoints  # Relative to experiment directory
  visualization_dir: visualizations  # Relative to experiment directory

model:
  backbone_name: resnet50
  pretrained_backbone: false
  num_classes: 91  # 90 COCO classes + 1 background
  hidden_dim: 256
  nheads: 8
  num_encoder_layers: 6
  num_decoder_layers: 6
  dim_feedforward: 2048
  dropout: 0.1
  num_queries: 100
  learnable_tgt: false

  position_embedding:
    type: sine
    normalize: true
  
  bbox_predictor:
    num_layers: 3
    hidden_dim: 256
  
  init:
    xavier_uniform: true
    prior_prob: 0.01

training:
  max_epochs: 300
  batch_size: 2  # Per GPU batch size
  num_workers: 4
  seed: 17
  deterministic: false
  gradient_clip_val: 0.1
  accumulate_grad_batches: 1
  check_val_every_n_epoch: 1
  precision: 32

optimizer:
  name: adamw
  lr: 1.0e-4  # Base learning rate
  weight_decay: 1.0e-4
  backbone_lr_factor: 0.1  # Factor to multiply with base lr
  
  lr_scheduler:
    name: step
    step_size: 200
    gamma: 0.1
    warmup_epochs: 0
    warmup_factor: 1.0
    min_lr: 0.0
```

## Model Architecture

### 1. Backbone Models
The project supports multiple ResNet variants as the CNN backbone:
- ResNet-18 (512 channels)
- ResNet-34 (512 channels)
- ResNet-50 (2048 channels)

Each backbone removes the classification head and returns feature maps with shape `[B, C, H/32, W/32]` where C depends on the backbone variant.

### 2. Transformer Architecture
The transformer consists of:
- **Encoder**: Stack of transformer encoder layers that process image features
  - Input shape: `[HW, B, D]` (image features and position encodings)
  - Output shape: `[HW, B, D]` (encoded image features)
  
- **Decoder**: Stack of transformer decoder layers that generate object detections
  - Input shapes:
    - Query embeddings: `[N, B, D]` (learnable object queries)
    - Memory: `[HW, B, D]` (encoded image features)
    - Position encodings: `[HW, B, D]`
  - Output shape: `[N, B, D]` (object detection features)

## Running the Training

### Using the Start Script
The project includes a `start_train.sh` script for easy training:
```bash
# Default usage
./start_train.sh

# With specific experiment
EXPERIMENT=resnet_34_ddp ./start_train.sh
```

### Basic Usage
```bash
# Train with default configuration (ResNet-50)
python train.py --config-name default

# Train with ResNet-18 (lightweight model)
python train.py --config-name default +experiment=resnet_18

# Train with ResNet-34 and DDP (multi-GPU)
python train.py --config-name default +experiment=resnet_34_ddp
```

### Configuration Override Patterns
```bash
# Method 1: Basic experiment override
python train.py --config-name default +experiment=resnet_18

# Method 2: Override experiment with additional parameters
python train.py --config-name default +experiment=resnet_18 \
    training.batch_size=4 \
    optimizer.lr=2e-4

# Method 3: Override data paths
python train.py --config-name default +experiment=resnet_18 \
    data.data_root_dir=/path/to/coco

# Method 4: DDP training with specific GPU count
python train.py --config-name default +experiment=resnet_50_ddp \
    distributed.num_gpus=4
```

### Environment Configuration
```bash
# Set data path via environment variable
export data_root_dir=/path/to/coco
python train.py --config-name default +experiment=resnet_18

# Debug/Development mode
python train.py --config-name default +experiment=resnet_18 \
    training.max_epochs=1 \
    training.limit_train_batches=10 \
    training.limit_val_batches=10
```

## Output Directory Structure
Each experiment creates its own directory under the base output directory with the following structure:
```
output/
└── {experiment_name}/
    ├── logs/
    │   └── train.log         # Training process logs
    ├── tensorboard/         # TensorBoard event files
    ├── checkpoints/         # Model checkpoints
    │   ├── checkpoint_epoch_{N}.pth
    │   └── best_model.pth
    └── visualizations/      # Debug visualizations
```

## Monitoring

### Available Tools
1. TensorBoard logs (default: `tensorboard/`)
   ```bash
   tensorboard --logdir output/{experiment_name}/tensorboard
   ```

2. Checkpoints (default: `checkpoints/`)
   - Best models saved based on validation mAP
   - Regular epoch checkpoints
   - Includes full state for training resumption:
     - Model state
     - Optimizer state
     - Learning rate scheduler state
     - Training state (epoch, step)

3. Metrics
   - COCO AP metrics (IoU=0.50:0.95, 0.50, 0.75)
   - Per-size AP (small/medium/large objects)
   - Loss components (classification, box regression, GIoU)
   - Training throughput and GPU usage

## Next Steps

1. Training Optimization
   - Learning rate warmup strategies
   - Advanced augmentation pipelines
   - Hyperparameter optimization

2. Feature Enhancements
   - Additional backbone architectures
   - Distributed training optimizations
   - Extended evaluation metrics

## Tensor Flow and Loss Computation

### 1. Input Processing and Backbone
1. **Input Tensors**:
   - Images: `[B × 3 × H × W]`
   - Padding Masks: `[B × H × W]` (True indicates padding)

2. **Backbone Processing**:
   ```
   Input Image [B × 3 × H × W]
   → ResNet Backbone
   Features [B × C × H' × W']  where H'=H/32, W'=W/32  (C=512 for ResNet34, 2048 for ResNet50)
   → 1×1 Convolution
   Projected Features [B × hidden_dim × H' × W']
   ```

### 2. Transformer Input Preparation
1. **Position Embeddings**:
   ```
   Features [B × hidden_dim × H' × W']
   → PositionEmbeddingSine
   Position Encodings [B × hidden_dim × H' × W']
   + Features
   Final Features [B × hidden_dim × H' × W']
   ```

2. **Mask Processing**:
   ```
   Padding Mask [B × H × W]
   → Spatial Downsampling (via interpolate)
   Feature Mask [B × H' × W']  where H'=H/32, W'=W/32
   → Flatten
   Attention Mask [B × H'W']
   ```

3. **Feature Preparation**:
   ```
   Features [B × hidden_dim × H' × W']  where H'=H/32, W'=W/32
   → Flatten + Permute
   Flattened Features [H'W' × B × hidden_dim]  # H'W' positions for each batch
   → LayerNorm
   Normalized Features [H'W' × B × hidden_dim]
   ```

### 3. Transformer Processing
1. **Encoder**:
   ```
   Input: 
   - src: Normalized Features [H'W' × B × hidden_dim]
   - pos: Position Encodings [H'W' × B × hidden_dim]
   - mask: Attention Mask [B × H'W']
   → Self-Attention (src + pos as query/key, src as value)
   → FFN
   (×6 layers)
   Output: Memory [H'W' × B × hidden_dim]
   ```

2. **Decoder**:
   ```
   Input:
   - tgt: Zero Tensor [N × B × hidden_dim]
   - memory: Encoded Features [H'W' × B × hidden_dim]
   - query_pos: Object Queries [N × B × hidden_dim]  # Learnable embeddings (nn.Embedding)
   - pos: Position Encodings [H'W' × B × hidden_dim]  # Fixed sinusoidal encodings
   - mask: Attention Mask [B × H'W']
   → Self-Attention (tgt + query_pos as query/key, tgt as value)
   → Cross-Attention ((tgt + query_pos) as query, (memory + pos) as key, memory as value)
   → FFN
   (×6 layers)
   Output: Detection Features [N × B × hidden_dim]
   ```

### 4. Prediction Heads
```
Detection Features [N × B × hidden_dim]
→ Linear + Softmax
Class Predictions [B × N × (num_classes + 1)]
→ MLP + Sigmoid
Box Predictions [B × N × 4] (normalized cxcywh format)
```

### 5. Loss Computation
1. **Hungarian Matching**:
   ```
   Simple Example (B=1, N=4 predictions, M=3 targets):
   
   Predictions:                              Ground Truth:
   - pred_1: "cat" at [0.2, 0.3]            - gt_1: "cat" at [0.2, 0.3]     # Perfect match
   - pred_2: "dog" at [0.4, 0.8]            - gt_2: "dog" at [0.5, 0.8]     # Right class, small offset
   - pred_3: "cat" at [0.7, 0.8]            - gt_3: "cat" at [0.7, 0.7]     # Right class, small offset
   - pred_4: "∅" at [0.1, 0.2]              (∅ = no object)
   
   Box coordinates are normalized [x, y] center positions in range [0, 1]

   Building Cost Matrix:
   For each prediction i in N:
     For each target j in M:
       cost_matrix[i,j] = compute_cost(pred_i, gt_j)

   Cost Matrix Calculation Examples:
   Good Match (pred_1 → gt_1):
   1. Classification Cost: -P(correct_class) = -0.9 = -0.9  # High probability for "cat"
   2. L1 Box Cost: |0.2-0.2| + |0.3-0.3| = 0              # Perfect position match
   3. Total Cost = λ₁×Class_Cost + λ₂×Box_Cost
                = 1×(-0.9) + 1×(0) = 0.1                   # Low total cost = good match

   Bad Match (pred_2 → gt_1):
   1. Classification Cost: -P(correct_class) = -0.1 = -0.1  # Low probability for "cat"
   2. L1 Box Cost: |0.4-0.2| + |0.8-0.3| = 0.7            # Large position difference
   3. Total Cost = λ₁×Class_Cost + λ₂×Box_Cost
                = 1×(-0.1) + 1×(0.7) = 2.0                 # High total cost = bad match

   Complete Cost Matrix [N×M] (after computing all N×M=12 pairs):
                gt_1      gt_2      gt_3
   pred_1   [  0.1,     2.0,      1.8  ]  # Low cost: right class + perfect position
   pred_2   [  2.0,     0.2,      2.0  ]  # Low cost: right class + small offset
   pred_3   [  1.8,     2.0,      0.1  ]  # Low cost: right class + small offset
   pred_4   [  2.0,     2.0,      2.0  ]  # High cost: wrong class + wrong position

   Hungarian Algorithm → Optimal Assignment:
   pred_1 ↔ gt_1 (cat)    # Lowest cost pair
   pred_2 ↔ gt_2 (dog)    # Second lowest cost pair
   pred_3 ↔ gt_3 (cat)    # Third lowest cost pair
   pred_4 ↔ ∅ (unmatched) # No good match, predict empty
   ```

   General Case:
   ```
   Inputs:
   - Predicted Classes [B × N × (num_classes + 1)]
   - Predicted Boxes [B × N × 4]
   - Target Classes [B × M]
   - Target Boxes [B × M × 4]
   → Compute Cost Matrix:
     * Classification Cost: -P(target_class)
     * L1 Box Cost: ||pred_box - target_box||₁
     * GIoU Cost: -GIoU(pred_box, target_box)
   Total Cost = λ₁×L1_Cost + λ₂×Class_Cost + λ₃×GIoU_Cost
   → Hungarian Algorithm
   Output: Optimal Prediction-to-Target Assignment
   ```

2. **Loss Components**:
   ```
   For each matched pair:
   → Classification: CrossEntropy(pred_class, target_class)
   → Box Regression: L1Loss(pred_box, target_box)
   → GIoU: 1 - GIoU(pred_box, target_box)
   
   Final Loss = λ₁×Classification + λ₂×Box_Regression + λ₃×GIoU
   ```