# Substep 3: Task Graph Matching for Task Verification

## Overview

This substep implements **Task Graph Matching** for the "From Mistake Detection to Task Verification" extension. This approach follows the extension specification:

1. **Encodes task graph descriptions** using CLIP text encoder (aligned video-text space)
2. **Matches visual steps to task graph nodes** using the Hungarian algorithm
3. **Uses matching quality** as a signal for error detection

**Key Requirement**: Uses **Perception/EgoVLP features** (768-dim) with **CLIP text encoder** (512-dim) because they share an aligned video-text embedding space, enabling meaningful visual-to-text matching.

## Architecture

```
Visual Steps (Perception features)       Task Graph (textual descriptions)
        │                                            │
        ▼                                            ▼
   Visual Proj                                  CLIP Text Encoder
   (768 → 256)                               (OpenAI ViT-B/32)
        │                                            │
        ▼                                            ▼
   Transformer                                   Text Proj
   (self-attention)                              (512 → 256)
        │                                            │
        └────────────────┬───────────────────────────┘
                         │
                    Cross-Attention
                         │
                         ▼
                 Hungarian Matching
                 (visual → text nodes)
                         │
                         ▼
                 Feature Fusion
        (visual features + matched text features + scores)
                         │
                         ▼
                   Classification
                   (error / no error)
```

## Key Components

### 1. TaskGraphMatcher Model (`model/task_graph_matcher.py`)

- **Visual Projection**: Projects 768-dim Perception features to 256-dim shared space
- **Text Encoder**: Uses CLIP ViT-B/32 (512-dim) for task graph descriptions
- **Transformer**: Contextualizes visual steps with self-attention
- **Cross-Attention**: Visual steps attend to task graph nodes
- **Hungarian Matching**: Optimal assignment between visual steps and graph nodes
- **Classification Head**: Binary classification for error detection

### 2. TextEncoder (`model/task_graph_matcher.py`)

Supports two modes:
- **CLIP** (default, recommended): 512-dim, aligned with Perception visual features
- **sentence-transformers** (fallback): 384-dim, `all-MiniLM-L6-v2`

## Files

```
substep3_step_detection/
├── model/
│   ├── __init__.py
│   ├── task_graph_matcher.py       # Main model with Hungarian matching
│   └── step_detector.py            # Legacy simple detector
├── train_task_graph_matching.py    # Training script (CPU)
├── train_task_graph_matching_gpu.py # Training script (GPU optimized)
├── eval_task_graph_matching.py     # Evaluation script
├── prepare_perception_embeddings.py # Generate perception step embeddings
├── activity_to_taskgraph.json      # Activity → Task Graph mapping
├── create_activity_mapping.py      # Script to generate mapping
└── README.md                       # This file
```

## Installation

```bash
# Install CLIP (required for text encoder)
pip install git+https://github.com/openai/CLIP.git

# Or use sentence-transformers as fallback
pip install sentence-transformers
```

## Usage

### 1. Prepare Perception Embeddings

```bash
python substep3_step_detection/prepare_perception_embeddings.py \
    --feat_folder data/video/perception \
    --output substep3_step_detection/step_embeddings_perception.npy
```

### 2. Training (CPU)

```bash
python substep3_step_detection/train_task_graph_matching.py \
    --npy extension_localization/data/step_embeddings_perception.npy \
    --annotations extension_localization/data/step_annotations.json \
    --epochs 30 \
    --ckpt_dir substep3_step_detection/checkpoints_graph
```

### 3. Training (GPU - Recommended)

```bash
python substep3_step_detection/train_task_graph_matching_gpu.py \
    --npy extension_localization/data/step_embeddings_perception.npy \
    --annotations extension_localization/data/step_annotations.json \
    --epochs 30 \
    --ckpt_dir substep3_step_detection/checkpoints_graph
```

### 4. Evaluation

```bash
python substep3_step_detection/eval_task_graph_matching.py \
    --checkpoint substep3_step_detection/checkpoints_graph/task_graph_matcher_recipe_1.pth \
    --test_recipe 1 \
    --output results/task_graph_eval.json
```

## Parameters

### Training
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--npy` | `step_embeddings_perception.npy` | Path to perception step embeddings |
| `--annotations` | `step_annotations.json` | Path to step annotations JSON |
| `--recording_csv` | `recording_id_step_idx.csv` | Recording → activity mapping |
| `--activity_mapping` | `activity_to_taskgraph.json` | Activity → task graph mapping |
| `--task_graph_dir` | `annotations/task_graphs` | Task graph JSON directory |
| `--text_model` | `clip` | Text encoder: `clip` or `all-MiniLM-L6-v2` |
| `--visual_dim` | `768` | Visual embedding dim (768 perception, 1024 omnivore) |
| `--epochs` | `30` | Number of training epochs |
| `--batch_size` | `8` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--hidden_dim` | `256` | Hidden dimension |

### Evaluation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | - | Path to model checkpoint |
| `--test_recipe` | - | Recipe ID for leave-one-out evaluation |
| `--threshold` | `0.5` | Classification threshold |
| `--output` | - | Path to save results JSON |

## Data Requirements

1. **Perception features** (`data/video/perception/*.npz`): 768-dim visual embeddings
2. **Step embeddings** (`step_embeddings_perception.npy`): Aggregated per-step features
3. **Step annotations** (`step_annotations.json`): Error labels per video
4. **Task graphs** (`annotations/task_graphs/*.json`): DAG with step descriptions
5. **Recording mapping** (`recording_id_step_idx.csv`): video_id → activity_id

## Feature Dimensions

| Feature Type | Dimension | Text Encoder | Notes |
|--------------|-----------|--------------|-------|
| Perception (default) | 768 | CLIP (512) | Aligned video-text space |
| Omnivore (legacy) | 1024 | sentence-transformers (384) | Not aligned |

## Key Differences from Substep 2

| Aspect | Substep 2 (RNN Baseline) | Substep 3 (Task Graph) |
|--------|-------------------------|------------------------|
| Visual features | Omnivore (1024-dim) | Perception (768-dim) |
| Text features | None | Task graph + CLIP |
| Matching | None | Hungarian algorithm |
| Architecture | BiLSTM | Transformer + Cross-attention |
| Input | Visual only | Visual + Textual |
| Interpretability | Low | High (matching scores) |

## Hungarian Algorithm

The Hungarian algorithm finds the optimal one-to-one assignment between:
- **Visual steps** (from Perception features)
- **Task graph nodes** (expected procedure steps)

This provides:
1. **Matching cost**: Lower cost = better procedure execution
2. **Unmatched nodes**: Missing steps or extra actions
3. **Matching quality**: Signal for error detection

## Notes

- **CLIP is recommended** for aligned video-text embeddings
- Training uses **leave-one-recipe-out** cross-validation
- Hungarian matching is computed offline (not differentiable)
- Cross-attention provides soft alignment for gradient flow
- Dataset: 384 videos, 24 recipes, ~220 with errors
