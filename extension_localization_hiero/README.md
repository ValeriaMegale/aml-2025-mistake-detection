# Step Localization with HiERO-style Clustering

Zero-shot approach for temporal step localization in recipe videos using hierarchical clustering.

## Overview

This module implements **Substep 1** of the extension task: recipe step localization. It uses a clustering-based approach (inspired by HiERO) to segment recipe videos into individual steps without requiring training or pre-trained models.

## Features

- **Zero-shot**: No training required, works directly on pre-extracted features
- **Hierarchical clustering**: Uses agglomerative clustering to find step boundaries
- **Automatic postprocessing**: Removes short segments, applies temporal NMS, limits segment count
- **Step embeddings**: Computes step-level embeddings by averaging features within segments

## Requirements

```bash
pip install numpy scipy scikit-learn matplotlib pyyaml tqdm
```

## Usage

### 1. Run Step Localization Pipeline

Process all videos and generate step segments + embeddings:

```bash
cd /home/pierpaolosorbellini/AML/aml-2025-mistake-detection
python extension_localization_hiero/run_step_localization.py
```

With custom config:

```bash
python extension_localization_hiero/run_step_localization.py \
    --config extension_localization_hiero/config_step_localization.yaml \
    --feat_folder data/video/perception \
    --split all \
    --output_segments extension_localization_hiero/data/step_segments_perception.json \
    --output_embeddings extension_localization_hiero/data/step_embeddings_perception.npy
```

### 2. Validate Results

Check statistics and optionally compare with ground truth:

```bash
python extension_localization_hiero/validate_step_localization.py \
    --segments extension_localization_hiero/data/step_segments_perception.json \
    --embeddings extension_localization_hiero/data/step_embeddings_perception.npy \
    --compare_gt \
    --plot
```

### 3. Use in Code

```python
from extension_localization_hiero import localize_steps_clustering, compute_step_embeddings
import numpy as np

# Load features (shape [T, D])
features = np.load('data/video/perception/video_id_360p.mp4_1s_1s.npz')['arr_0']

# Configuration
config = {
    'feat_stride': 1.0,
    'fps': 30.0,
    'clustering_method': 'hierarchical',
    'clustering_distance': 'cosine',
    'min_segment_duration': 2.0,
    'max_segments_per_video': 50
}

# Localize steps
segments, scores = localize_steps_clustering('video_id', features, config)
# segments: [(start1, end1), (start2, end2), ...]
# scores: [score1, score2, ...]

# Compute embeddings
embeddings, segment_info = compute_step_embeddings('video_id', segments, features, config)
# embeddings: numpy array [N, D] where N = number of segments
```

## Configuration

Edit `config_step_localization.yaml` to adjust parameters:

- **Clustering**: `clustering_method`, `clustering_distance`, `linkage_method`
- **Postprocessing**: `min_segment_duration`, `max_segments_per_video`, `nms_threshold`
- **Embeddings**: `pooling_method` (mean or max)

## Output Format

### Segments JSON

```json
{
  "video_id": {
    "segments": [[start1, end1], [start2, end2], ...],
    "scores": [score1, score2, ...],
    "num_segments": N,
    "meta": {
      "method": "hierarchical",
      "min_duration": 2.0,
      "max_segments": 50
    }
  }
}
```

### Embeddings NPY

Dictionary mapping `video_id` → numpy array of shape `[N, D]` where:
- `N` = number of segments for that video
- `D` = feature dimension (768 for perception encoder)

## Algorithm

1. **Feature Loading**: Load pre-extracted perception features (shape [T, 768])
2. **Clustering**: Apply hierarchical clustering on temporal features
   - Estimate number of clusters based on video duration (~1 step per 10-15 seconds)
   - Use cosine or euclidean distance
   - Convert cluster labels to temporal segments
3. **Postprocessing**:
   - Remove segments shorter than `min_segment_duration`
   - Apply temporal NMS to remove overlaps
   - Limit to `max_segments_per_video`
4. **Embedding Computation**: Average pool features within each segment boundary

## Notes

- Features should be extracted with 1-second stride (1s_1s.npz format)
- Clustering can be slow for very long videos (>1000 features) - consider downsampling
- Results are deterministic given same features and config
- No ground truth annotations required (fully unsupervised)
