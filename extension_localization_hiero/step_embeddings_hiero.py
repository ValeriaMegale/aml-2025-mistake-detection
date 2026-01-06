"""
Step Embeddings Computation
Compute step-level embeddings by averaging video features within step boundaries.
"""

import numpy as np


def time_to_feature_index(timestamp, feat_stride=1.0, fps=30.0):
    """
    Convert temporal timestamp (in seconds) to feature index.
    
    Args:
        timestamp: time in seconds
        feat_stride: seconds per feature (1.0 for perception 1s_1s.npz)
        fps: frames per second (for reference, not always used)
    
    Returns:
        feature_index: integer index in feature array
    """
    # For perception features with 1s stride: 1 second = 1 feature
    idx = int(timestamp / feat_stride)
    return idx


def compute_step_embeddings(video_id, segments, features, config):
    """
    Compute step-level embeddings by averaging features within segment boundaries.
    
    Args:
        video_id: string identifier for the video
        segments: list of (start, end) tuples in seconds
        features: numpy array of shape [T, D] where T is temporal length, D is feature dimension
        config: dict with configuration parameters:
            - feat_stride: seconds per feature (default: 1.0)
            - fps: frames per second (default: 30.0)
            - pooling_method: 'mean' or 'max' (default: 'mean')
    
    Returns:
        embeddings: numpy array of shape [N, D] where N is number of segments
        segment_info: list of dicts with metadata for each segment
    """
    # Extract config parameters with defaults
    feat_stride = config.get('feat_stride', 1.0)
    fps = config.get('fps', 30.0)
    pooling_method = config.get('pooling_method', 'mean')
    
    # Validate inputs
    if features is None or features.shape[0] == 0:
        print(f"Warning: Empty features for video {video_id}")
        return np.array([]), []
    
    if len(segments) == 0:
        return np.array([]), []
    
    T, D = features.shape
    embeddings = []
    segment_info = []
    
    for seg_idx, (start_time, end_time) in enumerate(segments):
        # Convert timestamps to feature indices
        start_idx = time_to_feature_index(start_time, feat_stride, fps)
        end_idx = time_to_feature_index(end_time, feat_stride, fps)
        
        # Clamp indices to valid range
        start_idx = max(0, min(start_idx, T - 1))
        end_idx = max(start_idx + 1, min(end_idx, T))
        
        # Extract features for this segment
        segment_features = features[start_idx:end_idx]
        
        # Handle edge case: empty segment
        if segment_features.shape[0] == 0:
            # Use single feature at start_idx as fallback
            segment_features = features[start_idx:start_idx+1]
            if segment_features.shape[0] == 0:
                # Last resort: use first feature
                segment_features = features[0:1]
        
        # Pool features (average by default)
        if pooling_method == 'mean':
            step_embedding = np.mean(segment_features, axis=0)
        elif pooling_method == 'max':
            step_embedding = np.max(segment_features, axis=0)
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")
        
        # Ensure embedding has correct shape
        if step_embedding.ndim == 0:
            step_embedding = np.array([step_embedding])
        elif step_embedding.ndim > 1:
            step_embedding = step_embedding.flatten()
        
        embeddings.append(step_embedding)
        
        # Store metadata
        segment_info.append({
            'segment_idx': seg_idx,
            'start_time': start_time,
            'end_time': end_time,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'num_features': segment_features.shape[0],
            'duration': end_time - start_time
        })
    
    # Convert to numpy array
    if len(embeddings) > 0:
        embeddings = np.array(embeddings)
        # Ensure shape is [N, D]
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
    else:
        embeddings = np.array([])
    
    return embeddings, segment_info


def batch_compute_step_embeddings(video_segments_dict, features_dict, config):
    """
    Compute step embeddings for multiple videos in batch.
    
    Args:
        video_segments_dict: dict mapping video_id -> list of (start, end) tuples
        features_dict: dict mapping video_id -> features array [T, D]
        config: configuration dict
    
    Returns:
        embeddings_dict: dict mapping video_id -> embeddings array [N, D]
        metadata_dict: dict mapping video_id -> list of segment info dicts
    """
    embeddings_dict = {}
    metadata_dict = {}
    
    for video_id in video_segments_dict.keys():
        if video_id not in features_dict:
            print(f"Warning: Features not found for video {video_id}")
            continue
        
        segments = video_segments_dict[video_id]
        features = features_dict[video_id]
        
        embeddings, segment_info = compute_step_embeddings(
            video_id, segments, features, config
        )
        
        embeddings_dict[video_id] = embeddings
        metadata_dict[video_id] = segment_info
    
    return embeddings_dict, metadata_dict
