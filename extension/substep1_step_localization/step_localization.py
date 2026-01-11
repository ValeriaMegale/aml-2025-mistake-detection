"""
Step Localization using Hierarchical Clustering (HiERO-style)
Zero-shot approach for temporal action localization without training.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import warnings
warnings.filterwarnings('ignore')


def compute_temporal_distance_matrix(features, distance_metric='cosine'):
    """
    Compute pairwise distance matrix between temporal feature vectors.
    
    Args:
        features: numpy array of shape [T, D] where T is temporal length, D is feature dim
        distance_metric: 'cosine' or 'euclidean'
    
    Returns:
        distance_matrix: [T, T] symmetric matrix
    """
    if distance_metric == 'cosine':
        # Normalize features
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        features_norm = features / norms
        
        # Cosine distance = 1 - cosine similarity
        similarity = np.dot(features_norm, features_norm.T)
        distance_matrix = 1 - similarity
    else:  # euclidean
        distance_matrix = squareform(pdist(features, metric='euclidean'))
    
    return distance_matrix


def estimate_num_clusters(features, video_duration, min_segment_duration=2.0, max_segments=50):
    """
    Estimate number of clusters based on video duration and heuristics.
    
    Args:
        features: [T, D] feature array
        video_duration: duration in seconds
        min_segment_duration: minimum segment duration in seconds
        max_segments: maximum number of segments
    
    Returns:
        n_clusters: estimated number of clusters
    """
    # Heuristic: ~1 step every 10-15 seconds
    estimated_by_duration = int(video_duration / 12.0)
    
    # Also consider feature variance (more variance = more potential steps)
    feature_variance = np.var(features, axis=0).mean()
    variance_factor = min(2.0, 1.0 + feature_variance / 100.0)
    estimated_by_variance = int(estimated_by_duration * variance_factor)
    
    # Constrain by min/max
    n_clusters = max(2, min(max_segments, estimated_by_variance))
    
    # Ensure we don't exceed temporal resolution
    max_by_resolution = int(video_duration / min_segment_duration)
    n_clusters = min(n_clusters, max_by_resolution)
    
    return n_clusters


def hierarchical_clustering_segmentation(features, n_clusters, distance_metric='cosine', linkage_method='ward'):
    """
    Perform hierarchical clustering on temporal features to find step boundaries.
    
    Args:
        features: [T, D] feature array
        n_clusters: number of clusters to form
        distance_metric: 'cosine' or 'euclidean'
        linkage_method: 'ward', 'complete', 'average', 'single'
    
    Returns:
        cluster_labels: [T] array with cluster assignment for each timestep
    """
    T = features.shape[0]
    
    if T < n_clusters:
        # If we have fewer timesteps than clusters, assign each timestep to its own cluster
        return np.arange(T)
    
    # For ward linkage, we need euclidean distance
    if linkage_method == 'ward':
        distance_metric = 'euclidean'
    
    # Compute distance matrix
    if T > 1000:
        # For very long videos, use AgglomerativeClustering directly (more memory efficient)
        # Note: AgglomerativeClustering doesn't support cosine with ward, so force euclidean
        if linkage_method == 'ward' and distance_metric == 'cosine':
            distance_metric = 'euclidean'
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method,
            metric=distance_metric if linkage_method != 'ward' else 'euclidean'
        )
        cluster_labels = clustering.fit_predict(features)
    else:
        # For shorter videos, use linkage + fcluster (more control)
        if distance_metric == 'cosine':
            # Cosine distance needs special handling
            distance_matrix = compute_temporal_distance_matrix(features, 'cosine')
            # Convert to condensed form for linkage
            condensed_distances = squareform(distance_matrix, checks=False)
            # Ward doesn't work with precomputed distances, use average instead
            if linkage_method == 'ward':
                Z = linkage(condensed_distances, method='average')
            else:
                Z = linkage(condensed_distances, method=linkage_method)
        else:
            Z = linkage(features, method=linkage_method, metric=distance_metric)
        
        cluster_labels = fcluster(Z, n_clusters, criterion='maxclust') - 1  # 0-indexed
    
    return cluster_labels


def cluster_labels_to_segments(cluster_labels, feat_stride=1.0, fps=30.0):
    """
    Convert cluster labels to temporal segments (start, end) in seconds.
    
    Args:
        cluster_labels: [T] array with cluster assignments
        feat_stride: seconds per feature (1.0 for perception 1s_1s.npz)
        fps: frames per second (for conversion)
    
    Returns:
        segments: list of (start, end) tuples in seconds
    """
    segments = []
    # Ensure cluster_labels is a numpy array of integers
    cluster_labels = np.asarray(cluster_labels, dtype=np.int32).flatten()
    T = len(cluster_labels)
    
    # Find boundaries where cluster changes using np.diff (more efficient)
    if T == 0:
        return segments
    
    # Find where cluster changes (diff != 0)
    diff = np.diff(cluster_labels)
    change_indices = np.where(diff != 0)[0] + 1  # +1 because diff is one element shorter
    
    # Build boundaries: start at 0, add all change points, end at T
    boundaries = [0] + change_indices.tolist() + [T]
    
    # Convert boundaries to temporal segments
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        
        # Convert indices to seconds
        start_time = start_idx * feat_stride
        end_time = end_idx * feat_stride
        
        segments.append((start_time, end_time))
    
    return segments


def temporal_nms(segments, scores=None, iou_threshold=0.3):
    """
    Apply Non-Maximum Suppression to remove overlapping segments.
    
    Args:
        segments: list of (start, end) tuples
        scores: optional list of scores for each segment
        iou_threshold: IoU threshold for removing overlaps
    
    Returns:
        filtered_segments: list of (start, end) tuples
        filtered_scores: list of scores (if provided)
    """
    if len(segments) == 0:
        return segments, scores if scores else []
    
    # Convert to numpy for easier manipulation
    segments_array = np.array(segments)
    starts = segments_array[:, 0]
    ends = segments_array[:, 1]
    durations = ends - starts
    
    # Compute IoU matrix
    n = len(segments)
    iou_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                iou_matrix[i, j] = 1.0
            else:
                # Compute intersection
                intersection_start = max(starts[i], starts[j])
                intersection_end = min(ends[i], ends[j])
                intersection = max(0, intersection_end - intersection_start)
                
                # Compute union
                union = durations[i] + durations[j] - intersection
                
                if union > 0:
                    iou_matrix[i, j] = intersection / union
                else:
                    iou_matrix[i, j] = 0.0
    
    # Greedy NMS: keep segments with highest score (or longest if no scores)
    # Convert scores to list if it's a numpy array
    if scores is not None:
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
    else:
        scores = durations  # Use duration as proxy for score
    
    keep = []
    remaining = set(range(n))
    
    while remaining:
        # Find segment with highest score
        best_idx = max(remaining, key=lambda i: scores[i])
        keep.append(best_idx)
        remaining.remove(best_idx)
        
        # Remove overlapping segments
        overlapping = [i for i in remaining if iou_matrix[best_idx, i] > iou_threshold]
        remaining -= set(overlapping)
    
    keep.sort()  # Sort by index to maintain temporal order
    filtered_segments = [segments[i] for i in keep]
    # Handle scores: check if it's not None and has elements
    if scores is not None and len(scores) > 0:
        # Convert to list if it's a numpy array
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        filtered_scores = [scores[i] for i in keep]
    else:
        filtered_scores = None
    
    return filtered_segments, filtered_scores


def postprocess_segments(segments, scores=None, min_duration=2.0, max_segments=50, nms_threshold=0.3):
    """
    Postprocess segments: filter by duration, apply NMS, limit count.
    
    Args:
        segments: list of (start, end) tuples
        scores: optional list of scores
        min_duration: minimum segment duration in seconds
        max_segments: maximum number of segments
        nms_threshold: IoU threshold for NMS
    
    Returns:
        filtered_segments: list of (start, end) tuples
        filtered_scores: list of scores (if provided)
    """
    if len(segments) == 0:
        return segments, scores if scores else []
    
    # Filter by minimum duration
    filtered = []
    filtered_scores = []
    for i, (start, end) in enumerate(segments):
        duration = end - start
        if duration >= min_duration:
            filtered.append((start, end))
            if scores:
                filtered_scores.append(scores[i])
    
    # Apply temporal NMS
    filtered, filtered_scores = temporal_nms(filtered, filtered_scores if scores else None, nms_threshold)
    
    # Limit to max_segments (keep longest/highest scoring)
    if len(filtered) > max_segments:
        if filtered_scores:
            # Sort by score (descending)
            sorted_indices = sorted(range(len(filtered)), key=lambda i: filtered_scores[i], reverse=True)
        else:
            # Sort by duration (descending)
            durations = [end - start for start, end in filtered]
            sorted_indices = sorted(range(len(filtered)), key=lambda i: durations[i], reverse=True)
        
        keep_indices = sorted(sorted_indices[:max_segments])  # Sort by time after selecting top
        filtered = [filtered[i] for i in keep_indices]
        if filtered_scores:
            filtered_scores = [filtered_scores[i] for i in keep_indices]
    
    # Ensure segments are sorted by start time
    sorted_indices = sorted(range(len(filtered)), key=lambda i: filtered[i][0])
    filtered = [filtered[i] for i in sorted_indices]
    if filtered_scores:
        filtered_scores = [filtered_scores[i] for i in sorted_indices]
    
    return filtered, filtered_scores


def localize_steps_clustering(video_id, features, config):
    """
    Main function to localize steps in a video using hierarchical clustering.
    
    Args:
        video_id: string identifier for the video
        features: numpy array of shape [T, D] where T is temporal length, D is feature dimension
        config: dict with configuration parameters:
            - feat_stride: seconds per feature (default: 1.0)
            - fps: frames per second (default: 30.0)
            - clustering_method: 'hierarchical' (default)
            - clustering_distance: 'cosine' or 'euclidean' (default: 'cosine')
            - linkage_method: 'ward', 'complete', 'average', 'single' (default: 'ward')
            - min_segment_duration: minimum segment duration in seconds (default: 2.0)
            - max_segments_per_video: maximum number of segments (default: 50)
            - nms_threshold: IoU threshold for NMS (default: 0.3)
            - video_duration: video duration in seconds (optional, estimated from features if not provided)
    
    Returns:
        segments: list of (start, end) tuples in seconds
        scores: list of confidence scores (optional, can be None)
    """
    # Extract config parameters with defaults
    feat_stride = config.get('feat_stride', 1.0)
    fps = config.get('fps', 30.0)
    clustering_method = config.get('clustering_method', 'hierarchical')
    clustering_distance = config.get('clustering_distance', 'cosine')
    linkage_method = config.get('linkage_method', 'ward')
    min_segment_duration = config.get('min_segment_duration', 2.0)
    max_segments = config.get('max_segments_per_video', 50)
    nms_threshold = config.get('nms_threshold', 0.3)
    
    # Estimate video duration from features if not provided
    if 'video_duration' in config:
        video_duration = config['video_duration']
    else:
        video_duration = features.shape[0] * feat_stride
    
    # Validate inputs
    if features is None or features.shape[0] == 0:
        print(f"Warning: Empty features for video {video_id}")
        return [], []
    
    T, D = features.shape
    
    # Estimate number of clusters
    n_clusters = estimate_num_clusters(features, video_duration, min_segment_duration, max_segments)
    
    # Perform clustering
    try:
        if clustering_method == 'hierarchical':
            # Fix: ward linkage doesn't work well with cosine, use average instead
            if linkage_method == 'ward' and clustering_distance == 'cosine':
                print(f"  Note: Using 'average' linkage instead of 'ward' with cosine distance")
                actual_linkage = 'average'
            else:
                actual_linkage = linkage_method
            
            cluster_labels = hierarchical_clustering_segmentation(
                features, n_clusters, clustering_distance, actual_linkage
            )
        else:
            raise ValueError(f"Unknown clustering method: {clustering_method}")
        
        # Convert cluster labels to segments
        segments = cluster_labels_to_segments(cluster_labels, feat_stride, fps)
        
        # Postprocess segments
        segments, scores = postprocess_segments(
            segments, None, min_segment_duration, max_segments, nms_threshold
        )
        
        # Generate simple confidence scores based on segment duration (longer = more confident)
        if scores is None:
            scores = [end - start for start, end in segments]
            # Normalize scores to [0, 1]
            if len(scores) > 0 and max(scores) > 0:
                max_score = max(scores)
                scores = [s / max_score for s in scores]
        
        return segments, scores
        
    except Exception as e:
        import traceback
        print(f"Error in clustering for video {video_id}: {e}")
        print(f"Traceback:")
        traceback.print_exc()
        # Return empty segments on error
        return [], []
