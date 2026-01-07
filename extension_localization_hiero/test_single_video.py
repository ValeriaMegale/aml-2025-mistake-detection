"""
Test script for single video step localization.
Useful for debugging and validation.
"""

import argparse
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from step_localization import localize_steps_clustering
from step_embeddings_hiero import compute_step_embeddings


def load_features_from_npz(path):
    """Load features from .npz file."""
    try:
        with np.load(path, allow_pickle=True) as data:
            for key in ['features', 'feats', 'embedding', 'arr_0', 'data']:
                if key in data:
                    return data[key]
            if len(data.files) > 0:
                return data[data.files[0]]
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None
    return None


def main():
    parser = argparse.ArgumentParser(description='Test step localization on single video')
    parser.add_argument('--video_id', type=str, required=True, help='Video ID (e.g., 10_16)')
    parser.add_argument('--feat_folder', type=str, default='data/video/perception', help='Feature folder')
    parser.add_argument('--feat_stride', type=float, default=1.0, help='Seconds per feature')
    parser.add_argument('--fps', type=float, default=30.0, help='Frames per second')
    
    args = parser.parse_args()
    
    # Build feature path
    feat_file = f"{args.video_id}_360p.mp4_1s_1s.npz"
    feat_path = os.path.join(args.feat_folder, feat_file)
    
    if not os.path.exists(feat_path):
        print(f"ERROR: Feature file not found: {feat_path}")
        return
    
    print(f"Loading features from {feat_path}")
    features = load_features_from_npz(feat_path)
    
    if features is None:
        print("ERROR: Could not load features")
        return
    
    print(f"Features shape: {features.shape}")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Temporal length: {features.shape[0]} seconds")
    
    # Configuration
    config = {
        'feat_stride': args.feat_stride,
        'fps': args.fps,
        'clustering_method': 'hierarchical',
        'clustering_distance': 'cosine',
        'linkage_method': 'ward',
        'min_segment_duration': 2.0,
        'max_segments_per_video': 50,
        'nms_threshold': 0.3
    }
    
    print("\nRunning step localization...")
    try:
        segments, scores = localize_steps_clustering(args.video_id, features, config)
        
        print(f"\nFound {len(segments)} segments:")
        for i, (start, end) in enumerate(segments):
            duration = end - start
            score = scores[i] if scores and i < len(scores) else 0.0
            print(f"  Segment {i+1}: [{start:.2f}s, {end:.2f}s] (duration: {duration:.2f}s, score: {score:.3f})")
        
        if len(segments) > 0:
            print("\nComputing step embeddings...")
            embeddings, segment_info = compute_step_embeddings(
                args.video_id, segments, features, config
            )
            
            print(f"Embeddings shape: {embeddings.shape}")
            print(f"Number of segments: {len(segments)}")
            print(f"Embedding dimension: {embeddings.shape[1]}")
            
            print("\nSegment info:")
            for info in segment_info[:5]:  # Show first 5
                print(f"  Segment {info['segment_idx']}: "
                      f"time=[{info['start_time']:.2f}s, {info['end_time']:.2f}s], "
                      f"indices=[{info['start_idx']}, {info['end_idx']}], "
                      f"features={info['num_features']}")
            if len(segment_info) > 5:
                print(f"  ... and {len(segment_info) - 5} more")
        
    except Exception as e:
        print(f"ERROR during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
