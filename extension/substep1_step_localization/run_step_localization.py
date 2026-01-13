"""
Main script to run step localization pipeline.
Processes all videos, performs clustering-based step localization, and computes step embeddings.
"""

import argparse
import json
import os
import yaml
import numpy as np
from tqdm import tqdm

from step_localization import localize_steps_clustering
from step_embeddings_hiero import compute_step_embeddings, batch_compute_step_embeddings


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_features_from_npz(path):
    """Load features from .npz file."""
    try:
        with np.load(path, allow_pickle=True) as data:
            # Try different possible keys
            for key in ['features', 'feats', 'embedding', 'arr_0', 'data']:
                if key in data:
                    return data[key]
            # If no standard key, use first available
            if len(data.files) > 0:
                return data[data.files[0]]
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None
    return None


def build_feature_file_map(feat_folder):
    """Build mapping from video_id to feature file path."""
    print(f"Indexing feature folder: {feat_folder}")
    if not os.path.exists(feat_folder):
        print(f"ERROR: Feature folder not found: {feat_folder}")
        return {}
    
    file_map = {}
    files = os.listdir(feat_folder)
    
    # Find common suffix
    suffix = ""
    for f in files:
        if f.endswith('.npz'):
            if '_360p.mp4_1s_1s.npz' in f:
                suffix = '_360p.mp4_1s_1s.npz'
            elif '_360p_1s_1s.mp4.npz' in f:
                suffix = '_360p_1s_1s.mp4.npz'
            elif '.npz' in f:
                suffix = '.npz'
            break
    
    print(f"Detected suffix: '{suffix}'")
    
    for filename in files:
        if filename.endswith(suffix):
            video_id = filename.replace(suffix, "")
            file_map[video_id] = os.path.join(feat_folder, filename)
    
    print(f"Mapped {len(file_map)} feature files")
    return file_map


def get_video_list(split='all'):
    """Get list of video IDs based on split."""
    if split == 'all':
        split_files = [
            'annotations/data_splits/recordings_data_split_combined.json',
            'annotations/data_splits/person_data_split_combined.json',
            'annotations/data_splits/environment_data_split_combined.json'
        ]
        all_videos = set()
        for split_file in split_files:
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    data = json.load(f)
                    for phase in ['train', 'val', 'test']:
                        if phase in data:
                            all_videos.update(data[phase])
        return list(all_videos)
    else:
        split_file = f'annotations/data_splits/recordings_data_split_combined.json'
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                data = json.load(f)
                if split in data:
                    return data[split]
        return []


def process_video(video_id, feat_path, config):
    """Process a single video: localize steps and compute embeddings."""
    features = load_features_from_npz(feat_path)
    if features is None:
        return None, None, None
    
    # Estimate video duration
    video_duration = features.shape[0] * config.get('feat_stride', 1.0)
    config['video_duration'] = video_duration
    
    # Localize steps
    segments, scores = localize_steps_clustering(video_id, features, config)
    
    if len(segments) == 0:
        print(f"Warning: No segments found for video {video_id}")
        return segments, scores, None
    
    # Compute step embeddings
    embeddings, segment_info = compute_step_embeddings(
        video_id, segments, features, config
    )
    
    return segments, scores, embeddings


def main(args):
    """Main pipeline execution."""
    if args.config:
        config = load_config(args.config)
    else:
        # Use default config
        config_path = os.path.join(
            os.path.dirname(__file__),
            'config_step_localization.yaml'
        )
        config = load_config(config_path)
    
    # Override config with command line arguments
    if args.feat_folder:
        config['feat_folder'] = args.feat_folder
    if args.split:
        config['split'] = args.split
    if args.output_segments:
        config['output_segments'] = args.output_segments
    if args.output_embeddings:
        config['output_embeddings'] = args.output_embeddings
    
    print("=" * 60)
    print("Step Localization Pipeline (HiERO-style)")
    print("=" * 60)
    print(f"Feature folder: {config['feat_folder']}")
    print(f"Split: {config['split']}")
    print(f"Clustering method: {config.get('clustering_method', 'hierarchical')}")
    print("=" * 60)
    
    # Build feature file map
    feat_folder = config['feat_folder']
    if not os.path.isabs(feat_folder):
        # Make relative to current working directory (project root)
        # __file__ is extension_localization_hiero/run_step_localization.py
        # Go up one level to get project root
        project_root = os.path.dirname(os.path.dirname(__file__))
        feat_folder = os.path.join(project_root, feat_folder)
    
    file_map = build_feature_file_map(feat_folder)
    
    if len(file_map) == 0:
        print("ERROR: No feature files found!")
        return
    
    video_list = get_video_list(config['split'])
    
    # Filter to videos that have features
    video_list = [vid for vid in video_list if vid in file_map]
    
    print(f"Processing {len(video_list)} videos...")
    
    # Process videos
    all_segments = {}
    all_embeddings = {}
    
    for video_id in tqdm(video_list, desc="Processing videos"):
        feat_path = file_map[video_id]
        
        segments, scores, embeddings = process_video(video_id, feat_path, config)
        
        if segments is not None and len(segments) > 0:
            all_segments[video_id] = {
                'segments': segments,
                'scores': scores if scores else [],
                'num_segments': len(segments),
                'meta': {
                    'method': config.get('clustering_method', 'hierarchical'),
                    'min_duration': config.get('min_segment_duration', 2.0),
                    'max_segments': config.get('max_segments_per_video', 50),
                    'clustering_distance': config.get('clustering_distance', 'cosine'),
                    'linkage_method': config.get('linkage_method', 'ward')
                }
            }
            
            if embeddings is not None and embeddings.shape[0] > 0:
                all_embeddings[video_id] = embeddings
    
    output_segments = config['output_segments']
    output_embeddings = config['output_embeddings']
    
        if not os.path.isabs(output_segments):
        if output_segments.startswith('extension_localization_hiero/'):
            output_segments = output_segments.replace('extension_localization_hiero/', '', 1)
        output_segments = os.path.join(os.path.dirname(__file__), output_segments)
    if not os.path.isabs(output_embeddings):
        if output_embeddings.startswith('extension_localization_hiero/'):
            output_embeddings = output_embeddings.replace('extension_localization_hiero/', '', 1)
        output_embeddings = os.path.join(os.path.dirname(__file__), output_embeddings)
    
    os.makedirs(os.path.dirname(output_segments), exist_ok=True)
    os.makedirs(os.path.dirname(output_embeddings), exist_ok=True)
    
    print(f"\nSaving segments to {output_segments}")
    with open(output_segments, 'w') as f:
        json.dump(all_segments, f, indent=2)
    
    print(f"Saving embeddings to {output_embeddings}")
    np.save(output_embeddings, all_embeddings)
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Processed videos: {len(all_segments)}")
    if len(all_segments) > 0:
        num_segments = [v['num_segments'] for v in all_segments.values()]
        print(f"Total segments: {sum(num_segments)}")
        print(f"Average segments per video: {np.mean(num_segments):.2f}")
        print(f"Min segments: {min(num_segments)}")
        print(f"Max segments: {max(num_segments)}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run step localization pipeline using hierarchical clustering'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML file (default: config_step_localization.yaml)'
    )
    parser.add_argument(
        '--feat_folder',
        type=str,
        default=None,
        help='Override feature folder from config'
    )
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        choices=['all', 'train', 'val', 'test'],
        help='Data split to process (default: all)'
    )
    parser.add_argument(
        '--output_segments',
        type=str,
        default=None,
        help='Override output segments path from config'
    )
    parser.add_argument(
        '--output_embeddings',
        type=str,
        default=None,
        help='Override output embeddings path from config'
    )
    
    args = parser.parse_args()
    main(args)
