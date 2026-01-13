"""
Validation and sanity check script for step localization results.
Computes statistics and optionally compares with ground truth annotations.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_segments(segments_path):
    """Load segments from JSON file."""
    with open(segments_path, 'r') as f:
        return json.load(f)


def load_embeddings(embeddings_path):
    """Load embeddings from NPY file."""
    return np.load(embeddings_path, allow_pickle=True).item()


def load_ground_truth():
    """Load ground truth step annotations."""
    gt_path = 'annotations/annotation_json/step_annotations.json'
    try:
        with open(gt_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Ground truth file not found: {gt_path}")
        return None


def compute_segment_statistics(segments_dict):
    """Compute statistics on segments."""
    stats = {
        'num_segments': [],
        'segment_durations': [],
        'total_videos': len(segments_dict)
    }
    
    for video_id, data in segments_dict.items():
        segments = data['segments']
        num_seg = len(segments)
        stats['num_segments'].append(num_seg)
        
        for start, end in segments:
            duration = end - start
            stats['segment_durations'].append(duration)
    
    stats['num_segments'] = np.array(stats['num_segments'])
    stats['segment_durations'] = np.array(stats['segment_durations'])
    
    return stats


def compare_with_ground_truth(pred_segments, gt_annotations):
    """Compare predicted segments with ground truth."""
    if gt_annotations is None:
        return None
    
    comparisons = {}
    
    for video_id in pred_segments.keys():
        if video_id not in gt_annotations:
            continue
        
        pred_segs = pred_segments[video_id]['segments']
        gt_steps = gt_annotations[video_id]['steps']
        
        gt_segs = []
        for step in gt_steps:
            if step['start_time'] >= 0 and step['end_time'] >= 0:
                gt_segs.append((step['start_time'], step['end_time']))
        
        if len(gt_segs) == 0:
            continue
        
        # Compute metrics
        num_pred = len(pred_segs)
        num_gt = len(gt_segs)
        
        # Compute temporal IoU for each predicted segment with best matching GT
        ious = []
        for pred_start, pred_end in pred_segs:
            best_iou = 0.0
            for gt_start, gt_end in gt_segs:
                # Compute IoU
                intersection_start = max(pred_start, gt_start)
                intersection_end = min(pred_end, gt_end)
                intersection = max(0, intersection_end - intersection_start)
                
                union_start = min(pred_start, gt_start)
                union_end = max(pred_end, gt_end)
                union = union_end - union_start
                
                if union > 0:
                    iou = intersection / union
                    best_iou = max(best_iou, iou)
            
            ious.append(best_iou)
        
        comparisons[video_id] = {
            'num_pred': num_pred,
            'num_gt': num_gt,
            'num_diff': abs(num_pred - num_gt),
            'mean_iou': np.mean(ious) if len(ious) > 0 else 0.0,
            'max_iou': np.max(ious) if len(ious) > 0 else 0.0,
            'min_iou': np.min(ious) if len(ious) > 0 else 0.0
        }
    
    return comparisons


def print_statistics(stats, comparisons=None):
    """Print statistics summary."""
    print("=" * 60)
    print("Step Localization Statistics")
    print("=" * 60)
    
    print(f"\nTotal videos processed: {stats['total_videos']}")
    
    print(f"\nNumber of segments per video:")
    print(f"  Mean: {np.mean(stats['num_segments']):.2f}")
    print(f"  Median: {np.median(stats['num_segments']):.2f}")
    print(f"  Std: {np.std(stats['num_segments']):.2f}")
    print(f"  Min: {np.min(stats['num_segments'])}")
    print(f"  Max: {np.max(stats['num_segments'])}")
    
    print(f"\nSegment durations (seconds):")
    print(f"  Mean: {np.mean(stats['segment_durations']):.2f}")
    print(f"  Median: {np.median(stats['segment_durations']):.2f}")
    print(f"  Std: {np.std(stats['segment_durations']):.2f}")
    print(f"  Min: {np.min(stats['segment_durations']):.2f}")
    print(f"  Max: {np.max(stats['segment_durations']):.2f}")
    
    if comparisons:
        print(f"\nComparison with Ground Truth:")
        mean_ious = [c['mean_iou'] for c in comparisons.values()]
        num_diffs = [c['num_diff'] for c in comparisons.values()]
        
        print(f"  Videos with GT: {len(comparisons)}")
        print(f"  Mean IoU: {np.mean(mean_ious):.3f}")
        print(f"  Median IoU: {np.median(mean_ious):.3f}")
        print(f"  Mean segment count difference: {np.mean(num_diffs):.2f}")
    
    print("=" * 60)


def plot_statistics(stats, output_dir=None):
    """Plot statistics visualizations."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot 1: Distribution of number of segments per video
        axes[0].hist(stats['num_segments'], bins=20, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Number of Segments per Video')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Number of Segments')
        axes[0].axvline(np.mean(stats['num_segments']), color='red', linestyle='--', label=f'Mean: {np.mean(stats["num_segments"]):.1f}')
        axes[0].legend()
        
        # Plot 2: Distribution of segment durations
        axes[1].hist(stats['segment_durations'], bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Segment Duration (seconds)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Segment Durations')
        axes[1].axvline(np.mean(stats['segment_durations']), color='red', linestyle='--', label=f'Mean: {np.mean(stats["segment_durations"]):.2f}s')
        axes[1].legend()
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'step_localization_stats.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nSaved plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")


def main(args):
    """Main validation function."""
    print(f"Loading segments from {args.segments}")
    segments_dict = load_segments(args.segments)
    
    embeddings_dict = None
    if args.embeddings:
        print(f"Loading embeddings from {args.embeddings}")
        embeddings_dict = load_embeddings(args.embeddings)
    
    print("\nComputing statistics...")
    stats = compute_segment_statistics(segments_dict)
    
    comparisons = None
    if args.compare_gt:
        print("\nLoading ground truth annotations...")
        gt_annotations = load_ground_truth()
        if gt_annotations:
            print("Comparing with ground truth...")
            comparisons = compare_with_ground_truth(segments_dict, gt_annotations)
    
    print_statistics(stats, comparisons)
    
    # Generate plots
    if args.plot:
        print("\nGenerating plots...")
        output_dir = os.path.dirname(args.segments) if args.segments else None
        plot_statistics(stats, output_dir)
    
    # Validate embeddings if provided
    if embeddings_dict:
        print("\nValidating embeddings...")
        valid_count = 0
        for video_id in segments_dict.keys():
            if video_id in embeddings_dict:
                pred_segs = segments_dict[video_id]['segments']
                embeddings = embeddings_dict[video_id]
                
                if len(pred_segs) == embeddings.shape[0]:
                    valid_count += 1
                else:
                    print(f"  Warning: {video_id} - {len(pred_segs)} segments but {embeddings.shape[0]} embeddings")
        
        print(f"  Valid embeddings: {valid_count}/{len(segments_dict)}")
    
    print("\nValidation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Validate step localization results'
    )
    parser.add_argument(
        '--segments',
        type=str,
        required=True,
        help='Path to segments JSON file'
    )
    parser.add_argument(
        '--embeddings',
        type=str,
        default=None,
        help='Path to embeddings NPY file (optional)'
    )
    parser.add_argument(
        '--compare_gt',
        action='store_true',
        help='Compare with ground truth annotations'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate visualization plots'
    )
    
    args = parser.parse_args()
    main(args)
