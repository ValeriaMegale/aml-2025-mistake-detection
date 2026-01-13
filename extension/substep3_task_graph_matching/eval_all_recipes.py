import argparse
import json
import os
import numpy as np
from pathlib import Path
from eval_task_graph_matching import (
    evaluate_model, main as eval_main,
    load_step_annotations, load_recording_to_activity_mapping, 
    load_activity_to_taskgraph, load_task_graph, TaskGraphMatchingDataset,
    collate_fn, TextEncoder, TaskGraphMatcher
)
from torch.utils.data import DataLoader
import torch


def get_available_checkpoints(ckpt_dir):
    """Trova tutti i checkpoints disponibili."""
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        return []
    
    checkpoints = []
    for ckpt_file in ckpt_path.glob("task_graph_matcher_recipe_*.pth"):
        # Estrai recipe ID dal nome file
        recipe_id = ckpt_file.stem.replace("task_graph_matcher_recipe_", "")
        checkpoints.append((recipe_id, str(ckpt_file)))
    
    return sorted(checkpoints, key=lambda x: int(x[0]) if x[0].isdigit() else 0)


def evaluate_all_recipes(args):
    """Valuta tutti i checkpoints disponibili."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Checkpoint directory: {args.ckpt_dir}")
    
    # Trova checkpoints disponibili
    available_ckpts = get_available_checkpoints(args.ckpt_dir)
    
    if not available_ckpts:
        print(f"No checkpoints found in {args.ckpt_dir}")
        return
    
    print(f"\nFound {len(available_ckpts)} checkpoints:")
    for recipe_id, ckpt_path in available_ckpts:
        print(f"  Recipe {recipe_id}: {Path(ckpt_path).name}")
    
    print("\nLoading shared data...")
    step_embeddings_raw = np.load(args.npy, allow_pickle=True).item()
    annotation_map = load_step_annotations(args.annotations)
    recording_to_activity = load_recording_to_activity_mapping(args.recording_csv)
    
    step_embeddings = {}
    for video_id, embeddings_array in step_embeddings_raw.items():
        if isinstance(embeddings_array, np.ndarray):
            step_embeddings[video_id] = [
                {'embedding': embeddings_array[i]} 
                for i in range(embeddings_array.shape[0])
            ]
        else:
            step_embeddings[video_id] = embeddings_array
    activity_to_taskgraph = load_activity_to_taskgraph(args.activity_mapping)
    
    print("Initializing text encoder...")
    text_encoder = TextEncoder(args.text_model)
    
    print("Precomputing task graph embeddings...")
    precomputed_text_emb = {}
    for act_id, tg_info in activity_to_taskgraph.items():
        task_graph = load_task_graph(args.task_graph_dir, tg_info['task_graph_file'])
        _, text_emb = text_encoder.encode_task_graph(task_graph)
        precomputed_text_emb[act_id] = text_emb
    
    all_videos = list(step_embeddings.keys())
    
    all_recipe_ids = sorted([r for r in activity_to_taskgraph.keys()], 
                          key=lambda x: int(x) if x.isdigit() else 0)
    
    print(f"\nTotal recipes in dataset: {len(all_recipe_ids)}")
    print(f"Checkpoints available: {len(available_ckpts)}")
    
    all_results = []
    all_metrics = []
    
    for recipe_id, ckpt_path in available_ckpts:
        print(f"\n{'='*70}")
        print(f"Evaluating Recipe {recipe_id}")
        print(f"{'='*70}")
        
        test_ids = [v for v in all_videos if v.startswith(f"{recipe_id}_")]
        
        if not test_ids:
            print(f"  No videos found for recipe {recipe_id}, skipping...")
            continue
        
        test_ds = TaskGraphMatchingDataset(
            step_embeddings, test_ids, annotation_map,
            recording_to_activity, activity_to_taskgraph,
            args.task_graph_dir, text_encoder, precomputed_text_emb
        )
        
        if len(test_ds) == 0:
            print(f"  Empty test dataset, skipping...")
            continue
        
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, 
            collate_fn=collate_fn
        )
        
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        visual_dim = args.visual_dim
        if 'args' in checkpoint and 'visual_dim' in checkpoint['args']:
            visual_dim = checkpoint['args']['visual_dim']
        else:
            if 'model_state_dict' in checkpoint:
                weight_shape = checkpoint['model_state_dict']['visual_proj.0.weight'].shape
                visual_dim = weight_shape[1]  # [hidden_dim, visual_dim]
            else:
                weight_shape = checkpoint['visual_proj.0.weight'].shape
                visual_dim = weight_shape[1]
        
        print(f"  Using visual_dim={visual_dim} (inferred from checkpoint)")
        
        model = TaskGraphMatcher(
            visual_dim=visual_dim,
            text_dim=text_encoder.dim,
            hidden_dim=args.hidden_dim
        ).to(device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch_info = checkpoint.get('epoch', 'unknown')
            print(f"  Loaded checkpoint from epoch {epoch_info}")
        else:
            model.load_state_dict(checkpoint)
            epoch_info = 'unknown'
        
        sample_batch = next(iter(test_loader))
        actual_vis_dim = sample_batch['visual_emb'].shape[-1]
        
        if actual_vis_dim != visual_dim:
            print(f"  WARNING: Embedding dimension mismatch!")
            print(f"    Checkpoint expects: {visual_dim}D")
            print(f"    Actual embeddings: {actual_vis_dim}D")
            print(f"    Applying zero-padding to adapt...")
            
            def adapt_batch(batch):
                vis_emb = batch['visual_emb']
                if vis_emb.shape[-1] < visual_dim:
                    # Zero-padding
                    padding = torch.zeros(vis_emb.shape[0], vis_emb.shape[1], 
                                        visual_dim - vis_emb.shape[-1], 
                                        dtype=vis_emb.dtype, device=vis_emb.device)
                    batch['visual_emb'] = torch.cat([vis_emb, padding], dim=-1)
                elif vis_emb.shape[-1] > visual_dim:
                    # Truncation
                    batch['visual_emb'] = vis_emb[..., :visual_dim]
                return batch
            
            class AdaptedLoader:
                def __init__(self, original_loader):
                    self.original_loader = original_loader
                def __iter__(self):
                    for batch in self.original_loader:
                        yield adapt_batch(batch)
                def __len__(self):
                    return len(self.original_loader)
            
            test_loader = AdaptedLoader(test_loader)
        
        metrics, predictions = evaluate_model(model, test_loader, device, threshold=args.threshold)
        
        result = {
            'recipe_id': recipe_id,
            'checkpoint': ckpt_path,
            'epoch': epoch_info,
            'num_test_videos': len(test_ids),
            'metrics': metrics,
            'predictions': predictions
        }
        all_results.append(result)
        all_metrics.append(metrics)
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
    
    # Compute aggregate metrics
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")
    
    if all_metrics:
        accuracies = [m['accuracy'] for m in all_metrics]
        precisions = [m['precision'] for m in all_metrics]
        recalls = [m['recall'] for m in all_metrics]
        f1s = [m['f1'] for m in all_metrics]
        aucs = [m['auc'] for m in all_metrics if 'auc' in m and m['auc'] > 0]
        
        aggregate = {
            'num_recipes_evaluated': len(all_metrics),
            'total_recipes': len(all_recipe_ids),
            'missing_recipes': [r for r in all_recipe_ids if r not in [c[0] for c in available_ckpts]],
            'accuracy': {
                'mean': float(np.mean(accuracies)),
                'std': float(np.std(accuracies)),
                'min': float(np.min(accuracies)),
                'max': float(np.max(accuracies))
            },
            'precision': {
                'mean': float(np.mean(precisions)),
                'std': float(np.std(precisions)),
                'min': float(np.min(precisions)),
                'max': float(np.max(precisions))
            },
            'recall': {
                'mean': float(np.mean(recalls)),
                'std': float(np.std(recalls)),
                'min': float(np.min(recalls)),
                'max': float(np.max(recalls))
            },
            'f1': {
                'mean': float(np.mean(f1s)),
                'std': float(np.std(f1s)),
                'min': float(np.min(f1s)),
                'max': float(np.max(f1s))
            }
        }
        
        if aucs:
            aggregate['auc'] = {
                'mean': float(np.mean(aucs)),
                'std': float(np.std(aucs)),
                'min': float(np.min(aucs)),
                'max': float(np.max(aucs))
            }
        
        print(f"\nRecipes evaluated: {aggregate['num_recipes_evaluated']}/{aggregate['total_recipes']}")
        if aggregate['missing_recipes']:
            print(f"Missing recipes: {', '.join(aggregate['missing_recipes'])}")
        
        print(f"\nAccuracy:  {aggregate['accuracy']['mean']:.4f} ± {aggregate['accuracy']['std']:.4f} "
              f"(min: {aggregate['accuracy']['min']:.4f}, max: {aggregate['accuracy']['max']:.4f})")
        print(f"Precision: {aggregate['precision']['mean']:.4f} ± {aggregate['precision']['std']:.4f}")
        print(f"Recall:    {aggregate['recall']['mean']:.4f} ± {aggregate['recall']['std']:.4f}")
        print(f"F1:        {aggregate['f1']['mean']:.4f} ± {aggregate['f1']['std']:.4f}")
        if aucs:
            print(f"AUC:       {aggregate['auc']['mean']:.4f} ± {aggregate['auc']['std']:.4f}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / 'all_recipes_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'aggregate': aggregate if all_metrics else {},
            'per_recipe': all_results
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    if all_results:
        import csv
        csv_file = output_dir / 'results_summary.csv'
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Recipe', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Num_Videos'])
            for r in all_results:
                m = r['metrics']
                writer.writerow([
                    r['recipe_id'],
                    f"{m['accuracy']:.4f}",
                    f"{m['precision']:.4f}",
                    f"{m['recall']:.4f}",
                    f"{m['f1']:.4f}",
                    f"{m.get('auc', 0.0):.4f}",
                    r['num_test_videos']
                ])
        print(f"Summary CSV saved to {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate Task Graph Matching for all available checkpoints'
    )
    
    parser.add_argument('--ckpt_dir', type=str,
                        default='extension/substep3_task_graph_matching/checkpoints_graph_test',
                        help='Directory containing checkpoints')
    parser.add_argument('--npy', type=str,
                        default='../substep1_step_localization/data/step_embeddings_perception.npy',
                        help='Path to step embeddings .npy file')
    parser.add_argument('--annotations', type=str,
                        default='annotations/annotation_json/step_annotations.json',
                        help='Path to step annotations JSON')
    parser.add_argument('--recording_csv', type=str,
                        default='annotations/annotation_csv/recording_id_step_idx.csv',
                        help='Path to recording_id_step_idx.csv')
    parser.add_argument('--activity_mapping', type=str,
                        default='extension/substep3_task_graph_matching/activity_to_taskgraph.json',
                        help='Path to activity_to_taskgraph.json')
    parser.add_argument('--task_graph_dir', type=str,
                        default='annotations/task_graphs',
                        help='Directory containing task graph JSON files')
    parser.add_argument('--text_model', type=str,
                        default='clip',
                        help='Text encoder: "clip" or "all-MiniLM-L6-v2"')
    
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--visual_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--output_dir', type=str,
                        default='results/extension/substep3_task_graph_matching',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    evaluate_all_recipes(args)
