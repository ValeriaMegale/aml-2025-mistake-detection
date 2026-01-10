"""
Evaluate GNN Classification for Task Verification (Substep 4).

Valuta tutti i checkpoints leave-one-out e genera metriche aggregate.
"""

import argparse
import json
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add current directory to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_classification_substep4.dataset import TaskGraphDataset
from graph_classification_substep4.model import DAGNNClassifier


def evaluate_model(model, dataloader, criterion, device):
    """Evaluation con metriche complete."""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_video_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            out = model(batch.x, batch.edge_index, batch.batch)
            
            # Reshape output: [batch, 1] -> [batch] for BCEWithLogitsLoss
            out = out.squeeze(-1) if out.dim() > 1 else out
            
            loss = criterion(out, batch.y.float())
            
            # Get probabilities (out is already [batch] shape after squeeze)
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).long()
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch.y.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            
            # Collect video IDs if available
            if hasattr(batch, 'video_id'):
                all_video_ids.extend(batch.video_id)
            
            total_loss += loss.item()
            n_batches += 1
    
    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # AUC
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0.0
    
    return {
        'loss': total_loss / max(n_batches, 1),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }, {
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'probabilities': all_probs.tolist(),
        'video_ids': all_video_ids
    }


def get_available_checkpoints(ckpt_dir):
    """Trova tutti i checkpoints disponibili."""
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        return []
    
    checkpoints = []
    for ckpt_file in ckpt_path.glob("gnn_classifier_recipe_*.pth"):
        recipe_id = ckpt_file.stem.replace("gnn_classifier_recipe_", "")
        checkpoints.append((recipe_id, str(ckpt_file)))
    
    return sorted(checkpoints, key=lambda x: int(x[0]) if x[0].isdigit() else 0)


def evaluate_all_checkpoints(args):
    """Valuta tutti i checkpoints disponibili."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find checkpoints
    available_ckpts = get_available_checkpoints(args.ckpt_dir)
    
    if not available_ckpts:
        print(f"No checkpoints found in {args.ckpt_dir}")
        return
    
    print(f"\nFound {len(available_ckpts)} checkpoints")
    
    # Load metadata
    metadata_file = Path(args.data_dir) / 'metadata.json'
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    all_recipes = sorted(metadata.get('recipes', []), 
                        key=lambda x: int(x) if x.isdigit() else 0)
    
    print(f"Total recipes: {len(all_recipes)}")
    
    # Evaluation results
    all_results = []
    all_metrics = []
    
    criterion = nn.BCEWithLogitsLoss()
    
    for recipe_id, ckpt_path in available_ckpts:
        print(f"\n{'='*70}")
        print(f"Evaluating Recipe {recipe_id}")
        print(f"{'='*70}")
        
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        # Get model configuration
        in_channels = checkpoint.get('in_channels', 1280)  # Default: text(512) + visual(768)
        hidden_channels = args.hidden_channels
        if 'args' in checkpoint:
            hidden_channels = checkpoint['args'].get('hidden_channels', hidden_channels)
        
        # Get test video IDs
        all_video_ids = list(metadata['graphs'].keys())
        test_video_ids = [vid for vid in all_video_ids 
                          if vid.startswith(f"{recipe_id}_")]
        
        if len(test_video_ids) == 0:
            print(f"No test videos for recipe {recipe_id}, skipping...")
            continue
        
        # Load test graphs
        test_graphs = []
        for vid in test_video_ids:
            graph_file = Path(args.data_dir) / 'test' / f"{vid}.pt"
            if not graph_file.exists():
                graph_file = Path(args.data_dir) / 'train' / f"{vid}.pt"
            if graph_file.exists():
                test_graphs.append(torch.load(graph_file, weights_only=False))
        
        if len(test_graphs) == 0:
            print(f"No test graphs found, skipping...")
            continue
        
        test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
        
        # Initialize model
        model = DAGNNClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=2,
            K=args.K,
            dropout=args.dropout
        ).to(device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"  Test samples: {len(test_graphs)}")
        
        # Evaluate
        metrics, predictions = evaluate_model(model, test_loader, criterion, device)
        
        # Store results
        result = {
            'recipe_id': recipe_id,
            'checkpoint': ckpt_path,
            'epoch': checkpoint.get('epoch', 'unknown'),
            'num_test_samples': len(test_graphs),
            'metrics': metrics,
            'predictions_detail': predictions
        }
        all_results.append(result)
        all_metrics.append(metrics)
        
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")
    
    # Aggregate metrics
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")
    
    if all_metrics:
        accuracies = [m['accuracy'] for m in all_metrics]
        precisions = [m['precision'] for m in all_metrics]
        recalls = [m['recall'] for m in all_metrics]
        f1s = [m['f1'] for m in all_metrics]
        aucs = [m['auc'] for m in all_metrics if m['auc'] > 0]
        
        aggregate = {
            'num_recipes_evaluated': len(all_metrics),
            'total_recipes': len(all_recipes),
            'missing_recipes': [r for r in all_recipes 
                               if r not in [c[0] for c in available_ckpts]],
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
        
        print(f"\nAccuracy:  {aggregate['accuracy']['mean']:.4f} ± {aggregate['accuracy']['std']:.4f}")
        print(f"Precision: {aggregate['precision']['mean']:.4f} ± {aggregate['precision']['std']:.4f}")
        print(f"Recall:    {aggregate['recall']['mean']:.4f} ± {aggregate['recall']['std']:.4f}")
        print(f"F1:        {aggregate['f1']['mean']:.4f} ± {aggregate['f1']['std']:.4f}")
        if aucs:
            print(f"AUC:       {aggregate['auc']['mean']:.4f} ± {aggregate['auc']['std']:.4f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_file = output_dir / 'all_recipes_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'aggregate': aggregate if all_metrics else {},
            'per_recipe': all_results
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Save summary CSV
    if all_results:
        csv_file = output_dir / 'results_summary.csv'
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Recipe', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Num_Samples'])
            for r in all_results:
                m = r['metrics']
                writer.writerow([
                    r['recipe_id'],
                    f"{m['accuracy']:.4f}",
                    f"{m['precision']:.4f}",
                    f"{m['recall']:.4f}",
                    f"{m['f1']:.4f}",
                    f"{m['auc']:.4f}",
                    r['num_test_samples']
                ])
        print(f"Summary CSV saved to {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate GNN Classification for all checkpoints'
    )
    
    parser.add_argument('--ckpt_dir', type=str,
                        default='substep4/checkpoints_gnn',
                        help='Directory containing checkpoints')
    parser.add_argument('--data_dir', type=str,
                        default='substep4/graph_classification_substep4/data',
                        help='Directory containing graph data')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--hidden_channels', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--K', type=int, default=10,
                        help='DAGNN propagation steps')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    parser.add_argument('--output_dir', type=str,
                        default='results/substep4_gnn_classification',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    evaluate_all_checkpoints(args)
