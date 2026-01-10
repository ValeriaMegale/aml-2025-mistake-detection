"""
Train GNN Classification for Task Verification (Substep 4).

Extension "From Mistake Detection to Task Verification" - Substep 4

Addestra DAGNN classifier su task graphs matched con leave-one-recipe-out evaluation.
"""

import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add parent directory to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_classification_substep4.dataset import TaskGraphDataset
from graph_classification_substep4.model import DAGNNClassifier, get_model_input_dim


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Training epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        
        # Reshape output: [batch, 1] -> [batch] for BCEWithLogitsLoss
        out = out.squeeze(-1) if out.dim() > 1 else out
        
        # BCEWithLogitsLoss expects float targets
        loss = criterion(out, batch.y.float())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def evaluate(model, dataloader, criterion, device):
    """Evaluation con metriche complete."""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            out = model(batch.x, batch.edge_index, batch.batch)
            
            # Reshape output: [batch, 1] -> [batch] for BCEWithLogitsLoss
            out = out.squeeze(-1) if out.dim() > 1 else out
            
            # BCEWithLogitsLoss
            loss = criterion(out, batch.y.float())
            
            # Get probabilities
            # Output is already [batch] shape after squeeze in forward
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).long()
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch.y.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            
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
    }


def train_one_fold(args, test_recipe_id, all_recipes):
    """Addestra modello per un fold (leave-one-out)."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Training for Recipe {test_recipe_id} (test set)")
    print(f"{'='*70}")
    
    # Load metadata to get all video IDs
    metadata_file = Path(args.data_dir) / 'metadata.json'
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    all_video_ids = list(metadata['graphs'].keys())
    
    # Split: test = recipe test_recipe_id, train = all others
    train_video_ids = [vid for vid in all_video_ids 
                       if not vid.startswith(f"{test_recipe_id}_")]
    test_video_ids = [vid for vid in all_video_ids 
                      if vid.startswith(f"{test_recipe_id}_")]
    
    if len(test_video_ids) == 0:
        print(f"No test videos for recipe {test_recipe_id}, skipping...")
        return None
    
    print(f"Train videos: {len(train_video_ids)}")
    print(f"Test videos: {len(test_video_ids)}")
    
    # Load datasets
    # We need to manually create the splits
    train_graphs = []
    test_graphs = []
    
    # Load all graphs and split
    for vid in train_video_ids:
        # Try train dir first, then test dir
        graph_file = Path(args.data_dir) / 'train' / f"{vid}.pt"
        if not graph_file.exists():
            graph_file = Path(args.data_dir) / 'test' / f"{vid}.pt"
        if graph_file.exists():
                train_graphs.append(torch.load(graph_file, weights_only=False))
    
    for vid in test_video_ids:
        graph_file = Path(args.data_dir) / 'test' / f"{vid}.pt"
        if not graph_file.exists():
            graph_file = Path(args.data_dir) / 'train' / f"{vid}.pt"
        if graph_file.exists():
                test_graphs.append(torch.load(graph_file, weights_only=False))
    
    if len(train_graphs) == 0 or len(test_graphs) == 0:
        print(f"Insufficient data, skipping...")
        return None
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
    
    # Get input dimension from first sample
    sample = train_graphs[0]
    in_channels = sample.x.size(1)
    
    print(f"Node feature dimension: {in_channels}")
    
    # Initialize model
    model = DAGNNClassifier(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        num_classes=2,
        K=args.K,
        dropout=args.dropout
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Binary classification: BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_f1 = 0
    best_epoch = 0
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            test_metrics = evaluate(model, test_loader, criterion, device)
            
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1}/{args.epochs} [{elapsed:.1f}s]: "
                  f"Train Loss={train_loss:.4f}, "
                  f"Test Acc={test_metrics['accuracy']:.3f}, "
                  f"F1={test_metrics['f1']:.3f}, "
                  f"AUC={test_metrics['auc']:.3f}")
            
            if test_metrics['f1'] > best_f1:
                best_f1 = test_metrics['f1']
                best_epoch = epoch + 1
                patience_counter = 0
                
                # Save checkpoint
                ckpt_dir = Path(args.ckpt_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / f"gnn_classifier_recipe_{test_recipe_id}.pth"
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'metrics': test_metrics,
                    'test_recipe': test_recipe_id,
                    'args': vars(args),
                    'in_channels': in_channels
                }, ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= args.patience and args.early_stop:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
    
    print(f"  Best F1: {best_f1:.4f} at epoch {best_epoch}")
    
    return {
        'recipe_id': test_recipe_id,
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'best_metrics': test_metrics if 'test_metrics' in locals() else {}
    }


def main(args):
    """Main training function con leave-one-out."""
    
    print("GNN Classification Training - Leave-One-Recipe-Out")
    print(f"Data directory: {args.data_dir}")
    print(f"Checkpoint directory: {args.ckpt_dir}")
    
    # Load metadata to get all recipes
    metadata_file = Path(args.data_dir) / 'metadata.json'
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    all_recipes = sorted(metadata.get('recipes', []), 
                        key=lambda x: int(x) if x.isdigit() else 0)
    
    print(f"\nTotal recipes: {len(all_recipes)}")
    
    # Train for each recipe (leave-one-out)
    all_results = []
    
    for recipe_id in all_recipes:
        result = train_one_fold(args, recipe_id, all_recipes)
        if result:
            all_results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    
    if all_results:
        f1s = [r['best_f1'] for r in all_results]
        print(f"Trained models: {len(all_results)}/{len(all_recipes)}")
        print(f"Mean F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"Min F1: {np.min(f1s):.4f}, Max F1: {np.max(f1s):.4f}")
        
        # Save summary
        summary_file = Path(args.ckpt_dir) / 'training_summary.json'
        with open(summary_file, 'w') as f:
            json.dump({
                'total_recipes': len(all_recipes),
                'trained': len(all_results),
                'results': all_results,
                'aggregate': {
                    'mean_f1': float(np.mean(f1s)),
                    'std_f1': float(np.std(f1s)),
                    'min_f1': float(np.min(f1s)),
                    'max_f1': float(np.max(f1s))
                }
            }, f, indent=2)
        print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train GNN Classification for Task Verification'
    )
    
    parser.add_argument('--data_dir', type=str,
                        default='substep4/graph_classification_substep4/data',
                        help='Directory containing graph data (with metadata.json)')
    parser.add_argument('--ckpt_dir', type=str,
                        default='substep4/checkpoints_gnn',
                        help='Directory to save checkpoints')
    
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--hidden_channels', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--K', type=int, default=10,
                        help='DAGNN propagation steps')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    parser.add_argument('--eval_every', type=int, default=5,
                        help='Evaluate every N epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--early_stop', action='store_true',
                        help='Enable early stopping')
    
    args = parser.parse_args()
    
    main(args)
