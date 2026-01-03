"""
Train Task Graph Matching for Task Verification

Extension "From Mistake Detection to Task Verification" - Substep 3

Questo script:
1. Carica step embeddings (perception features: 768-dim)
2. Carica task graphs e li encoda con CLIP text encoder (512-dim, spazio allineato)
3. Addestra il modello TaskGraphMatcher con Hungarian matching
4. Valida usando leave-one-recipe-out cross-validation

Features alignment: Le perception features e CLIP text embeddings condividono
uno spazio allineato video-testo, essenziale per il Hungarian matching.
"""

import argparse
import json
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# Import del modello
try:
    from substep3_step_detection.model.task_graph_matcher import TaskGraphMatcher, TextEncoder
except ImportError:
    from model.task_graph_matcher import TaskGraphMatcher, TextEncoder


# ==================== DATA LOADING ====================

def load_recording_to_activity_mapping(csv_path):
    """
    Carica il mapping recording_id -> activity_id da recording_id_step_idx.csv
    """
    mapping = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec_id = row['recording_id']
            act_id = row['activity_id']
            mapping[rec_id] = act_id
    return mapping


def load_activity_to_taskgraph(json_path):
    """
    Carica il mapping activity_id -> task_graph_file
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def load_task_graph(task_graph_dir, filename):
    """
    Carica un singolo task graph
    """
    filepath = os.path.join(task_graph_dir, filename)
    with open(filepath, 'r') as f:
        return json.load(f)


def load_step_annotations(json_path):
    """
    Carica le annotazioni degli step.
    Chiave: video_id (es. "1_7")
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def get_binary_label(video_id, annotation_map):
    """
    Determina se un video contiene errori.
    CORRETTO: cerca usando video_id come chiave.
    """
    if video_id not in annotation_map:
        return 0.0  # Default: no error
    
    video_data = annotation_map[video_id]
    
    # Controlla se il video ha il flag 'has_errors'
    if video_data.get('has_errors', False):
        return 1.0
    
    # Altrimenti, controlla gli step individuali
    if 'steps' in video_data:
        for step in video_data['steps']:
            if step.get('has_errors', False):
                return 1.0
    
    return 0.0


class TaskGraphMatchingDataset(Dataset):
    """
    Dataset per Task Graph Matching.
    
    Ogni sample contiene:
    - visual_emb: [N_steps, visual_dim] embedding degli step visivi (768-dim per perception)
    - text_emb: [N_nodes, text_dim] embedding dei nodi del task graph (512-dim per CLIP)
    - label: 0/1 se il video contiene errori
    """
    
    def __init__(
        self,
        step_embeddings,        # dict: video_id -> list of step dicts
        video_ids,              # list of video_ids to use
        annotation_map,         # dict: video_id -> annotation data
        recording_to_activity,  # dict: recording_id -> activity_id
        activity_to_taskgraph,  # dict: activity_id -> {name, task_graph_file}
        task_graph_dir,         # path to task graph directory
        text_encoder,           # TextEncoder instance
        precomputed_text_emb=None  # Optional: precomputed text embeddings per activity
    ):
        self.samples = []
        self.precomputed_text_emb = precomputed_text_emb or {}
        self.text_encoder = text_encoder
        self.task_graph_dir = task_graph_dir
        self.activity_to_taskgraph = activity_to_taskgraph
        
        for vid in tqdm(video_ids, desc="Preparing dataset"):
            if vid not in step_embeddings:
                continue
            
            steps = step_embeddings[vid]
            if len(steps) == 0:
                continue
            
            # Visual embeddings
            visual_emb = np.array([s['embedding'] for s in steps])
            
            # Get activity_id
            activity_id = recording_to_activity.get(vid)
            if activity_id is None:
                continue
            
            # Get task graph
            tg_info = activity_to_taskgraph.get(activity_id)
            if tg_info is None:
                continue
            
            # Get text embeddings (cached or compute)
            if activity_id in self.precomputed_text_emb:
                text_emb = self.precomputed_text_emb[activity_id]
            else:
                task_graph = load_task_graph(task_graph_dir, tg_info['task_graph_file'])
                _, text_emb = text_encoder.encode_task_graph(task_graph)
                self.precomputed_text_emb[activity_id] = text_emb
            
            # Label
            label = get_binary_label(vid, annotation_map)
            
            self.samples.append({
                'video_id': vid,
                'visual_emb': torch.tensor(visual_emb, dtype=torch.float32),
                'text_emb': torch.tensor(text_emb, dtype=torch.float32),
                'label': torch.tensor([label], dtype=torch.float32)
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """
    Collate function per padding variabile.
    """
    visual_embs = [s['visual_emb'] for s in batch]
    text_embs = [s['text_emb'] for s in batch]
    labels = torch.stack([s['label'] for s in batch])
    video_ids = [s['video_id'] for s in batch]
    
    # Pad visual embeddings
    padded_visual = pad_sequence(visual_embs, batch_first=True, padding_value=0)
    visual_lens = torch.tensor([len(v) for v in visual_embs])
    visual_mask = torch.arange(padded_visual.size(1))[None, :] >= visual_lens[:, None]
    
    # Pad text embeddings
    padded_text = pad_sequence(text_embs, batch_first=True, padding_value=0)
    text_lens = torch.tensor([len(t) for t in text_embs])
    text_mask = torch.arange(padded_text.size(1))[None, :] >= text_lens[:, None]
    
    return {
        'visual_emb': padded_visual,
        'text_emb': padded_text,
        'visual_mask': visual_mask,
        'text_mask': text_mask,
        'labels': labels,
        'video_ids': video_ids
    }


# ==================== TRAINING ====================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in dataloader:
        visual = batch['visual_emb'].to(device)
        text = batch['text_emb'].to(device)
        visual_mask = batch['visual_mask'].to(device)
        text_mask = batch['text_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        probs, matching_costs = model(visual, text, visual_mask, text_mask)
        
        loss = criterion(probs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds = []
    all_labels = []
    all_costs = []
    
    with torch.no_grad():
        for batch in dataloader:
            visual = batch['visual_emb'].to(device)
            text = batch['text_emb'].to(device)
            visual_mask = batch['visual_mask'].to(device)
            text_mask = batch['text_mask'].to(device)
            labels = batch['labels'].to(device)
            
            probs, matching_costs = model(visual, text, visual_mask, text_mask)
            
            loss = criterion(probs, labels)
            
            total_loss += loss.item()
            n_batches += 1
            
            all_preds.extend(probs.cpu().numpy().flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
            all_costs.extend(matching_costs.cpu().numpy().tolist())
    
    # Calcola metriche
    preds_binary = [1 if p > 0.5 else 0 for p in all_preds]
    
    tp = sum(1 for p, l in zip(preds_binary, all_labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds_binary, all_labels) if p == 1 and l == 0)
    tn = sum(1 for p, l in zip(preds_binary, all_labels) if p == 0 and l == 0)
    fn = sum(1 for p, l in zip(preds_binary, all_labels) if p == 0 and l == 1)
    
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    
    return {
        'loss': total_loss / max(n_batches, 1),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_matching_cost': np.mean(all_costs)
    }


# ==================== MAIN ====================

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    step_embeddings = np.load(args.npy, allow_pickle=True).item()
    annotation_map = load_step_annotations(args.annotations)
    recording_to_activity = load_recording_to_activity_mapping(args.recording_csv)
    activity_to_taskgraph = load_activity_to_taskgraph(args.activity_mapping)
    
    # Initialize text encoder
    print("Initializing text encoder...")
    text_encoder = TextEncoder(args.text_model)
    
    # Precompute all text embeddings
    print("Precomputing task graph embeddings...")
    precomputed_text_emb = {}
    for act_id, tg_info in activity_to_taskgraph.items():
        task_graph = load_task_graph(args.task_graph_dir, tg_info['task_graph_file'])
        _, text_emb = text_encoder.encode_task_graph(task_graph)
        precomputed_text_emb[act_id] = text_emb
    
    # Get all video IDs
    all_videos = list(step_embeddings.keys())
    recipes = sorted(list(set([v.split('_')[0] for v in all_videos])))
    
    print(f"\nFound {len(all_videos)} videos from {len(recipes)} recipes")
    print(f"Recipes: {recipes}")
    
    # Conta label distribution
    n_errors = sum(1 for vid in all_videos if get_binary_label(vid, annotation_map) == 1.0)
    print(f"Label distribution: {n_errors} errors / {len(all_videos) - n_errors} normal")
    
    # Leave-one-recipe-out cross-validation
    print(f"\n{'='*60}")
    print("Starting Leave-One-Recipe-Out Cross-Validation")
    print(f"{'='*60}")
    
    all_results = []
    
    for test_recipe in recipes:
        print(f"\n--- Fold: Recipe {test_recipe} held out for testing ---")
        
        train_ids = [v for v in all_videos if not v.startswith(f"{test_recipe}_")]
        test_ids = [v for v in all_videos if v.startswith(f"{test_recipe}_")]
        
        if len(test_ids) == 0:
            print(f"  No test samples for recipe {test_recipe}, skipping...")
            continue
        
        print(f"  Train: {len(train_ids)} videos, Test: {len(test_ids)} videos")
        
        # Create datasets
        train_ds = TaskGraphMatchingDataset(
            step_embeddings, train_ids, annotation_map,
            recording_to_activity, activity_to_taskgraph,
            args.task_graph_dir, text_encoder, precomputed_text_emb
        )
        
        test_ds = TaskGraphMatchingDataset(
            step_embeddings, test_ids, annotation_map,
            recording_to_activity, activity_to_taskgraph,
            args.task_graph_dir, text_encoder, precomputed_text_emb
        )
        
        if len(train_ds) == 0 or len(test_ds) == 0:
            print(f"  Empty dataset, skipping...")
            continue
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        
        # Model
        model = TaskGraphMatcher(
            visual_dim=args.visual_dim,
            text_dim=text_encoder.dim,
            hidden_dim=args.hidden_dim
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.BCELoss()
        
        # Training loop
        best_f1 = 0
        best_epoch = 0
        
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            
            if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                test_metrics = evaluate(model, test_loader, criterion, device)
                print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                      f"Test Acc={test_metrics['accuracy']:.3f}, F1={test_metrics['f1']:.3f}, "
                      f"Matching Cost={test_metrics['avg_matching_cost']:.3f}")
                
                if test_metrics['f1'] > best_f1:
                    best_f1 = test_metrics['f1']
                    best_epoch = epoch + 1
                    
                    # Save best checkpoint
                    ckpt_path = os.path.join(args.ckpt_dir, f"task_graph_matcher_recipe_{test_recipe}.pth")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': epoch,
                        'metrics': test_metrics,
                        'test_recipe': test_recipe
                    }, ckpt_path)
        
        # Final evaluation
        final_metrics = evaluate(model, test_loader, criterion, device)
        final_metrics['recipe'] = test_recipe
        final_metrics['best_epoch'] = best_epoch
        final_metrics['best_f1'] = best_f1
        all_results.append(final_metrics)
        
        print(f"  Best F1: {best_f1:.3f} at epoch {best_epoch}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Cross-Validation Summary")
    print(f"{'='*60}")
    
    avg_acc = np.mean([r['accuracy'] for r in all_results])
    avg_f1 = np.mean([r['f1'] for r in all_results])
    avg_prec = np.mean([r['precision'] for r in all_results])
    avg_rec = np.mean([r['recall'] for r in all_results])
    
    print(f"Average Accuracy: {avg_acc:.3f}")
    print(f"Average F1: {avg_f1:.3f}")
    print(f"Average Precision: {avg_prec:.3f}")
    print(f"Average Recall: {avg_rec:.3f}")
    
    # Save results
    results_path = os.path.join(args.ckpt_dir, "cv_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'per_fold': all_results,
            'average': {
                'accuracy': avg_acc,
                'f1': avg_f1,
                'precision': avg_prec,
                'recall': avg_rec
            }
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Task Graph Matching model")
    
    parser.add_argument('--npy', type=str, 
                        default='extension_localization/data/step_embeddings_perception.npy',
                        help='Path to step embeddings .npy file (perception features 768-dim)')
    parser.add_argument('--annotations', type=str,
                        default='extension_localization/data/step_annotations.json',
                        help='Path to step annotations JSON')
    parser.add_argument('--recording_csv', type=str,
                        default='annotations/annotation_csv/recording_id_step_idx.csv',
                        help='Path to recording_id_step_idx.csv')
    parser.add_argument('--activity_mapping', type=str,
                        default='substep3_step_detection/activity_to_taskgraph.json',
                        help='Path to activity_to_taskgraph.json')
    parser.add_argument('--task_graph_dir', type=str,
                        default='annotations/task_graphs',
                        help='Directory containing task graph JSON files')
    parser.add_argument('--text_model', type=str,
                        default='clip',
                        help='Text encoder: "clip" (512-dim, aligned with perception) or "all-MiniLM-L6-v2" (384-dim)')
    parser.add_argument('--ckpt_dir', type=str,
                        default='substep3_step_detection/checkpoints_graph',
                        help='Directory to save checkpoints')
    
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--visual_dim', type=int, default=768,
                        help='Visual embedding dimension: 768 for perception features, 1024 for omnivore')
    
    args = parser.parse_args()
    
    main(args)


# Esempio di esecuzione:
# python substep3_step_detection/train_task_graph_matching.py --npy extension_localization/data/step_embeddings.npy --annotations extension_localization/data/step_annotations.json --epochs 30
