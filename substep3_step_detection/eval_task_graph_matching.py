"""
Evaluate Task Graph Matching for Task Verification

Extension "From Mistake Detection to Task Verification" - Substep 3

Questo script valuta il modello TaskGraphMatcher sui dati di test.
"""

import argparse
import json
import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Import del modello
try:
    from substep3_step_detection.model.task_graph_matcher import TaskGraphMatcher, TextEncoder
    from substep3_step_detection.train_task_graph_matching import (
        TaskGraphMatchingDataset, collate_fn, 
        load_recording_to_activity_mapping, load_activity_to_taskgraph,
        load_task_graph, load_step_annotations, get_binary_label
    )
except ImportError:
    from model.task_graph_matcher import TaskGraphMatcher, TextEncoder
    from train_task_graph_matching import (
        TaskGraphMatchingDataset, collate_fn,
        load_recording_to_activity_mapping, load_activity_to_taskgraph,
        load_task_graph, load_step_annotations, get_binary_label
    )


def evaluate_model(model, dataloader, device, threshold=0.5):
    """
    Valutazione completa del modello.
    
    Returns:
        dict con tutte le metriche
        list di predictions dettagliate per ogni video
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_costs = []
    all_video_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            visual = batch['visual_emb'].to(device)
            text = batch['text_emb'].to(device)
            visual_mask = batch['visual_mask'].to(device)
            text_mask = batch['text_mask'].to(device)
            labels = batch['labels'].to(device)
            video_ids = batch['video_ids']
            
            probs, matching_costs = model(visual, text, visual_mask, text_mask)
            
            all_probs.extend(probs.cpu().numpy().flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
            all_costs.extend(matching_costs.cpu().numpy().tolist())
            all_video_ids.extend(video_ids)
    
    # Binary predictions
    all_preds = [1 if p > threshold else 0 for p in all_probs]
    
    # Metriche
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'avg_matching_cost': np.mean(all_costs),
        'threshold': threshold
    }
    
    # AUC (solo se ci sono entrambe le classi)
    if len(set(all_labels)) > 1:
        metrics['auc'] = roc_auc_score(all_labels, all_probs)
    else:
        metrics['auc'] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Predictions dettagliate
    predictions = []
    for vid, prob, pred, label, cost in zip(all_video_ids, all_probs, all_preds, all_labels, all_costs):
        predictions.append({
            'video_id': vid,
            'probability': prob,
            'prediction': pred,
            'ground_truth': int(label),
            'matching_cost': cost,
            'correct': pred == int(label)
        })
    
    return metrics, predictions


def print_results(metrics, predictions):
    """Stampa i risultati in formato leggibile."""
    
    print("\n" + "=" * 70)
    print("TASK GRAPH MATCHING - EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\nThreshold: {metrics['threshold']}")
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"\n  Avg Matching Cost: {metrics['avg_matching_cost']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"              Predicted")
    print(f"            Normal  Error")
    print(f"  Normal     {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"  Error      {cm[1][0]:5d}  {cm[1][1]:5d}")
    
    # Analisi errori
    errors = [p for p in predictions if not p['correct']]
    if errors:
        print(f"\nMisclassified samples ({len(errors)} total):")
        for e in errors[:10]:  # Mostra primi 10
            print(f"  {e['video_id']}: prob={e['probability']:.3f}, "
                  f"pred={e['prediction']}, gt={e['ground_truth']}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    print("\n" + "=" * 70)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    step_embeddings = np.load(args.npy, allow_pickle=True).item()
    annotation_map = load_step_annotations(args.annotations)
    recording_to_activity = load_recording_to_activity_mapping(args.recording_csv)
    activity_to_taskgraph = load_activity_to_taskgraph(args.activity_mapping)
    
    # Initialize text encoder
    print("Initializing text encoder...")
    text_encoder = TextEncoder(args.text_model)
    
    # Precompute text embeddings
    print("Precomputing task graph embeddings...")
    precomputed_text_emb = {}
    for act_id, tg_info in activity_to_taskgraph.items():
        task_graph = load_task_graph(args.task_graph_dir, tg_info['task_graph_file'])
        _, text_emb = text_encoder.encode_task_graph(task_graph)
        precomputed_text_emb[act_id] = text_emb
    
    # Get video IDs
    all_videos = list(step_embeddings.keys())
    
    if args.test_recipe:
        # Single recipe evaluation
        test_ids = [v for v in all_videos if v.startswith(f"{args.test_recipe}_")]
        print(f"\nEvaluating on recipe {args.test_recipe}: {len(test_ids)} videos")
    else:
        # All videos
        test_ids = all_videos
        print(f"\nEvaluating on all {len(test_ids)} videos")
    
    # Create dataset
    test_ds = TaskGraphMatchingDataset(
        step_embeddings, test_ids, annotation_map,
        recording_to_activity, activity_to_taskgraph,
        args.task_graph_dir, text_encoder, precomputed_text_emb
    )
    
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    
    model = TaskGraphMatcher(
        visual_dim=1024,
        text_dim=text_encoder.dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    # Evaluate
    metrics, predictions = evaluate_model(model, test_loader, device, threshold=args.threshold)
    
    # Print results
    print_results(metrics, predictions)
    
    # Save results
    if args.output:
        results = {
            'metrics': metrics,
            'predictions': predictions,
            'config': vars(args)
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Task Graph Matching model")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--npy', type=str,
                        default='extension_localization/data/step_embeddings.npy',
                        help='Path to step embeddings .npy file')
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
                        default='all-MiniLM-L6-v2',
                        help='Sentence transformer model name')
    
    parser.add_argument('--test_recipe', type=str, default=None,
                        help='Recipe ID to test on (leave-one-out style)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON')
    
    args = parser.parse_args()
    
    main(args)


# Esempio di esecuzione:
# python substep3_step_detection/eval_task_graph_matching.py --checkpoint substep3_step_detection/checkpoints_graph/task_graph_matcher_recipe_1.pth --test_recipe 1
