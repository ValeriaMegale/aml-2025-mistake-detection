"""
Evaluate Task Verification using HiERO step embeddings.
Adapted from eval_task_verification.py to work with new embedding format.
"""

import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

try:
    from extension_task_verification.model.task_verifier import TaskVerifier
except ImportError:
    from model.task_verifier import TaskVerifier


def load_annotations(json_path):
    """Load step annotations JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def get_recipe_level_label(video_id, annotation_map):
    """
    Get binary label for recipe-level correctness.
    Returns 1.0 if video has ANY errors, 0.0 otherwise.
    """
    if video_id not in annotation_map:
        return 0.0
    
    video_data = annotation_map[video_id]
    
    # Check if video has errors flag
    if video_data.get('has_errors', False):
        return 1.0
    
    # Check if any step has errors
    if 'steps' in video_data:
        for step in video_data['steps']:
            if step.get('has_errors', False):
                return 1.0
    
    return 0.0


def convert_hiero_embeddings_to_dict_format(hiero_embeddings_dict):
    """
    Convert HiERO embedding format to expected format.
    
    HiERO format: {video_id: numpy_array [N, 768]}
    Expected format: {video_id: [{'embedding': array}, {'embedding': array}, ...]}
    """
    converted = {}
    for video_id, embeddings_array in hiero_embeddings_dict.items():
        # embeddings_array is [N, 768]
        step_list = []
        for i in range(embeddings_array.shape[0]):
            step_list.append({
                'embedding': embeddings_array[i]  # [768] array
            })
        converted[video_id] = step_list
    return converted


class RecipeTaskDataset(Dataset):
    def __init__(self, data_dict, video_ids_list, annotation_map):
        self.samples = []
        for vid in video_ids_list:
            if vid not in data_dict:
                continue
            
            steps = data_dict[vid]
            # Extract embeddings from list of dicts
            embeddings = [s['embedding'] for s in steps]
            
            if len(embeddings) > 0:
                # Convert to numpy array then tensor
                seq = torch.tensor(np.array(embeddings), dtype=torch.float32)
                # Get recipe-level binary label
                lbl = get_recipe_level_label(vid, annotation_map)
                self.samples.append((seq, torch.tensor([lbl], dtype=torch.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    seqs, lbls = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    lens = torch.tensor([len(x) for x in seqs])
    mask = torch.arange(padded.size(1))[None, :] >= lens[:, None]
    return padded, torch.stack(lbls), mask


def run_evaluation(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print(f"Loading HiERO embeddings from {args.npy}...")
    # Load HiERO format: {video_id: numpy_array [N, 768]}
    hiero_embeddings = np.load(args.npy, allow_pickle=True).item()
    
    # Convert to expected format
    print("Converting embedding format...")
    data_dict = convert_hiero_embeddings_to_dict_format(hiero_embeddings)
    
    print(f"Loading annotations from {args.annotations}...")
    annotation_map = load_annotations(args.annotations)

    all_videos = list(data_dict.keys())
    # Extract recipe names (e.g., "1_25" -> "1")
    recipes = sorted(list(set([v.split('_')[0] for v in all_videos if '_' in v])))

    accuracies = []
    all_predictions = []
    all_labels = []

    print(f"\n{'='*60}")
    print(f"Starting Evaluation on {len(recipes)} folds")
    print(f"Checkpoints directory: {args.ckpt_dir}")
    print(f"{'='*60}")

    for fold_idx, test_recipe in enumerate(recipes):
        print(f"\n--- FOLD {fold_idx+1}/{len(recipes)}: Recipe {test_recipe} ---")
        
        # Test set = ONLY videos from current recipe
        test_ids = [v for v in all_videos if v.startswith(f"{test_recipe}_")]

        test_ds = RecipeTaskDataset(data_dict, test_ids, annotation_map)
        if len(test_ds) == 0:
            print(f"  No test samples for recipe {test_recipe}, skipping...")
            continue

        test_loader = DataLoader(
            test_ds, 
            batch_size=1, 
            shuffle=False, 
            collate_fn=collate_fn
        )

        # Load model checkpoint for this fold
        ckpt_path = os.path.join(args.ckpt_dir, f"model_holdout_{test_recipe}.pth")

        if not os.path.exists(ckpt_path):
            print(f"  Checkpoint not found: {ckpt_path}")
            print(f"  Skipping recipe {test_recipe}")
            continue

        # Initialize and load model
        model = TaskVerifier(input_dim=768).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        correct = 0
        total = 0
        fold_predictions = []
        fold_labels = []
        
        with torch.no_grad():
            for seqs, labels, mask in test_loader:
                seqs, labels, mask = seqs.to(device), labels.to(device), mask.to(device)

                preds = model(seqs, mask)
                predicted = (preds > 0.5).float()

                if predicted.item() == labels.item():
                    correct += 1
                total += 1
                
                fold_predictions.append(preds.item())
                fold_labels.append(labels.item())

        acc = correct / total if total > 0 else 0
        print(f"  Accuracy: {acc:.4f} ({correct}/{total})")
        accuracies.append(acc)
        all_predictions.extend(fold_predictions)
        all_labels.extend(fold_labels)

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    if len(accuracies) > 0:
        print(f"Folds evaluated: {len(accuracies)}")
        print(f"Mean Accuracy: {np.mean(accuracies):.4f}")
        print(f"Std Accuracy: {np.std(accuracies):.4f}")
        print(f"Min Accuracy: {np.min(accuracies):.4f}")
        print(f"Max Accuracy: {np.max(accuracies):.4f}")
        
        # Compute overall metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        binary_preds = (all_predictions > 0.5).astype(float)
        
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        precision = precision_score(all_labels, binary_preds, zero_division=0)
        recall = recall_score(all_labels, binary_preds, zero_division=0)
        f1 = f1_score(all_labels, binary_preds, zero_division=0)
        auc = roc_auc_score(all_labels, all_predictions) if len(np.unique(all_labels)) > 1 else 0.0
        
        print(f"\nOverall Metrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  AUC-ROC: {auc:.4f}")
    else:
        print("No results - check that checkpoints exist!")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate Task Verification using HiERO step embeddings'
    )
    parser.add_argument(
        '--npy', 
        required=True, 
        help='Path to .npy embeddings (HiERO format)'
    )
    parser.add_argument(
        '--annotations', 
        required=True, 
        help='Path to .json annotations (step_annotations.json)'
    )
    parser.add_argument(
        '--ckpt_dir', 
        default='extension_task_verification/checkpoints_hiero', 
        help='Folder containing saved models'
    )
    args = parser.parse_args()

    run_evaluation(args)
