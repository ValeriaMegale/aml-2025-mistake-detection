"""
Train Task Verification using HiERO step embeddings.
Adapted from train_task_verification.py to work with new embedding format.
"""

import argparse
import json
import os
import numpy as np

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

try:
    from extension.substep2_task_verification.model.task_verifier import TaskVerifier
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


def run_training(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create checkpoint directory
    os.makedirs(args.ckpt_dir, exist_ok=True)

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

    print(f"Found {len(recipes)} recipes for Leave-One-Out evaluation")
    print(f"Total videos: {len(all_videos)}")

    for fold_idx, test_recipe in enumerate(recipes):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx+1}/{len(recipes)}: Holding out Recipe {test_recipe}")
        print(f"{'='*60}")

        # Training set = All videos EXCEPT those from current recipe
        train_ids = [v for v in all_videos if not v.startswith(f"{test_recipe}_")]

        if len(train_ids) == 0:
            print(f"  Warning: No training videos for recipe {test_recipe}, skipping...")
            continue

        train_ds = RecipeTaskDataset(data_dict, train_ids, annotation_map)
        train_loader = DataLoader(
            train_ds, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )

        print(f"  Training videos: {len(train_ids)}")
        print(f"  Training batches: {len(train_loader)}")

        # Initialize Model (input_dim=768 for perception)
        model = TaskVerifier(input_dim=768).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Training Loop
        model.train()
        for epoch in range(args.epochs):
            epoch_loss = 0
            num_batches = 0
            
            for seqs, labels, mask in train_loader:
                seqs, labels, mask = seqs.to(device), labels.to(device), mask.to(device)

                optimizer.zero_grad()
                preds = model(seqs, mask)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
                print(f"  Epoch {epoch+1}/{args.epochs} - Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        ckpt_name = f"model_holdout_{test_recipe}.pth"
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

    print(f"\n{'='*60}")
    print("Training completed for all folds!")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Task Verification using HiERO step embeddings'
    )
    parser.add_argument(
        '--npy', 
        required=True, 
        help='Path to .npy embeddings (HiERO format: {video_id: array [N, 768]})'
    )
    parser.add_argument(
        '--annotations', 
        required=True, 
        help='Path to .json annotations (step_annotations.json)'
    )
    parser.add_argument(
        '--ckpt_dir', 
        default='extension/substep2_task_verification/checkpoints_hiero', 
        help='Folder to save models'
    )
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    import torch
    run_training(args)


