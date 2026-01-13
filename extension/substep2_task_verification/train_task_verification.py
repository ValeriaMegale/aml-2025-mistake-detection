import torch
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
    with open(json_path, 'r') as f:
        return json.load(f)


def get_binary_label(video_id, steps, annotation_map):
    if video_id not in annotation_map:
        return 0.0
    video_data = annotation_map[video_id]
    if video_data.get('has_errors', False):
        return 1.0
    if 'steps' in video_data:
        for step in video_data['steps']:
            if step.get('has_errors', False):
                return 1.0
    return 0.0


class RecipeTaskDataset(Dataset):
    def __init__(self, data_dict, video_ids_list, annotation_map):
        self.samples = []
        for vid in video_ids_list:
            if vid not in data_dict: continue
            steps = data_dict[vid]
            embeddings = [s['embedding'] for s in steps]
            if len(embeddings) > 0:
                seq = torch.tensor(np.array(embeddings), dtype=torch.float32)
                lbl = get_binary_label(vid, steps, annotation_map)
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

    os.makedirs(args.ckpt_dir, exist_ok=True)

    print(f"Loading data from {args.npy}...")
    data_dict = np.load(args.npy, allow_pickle=True).item()
    annotation_map = load_annotations(args.annotations)

    all_videos = list(data_dict.keys())
    recipes = sorted(list(set([v.split('_')[0] for v in all_videos])))

    print(f"Training Leave-One-Out on {len(recipes)} folds.")

    for test_recipe in recipes:
        print(f"\n--- FOLD: Holding out Recipe {test_recipe} ---")

        train_ids = [v for v in all_videos if not v.startswith(f"{test_recipe}_")]

        train_ds = RecipeTaskDataset(data_dict, train_ids, annotation_map)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        model = TaskVerifier().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        model.train()
        for epoch in range(args.epochs):
            epoch_loss = 0
            for seqs, labels, mask in train_loader:
                seqs, labels, mask = seqs.to(device), labels.to(device), mask.to(device)

                optimizer.zero_grad()
                preds = model(seqs, mask)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        ckpt_name = f"model_holdout_{test_recipe}.pth"
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy', required=True, help='Path to .npy embeddings')
    parser.add_argument('--annotations', required=True, help='Path to .json annotations')
    parser.add_argument('--ckpt_dir', default='checkpoints', help='Folder to save models')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    run_training(args)
