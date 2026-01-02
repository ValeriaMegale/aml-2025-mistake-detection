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


# --- DATASET & UTILS (Devono essere identici al train) ---
def load_annotations(json_path):
    with open(json_path, 'r') as f: return json.load(f)


def get_binary_label(video_id, steps, annotation_map):
    is_error = False
    for step in steps:
        lbl = str(step['label'])
        if lbl in annotation_map and annotation_map[lbl].get('has_errors', False):
            is_error = True;
            break
    return 1.0 if is_error else 0.0


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


# --- EVALUATION LOGIC ---
def run_evaluation(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading data from {args.npy}...")
    data_dict = np.load(args.npy, allow_pickle=True).item()
    annotation_map = load_annotations(args.annotations)

    all_videos = list(data_dict.keys())
    recipes = sorted(list(set([v.split('_')[0] for v in all_videos])))

    accuracies = []

    print(f"Starting Evaluation on {len(recipes)} folds using checkpoints in '{args.ckpt_dir}'")

    for test_recipe in recipes:
        # Test set = SOLO i video della ricetta corrente
        test_ids = [v for v in all_videos if v.startswith(f"{test_recipe}_")]

        test_ds = RecipeTaskDataset(data_dict, test_ids, annotation_map)
        if len(test_ds) == 0: continue

        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

        # Caricamento del modello specifico per questo fold
        ckpt_path = os.path.join(args.ckpt_dir, f"model_holdout_{test_recipe}.pth")

        if not os.path.exists(ckpt_path):
            print(f"Skipping {test_recipe}: Checkpoint not found at {ckpt_path}")
            continue

        model = TaskVerifier().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for seqs, labels, mask in test_loader:
                seqs, labels, mask = seqs.to(device), labels.to(device), mask.to(device)

                preds = model(seqs, mask)
                predicted = (preds > 0.5).float()

                if predicted == labels:
                    correct += 1
                total += 1

        acc = correct / total if total > 0 else 0
        print(f"Recipe {test_recipe} Accuracy: {acc:.4f} ({correct}/{total})")
        accuracies.append(acc)

    print("\n" + "=" * 30)
    if len(accuracies) > 0:
        print(f"FINAL MEAN ACCURACY: {np.mean(accuracies):.4f}")
    else:
        print("No results.")
    print("=" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy', required=True, help='Path to .npy embeddings')
    parser.add_argument('--annotations', required=True, help='Path to .json annotations')
    parser.add_argument('--ckpt_dir', default='checkpoints', help='Folder containing saved models')
    args = parser.parse_args()

    run_evaluation(args)

#esempio run
#python extension_task_verification/eval_task_verification.py --npy extension_localization/data/step_embeddings.npy --annotations extension_localization/data/step_annotations.json --ckpt_dir extension_task_verification/checkpoints
