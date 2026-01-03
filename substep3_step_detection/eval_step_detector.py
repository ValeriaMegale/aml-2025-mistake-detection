import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import average_precision_score, accuracy_score

# Import del modello
try:
    from extension_task_verification.model.step_detector import StepMistakeDetector
except ImportError:
    from model.step_detector import StepMistakeDetector


# --- DATASET & UTILS (Devono essere coerenti con il train) ---
def load_annotations(json_path):
    with open(json_path, 'r') as f: return json.load(f)


def get_step_labels(video_id, steps, annotation_map):
    labels = []
    for step in steps:
        lbl_id = str(step['label'])
        is_error = 0.0
        if lbl_id in annotation_map and annotation_map[lbl_id].get('has_errors', False):
            is_error = 1.0
        labels.append(is_error)
    return labels


class StepDetectionDataset(Dataset):
    def __init__(self, data_dict, video_ids_list, annotation_map):
        self.samples = []
        for vid in video_ids_list:
            if vid not in data_dict: continue
            steps = data_dict[vid]
            embeddings = [s['embedding'] for s in steps]

            if len(embeddings) > 0:
                seq = torch.tensor(np.array(embeddings), dtype=torch.float32)
                step_lbls = get_step_labels(vid, steps, annotation_map)
                lbl_seq = torch.tensor(step_lbls, dtype=torch.float32)
                self.samples.append((seq, lbl_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    seqs, lbls = zip(*batch)
    padded_seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
    # Pad labels con -1 (anche se useremo la maschera, è più sicuro)
    padded_lbls = pad_sequence(lbls, batch_first=True, padding_value=-1)
    lens = torch.tensor([len(x) for x in seqs])
    mask = torch.arange(padded_seqs.size(1))[None, :] >= lens[:, None]
    return padded_seqs, padded_lbls, mask


# --- EVALUATION LOGIC ---
def run_evaluation(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading data from {args.npy}...")
    data_dict = np.load(args.npy, allow_pickle=True).item()
    annotation_map = load_annotations(args.annotations)

    all_videos = list(data_dict.keys())
    recipes = sorted(list(set([v.split('_')[0] for v in all_videos])))

    fold_accuracies = []
    fold_aps = []

    print(f"Starting Step-Level Evaluation on {len(recipes)} folds.")

    for test_recipe in recipes:
        # Test set: solo i video della ricetta corrente
        test_ids = [v for v in all_videos if v.startswith(f"{test_recipe}_")]

        test_ds = StepDetectionDataset(data_dict, test_ids, annotation_map)
        if len(test_ds) == 0: continue

        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

        # Caricamento Checkpoint
        ckpt_path = os.path.join(args.ckpt_dir, f"step_model_{test_recipe}.pth")
        if not os.path.exists(ckpt_path):
            print(f"Skipping {test_recipe}: Checkpoint not found at {ckpt_path}")
            continue

        model = StepMistakeDetector().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        all_preds_probs = []
        all_labels_gt = []

        with torch.no_grad():
            for seqs, labels, mask in test_loader:
                seqs, labels, mask = seqs.to(device), labels.to(device), mask.to(device)

                # Forward
                logits = model(seqs, mask).squeeze(-1)  # [Batch, Seq]
                probs = torch.sigmoid(logits)  # Probabilità [0, 1]

                # --- MASKING CRUCIALE ---
                # Dobbiamo appiattire le predizioni e rimuovere i valori di padding
                # mask è True dove è padding -> valid_mask è True dove è valido
                valid_mask = ~mask

                # Seleziona solo gli elementi validi
                valid_probs = probs[valid_mask]
                valid_labels = labels[valid_mask]

                # Aggiungi alle liste globali per questo fold
                all_preds_probs.extend(valid_probs.cpu().numpy())
                all_labels_gt.extend(valid_labels.cpu().numpy())

        # Calcolo Metriche per questo fold
        if len(all_labels_gt) > 0:
            # Binarizzazione per Accuracy (soglia 0.5)
            binary_preds = [1 if p > 0.5 else 0 for p in all_preds_probs]

            acc = accuracy_score(all_labels_gt, binary_preds)

            # Average Precision (AP) richiede le probabilità, non le classi binarie
            # Gestisce il caso in cui non ci siano errori nel test set (AP indefinito)
            try:
                ap = average_precision_score(all_labels_gt, all_preds_probs)
                if np.isnan(ap): ap = 0.0
            except:
                ap = 0.0

            fold_accuracies.append(acc)
            fold_aps.append(ap)

            print(f"Recipe {test_recipe} -> Acc: {acc:.4f} | AP (Mistake): {ap:.4f}")
        else:
            print(f"Recipe {test_recipe} -> No valid steps found.")

    # Risultati Finali
    print("\n" + "=" * 40)
    if len(fold_accuracies) > 0:
        mean_acc = np.mean(fold_accuracies)
        mean_ap = np.mean(fold_aps)
        print(f"FINAL MEAN ACCURACY: {mean_acc:.4f}")
        print(f"FINAL MEAN AP (mAP): {mean_ap:.4f}")
    else:
        print("No results computed.")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy', required=True, help='Path to .npy embeddings')
    parser.add_argument('--annotations', required=True, help='Path to .json annotations')
    parser.add_argument('--ckpt_dir', default='checkpoints_step', help='Folder containing saved models')
    args = parser.parse_args()

    run_evaluation(args)

# esempio run
#python extension_task_verification/eval_step_detection.py --npy extension_localization/data/step_embeddings.npy --annotations extension_localization/data/step_annotations.json --ckpt_dir substep3_step_detection/checkpoints_step