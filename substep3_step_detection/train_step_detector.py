import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# Import del modello appena creato
try:
    from extension_task_verification.model.step_detector import StepMistakeDetector
except ImportError:
    from model.step_detector import StepMistakeDetector


# --- DATASET AGGIORNATO PER STEP-LEVEL ---
def load_annotations(json_path):
    with open(json_path, 'r') as f: return json.load(f)


def get_step_labels(video_id, steps, annotation_map):
    """
    Restituisce una lista di 0.0 (Normal) e 1.0 (Error) lunga quanto gli step.
    """
    labels = []
    for step in steps:
        lbl_id = str(step['label'])
        is_error = 0.0
        # Controlla flag 'has_errors'
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
                # ORA: label è un vettore [L], non un singolo scalare
                step_lbls = get_step_labels(vid, steps, annotation_map)
                lbl_seq = torch.tensor(step_lbls, dtype=torch.float32)

                self.samples.append((seq, lbl_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    seqs, lbls = zip(*batch)

    # Pad Features: [Batch, Max_Len, Dim]
    padded_seqs = pad_sequence(seqs, batch_first=True, padding_value=0)

    # Pad Labels: [Batch, Max_Len] - Usiamo -1 come valore di padding per le label
    padded_lbls = pad_sequence(lbls, batch_first=True, padding_value=-1)

    # Mask: True dove è padding (per il Transformer)
    lens = torch.tensor([len(x) for x in seqs])
    mask = torch.arange(padded_seqs.size(1))[None, :] >= lens[:, None]

    return padded_seqs, padded_lbls, mask


# --- TRAINING LOOP ---
def run_training(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print(f"Loading data from {args.npy}...")
    data_dict = np.load(args.npy, allow_pickle=True).item()
    annotation_map = load_annotations(args.annotations)

    all_videos = list(data_dict.keys())
    recipes = sorted(list(set([v.split('_')[0] for v in all_videos])))

    print(f"Training Step Detection (Mistake Identification) on {len(recipes)} folds.")

    for test_recipe in recipes:
        print(f"\n--- FOLD: Holding out Recipe {test_recipe} ---")
        train_ids = [v for v in all_videos if not v.startswith(f"{test_recipe}_")]

        train_ds = StepDetectionDataset(data_dict, train_ids, annotation_map)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        # Init Model
        model = StepMistakeDetector().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Loss: BCEWithLogitsLoss con reduction='none' per gestire il masking manualmente
        # oppure ignorando i valori specifici se supportato (ma per BCE standard meglio manuale)
        criterion = nn.BCEWithLogitsLoss(reduction='none')

        model.train()
        for epoch in range(args.epochs):
            epoch_loss = 0
            steps_count = 0

            for seqs, labels, mask in train_loader:
                seqs, labels, mask = seqs.to(device), labels.to(device), mask.to(device)
                # Labels shape: [Batch, Seq]
                # Logits shape: [Batch, Seq, 1] -> Squeeze a [Batch, Seq]

                optimizer.zero_grad()
                logits = model(seqs, mask).squeeze(-1)

                # Calcolo Loss
                raw_loss = criterion(logits, labels)  # [Batch, Seq]

                # Masking della Loss:
                # 'mask' è True dove c'è padding. Noi vogliamo validità (False -> 1, True -> 0)
                valid_mask = (~mask).float()

                masked_loss = raw_loss * valid_mask

                # Media solo sui token validi
                loss = masked_loss.sum() / (valid_mask.sum() + 1e-6)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # print(f"  Epoch {epoch+1} Loss: {epoch_loss:.4f}")

        # SALVATAGGIO
        ckpt_path = os.path.join(args.ckpt_dir, f"step_model_{test_recipe}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--ckpt_dir', default='checkpoints_step', help='Folder to save step models')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    run_training(args)

# esempio run
# python extension_task_verification/train_step_detection.py --npy extension_localization/data/step_embeddings.npy --annotations extension_localization/data/step_annotations.json --ckpt_dir substep3_step_detection/checkpoints_step --epochs 15