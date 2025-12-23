import os
import argparse
import logging
import numpy as np
import torch
import timm
from PIL import Image
from decord import VideoReader, cpu
from tqdm import tqdm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import Dataset, DataLoader

# Configurazione Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SemanticVideoEncoder:
    def __init__(self, model_name='vit_pe_core_base_patch16_224.fb', device=None):
        self.device = self._get_device(device)
        logger.info(f"Inizializzazione modello {model_name} su {self.device}...")

        # Caricamento Backbone
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model = self.model.to(self.device).eval()

        # Trasformazioni
        config = resolve_data_config({}, model=self.model)
        self.preprocess = create_transform(**config)

    def _get_device(self, requested_device):
        if requested_device: return torch.device(requested_device)
        if torch.cuda.is_available(): return torch.device('cuda')
        return torch.device('cpu')

    def run_inference_batch(self, batch_frames):
        """Inferenza ottimizzata con Mixed Precision."""
        # batch_frames arriva come [Batch, Channels, Height, Width]
        batch_frames = batch_frames.to(self.device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                output = self.model(batch_frames)

        return output.cpu().numpy()


# --- CLASSE DATASET PER PARALLELISMO ---
class VideoDataset(Dataset):
    def __init__(self, video_paths, preprocess_fn):
        self.video_paths = video_paths
        self.preprocess = preprocess_fn

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        try:
            # Lettura video (bottleneck CPU)
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = vr.get_avg_fps()
            num_frames = len(vr)
            duration = num_frames / fps

            # 1 frame al secondo
            indices = [int(min((sec + 0.5) * fps, num_frames - 1)) for sec in range(int(duration))]

            if not indices:
                return None, video_path

            # Decord batch reading (più veloce del loop)
            raw_frames = vr.get_batch(indices).asnumpy()

            # Preprocessing
            tensor_list = [self.preprocess(Image.fromarray(f)) for f in raw_frames]
            video_tensor = torch.stack(tensor_list)

            return video_tensor, video_path

        except Exception as e:
            print(f"Errore lettura {video_path}: {e}")
            return None, video_path


def collate_fn(batch):
    # Filtra eventuali None dovuti a errori di lettura
    batch = [item for item in batch if item[0] is not None]
    if not batch: return None
    return batch[0]  # Ritorna (tensor, path) del singolo video (batch_size=1 nel dataloader)


def main(args):
    if not os.path.exists(args.source):
        logger.error(f"Cartella non trovata: {args.source}")
        return
    os.makedirs(args.dest, exist_ok=True)

    encoder = SemanticVideoEncoder(device=args.device)

    # Raccolta file
    video_list = []
    for root, _, files in os.walk(args.source):
        for file in files:
            if file.lower().endswith('.mp4'):
                vpath = os.path.join(root, file)
                # Check resume rapido
                out_name = f"{os.path.basename(vpath)}_1s_1s.npz"
                if not os.path.exists(os.path.join(args.dest, out_name)):
                    video_list.append(vpath)

    logger.info(f"Video da elaborare: {len(video_list)}")

    # Dataset & DataLoader (Il segreto della velocità)
    # batch_size=1 perché ogni "item" è un intero video (che contiene N frame)
    # num_workers=2: Due processi CPU caricano i video mentre la GPU ne elabora un altro
    dataset = VideoDataset(video_list, encoder.preprocess)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    for data in tqdm(loader, desc="Processing"):
        if data is None: continue

        frames, vpath = data

        # Elaborazione a batch sulla GPU
        features_list = []
        for i in range(0, len(frames), args.batch_size):
            batch = frames[i: i + args.batch_size]
            emb = encoder.run_inference_batch(batch)
            features_list.append(emb)

        if features_list:
            full_features = np.vstack(features_list)
            out_name = f"{os.path.basename(vpath)}_1s_1s.npz"
            np.savez_compressed(os.path.join(args.dest, out_name), features=full_features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--dest', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size GPU (prova 16 o 32)")
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    main(args)

import torch;
print(f'GPU Disponibile: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nessuno"}')