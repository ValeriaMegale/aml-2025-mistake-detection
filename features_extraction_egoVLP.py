import os
import sys
import torch
import argparse
import logging
import numpy as np
import gc
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


sys.path.append(os.path.join(os.getcwd(), 'EgoVLP'))

try:
    from model.model import FrozenInTime
    from utils.util import state_dict_data_parallel_fix
except ImportError:
    print("❌ ERRORE: Non trovo i moduli di EgoVLP.")
    print("   Assicurati di aver clonato il repo: git clone https://github.com/showlab/EgoVLP.git")
    print("   E che questo script sia nella cartella superiore rispetto a 'EgoVLP'.")
    sys.exit(1)

# Se decord da problemi su Windows, si può usare il fallback OpenCV,
# ma qui proviamo a mantenere decord per coerenza con il tuo setup precedente.
try:
    from decord import VideoReader, cpu
except ImportError:
    print("⚠️ Decord non installato. Installa con: pip install decord")

# Configurazione Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EgoVLPFeatureExtractor:
    def __init__(self, weights_path, device=None, num_frames=16):
        self.device = self._get_device(device)
        self.num_frames = num_frames

        logger.info(f"🔧 Inizializzazione EgoVLP su {self.device} con {num_frames} frames/clip...")

        # 1. Configurazione Modello (presa dal notebook originale)
        video_params = {
            "model": "SpaceTimeTransformer",
            "arch_config": "base_patch16_224",
            "num_frames": num_frames,
            "pretrained": True,
            "time_init": "zeros"
        }
        text_params = {
            "model": "distilbert-base-uncased",
            "pretrained": True,
            "input": "text"
        }

        # 2. Istanziazione
        self.model = FrozenInTime(video_params, text_params, projection_dim=256, projection='minimal',
                                  load_checkpoint=None)

        # 3. Caricamento Pesi
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Non trovo i pesi in: {weights_path}")

        logger.info(f"Caricamento checkpoint: {weights_path}")
        checkpoint = torch.load(weights_path, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        # Fix per nomi parametri se salvati in DataParallel
        state_dict = state_dict_data_parallel_fix(state_dict, self.model.state_dict())
        self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()

        # 4. Trasformazioni (Standard ImageNet Normalization)
        # EgoVLP si aspetta i tensori normalizzati
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_device(self, requested_device):
        if requested_device: return torch.device(requested_device)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run_inference_batch(self, batch_clips):
        """
        Input: [Batch, T, C, H, W] -> Clip video
        Output: [Batch, 256] -> Feature Vector
        """
        # Spostiamo i canali per matchare ciò che si aspetta EgoVLP se necessario
        # FrozenInTime di solito vuole [B, C, T, H, W]
        # batch_clips arriva come [B, T, C, H, W] dal DataLoader
        batch_clips = batch_clips.permute(0, 2, 1, 3, 4).float()  # -> [B, C, T, H, W]
        batch_clips = batch_clips.to(self.device, non_blocking=True)

        with torch.no_grad():
            # Nota: compute_video restituisce l'embedding proiettato
            output = self.model.compute_video(batch_clips)

        return output.cpu().numpy()


class ClipDataset(Dataset):
    def __init__(self, video_paths, transform, num_frames=16):
        self.video_paths = video_paths
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = vr.get_avg_fps()
            total_frames = len(vr)
            duration = total_frames / fps

            clips_list = []

            # Per ogni secondo, estraiamo una CLIP di 'num_frames' centrata
            for sec in range(int(np.ceil(duration))):
                center_frame = int((sec + 0.5) * fps)

                # Calcoliamo indici clip (es. 16 frame attorno al centro)
                start = max(0, center_frame - self.num_frames // 2)
                end = min(total_frames, start + self.num_frames)

                # Se siamo alla fine e mancano frame, torniamo indietro
                if end - start < self.num_frames and start > 0:
                    start = max(0, end - self.num_frames)

                indices = list(range(start, end))

                # Padding se il video è cortissimo (meno di 16 frame totali)
                while len(indices) < self.num_frames:
                    indices.append(indices[-1])

                # Estrazione buffer
                buffer = vr.get_batch(indices).asnumpy()  # [T, H, W, C]

                # Trasformazione: [T, H, W, C] -> [T, C, H, W] normalizzato
                processed_clip = []
                for frame in buffer:
                    img = Image.fromarray(frame)  # PIL
                    processed_clip.append(self.transform(img))

                # Stack temporale: [T, C, H, W]
                clips_list.append(torch.stack(processed_clip))

            # Stack finale del video: [N_Secondi, T, C, H, W]
            if not clips_list:
                return None, video_path

            return torch.stack(clips_list), video_path

        except Exception as e:
            logger.error(f"Errore lettura {os.path.basename(video_path)}: {e}")
            return None, video_path


def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch: return None
    return batch[0]  # Batch size 1 nel dataloader esterno


def main(args):
    # Setup cartelle
    if not os.path.exists(args.source):
        logger.error(f"Cartella sorgente non trovata: {args.source}")
        return
    os.makedirs(args.dest, exist_ok=True)

    # Inizializza Modello
    try:
        encoder = EgoVLPFeatureExtractor(
            weights_path=args.weights,
            device=args.device,
            num_frames=16  # EgoVLP standard usa 16 frame
        )
    except Exception as e:
        logger.error(f"Fallita inizializzazione modello: {e}")
        return

    # Lista Video
    video_list = []
    for root, _, files in os.walk(args.source):
        for file in files:
            if file.lower().endswith('.mp4'):
                vpath = os.path.join(root, file)
                out_name = f"{os.path.basename(vpath)}_egovlp.npz"
                if not os.path.exists(os.path.join(args.dest, out_name)):
                    video_list.append(vpath)

    logger.info(f"Trovati {len(video_list)} video da elaborare.")

    # Dataset & Loader
    dataset = ClipDataset(video_list, encoder.transform, num_frames=16)
    # num_workers=0 per stabilità su Windows
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    for data in tqdm(loader, desc="EgoVLP Processing"):
        if data is None: continue

        # clips shape: [N_Seconds, T, C, H, W]
        clips, vpath = data

        features_list = []
        try:
            # Batch inferenza: processiamo 'args.batch_size' Secondi alla volta
            # Ogni "elemento" del batch è in realtà una clip di 16 frame
            for i in range(0, len(clips), args.batch_size):
                batch_clips = clips[i: i + args.batch_size]
                emb = encoder.run_inference_batch(batch_clips)
                features_list.append(emb)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n⚠️ GPU OOM! Riduci --batch_size.")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

        if features_list:
            full_features = np.vstack(features_list)
            # Salvataggio
            out_name = f"{os.path.basename(vpath)}_egovlp.npz"
            np.savez_compressed(os.path.join(args.dest, out_name), features=full_features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EgoVLP Feature Extraction")
    parser.add_argument('--source', type=str, required=True, help='Cartella video input')
    parser.add_argument('--dest', type=str, required=True, help='Cartella output')
    parser.add_argument('--weights', type=str, required=True, help='Percorso file .pth di EgoVLP')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (attenzione: 4 clips = 4*16 frames!)')
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()
    main(args)