import argparse
import os
import numpy as np
from tqdm import tqdm

def load_features_from_npz(path):
    try:
        with np.load(path, allow_pickle=True) as data:
            for key in ['features', 'feats', 'embedding', 'arr_0', 'data']:
                if key in data:
                    return data[key]
            if len(data.files) > 0:
                return data[data.files[0]]
    except Exception as e:
        return None
    return None

def build_file_map(feat_folder):
    print(f"Indicizzazione cartella: {feat_folder}")
    if not os.path.exists(feat_folder):
        print("ERRORE: Cartella non trovata.")
        return {}
    files = os.listdir(feat_folder)
    suffix = ""
    for f in files:
        if f.endswith('.npz'):
            if '_360p.mp4_1s_1s.npz' in f:
                suffix = '_360p.mp4_1s_1s.npz'
            elif '_360p_1s_1s.mp4.npz' in f:
                suffix = '_360p_1s_1s.mp4.npz'
            elif '.npz' in f:
                suffix = '.npz'
            break
    print(f"Suffisso rilevato: '{suffix}'")
    file_map = {}
    for filename in files:
        if filename.endswith(suffix):
            vid_id = filename.replace(suffix, "")
            file_map[vid_id] = os.path.join(feat_folder, filename)
    print(f"File mappati: {len(file_map)} (es. {list(file_map.keys())[:3]})")
    return file_map

def main(args):
    feat_map = build_file_map(args.feat_folder)
    if not feat_map: return
    all_step_embeddings = {}
    for vid, path in tqdm(feat_map.items()):
        feats = load_features_from_npz(path)
        if feats is None:
            continue
        # Ogni video: lista di step, qui 1 step = tutto il video
        entry = {
            'video_id': vid,
            'embedding': np.mean(feats, axis=0)
        }
        all_step_embeddings[vid] = [entry]
    print(f"\nSalvataggio {len(all_step_embeddings)} video in {args.output}")
    if len(all_step_embeddings) > 0:
        np.save(args.output, all_step_embeddings)
    else:
        print("Nessun video salvato. Controlla il path delle features.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_folder', required=True, help='Cartella perception features')
    parser.add_argument('--output', required=True, help='File output .npy')
    args = parser.parse_args()
    main(args)

# Esempio:
# python prepare_perception_embeddings.py --feat_folder ../data/video/perception --output step_embeddings_perception.npy
