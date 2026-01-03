import argparse
import os
import pandas as pd
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
    """Crea mappa {video_id: path} gestendo i suffissi."""
    print(f"Indicizzazione cartella: {feat_folder}")
    if not os.path.exists(feat_folder):
        print("ERRORE: Cartella non trovata.")
        return {}

    files = os.listdir(feat_folder)
    # Cerca suffisso comune
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
    # --- VERSIONE OMNIVORE ---
    # feat_map = build_file_map('data/video/omnivore')
    # print("[OMNIVORE] Lettura CSV:", args.preds_csv)
    # df = pd.read_csv(args.preds_csv)
    # ...existing code...
    # np.save(args.output, all_step_embeddings)

    # --- VERSIONE PERCEPTION ---
    feat_map = build_file_map('data/video/perception')
    print("[PERCEPTION] Lettura CSV:", args.preds_csv)
    df = pd.read_csv(args.preds_csv)
    col_map = {
        'recording_id': 'video_id',
        'video-id': 'video_id',
        'start_time': 'start',
        't-start': 'start',
        'end_time': 'end',
        't-end': 'end',
        'step_id': 'label',
        'label': 'label'
    }
    df.rename(columns=col_map, inplace=True)
    df.columns = df.columns.str.strip()
    required = ['video_id', 'start', 'end']
    if not all(col in df.columns for col in required):
        print(f"ERRORE: Colonne mancanti. Trovate: {df.columns}. Servono: {required}")
        return

    # Se è il file annotations, non filtriamo per score (non c'è)
    if 'score' in df.columns:
        print(f"Filtro score >= {args.threshold}...")
        df = df[df['score'] >= args.threshold]
    else:
        print("Modalità Ground Truth (nessuno score da filtrare).")

    all_step_embeddings = {}
    current_vid = None
    current_feats = None

    # 3. Estrazione
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        video_id = str(row['video_id']).strip()

        if video_id != current_vid:
            current_vid = video_id
            feat_path = feat_map.get(video_id)
            # Tentativi di fallback (- vs _)
            if not feat_path: feat_path = feat_map.get(video_id.replace('-', '_'))
            if not feat_path: feat_path = feat_map.get(video_id.replace('_', '-'))

            if feat_path:
                current_feats = load_features_from_npz(feat_path)
            else:
                current_feats = None

        if current_feats is None: continue

        # Estrazione temporale
        start_frame = int(row['start'] * args.fps)
        end_frame = int(row['end'] * args.fps)
        s_idx = max(0, start_frame // args.feat_stride)
        e_idx = max(0, end_frame // args.feat_stride)

        n_feats = current_feats.shape[0]
        s_idx = min(s_idx, n_feats - 1)
        e_idx = min(e_idx, n_feats)

        grid = current_feats[s_idx:e_idx]
        if grid.shape[0] == 0: grid = current_feats[s_idx:s_idx + 1]

        step_emb = np.mean(grid, axis=0)

        entry = {
            'video_id': video_id,
            'start': row['start'],
            'end': row['end'],
            'label': row['label'],
            'embedding': step_emb
        }
        # Aggiungiamo info extra utili per il task verification
        if 'has_errors' in row:
            entry['has_errors'] = row['has_errors']

        if video_id not in all_step_embeddings: all_step_embeddings[video_id] = []
        all_step_embeddings[video_id].append(entry)

    # 4. Output
    print(f"\nSalvataggio {len(all_step_embeddings)} video in {args.output}")
    if len(all_step_embeddings) > 0:
        np.save(args.output, all_step_embeddings)
    else:
        print("Nessun video salvato. Controlla il path delle features.")
PATH= '../extension_localization/data/step_embeddings.npy'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_csv', required=True)
    parser.add_argument('--feat_folder', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--feat_stride', type=int, default=30)
    parser.add_argument('--threshold', type=float, default=0.1)
    args = parser.parse_args()
    main(args)

#esempio run python extension_localization/step_embeddings.py --preds_csv extension_localization/data/libs/model_outputs/preds.csv --feat_folder data/video/omnivore --output extension_localization/data/step_embeddings.npy --threshold 0.15