import pandas as pd
import numpy as np
import os


def compute_step_embeddings(preds_csv_path, features_folder, output_path, fps=30, feat_stride=30):
    """
    Calcola l'embedding per ogni step.
    Specifico per file .npz con chiave 'arr_0' e shape (Tempo, Canali).
    """

    # 1. Carica le predizioni
    print(f"Lettura CSV: {preds_csv_path}")
    df = pd.read_csv(preds_csv_path)

    all_step_embeddings = {}
    unique_videos = df['video-id'].unique()

    print(f"In elaborazione {len(unique_videos)} video...")

    processed_count = 0
    missing_count = 0

    for video_id in unique_videos:
        # 2. Costruzione nome file (formato confermato dall'utente)
        filename = f"{video_id}_360p.mp4_1s_1s.npz"
        feat_file = os.path.join(features_folder, filename)

        if not os.path.exists(feat_file):
            print(f"MANCANTE: {filename}")
            missing_count += 1
            continue

        try:
            # 3. Caricamento .npz usando la chiave 'arr_0'
            with np.load(feat_file) as data:
                features = data['arr_0']

            # Assicuriamoci sia float32
            features = features.astype(np.float32)

        except Exception as e:
            print(f"Errore lettura {filename}: {e}")
            continue

        # NOTA: Abbiamo confermato che la shape è (Tempo, Canali) = (604, 1024).
        # NON facciamo trasposizione automatica basata sulle dimensioni.

        # 4. Loop sugli step del video
        video_preds = df[df['video-id'] == video_id]
        embeddings_list = []

        for index, row in video_preds.iterrows():
            t_start = row['t-start']
            t_end = row['t-end']

            # Calcolo indici (slicing temporale)
            # Con fps=30 e stride=30, il fattore è 1.0 (1 feature al secondo)
            start_idx = int(np.floor((t_start * fps) / feat_stride))
            end_idx = int(np.ceil((t_end * fps) / feat_stride))

            # Controlli per non uscire dalla matrice
            start_idx = max(0, start_idx)
            end_idx = min(features.shape[0], end_idx)

            # Se lo step è troppo breve, prendiamo almeno 1 frame
            if end_idx <= start_idx:
                end_idx = start_idx + 1

            if start_idx >= features.shape[0]:
                continue

                # 5. Estrazione e Media (Average Pooling)
            # Slicing sulla prima dimensione (Tempo)
            step_feats = features[start_idx:end_idx, :]

            if step_feats.shape[0] == 0:
                continue

            # Media lungo l'asse 0 (tempo) per ottenere un vettore (1024,)
            step_embedding = np.mean(step_feats, axis=0)

            embeddings_list.append({
                'row_id': index,
                'video_id': video_id,
                'label': row['label'],
                'score': row['score'],
                'embedding': step_embedding
            })

        all_step_embeddings[video_id] = embeddings_list
        processed_count += 1

    # 6. Salva risultati
    print(f"\nOperazione completata!")
    print(f"Video processati correttamente: {processed_count}")
    print(f"Video mancanti: {missing_count}")

    np.save(output_path, all_step_embeddings)
    print(f"File salvato in: {output_path}")

PREDS_FILE = 'libs/utils/model_outputs/preds.csv'
FEAT_FOLDER = '..\data\\video\omnivore'
OUTPUT_FILE = 'data/step_embeddings.npy'

# Parametri standard (1 feature al secondo)
FPS = 30
STRIDE = 30

if __name__ == "__main__":
    compute_step_embeddings(PREDS_FILE, FEAT_FOLDER, OUTPUT_FILE, fps=FPS, feat_stride=STRIDE)