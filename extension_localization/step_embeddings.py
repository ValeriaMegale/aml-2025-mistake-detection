import pandas as pd
import numpy as np
import os


def compute_step_embeddings(preds_csv_path, features_folder, output_path, fps=30, feat_stride=30):
    """
    Calcola l'embedding per ogni step caricando file .npz.
    Gestisce lunghezze temporali variabili ancorandosi alla feature_dim fissa (1024).
    """

    # Dimensione feature fissa nota (Omnivore/I3D solitamente 1024)
    KNOWN_FEATURE_DIM = 1024

    print(f"Lettura CSV: {preds_csv_path}")
    df = pd.read_csv(preds_csv_path)

    all_step_embeddings = {}
    unique_videos = df['video-id'].unique()

    print(f"In elaborazione {len(unique_videos)} video...")

    missing_files = 0
    processed_files = 0

    for video_id in unique_videos:
        # 1. Costruzione nome file
        filename = f"{video_id}_360p.mp4_1s_1s.npz"
        feat_file = os.path.join(features_folder, filename)

        if not os.path.exists(feat_file):
            print(f"⚠️ File non trovato: {filename}")
            missing_files += 1
            continue

        try:
            # 2. Caricamento .npz
            with np.load(feat_file) as data:
                keys = list(data.files)  # .files è l'attributo corretto per vedere le chiavi

                # Cerchiamo la chiave delle feature
                if 'features' in keys:
                    features = data['features']
                elif 'arr_0' in keys:
                    features = data['arr_0']
                else:
                    # Fallback: prendiamo la prima chiave che ha senso (non vuota)
                    features = data[keys[0]]

                # Assicuriamoci che sia float32
                features = features.astype(np.float32)

        except Exception as e:
            print(f"❌ Errore leggendo {filename}: {e}")
            continue

        # 3. Gestione Dimensioni (CRUCIALE: Ancoraggio a 1024)

        # Caso A: Feature Spaziali (T, H, W, C) o simili -> Flatten spaziale
        if features.ndim > 2:
            # Media su tutte le dimensioni tranne la prima (Tempo) e l'ultima (Canali)
            # o semplice media spaziale se shape è (T, H, W, C)
            # Qui assumiamo che l'ultima dimensione sia 1024.
            # Se è (1, 1024, T) è un caso raro, ma gestiamo il caso standard (T, C, H, W) -> mean
            features = features.mean(axis=tuple(range(1, features.ndim - 1)))

            # Caso B: Shape 2D (T, C) o (C, T)
        if features.ndim == 2:
            rows, cols = features.shape

            if cols == KNOWN_FEATURE_DIM:
                # Shape è (T, 1024) -> OK, non toccare
                pass
            elif rows == KNOWN_FEATURE_DIM:
                # Shape è (1024, T) -> Trasponi
                features = features.T
            else:
                # Se nessuna dimensione è 1024, c'è un problema nel file o nel modello
                print(f"⚠️ Shape anomala per {filename}: {features.shape}. Atteso dim {KNOWN_FEATURE_DIM}.")
                # (Opzionale) Possiamo provare a indovinare che la dim più grande è il tempo
                if rows < cols:
                    features = features.T

        # Controllo finale post-processing
        if features.shape[1] != KNOWN_FEATURE_DIM:
            print(f"❌ Errore critico dimensione feature per {video_id}: {features.shape}. Salto.")
            continue

        processed_files += 1

        # 4. Estrazione segmenti (Loop sugli step)
        video_preds = df[df['video-id'] == video_id]
        embeddings_list = []

        max_time_steps = features.shape[0]

        for index, row in video_preds.iterrows():
            t_start = row['t-start']
            t_end = row['t-end']

            # Conversione secondi -> indici
            # Nota: feat_stride=30 con fps=30 significa 1 feature al secondo.
            # Formula: (secondi * fps) / stride
            start_idx = int(np.floor((t_start * fps) / feat_stride))
            end_idx = int(np.ceil((t_end * fps) / feat_stride))

            # Clamp degli indici (non andare sotto 0 o oltre la fine del file)
            start_idx = max(0, start_idx)
            end_idx = min(max_time_steps, end_idx)

            # Se il segmento è non valido (start >= end), proviamo a prendere almeno 1 frame
            if start_idx >= end_idx:
                if start_idx < max_time_steps:
                    end_idx = start_idx + 1
                else:
                    # Il timestamp del CSV va oltre la durata del video estratto
                    continue

            # Slice
            step_feats = features[start_idx:end_idx, :]

            # Average Pooling temporale per ottenere un solo vettore per lo step
            if step_feats.shape[0] > 0:
                step_embedding = np.mean(step_feats, axis=0)

                embeddings_list.append({
                    'row_id': index,
                    'video_id': video_id,
                    'label': row['label'],
                    'score': row['score'],
                    'embedding': step_embedding
                })

        all_step_embeddings[video_id] = embeddings_list

    # 5. Salvataggio
    print(f"\nFinito! Processati {processed_files} video. {missing_files} mancanti.")

    # Crea cartella output se non esiste
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Salviamo
    np.save(output_path, all_step_embeddings)
    print(f"Output salvato in: {output_path}")


# --- CONFIGURAZIONE ---
# Uso r'' per stringhe raw (gestisce meglio i backslash di Windows)
PREDS_FILE = r'libs/utils/model_outputs/preds.csv'
FEAT_FOLDER = r'..\data\video\omnivore'
OUTPUT_FILE = r'data/step_embeddings.npy'

FPS = 30
STRIDE = 30

if __name__ == "__main__":
    if os.path.exists(FEAT_FOLDER):
        compute_step_embeddings(PREDS_FILE, FEAT_FOLDER, OUTPUT_FILE, fps=FPS, feat_stride=STRIDE)
    else:
        print(f"Attenzione: Cartella '{FEAT_FOLDER}' non trovata.")