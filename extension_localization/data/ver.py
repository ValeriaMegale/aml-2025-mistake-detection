import numpy as np
import pandas as pd


def verify_output(npy_path, csv_path):
    print(f"--- VERIFICA FILE: {npy_path} ---")

    # 1. Caricamento
    try:
        # allow_pickle=True è necessario perché abbiamo salvato un dizionario
        data = np.load(npy_path, allow_pickle=True).item()
        print("✅ File caricato correttamente.")
    except Exception as e:
        print(f"❌ Errore caricamento file .npy: {e}")
        return

    # 2. Controllo Struttura Generale
    df = pd.read_csv(csv_path)
    csv_videos = set(df['video-id'].unique())
    npy_videos = set(data.keys())

    print(f"\nVideo nel CSV originale: {len(csv_videos)}")
    print(f"Video nel file Output:   {len(npy_videos)}")

    missing = csv_videos - npy_videos
    if missing:
        print(f"⚠️ Attenzione: {len(missing)} video mancano nell'output (probabilmente mancavano i file feature).")
    else:
        print("✅ Tutti i video del CSV sono presenti nell'output.")

    # 3. Analisi Approfondita del Contenuto
    total_steps = 0
    total_nans = 0
    shapes_found = set()

    # Prendiamo un video a caso per mostrare un esempio
    example_vid = list(data.keys())[0]

    for vid_id, steps in data.items():
        total_steps += len(steps)
        for item in steps:
            emb = item['embedding']

            # Raccogli shape
            shapes_found.add(emb.shape)

            # Cerca NaNs (Not a Number) o Infiniti
            if np.isnan(emb).any() or np.isinf(emb).any():
                total_nans += 1

    print(f"\n--- STATISTICHE CONTENUTO ---")
    print(f"Totale step processati: {total_steps}")
    print(f"Dimensioni embedding trovate: {shapes_found}")

    # VERDETTO DIMENSIONI
    if len(shapes_found) == 1 and (1024,) in shapes_found:
        print("✅ DIMENSIONI OK: Tutti gli embedding sono vettori di lunghezza 1024.")
    else:
        print(f"❌ ERRORE DIMENSIONI: Trovate forme incoerenti: {shapes_found}")

    # VERDETTO VALORI NUMERICI
    if total_nans == 0:
        print("✅ VALORI OK: Nessun NaN o Inf trovato.")
    else:
        print(f"❌ ERRORE VALORI: Trovati {total_nans} embedding corrotti (NaN/Inf).")

    # 4. Esempio di un dato
    print(f"\n--- ESEMPIO DI OUTPUT ({example_vid}) ---")
    if len(data[example_vid]) > 0:
        ex_step = data[example_vid][0]
        print(f"Step Label: {ex_step['label']}")
        print(f"Step Score: {ex_step['score']}")
        print(f"Embedding (primi 5 valori): {ex_step['embedding'][:5]}")
        print(f"Media valore embedding: {np.mean(ex_step['embedding']):.6f}")
    else:
        print("Questo video non ha step validi salvati.")


# --- ESEGUI ---
VERIFY_FILE = 'step_embeddings.npy'  # Il file che hai appena creato
ORIGINAL_CSV = '../libs/utils/model_outputs/gt.csv'  # Il tuo CSV originale

if __name__ == "__main__":
    verify_output(VERIFY_FILE, ORIGINAL_CSV)