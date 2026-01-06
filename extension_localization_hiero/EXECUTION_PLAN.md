# Piano di Esecuzione - Step Localization HiERO

## 📋 Checklist Completa

### FASE 1: Setup e Verifica Dipendenze

- [ ] **Step 1.1**: Verifica che conda environment `aml` sia attivo
  ```bash
  conda activate aml
  python --version  # Dovrebbe essere 3.11
  ```

- [ ] **Step 1.2**: Installa dipendenze mancanti
  ```bash
  pip install scikit-learn scipy pyyaml matplotlib
  ```

- [ ] **Step 1.3**: Verifica che le features perception esistano
  ```bash
  ls data/video/perception/*.npz | wc -l  # Dovrebbe mostrare numero di file
  ls data/video/perception/*.npz | head -3  # Verifica formato nomi file
  ```

---

### FASE 2: Test su Singolo Video

- [ ] **Step 2.1**: Test import del modulo
  ```bash
  cd /home/pierpaolosorbellini/AML/aml-2025-mistake-detection
  python -c "import extension_localization_hiero; print('OK')"
  ```
  **Se funziona**: ✅ Procedi al passo successivo  
  **Se errore**: Controlla che sklearn sia installato

- [ ] **Step 2.2**: Test su un singolo video
  ```bash
  python extension_localization_hiero/test_single_video.py --video_id 10_16
  ```
  **Cosa aspettarsi**:
  - Carica features
  - Mostra shape features
  - Esegue clustering
  - Mostra segmenti trovati
  - Calcola embeddings
  
  **Se funziona**: ✅ Procedi alla Fase 3  
  **Se errore**: Controlla log, probabilmente problema con sklearn o formato features

---

### FASE 3: Test su Piccolo Batch

- [ ] **Step 3.1**: Test su 5-10 video (split test)
  ```bash
  python extension_localization_hiero/run_step_localization.py \
      --split test \
      --output_segments extension_localization_hiero/data/step_segments_test.json \
      --output_embeddings extension_localization_hiero/data/step_embeddings_test.npy
  ```
  **Cosa aspettarsi**:
  - Processa video del test set
  - Salva segments JSON e embeddings NPY
  - Mostra summary con statistiche
  
  **Se funziona**: ✅ Procedi alla Fase 4  
  **Se errore**: Controlla:
    - Path features corretto
    - Formato file features
    - Permessi scrittura cartella data/

---

### FASE 4: Validazione Risultati

- [ ] **Step 4.1**: Valida risultati test
  ```bash
  python extension_localization_hiero/validate_step_localization.py \
      --segments extension_localization_hiero/data/step_segments_test.json \
      --embeddings extension_localization_hiero/data/step_embeddings_test.npy \
      --compare_gt \
      --plot
  ```
  **Cosa aspettarsi**:
  - Statistiche su numero segmenti per video
  - Statistiche su durata segmenti
  - Confronto con ground truth (se disponibile)
  - Plot distribuzioni (se --plot)
  
  **Verifica**:
  - Numero segmenti ragionevole (non 0, non 500)
  - Durata segmenti ragionevole (2-30 secondi tipicamente)
  - Embeddings hanno shape corretta [N, 768]

---

### FASE 5: Processing Completo

- [ ] **Step 5.1**: Processa tutti i video (split 'all')
  ```bash
  python extension_localization_hiero/run_step_localization.py \
      --split all \
      --output_segments extension_localization_hiero/data/step_segments_perception.json \
      --output_embeddings extension_localization_hiero/data/step_embeddings_perception.npy
  ```
  **Tempo stimato**: 10-30 minuti (dipende da numero video)
  
  **Monitoraggio**: Guarda output per:
  - Progress bar
  - Errori su singoli video
  - Summary finale

- [ ] **Step 5.2**: Validazione risultati completi
  ```bash
  python extension_localization_hiero/validate_step_localization.py \
      --segments extension_localization_hiero/data/step_segments_perception.json \
      --embeddings extension_localization_hiero/data/step_embeddings_perception.npy \
      --compare_gt \
      --plot
  ```

---

### FASE 6: Fine-tuning Parametri (Opzionale)

Se i risultati non sono soddisfacenti, modifica `config_step_localization.yaml`:

- [ ] **Step 6.1**: Aggiusta `min_segment_duration` (default: 2.0)
  - Troppi segmenti corti? → Aumenta (es. 3.0)
  - Troppi segmenti persi? → Diminuisci (es. 1.5)

- [ ] **Step 6.2**: Aggiusta `max_segments_per_video` (default: 50)
  - Troppi segmenti? → Diminuisci (es. 30)
  - Troppo pochi? → Aumenta (es. 100)

- [ ] **Step 6.3**: Prova `clustering_distance` (default: 'cosine')
  - Prova 'euclidean' se 'cosine' non funziona bene

- [ ] **Step 6.4**: Rilancia con nuovi parametri
  ```bash
  python extension_localization_hiero/run_step_localization.py --split test
  ```

---

## 🚨 Troubleshooting

### Problema: "ModuleNotFoundError: No module named 'sklearn'"
**Soluzione**: 
```bash
conda activate aml
pip install scikit-learn
```

### Problema: "No feature files found"
**Soluzione**: Verifica path features
```bash
ls data/video/perception/*.npz | head -5
# Se non trova, controlla path in config o usa --feat_folder
```

### Problema: "Empty segments for all videos"
**Soluzione**: 
- Controlla che features non siano vuote
- Diminuisci `min_segment_duration` in config
- Verifica formato features (dovrebbero essere [T, 768])

### Problema: Clustering troppo lento
**Soluzione**: 
- Usa downsampling features (modifica `feat_stride` o downsample features prima)
- Processa split più piccoli (train/val/test separati)

---

## 📊 Output Attesi

### Files Generati:
- `extension_localization_hiero/data/step_segments_perception.json`
- `extension_localization_hiero/data/step_embeddings_perception.npy`

### Statistiche Tipiche:
- **Numero segmenti per video**: 5-20 (dipende da durata video)
- **Durata segmenti**: 2-15 secondi (media ~5-8 secondi)
- **Embedding dimension**: 768 (perception encoder)

---

## ✅ Criteri di Successo

Il sistema funziona correttamente se:
1. ✅ Import modulo senza errori
2. ✅ Test singolo video produce segmenti ragionevoli
3. ✅ Batch processing completa senza crash
4. ✅ Statistiche non degenerate (non 0 segmenti, non 500 segmenti)
5. ✅ Embeddings hanno shape corretta [N, 768]
6. ✅ Segmenti ordinati temporalmente
7. ✅ Nessun overlap eccessivo tra segmenti

---

## 🎯 Prossimo Step Dopo Completamento

Una volta completato Substep 1, puoi procedere a:
- **Substep 2**: Task verification baselines usando gli step embeddings generati
- Gli embeddings sono pronti in formato NPY per essere usati nei modelli downstream
