# Piano di Pulizia e Refactoring Repository

**STATO: IN ESECUZIONE** - Refactoring completato per le fasi principali.

## Analisi Struttura Attuale vs Struttura Richiesta

### Struttura Attuale
```
aml-2025-mistake-detection/
├── core/                    ✅ Core baselines (MLP, Transformer, RNN)
├── dataloader/              ✅ Dataset loaders
├── annotations/             ✅ Dataset annotations
├── results/                 ✅ Evaluation results
│   ├── error_type_analysis/
│   ├── substep2_task_verification/
│   ├── substep3_task_graph_matching/
│   └── substep4_gnn_classification/
├── extension_localization/          ⚠️  Vecchia? (da verificare/rimuovere)
├── extension_localization_hiero/    ✅ Substep 1: Step localization (HiERO)
├── extension_task_verification/     ✅ Substep 2: Task verification baseline
├── substep3_step_detection/         ✅ Substep 3: Task graph matching
├── substep4/                        ✅ Substep 4: GNN classification
├── nohup_*.out (18 files)           ❌ File log temporanei
├── run_*.sh (3 files)               ⚠️  Script non documentati
├── traccia-aml.txt                  ⚠️  Dovrebbe essere in docs/
├── checkpoints_selected/            ⚠️  Non usato?
└── er_annotations/                  ⚠️  Non usato?
```

### Struttura Target (basata su traccia)

```
aml-2025-mistake-detection/
├── README.md                        # Setup e quickstart
├── REPORT.md                        # Report finale
├── requirements.txt                 # Dependencies
│
├── core/                            # Step 2: Baselines
│   ├── models/                     # MLP, Transformer, RNN
│   ├── evaluate.py
│   └── evaluate_by_error_type.py
│
├── dataloader/                     # Dataset loaders
├── annotations/                    # Dataset (non modificare)
│
├── extension/                      # Extension: Task Verification
│   ├── substep1_step_localization/
│   │   ├── README.md
│   │   └── data/
│   │
│   ├── substep2_task_verification/
│   │   ├── README.md
│   │   ├── train_task_verification_hiero.py
│   │   ├── eval_task_verification_hiero.py
│   │   └── model/
│   │
│   ├── substep3_task_graph_matching/
│   │   ├── README.md
│   │   ├── train_task_graph_matching_gpu.py
│   │   ├── eval_all_recipes.py
│   │   └── model/
│   │
│   └── substep4_gnn_classification/
│       ├── README.md
│       ├── prepare_graph_data.py
│       ├── train_gnn_classification.py
│       ├── eval_gnn_classification.py
│       └── model/
│
├── results/                        # Tutti i risultati
│   ├── baselines/                  # Step 2 results
│   └── extension/                  # Extension results
│
├── scripts/                        # Script utility
│   ├── train_baselines.sh
│   └── monitor_status.py
│
└── docs/                           # Documentazione
    └── traccia-aml.txt
```

---

## Piano di Azione

### Fase 1: Pulizia File Temporanei e Log

**Obiettivo**: Rimuovere file non necessari dal repository

**Azioni**:
1. ✅ Aggiungere `*.out` al `.gitignore`
2. ✅ Aggiungere `nohup*.out` al `.gitignore`
3. Rimuovere tutti i file `nohup_*.out` dal root (18 file)
4. Verificare e rimuovere altri file temporanei

**Files da rimuovere**:
- `nohup_LR1E3_*.out` (6 files)
- `nohup_NEW_*.out` (6 files)
- `nohup_OLD_*.out` (6 files)
- `nohup_mlp.out`, `nohup_rnn.out`, `nohup_transformer.out`

---

### Fase 2: Consolidamento Directory Extension

**Obiettivo**: Unificare struttura extension in directory unica

**Azioni**:
1. Verificare se `extension_localization/` è ancora usata
   - Se no: rimuoverla
   - Se sì: consolidarla con `extension_localization_hiero/`
2. Rinominare directory per coerenza:
   - `extension_localization_hiero/` → `extension/substep1_step_localization/`
   - `extension_task_verification/` → `extension/substep2_task_verification/`
   - `substep3_step_detection/` → `extension/substep3_task_graph_matching/`
   - `substep4/` → `extension/substep4_gnn_classification/`

**Note**: Aggiornare tutti gli import e path nei file Python

---

### Fase 3: Organizzazione Script e Utility

**Obiettivo**: Raggruppare script utility in directory dedicata

**Azioni**:
1. Creare directory `scripts/`
2. Spostare script shell:
   - `run_all_trainings.sh` → `scripts/train_baselines.sh`
   - `run_all_trainings_fast.sh` → `scripts/train_baselines_fast.sh`
   - `run_missing_trainings_lr1e3.sh` → `scripts/train_missing_lr1e3.sh`
   - `monitor_trainings.sh` → `scripts/monitor_trainings.sh`
3. Spostare utility Python:
   - `monitor_status.py` → `scripts/monitor_status.py`
4. Documentare script in `scripts/README.md`

---

### Fase 4: Organizzazione Documentazione

**Obiettivo**: Spostare documentazione in directory dedicata

**Azioni**:
1. Creare directory `docs/`
2. Spostare `traccia-aml.txt` → `docs/traccia-aml.txt`
3. Verificare altri file di documentazione da spostare

---

### Fase 5: Organizzazione Risultati

**Obiettivo**: Strutturare risultati in modo logico

**Azioni**:
1. Creare struttura:
   - `results/baselines/` (per Step 2)
   - `results/extension/` (per Extension)
2. Spostare risultati:
   - `results/error_type_analysis/` → `results/baselines/error_type_analysis/`
   - `results/substep2_task_verification/` → `results/extension/substep2_task_verification/`
   - `results/substep3_task_graph_matching/` → `results/extension/substep3_task_graph_matching/`
   - `results/substep4_gnn_classification/` → `results/extension/substep4_gnn_classification/`

---

### Fase 6: Rimozione Directory Non Utilizzate

**Obiettivo**: Rimuovere directory obsolete

**Azioni**:
1. Verificare uso di `checkpoints_selected/`
   - Se non usato: rimuovere
2. Verificare uso di `er_annotations/`
   - Se non usato: rimuovere
3. Verificare uso di `extension_localization/` (non _hiero)
   - Se non usato: rimuovere

---

### Fase 7: Aggiornamento README e Documentazione

**Obiettivo**: Aggiornare README con struttura finale

**Azioni**:
1. Aggiornare `README.md` con:
   - Struttura directory finale
   - Istruzioni per Step 2 (Baselines)
   - Istruzioni per Extension (Substep 1-4)
   - Path corretti per script e risultati
2. Verificare che ogni substep abbia `README.md` aggiornato
3. Creare `scripts/README.md` per documentare script utility

---

### Fase 8: Aggiornamento Import e Path

**Obiettivo**: Aggiornare tutti gli import dopo refactoring

**Azioni**:
1. Cercare tutti gli import relativi alle directory rinominate
2. Aggiornare path nei file Python:
   - `train_er.py`
   - Script di evaluation
   - Script di training extension
3. Aggiornare path nei file di configurazione
4. Testare che tutto funzioni dopo refactoring

---

## Ordine di Esecuzione

1. **Fase 1**: Pulizia file temporanei (semplice, nessun rischio)
2. **Fase 6**: Rimozione directory non usate (verifica prima)
3. **Fase 3**: Organizzazione script (low risk)
4. **Fase 4**: Organizzazione docs (low risk)
5. **Fase 5**: Organizzazione risultati (medium risk - aggiornare path)
6. **Fase 2**: Consolidamento extension (high risk - molti import da aggiornare)
7. **Fase 7**: Aggiornamento README
8. **Fase 8**: Aggiornamento import e test finale

---

## Checklist Pre-Refactoring

- [ ] Backup repository (branch o commit)
- [ ] Verificare che tutti i test/evaluation funzionino
- [ ] Identificare tutti i file che referenziano path vecchi
- [ ] Preparare lista di find/replace necessari

## Checklist Post-Refactoring

- [ ] Tutti gli script funzionano
- [ ] Tutti gli import corretti
- [ ] README aggiornato
- [ ] Test rapido: train baselines
- [ ] Test rapido: eval baselines
- [ ] Test rapido: extension substep 1-4
- [ ] Commit e push

---

## Note

- **Checkpoint**: Già gestiti in `.gitignore`, non da spostare
- **Data**: Directory `data/` già in `.gitignore`, non da toccare
- **Wandb**: Directory `wandb/` già in `.gitignore`, non da toccare
- **Annotations**: Directory `annotations/` non modificare (dataset originale)
