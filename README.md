# AML/DAAI 2025 - Mistake Detection Project

Progetto per la rilevazione di errori in attività procedurali utilizzando il dataset CaptainCook4D.

## Setup Iniziale

### 1. Ambiente Virtuale

```bash
python -m venv .venv
source .venv/bin/activate  # Su Linux/Mac
pip install -r requirements.txt
```

### 2. Dati

Scarica le feature pre-estratte per segmenti di 1 secondo e posizionale nella directory `data/features`.

Scarica i checkpoint ufficiali migliori da [qui](https://utdallas.app.box.com/s/uz3s1alrzucz03sleify8kazhuc1ksl3) (directory `error_recognition_best`) e posizionali in `checkpoints/error_recognition_best/`.

---

## Parte 1: Error Recognition (Task Principale)

L'obiettivo è rilevare errori a livello di step o recording utilizzando modelli di classificazione binaria.

### Architetture Disponibili

- **MLP**: Multi-Layer Perceptron con aggregazione delle feature
- **Transformer**: ErFormer con attention temporale
- **RNN**: LSTM per catturare dipendenze temporali

### Training

#### Training Singolo Modello

```bash
python train_er.py \
    --variant MLP \
    --backbone omnivore \
    --split step \
    --modality video \
    --num_epochs 100 \
    --lr 0.0005 \
    --weight_decay 0.005 \
    --batch_size 128
```

**Parametri principali:**
- `--variant`: `MLP`, `RNN`, o `Transformer`
- `--backbone`: `omnivore` (1024-dim) o `perception` (768-dim)
- `--split`: `step`, `recordings`, `person`, `environment`
- `--modality`: `video` (default)
- `--num_epochs`: numero di epoche (default: 100)
- `--lr`: learning rate (default: 0.0005)
- `--weight_decay`: weight decay (default: 0.005)
- `--batch_size`: dimensione batch (default: 128)

#### Training Automatico (Tutti i Modelli)

Per eseguire tutti i training in parallelo:

```bash
# Versione standard (2 training paralleli)
./scripts/run_all_trainings.sh

# Versione veloce (3 training paralleli)
./scripts/run_all_trainings_fast.sh

# Monitoraggio progressi
./scripts/monitor_trainings.sh
```

I checkpoint vengono salvati in `checkpoints/error_recognition/{VARIANT}/{BACKBONE}/`.

### Valutazione

#### Valutazione Globale

```bash
python -m core.evaluate \
    --variant MLP \
    --backbone omnivore \
    --ckpt checkpoints/error_recognition_best/MLP/omnivore/error_recognition_MLP_omnivore_step_epoch_43.pt \
    --split step \
    --threshold 0.6
```

**Soglie consigliate:**
- `step` split: `0.6`
- `recordings` split: `0.4`

#### Valutazione per Tipo di Errore

```bash
python -m core.evaluate_by_error_type \
    --split step \
    --backbone omnivore \
    --variant MLP \
    --ckpt checkpoints/error_recognition/MLP/omnivore/error_recognition_MLP_omnivore_step_epoch_35.pt
```

Questo comando genera metriche dettagliate per ogni categoria di errore:
- Technique Error
- Preparation Error
- Temperature Error
- Measurement Error
- Timing Error
- No Error (Normal)

### Risultati Attesi

**Step Split (Omnivore):**

| Modello | Accuracy | F1 | AUC | Precision | Recall |
|---------|----------|----|----|-----------|--------|
| MLP | 67.29% | 0.51 | 0.68 | 0.48 | 0.55 |
| Transformer | 50.13% | 0.51 | 0.65 | 0.37 | 0.82 |
| RNN | 61.15% | **0.54** | 0.68 | 0.43 | 0.72 |

---

## Parte 2: Extension - Task Verification

Estensione che verifica se un'intera ricetta è stata eseguita correttamente analizzando video e task graph congiuntamente.

### Pipeline Completa

```
Video → Step Localization → Step Embeddings → Task Graph Matching → Graph Realization → Classification
```

### Substep 1: Step Localization

Segmentazione temporale dei video in step utilizzando clustering gerarchico (stile HiERO).

**Esecuzione:**

```bash
python extension/substep1_step_localization/run_step_localization.py \
    --config extension/substep1_step_localization/config_step_localization.yaml \
    --feat_folder data/video/perception \
    --split all \
    --output_segments extension/substep1_step_localization/data/step_segments_perception.json \
    --output_embeddings extension/substep1_step_localization/data/step_embeddings_perception.npy
```

**Validazione:**

```bash
python extension/substep1_step_localization/validate_step_localization.py \
    --segments extension/substep1_step_localization/data/step_segments_perception.json \
    --embeddings extension/substep1_step_localization/data/step_embeddings_perception.npy \
    --compare_gt \
    --plot
```

**Output:**
- `step_segments_perception.json`: segmenti temporali per ogni video
- `step_embeddings_perception.npy`: embeddings aggregati per step (768-dim)

### Substep 2: Simple Task Verification Baseline

Baseline Transformer per classificazione binaria a livello di ricetta.

**Training (Leave-One-Out):**

```bash
python extension/substep2_task_verification/train_task_verification_hiero.py \
    --npy extension/substep1_step_localization/data/step_embeddings_perception.npy \
    --annotations annotations/annotation_json/step_annotations.json \
    --ckpt_dir extension/substep2_task_verification/checkpoints_hiero \
    --epochs 15 \
    --batch_size 8 \
    --lr 1e-4
```

**Valutazione:**

```bash
python extension/substep2_task_verification/eval_task_verification_hiero.py \
    --npy extension/substep1_step_localization/data/step_embeddings_perception.npy \
    --annotations annotations/annotation_json/step_annotations.json \
    --ckpt_dir extension/substep2_task_verification/checkpoints_hiero
```

**Risultati Attesi:**
- Accuracy: 0.5855 ± 0.1431
- F1: 0.6747
- AUC: 0.5656

### Substep 3: Task Graph Matching

Matching tra step visivi e nodi del task graph utilizzando algoritmo ungherese.

**Preparazione Embeddings:**

```bash
python extension/substep3_task_graph_matching/prepare_perception_embeddings.py \
    --feat_folder data/video/perception \
    --output extension/substep3_task_graph_matching/step_embeddings_perception.npy
```

**Training (GPU - Consigliato):**

```bash
python extension/substep3_task_graph_matching/train_task_graph_matching_gpu.py \
    --npy extension/substep1_step_localization/data/step_embeddings_perception.npy \
    --annotations annotations/annotation_json/step_annotations.json \
    --epochs 30 \
    --ckpt_dir extension/substep3_task_graph_matching/checkpoints_graph \
    --batch_size 8 \
    --lr 1e-4
```

**Valutazione:**

```bash
python extension/substep3_task_graph_matching/eval_task_graph_matching.py \
    --checkpoint extension/substep3_task_graph_matching/checkpoints_graph/task_graph_matcher_recipe_1.pth \
    --test_recipe 1 \
    --output results/task_graph_eval.json
```

**Risultati Attesi (10 ricette):**
- Accuracy: 0.5395 ± 0.1472
- F1: 0.5074 ± 0.3130
- AUC: 0.5565 ± 0.1088

### Substep 4: GNN Classification

Classificazione basata su Graph Neural Network (DAGNN) sul task graph realizzato.

**Preparazione Dati:**

```bash
python extension/substep4_gnn_classification/prepare_graph_data.py \
    --step_embeddings_npy extension/substep1_step_localization/data/step_embeddings_perception.npy \
    --output_dir extension/substep4_gnn_classification/graph_classification_substep4/data \
    --feature_dim 1280 \
    --test_recipes "1,2,3"  # Opzionale
```

**Training:**

```bash
python extension/substep4_gnn_classification/train_gnn_classification.py \
    --data_dir extension/substep4_gnn_classification/graph_classification_substep4/data \
    --ckpt_dir extension/substep4_gnn_classification/checkpoints_gnn \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-3 \
    --hidden_channels 128 \
    --K 10 \
    --dropout 0.3 \
    --early_stop
```

**Valutazione:**

```bash
python extension/substep4_gnn_classification/eval_gnn_classification.py \
    --ckpt_dir extension/substep4_gnn_classification/checkpoints_gnn \
    --data_dir extension/substep4_gnn_classification/graph_classification_substep4/data \
    --output_dir results/substep4_gnn_classification \
    --batch_size 32
```

**Risultati Attesi (24 ricette):**
- Accuracy: 0.5748 ± 0.1102
- F1: 0.7234 ± 0.0949
- Recall: 1.0000 (perfetto)

---

## Workflow Completo

### 1. Error Recognition (Task Principale)

```bash
# Training
python train_er.py --variant RNN --backbone omnivore --split step --num_epochs 100

# Valutazione globale
python -m core.evaluate --variant RNN --backbone omnivore --ckpt checkpoints/error_recognition/RNN/omnivore/error_recognition_RNN_omnivore_step_epoch_14.pt --split step --threshold 0.6

# Valutazione per tipo di errore
python -m core.evaluate_by_error_type --split step --backbone omnivore --variant RNN --ckpt checkpoints/error_recognition/RNN/omnivore/error_recognition_RNN_omnivore_step_epoch_14.pt
```

### 2. Task Verification (Extension)

```bash
# Step 1: Localizzazione step
python extension/substep1_step_localization/run_step_localization.py

# Step 2: Baseline semplice
python extension/substep2_task_verification/train_task_verification_hiero.py
python extension/substep2_task_verification/eval_task_verification_hiero.py

# Step 3: Task graph matching
python extension/substep3_task_graph_matching/train_task_graph_matching_gpu.py
python extension/substep3_task_graph_matching/eval_task_graph_matching.py

# Step 4: GNN classification
python extension/substep4_gnn_classification/prepare_graph_data.py
python extension/substep4_gnn_classification/train_gnn_classification.py
python extension/substep4_gnn_classification/eval_gnn_classification.py
```

---

## Struttura Directory

```
aml-2025-mistake-detection/
├── core/                    # Moduli core (evaluate, models, utils)
├── dataloader/              # Dataset loaders
├── extension/               # Extension task verification
│   ├── substep1_step_localization/
│   ├── substep2_task_verification/
│   ├── substep3_task_graph_matching/
│   └── substep4_gnn_classification/
├── annotations/             # Annotazioni dataset
├── checkpoints/             # Checkpoint modelli
├── results/                 # Risultati valutazioni
├── scripts/                 # Script automatizzati
├── train_er.py             # Script training principale
└── requirements.txt        # Dipendenze
```

---

## Note Importanti

- **Backbone**: `omnivore` (1024-dim) e `perception` (768-dim) sono supportati
- **Split**: `step` è il principale, ma sono disponibili anche `recordings`, `person`, `environment`
- **Extension**: Richiede che Substep 1 sia completato prima degli altri substep
- **Leave-One-Out**: I modelli di extension usano cross-validation leave-one-recipe-out
- **Checkpoint**: I checkpoint vengono salvati automaticamente durante il training

---

## Riferimenti

- **Dataset**: CaptainCook4D - Peddi et al., "CaptainCook4D: A Dataset for Understanding Errors in Procedural Activities", NeurIPS 2024
- **Backbone**: Omnivore (Girdhar et al., CVPR 2022), SlowFast (Feichtenhofer et al., ICCV 2019)
- **Error Recognition**: https://github.com/CaptainCook4D/error_recognition
- **Feature Extraction**: https://github.com/CaptainCook4D/feature_extractors