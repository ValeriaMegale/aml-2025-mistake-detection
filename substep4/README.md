# Substep 4: GNN Classification for Task Verification

## Overview

Substep 4 completa l'estensione "From Mistake Detection to Task Verification" implementando un **Graph Neural Network (GNN)** basato su **DAGNN** per classificare se un'intera ricetta è stata eseguita correttamente o meno, basandosi sulla "realization" del task graph.

## Architettura

```
Task Graph (DAG) + Matched Visual Steps
        │
        ▼
Node Features: [Text Features + Matched Visual Features]
        │
        ▼
Learnable Projection: Text(512) + Visual(768) → Hidden(128)
        │
        ▼
DAGNN Layer: Propagazione informazioni attraverso DAG
        │
        ▼
Global Mean Pooling: Graph Embedding
        │
        ▼
Classifier: Binary Classification (Correct/Incorrect)
```

## Componenti

### 1. Data Preparation (`prepare_graph_data.py`)

Converte task graphs matched in formato PyTorch Geometric Data.

**Input:**
- Step embeddings da Substep 1 (`extension_localization_hiero/data/step_embeddings_perception.npy`)
- Task graphs (`annotations/task_graphs/*.json`)
- Annotations per labels (`annotations/annotation_json/step_annotations.json`)

**Processo:**
1. Per ogni video, carica visual step embeddings
2. Carica task graph corrispondente
3. Calcola matching visual steps → graph nodes (Hungarian algorithm)
4. Crea node features: concatenazione di text features (CLIP) + matched visual features (perception)
5. Crea PyTorch Geometric `Data` object con:
   - `x`: node features [N_nodes, text_dim + visual_dim]
   - `edge_index`: edge connectivity [2, N_edges] dal task graph
   - `y`: graph-level binary label [1] (0=correct, 1=error)

**Output:**
- Directory `graph_classification_substep4/data/` con:
  - `train/` - grafi per training
  - `test/` - grafi per test (o tutti in train se non specificato)
  - `metadata.json` - mapping video_id → file, labels, etc.

**Usage:**
```bash
python substep4/prepare_graph_data.py \
    --step_embeddings_npy extension_localization_hiero/data/step_embeddings_perception.npy \
    --output_dir substep4/graph_classification_substep4/data \
    --feature_dim 1280 \
    --test_recipes "1,2,3"  # Opzionale: ricette per test split
```

### 2. Dataset (`graph_classification_substep4/dataset.py`)

Carica grafi preparati in formato PyTorch Geometric.

**Features:**
- Carica grafi da directory train/test
- Supporta leave-one-out split tramite metadata
- Compatibile con PyTorch Geometric DataLoader

**Usage:**
```python
from substep4.graph_classification_substep4.dataset import TaskGraphDataset

dataset = TaskGraphDataset(
    root='substep4/graph_classification_substep4/data',
    split='train'
)
```

### 3. Model (`graph_classification_substep4/model.py`)

**DAGNNClassifier**: Modello GNN per graph-level classification.

**Architettura:**
- **Input Projection**: Proietta node features (text + visual concatenate) a hidden_dim
- **DAGNN Layer**: `DAGNNConv` per processare Directed Acyclic Graph
- **Global Pooling**: Mean pooling per aggregare node embeddings a graph embedding
- **Classifier**: Binary classification head

**Parametri:**
- `in_channels`: text_dim + visual_dim (default: 512 + 768 = 1280)
- `hidden_channels`: dimensione hidden/embedding (default: 128)
- `num_classes`: 2 (binary classification)
- `K`: numero propagazioni DAGNN (default: 10)
- `dropout`: dropout rate (default: 0.3)

### 4. Training (`train_gnn_classification.py`)

Addestra DAGNN classifier con **leave-one-recipe-out** cross-validation.

**Features:**
- Training per ogni ricetta (k-1 ricette per training, ricetta k per test)
- Salvataggio checkpoint per ogni fold
- Early stopping opzionale
- Metriche: Accuracy, Precision, Recall, F1, AUC

**Usage:**
```bash
python substep4/train_gnn_classification.py \
    --data_dir substep4/graph_classification_substep4/data \
    --ckpt_dir substep4/checkpoints_gnn \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-3 \
    --hidden_channels 128 \
    --K 10 \
    --dropout 0.3 \
    --early_stop
```

**Output:**
- Checkpoints: `substep4/checkpoints_gnn/gnn_classifier_recipe_{recipe_id}.pth`
- Training summary: `substep4/checkpoints_gnn/training_summary.json`

### 5. Evaluation (`eval_gnn_classification.py`)

Valuta tutti i checkpoints leave-one-out e genera metriche aggregate.

**Usage:**
```bash
python substep4/eval_gnn_classification.py \
    --ckpt_dir substep4/checkpoints_gnn \
    --data_dir substep4/graph_classification_substep4/data \
    --output_dir results/substep4_gnn_classification \
    --batch_size 32
```

**Output:**
- `results/substep4_gnn_classification/all_recipes_results.json` - risultati dettagliati
- `results/substep4_gnn_classification/results_summary.csv` - summary CSV

## Pipeline Completo

### Step 1: Preparazione Dati
```bash
# Assicurati che Substep 1 sia completato (step embeddings)
# Poi prepara i grafi:
python substep4/prepare_graph_data.py
```

### Step 2: Training
```bash
# Addestra modelli per tutte le ricette (leave-one-out)
python substep4/train_gnn_classification.py --epochs 50
```

### Step 3: Evaluation
```bash
# Valuta tutti i checkpoints
python substep4/eval_gnn_classification.py
```

## Testing

Verifica che il pipeline funzioni correttamente:

```bash
python substep4/test_pipeline.py
```

Questo script verifica:
- Import corretti
- Creazione modello
- Forward pass
- Dataset loading
- Integrazione componenti

## Dettagli Implementativi

### Node Features

Per ogni nodo del task graph:
- **Text Feature**: CLIP encoding della descrizione del nodo [512-dim]
- **Matched Visual Feature**: Embedding dello step visivo matched (se presente) [768-dim], altrimenti zero
- **Node Feature**: Concatenazione [text_feat, visual_feat] = [1280-dim]

Il modello applica una proiezione learnable per fondere text + visual features nello spazio hidden.

### Matching

Il matching visual steps → graph nodes viene calcolato usando:
1. Cosine similarity tra visual embeddings e text embeddings
2. Hungarian algorithm per matching ottimale one-to-one

### DAGNN

Usa `DAGNNConv` da PyTorch Geometric, specificamente progettato per Directed Acyclic Graphs (DAG). Questo layer propaga informazioni attraverso il grafo rispettando la struttura DAG.

## Requisiti

- PyTorch
- PyTorch Geometric
- NumPy
- Scikit-learn
- CLIP (per text encoding)
- Scipy (per Hungarian matching)

```bash
pip install torch torch-geometric numpy scikit-learn scipy
pip install git+https://github.com/openai/CLIP.git
```

## File Structure

```
substep4/
├── README.md                           # Questo file
├── prepare_graph_data.py               # Preparazione dati
├── train_gnn_classification.py        # Training script
├── eval_gnn_classification.py         # Evaluation script
├── test_pipeline.py                   # Test pipeline
├── graph_classification_substep4/
│   ├── dataset.py                     # Dataset loader
│   ├── model.py                       # DAGNNClassifier
│   ├── utils.py                       # Utility functions
│   └── data/                          # Graph data (generated)
│       ├── train/                     # Training graphs
│       ├── test/                      # Test graphs
│       └── metadata.json              # Metadata
└── checkpoints_gnn/                   # Model checkpoints (generated)
```

## Note

- Il modello usa **binary classification** (correct/incorrect) a livello di ricetta
- **Leave-one-out** evaluation: per ogni ricetta, addestra su tutte le altre
- Le dimensioni delle features sono basate su:
  - Text: CLIP ViT-B/32 (512-dim) o sentence-transformers (384-dim)
  - Visual: Perception Encoder (768-dim) o Omnivore (1024-dim)
- Il matching è calcolato durante la preparazione dati, ma potrebbe essere migliorato usando il modello Substep 3 addestrato per le proiezioni learnable

## Integrazione con Substep 3

Substep 4 può essere migliorato integrando il modello Substep 3 addestrato per:
1. Usare le proiezioni learnable del TaskGraphMatcher per fondere text + visual
2. Usare i matching scores dal modello addestrato invece del matching basato su cosine similarity

Questo è un'estensione futura opzionale.
