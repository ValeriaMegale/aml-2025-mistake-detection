# Task Verification with HiERO Step Embeddings

Adattamento del codice di task verification per usare gli step embeddings generati da HiERO (Substep 1).

## Differenze dalla versione originale

- **Input embeddings**: Formato HiERO `{video_id: array [N, 768]}` invece di `{video_id: [{'embedding': ...}, ...]}`
- **Automatic conversion**: Il codice converte automaticamente il formato
- **Input dimension**: 768 (perception encoder) invece di 1024 (omnivore)

## File

- `train_task_verification_hiero.py` - Training con leave-one-out
- `eval_task_verification_hiero.py` - Valutazione dei modelli addestrati

## Usage

### 1. Training (Leave-One-Out)

```bash
cd /home/pierpaolosorbellini/AML/aml-2025-mistake-detection

python extension_task_verification/train_task_verification_hiero.py \
    --npy extension_localization_hiero/data/step_embeddings_perception.npy \
    --annotations annotations/annotation_json/step_annotations.json \
    --ckpt_dir extension_task_verification/checkpoints_hiero \
    --epochs 15 \
    --batch_size 8 \
    --lr 1e-4
```

Questo eseguirà **Leave-One-Out evaluation**: per ogni ricetta, addestra un modello su tutte le altre ricette e salva il checkpoint.

### 2. Evaluation

```bash
python extension_task_verification/eval_task_verification_hiero.py \
    --npy extension_localization_hiero/data/step_embeddings_perception.npy \
    --annotations annotations/annotation_json/step_annotations.json \
    --ckpt_dir extension_task_verification/checkpoints_hiero
```

Questo caricherà ogni checkpoint e valuterà sul test set corrispondente (ricetta hold-out).

## Output

### Checkpoints
Salvati in `extension_task_verification/checkpoints_hiero/`:
- `model_holdout_1.pth`
- `model_holdout_2.pth`
- ... (uno per ogni ricetta)

### Evaluation Results
- Mean accuracy across all folds
- Precision, Recall, F1, AUC-ROC

## Model Architecture

Il modello `TaskVerifier`:
- **Input**: Sequence of step embeddings [batch, seq_len, 768]
- **Transformer Encoder**: 2 layers, 4 heads, hidden_dim=256
- **Classification Head**: Binary classifier (correct/incorrect recipe)
- **Output**: Probability [0, 1] of recipe being incorrect

## Note

- Il modello usa **recipe-level binary labels**: 1 se il video ha errori, 0 altrimenti
- **Leave-One-Out**: Per ogni ricetta, addestra su tutte le altre (k-1) e testa sulla ricetta hold-out
- Gli embeddings vengono automaticamente convertiti dal formato HiERO al formato atteso
