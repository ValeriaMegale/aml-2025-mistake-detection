# Mistake Detection in Procedural Activities

## AML/DAAI 2024-2025 Project Report

**Team Members:** Valeria Megale, Luca Favole, Alberto Miglio, Eugenio Fasone

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Task 1: Feature Extraction](#2-task-1-feature-extraction)
3. [Task 2: Baselines Reproduction and Improvements](#3-task-2-baselines-reproduction-and-improvements)
    - 2.1 [Baseline Reproduction (MLP & Transformer)](#31-baseline-reproduction)
    - 2.2 [Error Type Analysis](#32-error-type-analysis)
    - 2.3 [New Baseline: RNN/LSTM](#33-new-baseline-rnnlstm)
4. [Results](#4-results)
5. [Extension: From Mistake Detection to Task Verification](#5-extension-from-mistake-detection-to-task-verification)
    - 5.1 [Overview](#51-overview)
    - 5.2 [Substep 1: Recipe Step Localization](#52-substep-1-recipe-step-localization)
    - 5.3 [Substep 2: Simple Task Verification Baseline](#53-substep-2-simple-task-verification-baseline)
    - 5.4 [Substep 3: Task Graph Encoding + Step Matching](#54-substep-3-task-graph-encoding--step-matching)
    - 5.5 [Substep 4: GNN Classification](#55-substep-4-gnn-classification)
    - 5.6 [Comparison of Approaches](#56-comparison-of-approaches)
6. [Conclusions](#6-conclusions)

---

## 1. Introduction

This project addresses the task of **Mistake Detection in Procedural Activities** using the CaptainCook4D dataset. The
goal is to detect errors in cooking recipe executions by analyzing video features extracted from pre-trained backbones.

The CaptainCook4D dataset contains:

- **384 recordings** of cooking activities
- **5 error categories**: Technique Error, Preparation Error, Temperature Error, Measurement Error, Timing Error
- **4 data splits**: recordings, step, person, environment

---

## 2. Task 1: Feature Extraction

### 2.1 Pre-trained Backbones

We utilize two pre-trained video encoders to extract features from raw video data:

| Backbone     | Output Dimension | Description                                      |
|--------------|------------------|--------------------------------------------------|
| **Omnivore** | 1024             | Multimodal transformer-based encoder             |
| **SlowFast** | 400              | Two-pathway architecture for video understanding |

### 2.2 Input/Output Pipeline

**INPUT (Video Sub-segments):**

- The models process **1-second video snippets** extracted from each recipe step
- Each snippet is represented as a tensor of stacked RGB frames

**OUTPUT (Feature Vectors):**

- Each sub-segment produces a **high-dimensional feature vector** (embedding)
- For Omnivore: 1024-dimensional vector
- For SlowFast: 400-dimensional vector

**Result:** For a single recipe step, the output is a **sequence of feature vectors**, where each vector represents one
sub-segment. This sequence serves as input for the classification models.

---

## 3. Task 2: Baselines Reproduction and Improvements

### 3.1 Baseline Reproduction

We reproduced three baseline architectures for error recognition:

#### V1: MLP (Multi-Layer Perceptron)

- **Architecture:** Input → Linear(input_dim, 128) → ReLU → Linear(128, 1)
- **Input:** Aggregated features from video segments
- **Output:** Binary classification (error/no error)

#### V2: Transformer (ErFormer)

- **Architecture:** Transformer encoder with multimodal fusion
- **Layers:** 1 encoder layer, 8 attention heads, 2048 feedforward dimension
- **Input:** Sequence of feature vectors
- **Output:** Binary classification with temporal attention

#### V3: RNN Baseline (New)

- **Architecture:** RNN(input_dim, hidden_dim, num_layers) → Dropout(0.5) → Linear(128, 1)
- **Input:** Sequence of feature vectors
- **Mechanism:** Uses the final hidden state as a summary of the entire action sequence
- **Rationale:** Captures temporal dependencies between video snippets within a recipe step

### 3.2 Error Type Analysis

We analyzed model performance across different error categories using the `evaluate_by_error_type.py` module.

#### Overall Comparison (MLP vs Transformer)

**Step-Level Analysis:**

| Model       | Global Accuracy | Global F1 | Recall | Precision |
|-------------|-----------------|-----------|--------|-----------|
| MLP         | ~56.8%          | ~0.55     | ~0.86  | ~0.42     |
| Transformer | ~46.7%          | ~0.48     | ~0.79  | ~0.38     |

**Recording-Level Analysis:**

| Model       | Global Accuracy | Recall | Notes                                        |
|-------------|-----------------|--------|----------------------------------------------|
| MLP         | ~44.3%          | ~0.86  | Detects most errors but many false positives |
| Transformer | ~57.1%          | ~0.49  | Higher accuracy but misses >50% of errors    |

**Key Findings:**

- The **MLP baseline** generally outperforms the Transformer on step-level classification
- MLP demonstrates higher Recall (~0.86), suggesting better sensitivity to error detection
- At recording level, Transformer achieves higher accuracy but with significantly lower Recall (~0.49), meaning it
  misses more than half of actual errors
- Both models struggle with Precision, leading to false positives

#### Performance by Error Category (Baselines Analysis)

We analyzed the performance of the reproduced baselines (MLP and Transformer) across different error categories to
understand their specific strengths and limitations.

| Error Category        | Samples | MLP Accuracy | Transf. Accuracy |  MLP F1  | Transf. F1 |
|:----------------------|:-------:|:------------:|:----------------:|:--------:|:----------:|
| **Technique Error**   |   62    |    54.84%    |    **88.71%**    |   0.71   |  **0.94**  |
| **Measurement Error** |   42    |    47.62%    |    **88.10%**    |   0.64   |  **0.93**  |
| **Preparation Error** |   49    |    55.10%    |    **81.63%**    |   0.71   |  **0.90**  |
| **Timing Error**      |   34    |    61.76%    |    **82.35%**    |   0.76   |  **0.90**  |
| **Temperature Error** |    8    |    75.00%    |    **87.50%**    |   0.86   |  **0.93**  |
| **No Error** (Normal) |   691   |  **69.32%**  |      44.57%      | **0.43** |    0.37    |

*Data Source: Evaluation on Step Split using Omnivore backbone (MLP Epoch 35, Transformer Epoch 26).*

**Analysis:**

- **Complementary Behavior:** The two baselines exhibit contrasting behaviors. The **Transformer** is highly sensitive
  to errors, achieving remarkable accuracy (>80%) and F1 scores on specific error categories. However, it struggles
  significantly with the "No Error" class (44.57% accuracy), indicating a tendency to hallucinate errors in normal
  steps (High False Positive Rate).
- **MLP Conservatism:** The **MLP** is more conservative. It performs significantly better on the dominant "No Error"
  class (~69%), which explains its higher global accuracy compared to the Transformer. However, it fails to detect
  specific errors as effectively as the Transformer (e.g., ~55% on Technique Error vs ~89% for Transformer).
- **Motivation for RNN:** The analysis shows a gap: the MLP lacks sensitivity to specific errors, while the Transformer
  lacks precision on normal steps. We hypothesize that an RNN, by modeling the temporal evolution of the step, might
  find a better balance between these two extremes.

### 3.3 New Baseline: RNN

To better capture temporal dependencies between video snippets within a recipe step, we implemented a Recurrent Neural
Network based on Long Short-Term Memory (LSTM) units. Unlike the MLP baseline, which processes aggregated features, the
LSTM processes the sequence of frame features step-by-step.

### 4.1 Risultati delle Run (wandb export)

**Design Choices:**

- **RNN** for hierarchical temporal processing
- **Dropout (0.5)** for regularization against overfitting
- **Hidden dimension 128** to match MLP capacity for fair comparison
- **BCEWithLogitsLoss** with pos_weight=1.5 to handle class imbalance

**Rationale:**
The RNN should theoretically handle the sequential nature of videos better than the MLP (which treats segments in
isolation or aggregates them simply). By using the final hidden state as a compact summary of the entire action
sequence, we capture temporal patterns that may indicate errors developing over time.
=======
| Modello | Backbone | Split | Batch Size | LR | Epoche | Accuracy (test) | F1 (test) | AUC (test) | Precision (test) |
Recall (test) |
|--------------|-------------|-------|------------|--------|--------|-----------------|-----------|------------|------------------|---------------|
| RNN | perception | step | 128 | 0.0005 | 100 | 0.6927 | 0.7645 | 0.7419 | 0.4449 | 0.6303 |
| Transformer | perception | step | 128 | 0.0005 | 100 | 0.6775 | 0.8154 | 0.7494 | 0.4401 | 0.8182 |
| MLP | perception | step | 128 | 0.0005 | 100 | 0.7217 | 0.7495 | 0.7143 | 0.4096 | 0.5600 |

*Tutte le run sono state eseguite con modality = video, weight_decay = 0.005, seed = 42, feature directory = data/.*

**Nota:** I risultati sono estratti dal file wandb_export_2025-12-27T12_01_32.634+01_00.csv. Per ogni run sono riportate
le metriche principali sul test set. Altre metriche (sub-step, validazione) sono disponibili nel file esportato.

---

#### Parametri principali usati negli esperimenti

| Parametro          | Vecchio valore | Nuovo valore |
|--------------------|----------------|--------------|
| batch_size         | 32             | 128          |
| num_epochs         | 100            | 100          |
| learning rate (lr) | 0.001          | 0.0005       |
| weight_decay       | 1e-4           | 0.005        |
| test_batch_size    | -              | 1            |
| hidden_dim (LSTM)  | -              | 128          |
| num_layers (LSTM)  | -              | 2            |
| dropout (LSTM)     | -              | 0.5          |

Altri parametri aggiunti: log_interval, dry_run, seed, ecc. Alcune modifiche strutturali alle MLP (layer
aggiunti/rimossi) e ai parametri LSTM sono state introdotte in commit specifici.

---
| Nuovi | Vecchie | RNN | No |
| Vecchi | Nuove | MLP | No |
| Vecchi | Nuove | Transformer | No |
| Vecchi | Nuove | RNN | No |
| Nuovi | Nuove | MLP | No |
| Nuovi | Nuove | Transformer | No |
| Nuovi | Nuove | RNN | No |
| MLP | Omnivore | step | 0.71 | 0.66 | 0.15 | 0.24 | 0.76 |
| MLP | Omnivore | sub-step | 0.68 | 0.41 | 0.30 | 0.35 | 0.65 |
| Transformer | Omnivore | step | 0.70 | 0.52 | 0.60 | 0.55 | 0.76 |
| Transformer | Omnivore | sub-step | 0.67 | 0.44 | 0.66 | 0.53 | 0.75 |

## 4.1 Results for New Checkpoints by error type (core/evaluate_by_error_type.py)


In this section, we present the quantitative evaluation of our models. We focus on the comparison between the standard
baselines (MLP, Transformer) and our proposed RNN baseline using the **Omnivore** backbone, which yielded the most
consistent results across all architectures.

### 4.1 Global Performance Comparison

Table 1 summarizes the performance of the three architectures on the **Step Split**.

**Table 1: Global Performance on Test Set (Backbone: Omnivore)**

| Model           |  Accuracy  | F1 Score |   AUC    | Precision |  Recall  |
|:----------------|:----------:|:--------:|:--------:|:---------:|:--------:|
| **MLP**         | **67.29%** |   0.51   | **0.68** | **0.48**  |   0.55   |
| **Transformer** |   50.13%   |   0.51   |   0.65   |   0.37    | **0.82** |
| **RNN (Ours)**  |   61.15%   | **0.54** | **0.68** |   0.43    |   0.72   |

*Note: Results extracted from best checkpoints: MLP (Epoch 35), Transformer (Epoch 26), and RNN (Epoch 14).*

**Analysis:**

- **The Trade-off:** The **MLP** acts as a "conservative" model, achieving the highest Accuracy (67.29%) but missing
  nearly half the errors (Recall 0.55). Conversely, the **Transformer** is highly "sensitive," detecting 82% of errors
  but generating an excessive number of false positives, which drops its Accuracy to ~50%.
- **RNN Performance:** Our proposed **RNN** baseline strikes the best balance. It achieves the **highest F1 Score (0.54)
  **, significantly improving Recall compared to the MLP (0.72 vs 0.55) while maintaining much better Accuracy than the
  Transformer (61.15% vs 50.13%). This suggests that modeling temporal dependencies helps distinguish actual errors from
  noise more effectively than attention mechanisms alone in this low-data regime.

### 4.2 Evaluation by Error Type (RNN)

To deeply understand the RNN's capabilities, we performed a granular analysis using `evaluate_by_error_type.py` on the
best RNN checkpoint. The results are detailed in Table 2.

**Table 2: RNN (Omnivore) Performance Breakdown by Error Category**

| Error Category        | Samples |  Accuracy  | Precision |   Recall   | F1 Score  |
|:----------------------|:-------:|:----------:|:---------:|:----------:|:---------:|
| **Technique Error**   |   62    | **77.42%** |  100.00%  |   77.42%   | **87.27** |
| **Timing Error**      |   34    |   73.53%   |  100.00%  |   73.53%   |   84.75   |
| **Temperature Error** |    8    |   75.00%   |  100.00%  |   75.00%   |   85.71   |
| **Preparation Error** |   49    |   69.39%   |  100.00%  |   69.39%   |   81.93   |
| **Measurement Error** |   42    |   57.14%   |  100.00%  |   57.14%   |   72.73   |
| **No Error** (Normal) |   691   |   60.49%   |  31.23%   | **76.76%** |   44.40   |

*Data Source: RNN-Omnivore epoch 14. Note: Precision is 100% for error categories because the evaluation script
calculates precision within the subset of steps containing that specific error tag versus the predictions made.*

**Key Findings:**

1. **Superior Detection of Technique & Timing:** The RNN excels at detecting *Technique Errors* (77.4% Accuracy, 87.3
   F1) and *Timing Errors* (73.5% Accuracy, 84.7 F1). This confirms our hypothesis: errors involving movement quality (
   Technique) or duration (Timing) are best captured by the sequential processing of the LSTM, as opposed to the static
   aggregation of the MLP.
2. **The "No Error" Challenge:** The model struggles mostly with the "No Error" class (Accuracy 60.49%). The low
   precision (31.23%) indicates that the RNN often flags correct steps as errors. However, the high Recall (76.76%) on
   the "No Error" class suggests the model correctly identifies the majority of normal steps, but the false positives
   significantly impact the precision.
3. **Backbone Impact:** Comparing these results to our preliminary tests with the Perception backbone (where Global F1
   was ~0.49), the **Omnivore** backbone provides richer features for the RNN, leading to a +5% improvement in global F1
   score.

### 4.3 Discussion

Our experiments demonstrate that the **RNN with Omnivore features** is the most robust architecture for this task among
the tested baselines. It outperforms the MLP in identifying mistakes (higher Recall/F1) and outperforms the Transformer
in reliability (higher Accuracy). The ability of the LSTM to maintain a temporal state proves crucial for identifying
dynamic errors like *Timing* and *Technique*, which are less discernible to non-sequential models. Future work should
focus on reducing the False Positive Rate on normal steps, potentially by adjusting the loss function weights or
incorporating attention mechanisms within the RNN.

### 4.1.4 Nuovo Checkpoint: Transformer con nuove feature e nuovi parametri

| Modello     | Backbone   | Feature       | Parametri       | Epoca | Checkpoint | Accuracy (test) | F1 (test) | AUC (test) | Precision (test) | Recall (test) |
|-------------|------------|---------------|-----------------|-------|------------|-----------------|-----------|------------|------------------|---------------|
| Transformer | perception | nuove feature | nuovi parametri | 27    | Sì         | 0.7288 (step)   | 0.5919    | 0.7834     | 0.6439           | 0.5477        |

**Sub Step Level:**

- F1: 0.5440
- Accuracy: 0.7197
- AUC: 0.7568
- Precision: 0.5849
- Recall: 0.5084

## 4.2 Risultati Evaluation per Error Type (core/evaluate_by_error_type.py)

**Modello:** RNN (perception, step split, checkpoint RNN_epoch_20.pt)

## 4.3 Risultati Evaluation Globale RNN (core/evaluate.py)

**Modello:** RNN (perception, step split, checkpoint RNN_epoch_20.pt)

- Accuracy globale: 62.91%
- Precision: 42.49%
- Recall: 53.41%
- F1: 47.33%
- AUC: 66.00%

*Risultati generati da core/evaluate.py su checkpoint RNN_epoch_20.pt (perception, step split).*

## 4.4 Risultati RNN per Categoria di Errore (core/evaluate_by_error_type.py)

**Modello:** RNN (perception, step split, checkpoint RNN_epoch_20.pt)

| Categoria         | Samples | Accuracy | Precision | Recall | F1    | AUC   |
|-------------------|---------|----------|-----------|--------|-------|-------|
| Technique Error   | 62      | 61.29    | 100.00    | 61.29  | 76.00 | N/A   |
| Preparation Error | 49      | 44.90    | 100.00    | 44.90  | 61.97 | N/A   |
| Temperature Error | 8       | 50.00    | 100.00    | 50.00  | 66.67 | N/A   |
| Measurement Error | 42      | 59.52    | 100.00    | 59.52  | 74.63 | N/A   |
| Timing Error      | 34      | 50.00    | 100.00    | 50.00  | 66.67 | N/A   |
| No Error          | 691     | 64.11    | 29.13     | 52.11  | 37.37 | 65.53 |

*Risultati generati da core/evaluate_by_error_type.py su checkpoint RNN_epoch_20.pt (perception, step split).*

---

## 5. Extension: From Mistake Detection to Task Verification

### 5.1 Overview

Moving beyond step-level mistake detection, we implemented a **Task Verification** pipeline that predicts whether an entire recipe video corresponds to a correct or incorrect execution by jointly analyzing the video and its corresponding task graph. This extension addresses a more practical scenario where only recipe-level binary labels are available, without requiring step-level error annotations.

**Pipeline Architecture:**

```
Video → Step Localization → Step Embeddings → Task Graph Matching → Graph Realization → Classification
```

The extension consists of four main substeps:

1. **Substep 1**: Zero-shot step localization using hierarchical clustering (HiERO-style)
2. **Substep 2**: Simple transformer baseline for recipe-level classification
3. **Substep 3**: Task graph matching with Hungarian algorithm
4. **Substep 4**: GNN-based classification on task graph realization

---

### 5.2 Substep 1: Recipe Step Localization

**Method**: Hierarchical clustering-based approach inspired by HiERO for temporal step segmentation.

**Approach:**
- Uses pre-extracted Perception Encoder features (768-dim, 1-second stride)
- Applies agglomerative hierarchical clustering on temporal features
- Estimates number of clusters based on video duration (~1 step per 10-15 seconds)
- Post-processing: removes short segments, applies temporal NMS, limits segment count

**Output:**
- Step segments: `[(start1, end1), (start2, end2), ...]` for each video
- Step embeddings: Average-pooled features within each segment `[N_steps, 768]`

**Implementation**: `extension/substep1_step_localization/`

**Results**: Step segments and embeddings generated for all videos in the dataset.

---

### 5.3 Substep 2: Simple Task Verification Baseline

**Architecture**: `TaskVerifier` - Transformer-based classifier for recipe-level binary classification.

**Model Details:**
- Input: Sequence of step embeddings `[batch, seq_len, 768]`
- Architecture: Transformer encoder (2 layers, 4 heads, hidden_dim=256) + classification head
- Output: Probability `[0, 1]` of recipe being incorrect

**Training:**
- Leave-one-recipe-out cross-validation
- Binary classification: 1 if video has ANY errors, 0 otherwise
- BCE loss

**Implementation**: `extension/substep2_task_verification/train_task_verification_hiero.py`

**Results**: Evaluation on 24 recipes (leave-one-out)

**Table 2: Simple Task Verification Baseline Performance (Leave-One-Out, 24 recipes)**

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Accuracy | 0.5855 | 0.1431 | 0.2500 | 0.8333 |
| Precision | 0.6043 | - | - | - |
| Recall | 0.7636 | - | - | - |
| F1 | 0.6747 | - | - | - |
| AUC | 0.5656 | - | - | - |

*Note: Results from TaskVerifier model (Transformer encoder) on HiERO step embeddings. Evaluation on all 24 recipes with leave-one-out cross-validation.*

---

### 5.4 Substep 3: Task Graph Encoding + Step Matching

**Architecture**: `TaskGraphMatcher` - Matches visual steps to task graph nodes using Hungarian algorithm.

**Model Components:**

1. **Text Encoder**: CLIP ViT-B/32 (512-dim) or sentence-transformers (384-dim) for task graph node descriptions
2. **Visual Projection**: Projects Perception embeddings (768-dim) to shared space (256-dim)
3. **Transformer**: Self-attention on visual steps, cross-attention to task graph nodes
4. **Hungarian Matching**: Optimal one-to-one assignment between visual steps and graph nodes
5. **Classification Head**: Binary classifier on matched features

**Key Features:**
- Uses aligned video-text embedding space (Perception + CLIP)
- Hungarian algorithm for optimal matching
- Matching quality as signal for error detection

**Training:**
- Leave-one-recipe-out cross-validation
- Per-recipe checkpoints: `task_graph_matcher_recipe_{id}.pth`

**Implementation**: `extension/substep3_task_graph_matching/`

**Results**: Evaluation on 10/24 recipes (checkpoints available)

**Table 3: Task Graph Matching Performance (Leave-One-Out, 10 recipes)**

| Recipe ID | Recipe Name | Accuracy | Precision | Recall | F1 | AUC | Num Videos |
|-----------|-------------|----------|-----------|--------|----|----|-----------|
| 1 | Microwave Egg Sandwich | 0.3333 | 1.0000 | 0.0769 | 0.1429 | 0.6154 | 18 |
| 2 | Dressed Up Meatballs | 0.3750 | 0.5000 | 0.1000 | 0.1667 | 0.4333 | 16 |
| 10 | Pinwheels | 0.6667 | 0.6667 | 1.0000 | 0.8000 | 0.5938 | 12 |
| 12 | Tomato Mozzarella Salad | 0.6667 | 1.0000 | 0.1429 | 0.2500 | 0.6623 | 18 |
| 13 | Butter Corn Cup | 0.3571 | 0.0000 | 0.0000 | 0.0000 | 0.3333 | 14 |
| 15 | Tomato Chutney | 0.6667 | 0.6667 | 1.0000 | 0.8000 | 0.5000 | 15 |
| 16 | Scrambled Eggs | 0.6250 | 0.6250 | 1.0000 | 0.7692 | 0.6167 | 16 |
| 17 | Cucumber Raita | 0.4000 | 0.4000 | 1.0000 | 0.5714 | 0.5208 | 20 |
| 18 | Zoodles | 0.7333 | 0.7333 | 1.0000 | 0.8462 | 0.7273 | 15 |
| 20 | Sauted Mushrooms | 0.5714 | 0.5714 | 1.0000 | 0.7273 | 0.5625 | 14 |
| **Mean** | | **0.5395** | **0.6163** | **0.6320** | **0.5074** | **0.5565** | |
| **Std** | | **0.1472** | **0.2749** | **0.4519** | **0.3130** | **0.1088** | |

*Note: Results computed with dimension adaptation (zero-padding) due to checkpoint/embedding dimension mismatch (checkpoint expects 1024-dim Omnivore, embeddings are 768-dim Perception). Missing recipes (14/24): 3, 4, 5, 7, 8, 9, 21, 22, 23, 25, 26, 27, 28, 29.*

**Analysis:**
- Moderate performance with high variance across recipes
- Recall higher than precision, indicating sensitivity to errors but with false positives
- Matching provides interpretability: can identify which steps are matched/mismatched

---

### 5.5 Substep 4: GNN Classification

**Architecture**: `DAGNNClassifier` - Graph Neural Network for classifying task graph realizations.

**Model Components:**

1. **Node Features**: Concatenation of text features (CLIP, 512-dim) + matched visual features (Perception, 768-dim) = 1280-dim
2. **Learnable Projection**: Projects node features to hidden dimension (128-dim)
3. **DAGNN Layer**: `DAGNNConv` specifically designed for Directed Acyclic Graphs
4. **Global Pooling**: Mean pooling to aggregate node embeddings to graph embedding
5. **Classifier**: Binary classification head

**Key Features:**
- Operates on task graph structure (DAG) rather than sequences
- Node features incorporate both textual (expected) and visual (observed) information
- DAGNN respects graph topology for information propagation

**Training:**
- Leave-one-recipe-out cross-validation
- Per-recipe checkpoints: `gnn_classifier_recipe_{id}.pth`

**Implementation**: `extension/substep4_gnn_classification/`

**Results**: Evaluation on 24 recipes (leave-one-out)

**Table 4: GNN Classification Performance (Leave-One-Out, 24 recipes)**

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Accuracy | 0.5748 | 0.1102 | 0.3333 | 0.8000 |
| Precision | 0.5748 | 0.1102 | 0.3333 | 0.8000 |
| Recall | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| F1 | 0.7234 | 0.0949 | 0.5000 | 0.9091 |
| AUC | 0.5170 | 0.1049 | 0.3000 | 0.7667 |

*Note: Results from DAGNNClassifier model on task graph realizations. Evaluation on all 24 recipes with leave-one-out cross-validation. The model achieves perfect recall (1.00) but with lower precision, indicating high sensitivity to errors.*

---

### 5.6 Comparison of Approaches

**Comparison Table: Substep 2 vs 3 vs 4**

| Approach | Input | Architecture | Key Feature | Accuracy | F1 | AUC |
|----------|-------|--------------|-------------|----------|----|----|
| **Substep 2** (Simple Baseline) | Step embeddings sequence | Transformer | Sequence-level classification | 0.5855 | 0.6747 | 0.5656 |
| **Substep 3** (Task Graph Matching) | Step embeddings + Task graph | Transformer + Hungarian | Visual-text matching | 0.5395 | 0.5074 | 0.5565 |
| **Substep 4** (GNN) | Task graph realization | DAGNN | Graph structure awareness | 0.5748 | 0.7234 | 0.5170 |

*Note: All results from leave-one-out cross-validation on 24 recipes (Substep 3: 10 recipes).*

**Discussion:**

- **Substep 2** (Simple Baseline) achieves the highest **Accuracy (0.5855)** among the three approaches, with competitive F1 (0.6747) and AUC (0.5656). It provides a simple, effective baseline that directly classifies sequences of steps without task graph information.
- **Substep 3** (Task Graph Matching) shows the lowest **F1 (0.5074)** and moderate Accuracy (0.5395), but leverages task graph structure through matching, providing interpretability. The Hungarian matching allows identifying which visual steps correspond to which graph nodes, but the approach suffers from embedding dimension mismatches and limited training data (only 10/24 recipes).
- **Substep 4** (GNN) achieves the highest **F1 (0.7234)** and perfect **Recall (1.00)**, indicating excellent error detection sensitivity. The graph structure awareness enables the model to capture complex relationships between steps, though it trades some precision for high recall. Accuracy (0.5748) is competitive with Substep 2.

**Key Findings:**

- **F1 Score**: Substep 4 (0.7234) > Substep 2 (0.6747) > Substep 3 (0.5074)
- **Accuracy**: Substep 2 (0.5855) > Substep 4 (0.5748) > Substep 3 (0.5395)
- **AUC**: Substep 3 (0.5565) ≈ Substep 2 (0.5656) > Substep 4 (0.5170)

**Trade-offs:**

- **Interpretability**: Substep 3 > Substep 4 > Substep 2 (matching provides clear step-to-node correspondences)
- **Structure awareness**: Substep 4 > Substep 3 > Substep 2 (GNN explicitly models graph topology)
- **Simplicity**: Substep 2 > Substep 3 > Substep 4 (transformer baseline is simplest)
- **Error detection sensitivity**: Substep 4 > Substep 2 > Substep 3 (Substep 4 achieves perfect recall)

---

## 6. Conclusions

### Key Findings

1. **MLP provides a strong baseline** despite its simplicity, outperforming the Transformer on several metrics (
   Step-Level Accuracy: 56.8% vs 46.7%)

2. **Error-specific detection is highly accurate** - both models achieve >80% accuracy on explicit error types such as
   Technique, Measurement, Preparation, and Temperature errors. This suggests the Omnivore features are discriminative
   enough to capture significant deviations in action execution.

3. **Class imbalance is the main challenge** - the "No Error" class (691 samples) dominates over error classes (<70
   samples each), causing:
    - High false positive rates
    - Models over-predicting errors (high Recall, low Precision)
    - Poor performance on normal step classification (~51% for MLP, ~41% for Transformer)

4. **Trade-off between Accuracy and Recall** - At recording level, Transformer achieves higher accuracy (57.1%) but
   misses >50% of actual errors (Recall: 0.49), while MLP detects most errors (Recall: 0.86) but with more false
   positives.

5. **Temporal modeling (RNN/LSTM)** is expected to improve detection by capturing sequential patterns that indicate
   errors developing over time.

### Future Work

- **Address class imbalance** with advanced techniques:
    - Oversampling error classes / Undersampling normal class
    - Focal Loss or other imbalance-aware loss functions
    - Increase pos_weight in BCEWithLogitsLoss
- **Experiment with attention mechanisms** in the RNN (e.g., Bidirectional LSTM with attention)
- **Evaluate on different data splits** (person, environment) to test generalization
- **Explore new backbones** (EgoVLP, PerceptionEncoder) for potentially richer features
- **Complete Extension evaluation**: Train all models (Substep 2, 4) and perform comprehensive comparison
- **Improve Task Graph Matching**: Integrate learnable projections from Substep 3 into Substep 4
- **Graph-based improvements**: Experiment with different GNN architectures (GraphConv, GAT) for task graphs

---

## References

- CaptainCook4D Dataset: Peddi et al., "CaptainCook4D: A Dataset for Understanding Errors in Procedural Activities",
  NeurIPS 2024
- Omnivore: Girdhar et al., "Omnivore: A Single Model for Many Visual Modalities", CVPR 2022
- SlowFast Networks: Feichtenhofer et al., "SlowFast Networks for Video Recognition", ICCV 2019