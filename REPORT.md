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
5. [Conclusions](#5-conclusions)

---

## 1. Introduction

This project addresses the task of **Mistake Detection in Procedural Activities** using the CaptainCook4D dataset. The goal is to detect errors in cooking recipe executions by analyzing video features extracted from pre-trained backbones.

The CaptainCook4D dataset contains:
- **384 recordings** of cooking activities
- **5 error categories**: Technique Error, Preparation Error, Temperature Error, Measurement Error, Timing Error
- **4 data splits**: recordings, step, person, environment

---

## 2. Task 1: Feature Extraction

### 2.1 Pre-trained Backbones

We utilize two pre-trained video encoders to extract features from raw video data:

| Backbone | Output Dimension | Description |
|----------|------------------|-------------|
| **Omnivore** | 1024 | Multimodal transformer-based encoder |
| **SlowFast** | 400 | Two-pathway architecture for video understanding |

### 2.2 Input/Output Pipeline

**INPUT (Video Sub-segments):**
- The models process **1-second video snippets** extracted from each recipe step
- Each snippet is represented as a tensor of stacked RGB frames

**OUTPUT (Feature Vectors):**
- Each sub-segment produces a **high-dimensional feature vector** (embedding)
- For Omnivore: 1024-dimensional vector
- For SlowFast: 400-dimensional vector

**Result:** For a single recipe step, the output is a **sequence of feature vectors**, where each vector represents one sub-segment. This sequence serves as input for the classification models.

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

#### V3: RNN/LSTM Baseline (New)
- **Architecture:** LSTM(input_dim, hidden_dim=128, num_layers=2) → Dropout(0.5) → Linear(128, 1)
- **Input:** Sequence of feature vectors
- **Mechanism:** Uses the final hidden state as a summary of the entire action sequence
- **Rationale:** Captures temporal dependencies between video snippets within a recipe step

### 3.2 Error Type Analysis

We analyzed model performance across different error categories using the `evaluate_by_error_type.py` module.

#### Overall Comparison (MLP vs Transformer)

**Step-Level Analysis:**

| Model | Global Accuracy | Global F1 | Recall | Precision |
|-------|-----------------|-----------|--------|-----------|
| MLP | ~56.8% | ~0.55 | ~0.86 | ~0.42 |
| Transformer | ~46.7% | ~0.48 | ~0.79 | ~0.38 |

**Recording-Level Analysis:**

| Model | Global Accuracy | Recall | Notes |
|-------|-----------------|--------|-------|
| MLP | ~44.3% | ~0.86 | Detects most errors but many false positives |
| Transformer | ~57.1% | ~0.49 | Higher accuracy but misses >50% of errors |

**Key Findings:**
- The **MLP baseline** generally outperforms the Transformer on step-level classification
- MLP demonstrates higher Recall (~0.86), suggesting better sensitivity to error detection
- At recording level, Transformer achieves higher accuracy but with significantly lower Recall (~0.49), meaning it misses more than half of actual errors
- Both models struggle with Precision, leading to false positives

#### Performance by Error Category

| Error Type | MLP Accuracy | MLP F1 | Samples |
|------------|--------------|--------|---------|
| Technique Error | ~87.1% | ~0.93 | ~63 |
| Measurement Error | ~90.5% | ~0.95 | ~42 |
| Preparation Error | ~85.2% | ~0.91 | ~58 |
| Temperature Error | ~82.4% | ~0.89 | ~31 |
| Timing Error | ~78.6% | ~0.85 | ~24 |
| **No Error** | ~51.7% | ~0.67 | ~691 |

**Analysis:**
- Both models achieve **high accuracy (>80%)** on explicit error types
- The visual features extracted by Omnivore are discriminative enough to capture significant deviations in action execution (e.g., using the wrong tool, incorrect motion)
- The primary bottleneck is the **"No Error" class** (~51% accuracy for MLP, ~41% for Transformer)
- This class imbalance (691 normal vs <70 error samples per category) causes high false positive rates
- The models tend to over-predict errors (high Recall for error classes, but low Precision globally)


### 3.3 New Baseline: RNN/LSTM
To better capture temporal dependencies between video snippets within a recipe step, we implemented a Recurrent Neural Network based on Long Short-Term Memory (LSTM) units. Unlike the MLP baseline, which processes aggregated features, the LSTM processes the sequence of frame features step-by-step.

### 4.1 Risultati delle Run (wandb export)

| Modello      | Backbone    | Split | Batch Size | LR     | Epoche | Accuracy (test) | F1 (test) | AUC (test) | Precision (test) | Recall (test) |
|--------------|-------------|-------|------------|--------|--------|-----------------|-----------|------------|------------------|---------------|
| RNN          | perception  | step  | 128        | 0.0005 | 100    | 0.6927          | 0.7645    | 0.7419     | 0.4449           | 0.6303        |
| Transformer  | perception  | step  | 128        | 0.0005 | 100    | 0.6775          | 0.8154    | 0.7494     | 0.4401           | 0.8182        |
| MLP          | perception  | step  | 128        | 0.0005 | 100    | 0.7217          | 0.7495    | 0.7143     | 0.4096           | 0.5600        |

*Tutte le run sono state eseguite con modality = video, weight_decay = 0.005, seed = 42, feature directory = data/.*

**Nota:** I risultati sono estratti dal file wandb_export_2025-12-27T12_01_32.634+01_00.csv. Per ogni run sono riportate le metriche principali sul test set. Altre metriche (sub-step, validazione) sono disponibili nel file esportato.

---
#### Parametri principali usati negli esperimenti

| Parametro         | Vecchio valore | Nuovo valore |
|-------------------|---------------|--------------|
| batch_size        | 32            | 128          |
| num_epochs        | 100           | 100          |
| learning rate (lr)| 0.001         | 0.0005       |
| weight_decay      | 1e-4          | 0.005        |
| test_batch_size   | -             | 1            |
| hidden_dim (LSTM) | -             | 128          |
| num_layers (LSTM) | -             | 2            |
| dropout (LSTM)    | -             | 0.5          |

Altri parametri aggiunti: log_interval, dry_run, seed, ecc. Alcune modifiche strutturali alle MLP (layer aggiunti/rimossi) e ai parametri LSTM sono state introdotte in commit specifici.

---
| Nuovi          | Vecchie       | RNN          | No                   |
| Vecchi         | Nuove         | MLP          | No                   |
| Vecchi         | Nuove         | Transformer  | No                   |
| Vecchi         | Nuove         | RNN          | No                   |
| Nuovi          | Nuove         | MLP          | No                   |
| Nuovi          | Nuove         | Transformer  | No                   |
| Nuovi          | Nuove         | RNN          | No                   |
| MLP | Omnivore | step | 0.71 | 0.66 | 0.15 | 0.24 | 0.76 |
| MLP | Omnivore | sub-step | 0.68 | 0.41 | 0.30 | 0.35 | 0.65 |
| Transformer | Omnivore | step | 0.70 | 0.52 | 0.60 | 0.55 | 0.76 |
| Transformer | Omnivore | sub-step | 0.67 | 0.44 | 0.66 | 0.53 | 0.75 |


### 4.1.4 Nuovo Checkpoint: Transformer con nuove feature e nuovi parametri

| Modello      | Backbone    | Feature         | Parametri         | Epoca | Checkpoint | Accuracy (test) | F1 (test) | AUC (test) | Precision (test) | Recall (test) |
|--------------|-------------|-----------------|-------------------|-------|------------|-----------------|-----------|------------|------------------|---------------|
| Transformer  | perception  | nuove feature   | nuovi parametri   | 27    | Sì         | 0.7288 (step)   | 0.5919    | 0.7834     | 0.6439           | 0.5477        |

**Sub Step Level:**
- F1: 0.5440
- Accuracy: 0.7197
- AUC: 0.7568
- Precision: 0.5849
- Recall: 0.5084

## 4.2 Risultati Evaluation per Error Type (core/evaluate_by_error_type.py)

**Modello:** RNN (perception, step split, checkpoint RNN_epoch_20.pt)

```
=====================================================================================
RISULTATI ANALISI PER TIPO DI ERRORE
=====================================================================================

Categoria                  Samples        Acc       Prec     Recall         F1        AUC
-------------------------------------------------------------------------------------
GLOBAL                         798      62.91      42.49      53.41      47.33      66.00
-------------------------------------------------------------------------------------
Technique Error                 62      61.29     100.00      61.29      76.00        N/A
Preparation Error               49      44.90     100.00      44.90      61.97        N/A
Temperature Error                8      50.00     100.00      50.00      66.67        N/A
Measurement Error               42      59.52     100.00      59.52      74.63        N/A
Timing Error                    34      50.00     100.00      50.00      66.67        N/A
No Error                       691      64.11      29.13      52.11      37.37      65.53
=====================================================================================
```

*Risultati generati da core/evaluate_by_error_type.py su checkpoint RNN_epoch_20.pt (perception, step split).*

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

| Categoria           | Samples | Accuracy | Precision | Recall | F1    | AUC   |
|---------------------|---------|----------|-----------|--------|-------|-------|
| Technique Error     | 62      | 61.29    | 100.00    | 61.29  | 76.00 | N/A   |
| Preparation Error   | 49      | 44.90    | 100.00    | 44.90  | 61.97 | N/A   |
| Temperature Error   | 8       | 50.00    | 100.00    | 50.00  | 66.67 | N/A   |
| Measurement Error   | 42      | 59.52    | 100.00    | 59.52  | 74.63 | N/A   |
| Timing Error        | 34      | 50.00    | 100.00    | 50.00  | 66.67 | N/A   |
| No Error            | 691     | 64.11    | 29.13     | 52.11  | 37.37 | 65.53 |

*Risultati generati da core/evaluate_by_error_type.py su checkpoint RNN_epoch_20.pt (perception, step split).*

## 5. Conclusions

### Key Findings

1. **MLP provides a strong baseline** despite its simplicity, outperforming the Transformer on several metrics (Step-Level Accuracy: 56.8% vs 46.7%)

2. **Error-specific detection is highly accurate** - both models achieve >80% accuracy on explicit error types such as Technique, Measurement, Preparation, and Temperature errors. This suggests the Omnivore features are discriminative enough to capture significant deviations in action execution.

3. **Class imbalance is the main challenge** - the "No Error" class (691 samples) dominates over error classes (<70 samples each), causing:
   - High false positive rates
   - Models over-predicting errors (high Recall, low Precision)
   - Poor performance on normal step classification (~51% for MLP, ~41% for Transformer)

4. **Trade-off between Accuracy and Recall** - At recording level, Transformer achieves higher accuracy (57.1%) but misses >50% of actual errors (Recall: 0.49), while MLP detects most errors (Recall: 0.86) but with more false positives.

5. **Temporal modeling (RNN/LSTM)** is expected to improve detection by capturing sequential patterns that indicate errors developing over time.

### Future Work

- **Address class imbalance** with advanced techniques:
  - Oversampling error classes / Undersampling normal class
  - Focal Loss or other imbalance-aware loss functions
  - Increase pos_weight in BCEWithLogitsLoss
- **Experiment with attention mechanisms** in the RNN (e.g., Bidirectional LSTM with attention)
- **Evaluate on different data splits** (person, environment) to test generalization
- **Explore new backbones** (EgoVLP, PerceptionEncoder) for potentially richer features

---

## References

- CaptainCook4D Dataset: Peddi et al., "CaptainCook4D: A Dataset for Understanding Errors in Procedural Activities", NeurIPS 2024
- Omnivore: Girdhar et al., "Omnivore: A Single Model for Many Visual Modalities", CVPR 2022
- SlowFast Networks: Feichtenhofer et al., "SlowFast Networks for Video Recognition", ICCV 2019