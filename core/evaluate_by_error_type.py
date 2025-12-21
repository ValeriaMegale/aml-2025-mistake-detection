"""
Modulo per l'analisi delle performance dei modelli per tipo di errore.
Calcola Accuracy, Precision, Recall, F1 e AUC per ogni categoria di errore.

Segue la stessa architettura di core/evaluate.py e base.py
"""

import argparse
import json
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from tqdm import tqdm

from base import fetch_model
from constants import Constants as const
from dataloader.CaptainCookStepDataset import CaptainCookStepDataset


# ===================== CONFIG (stesso stile di evaluate.py) =====================

class Config(object):
    """Wrapper class for model hyperparameters - stesso stile di core/evaluate.py"""
    
    def __init__(self):
        self.backbone = "omnivore"
        self.modality = "video"
        self.phase = "test"
        self.segment_length = 1
        self.segment_features_directory = "data/"
        
        self.ckpt_directory = "./checkpoints/"
        self.split = "recordings"
        self.batch_size = 1
        self.test_batch_size = 1
        self.ckpt = None
        self.seed = 1000
        self.device = "cuda"
        
        self.variant = const.MLP_VARIANT
        self.task_name = const.ERROR_RECOGNITION
        self.error_category = None


# ===================== ERROR CATEGORY MAPPING =====================

ERROR_CATEGORY_NAME_LABEL_MAP = {
    const.TECHNIQUE_ERROR: 6,
    const.PREPARATION_ERROR: 2,
    const.TEMPERATURE_ERROR: 3,
    const.MEASUREMENT_ERROR: 4,
    const.TIMING_ERROR: 5
}

ERROR_CATEGORY_LABEL_NAME_MAP = {
    6: const.TECHNIQUE_ERROR,
    2: const.PREPARATION_ERROR,
    3: const.TEMPERATURE_ERROR,
    4: const.MEASUREMENT_ERROR,
    5: const.TIMING_ERROR,
    0: "No Error"
}


def load_error_annotations():
    """Carica le annotazioni degli errori dal file JSON."""
    with open('annotations/annotation_json/error_annotations.json', 'r') as f:
        error_annotations = json.load(f)
    
    # Costruisci dizionario recording_id -> step_id -> set(error_categories)
    recording_step_errors = {}
    for recording_dict in error_annotations:
        recording_id = recording_dict['recording_id']
        recording_step_errors[recording_id] = {}
        
        for step_dict in recording_dict['step_annotations']:
            step_id = step_dict['step_id']
            error_labels = set()
            
            if "errors" not in step_dict:
                error_labels.add(0)  # No Error
            else:
                for error in step_dict['errors']:
                    tag = error['tag']
                    if tag in ERROR_CATEGORY_NAME_LABEL_MAP:
                        error_labels.add(ERROR_CATEGORY_NAME_LABEL_MAP[tag])
                    else:
                        error_labels.add(0)
            
            recording_step_errors[recording_id][step_id] = error_labels
    
    return recording_step_errors


# ===================== FUNZIONE DI VALUTAZIONE PER ERROR TYPE =====================

def test_er_model_by_error_type(model, test_loader, device, threshold=0.5, 
                                 step_normalization=True, sub_step_normalization=True):
    """
    Valuta il modello calcolando metriche per ogni tipo di errore.
    Basato su test_er_model in base.py ma con analisi per categoria.
    """
    model.eval()
    
    # Carica annotazioni errori
    recording_step_errors = load_error_annotations()
    
    all_targets = []
    all_outputs = []
    test_step_start_end_list = []
    step_error_categories = []  # Lista delle categorie per ogni step
    
    counter = 0
    test_loader_tqdm = tqdm(test_loader, desc="Evaluating by error type")
    
    with torch.no_grad():
        for data, target in test_loader_tqdm:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            sigmoid_output = output.sigmoid()
            all_outputs.append(sigmoid_output.detach().cpu().numpy().reshape(-1))
            all_targets.append(target.detach().cpu().numpy().reshape(-1))
            
            test_step_start_end_list.append((counter, counter + data.shape[0]))
            counter += data.shape[0]
    
    # Flatten
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    
    # =================== STEP LEVEL PROCESSING ===================
    # Ottieni error categories per ogni step dal dataset
    step_error_cats = []
    for idx in range(len(test_loader.dataset)):
        recording_id = test_loader.dataset._step_dict[idx][0]
        step_data = test_loader.dataset._step_dict[idx][1]
        # Prendi le categorie dal primo elemento (sono uguali per tutto lo step)
        error_cats = step_data[0][3]  # (start, end, has_errors, error_category_labels)
        step_error_cats.append(error_cats)
    
    # Calcola output e target a livello di step (come in base.py)
    all_step_targets = []
    all_step_outputs = []
    
    for i, (start, end) in enumerate(test_step_start_end_list):
        step_output = all_outputs[start:end]
        step_target = all_targets[start:end]
        
        step_output = np.array(step_output)
        if end - start > 1 and sub_step_normalization:
            prob_range = np.max(step_output) - np.min(step_output)
            if prob_range > 0:
                step_output = (step_output - np.min(step_output)) / prob_range
        
        mean_step_output = np.mean(step_output)
        step_target_val = 1 if np.mean(step_target) > 0.95 else 0
        
        all_step_outputs.append(mean_step_output)
        all_step_targets.append(step_target_val)
    
    all_step_outputs = np.array(all_step_outputs)
    all_step_targets = np.array(all_step_targets)
    
    # Normalizzazione globale
    if step_normalization:
        prob_range = np.max(all_step_outputs) - np.min(all_step_outputs)
        if prob_range > 0:
            all_step_outputs = (all_step_outputs - np.min(all_step_outputs)) / prob_range
    
    # =================== CALCOLO METRICHE ===================
    results = {}
    
    # Metriche globali
    pred_step_labels = (all_step_outputs > threshold).astype(int)
    results['global'] = compute_metrics(all_step_targets, all_step_outputs, pred_step_labels)
    results['global']['num_samples'] = len(all_step_targets)
    
    # Metriche per ogni categoria di errore
    for cat_label, cat_name in ERROR_CATEGORY_LABEL_NAME_MAP.items():
        # Filtra step che contengono questa categoria
        indices = [i for i, cats in enumerate(step_error_cats) if cat_label in cats]
        
        if len(indices) > 0:
            cat_targets = all_step_targets[indices]
            cat_outputs = all_step_outputs[indices]
            cat_preds = pred_step_labels[indices]
            
            results[cat_name] = compute_metrics(cat_targets, cat_outputs, cat_preds)
            results[cat_name]['num_samples'] = len(indices)
    
    return results


def compute_metrics(targets, outputs, predictions):
    """Calcola tutte le metriche - stesso stile di base.py"""
    metrics = {
        const.ACCURACY: accuracy_score(targets, predictions),
        const.PRECISION: precision_score(targets, predictions, zero_division=0),
        const.RECALL: recall_score(targets, predictions, zero_division=0),
        const.F1: f1_score(targets, predictions, zero_division=0),
    }
    
    # AUC richiede almeno 2 classi
    if len(np.unique(targets)) > 1:
        metrics[const.AUC] = roc_auc_score(targets, outputs)
    else:
        metrics[const.AUC] = None
    
    return metrics


def print_results(results):
    """Stampa i risultati in formato tabellare."""
    print("\n" + "="*85)
    print("RISULTATI ANALISI PER TIPO DI ERRORE")
    print("="*85)
    
    # Header
    print(f"\n{'Categoria':<25} {'Samples':>8} {'Acc':>10} {'Prec':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print("-"*85)
    
    # Global first
    g = results['global']
    auc_str = f"{g[const.AUC]*100:.2f}" if g[const.AUC] is not None else "N/A"
    print(f"{'GLOBAL':<25} {g['num_samples']:>8} {g[const.ACCURACY]*100:>10.2f} "
          f"{g[const.PRECISION]*100:>10.2f} {g[const.RECALL]*100:>10.2f} "
          f"{g[const.F1]*100:>10.2f} {auc_str:>10}")
    print("-"*85)
    
    # Per category
    for cat_name in [const.TECHNIQUE_ERROR, const.PREPARATION_ERROR, const.TEMPERATURE_ERROR, 
                     const.MEASUREMENT_ERROR, const.TIMING_ERROR, "No Error"]:
        if cat_name in results:
            m = results[cat_name]
            auc_str = f"{m[const.AUC]*100:.2f}" if m[const.AUC] is not None else "N/A"
            print(f"{cat_name:<25} {m['num_samples']:>8} {m[const.ACCURACY]*100:>10.2f} "
                  f"{m[const.PRECISION]*100:>10.2f} {m[const.RECALL]*100:>10.2f} "
                  f"{m[const.F1]*100:>10.2f} {auc_str:>10}")
    
    print("="*85)


def save_results(results, config, output_dir="results/error_type_analysis"):
    """Salva i risultati in JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(
        output_dir, 
        f"{config.variant}_{config.backbone}_{config.split}_error_analysis.json"
    )
    
    # Converti per JSON serialization
    json_results = {}
    for k, v in results.items():
        json_results[k] = {mk: float(mv) if mv is not None else None for mk, mv in v.items()}
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    return results_file


# ===================== MAIN (stesso stile di evaluate.py) =====================

def eval_er_by_error_type(config, threshold):
    """Funzione principale di valutazione - stesso pattern di eval_er in evaluate.py"""
    
    # Carica modello
    model = fetch_model(config)
    model.load_state_dict(torch.load(config.ckpt, map_location=config.device))
    model.eval()
    print(f"Loaded model from {config.ckpt}")
    
    # Carica dataset test - usando CaptainCookStepDataset esistente
    from dataloader.CaptainCookStepDataset import collate_fn
    test_dataset = CaptainCookStepDataset(config, const.TEST, config.split)
    # IMPORTANTE: Non usare shuffle=True, altrimenti l'associazione 
    # tra predizioni e categorie di errore sarà sbagliata
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, collate_fn=collate_fn)
    print(f"Test dataset size: {len(test_dataset)} steps")
    
    # Valutazione per error type
    results = test_er_model_by_error_type(
        model, test_loader, config.device, 
        threshold=threshold,
        step_normalization=True, 
        sub_step_normalization=True
    )
    
    # Stampa e salva risultati
    print_results(results)
    save_results(results, config)
    
    return results


if __name__ == "__main__":
    # Parser argomenti - stesso stile di evaluate.py
    parser = argparse.ArgumentParser(description="Evaluate model by error type")
    parser.add_argument("--split", type=str, 
                        choices=[const.STEP_SPLIT, const.RECORDINGS_SPLIT, const.PERSON_SPLIT, const.ENVIRONMENT_SPLIT], 
                        required=True)
    parser.add_argument("--backbone", type=str, 
                        choices=[const.SLOWFAST, const.OMNIVORE], 
                        required=True)
    parser.add_argument("--variant", type=str, 
                        choices=[const.MLP_VARIANT, const.TRANSFORMER_VARIANT, const.RNN_VARIANT], 
                        required=True)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Configura - stesso pattern di evaluate.py
    conf = Config()
    conf.split = args.split
    conf.backbone = args.backbone
    conf.variant = args.variant
    conf.ckpt = args.ckpt
    conf.device = args.device
    
    eval_er_by_error_type(conf, args.threshold)
