# In base.py

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import numpy as np
import torch
from constants import Constants as const

def calculate_metrics(targets, predictions, threshold=0.5):
    """Helper function per calcolare metriche standard"""
    pred_labels = (predictions > threshold).astype(int)
    return {
        const.PRECISION: precision_score(targets, pred_labels, zero_division=0),
        const.RECALL: recall_score(targets, pred_labels, zero_division=0),
        const.F1: f1_score(targets, pred_labels, zero_division=0),
        const.ACCURACY: accuracy_score(targets, pred_labels),
        const.AUC: roc_auc_score(targets, predictions) if len(np.unique(targets)) > 1 else 0.0
    }

def test_er_model(model, test_loader, criterion, device, phase, step_normalization=True, sub_step_normalization=True, threshold=0.6):
    model.eval() # Assicuriamoci che il modello sia in eval mode
    total_samples = 0
    all_targets = []
    all_outputs = []
    
    # Lista per tracciare i tipi di errore per ogni step
    all_error_types = [] 
    
    test_losses = []
    test_step_start_end_list = []
    counter = 0

    # NOTA: Qui assumiamo che il DataLoader o il Dataset ritorni anche l'informazione sul tipo di errore.
    # Se il tuo dataset ritorna solo (data, target), dovrai modificare CaptainCookStepDataset.py 
    # per ritornare (data, target, meta) dove meta contiene 'error_type'.
    # Per ora, modifico il ciclo aspettandomi che tu possa adattare il return del dataloader.
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"{phase} Progress"):
            # Gestione flessibile del batch se contiene metadata extra
            if len(batch) == 3:
                data, target, metadata = batch
                # Estrai error_type dai metadata se presenti, altrimenti usa placeholder
                current_error_types = metadata if isinstance(metadata, list) else [metadata] * data.shape[0]
            else:
                data, target = batch
                current_error_types = ["Unknown"] * data.shape[0] # Placeholder se non disponibile

            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_losses.append(loss.item())

            sigmoid_output = output.sigmoid()
            all_outputs.append(sigmoid_output.detach().cpu().numpy().reshape(-1))
            all_targets.append(target.detach().cpu().numpy().reshape(-1))
            
            # Nota: Questo funziona a livello di sub-step. Per l'analisi a livello di step, 
            # propagheremo queste info dopo il loop.
            
            test_step_start_end_list.append((counter, counter + data.shape[0]))
            counter += data.shape[0]

    # Flatten lists
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    
    # --- Calcolo Metriche STEP LEVEL ---
    all_step_targets = []
    all_step_outputs = []
    
    # Qui dovresti avere una lista di tipi di errore allineata con gli step
    # Poiché la logica originale aggrega i sub-step, dobbiamo fare lo stesso per i tipi di errore.
    # Assumiamo di avere accesso alla lista dei task/errori dal dataset originale se non passati nel loop.
    
    # -------------------------------------------------------------
    # LOGICA AGGREGATA STEP-LEVEL (Come prima)
    # -------------------------------------------------------------
    for start, end in test_step_start_end_list:
        step_output = all_outputs[start:end]
        step_target = all_targets[start:end]

        if start - end > 1 and sub_step_normalization:
             prob_range = np.max(step_output) - np.min(step_output)
             if prob_range > 0:
                step_output = (step_output - np.min(step_output)) / prob_range

        mean_step_output = np.mean(step_output)
        # Target è 1 se la maggioranza dei sub-step sono errore (o logica > 0.95)
        step_target_label = 1 if np.mean(step_target) > 0.95 else 0

        all_step_outputs.append(mean_step_output)
        all_step_targets.append(step_target_label)

    all_step_outputs = np.array(all_step_outputs)
    if step_normalization:
        prob_range = np.max(all_step_outputs) - np.min(all_step_outputs)
        if prob_range > 0:
            all_step_outputs = (all_step_outputs - np.min(all_step_outputs)) / prob_range
    
    all_step_targets = np.array(all_step_targets)
    
    # Calcolo metriche globali
    step_metrics = calculate_metrics(all_step_targets, all_step_outputs, threshold)
    
    # --- NUOVA SEZIONE: ANALISI PER CATEGORIA ---
    # Per far funzionare questo, devi recuperare la lista degli error_types per ogni step processato.
    # Puoi farlo accedendo direttamente al dataset tramite gli indici se il dataloader non li passa.
    
    print("\n--- Analisi per Categoria di Errore ---")
    
    # Recuperiamo le annotazioni direttamente dal dataset usando gli indici processati
    # Nota: Questo richiede che test_loader non sia shuffled per corrispondere sequenzialmente,
    # oppure che testiamo su tutto il dataset in ordine.
    dataset_error_types = []
    if hasattr(test_loader.dataset, 'get_error_types'):
        # Assumiamo di aggiungere un metodo get_error_types al dataset o accedere a un attributo
        dataset_error_types = test_loader.dataset.get_error_types() 
    
    # Se riusciamo a ottenere i tipi, calcoliamo le metriche
    if len(dataset_error_types) == len(all_step_targets):
        unique_errors = set(dataset_error_types)
        if "Normal" in unique_errors: unique_errors.remove("Normal")
        
        category_metrics = {}
        
        for err_type in unique_errors:
            # Creiamo un sottoinsieme: Solo campioni "Normali" + Campioni di "Questo Errore"
            # Ignoriamo gli altri tipi di errore per vedere quanto bene il modello distingue QUESTO errore dal normale.
            indices = [i for i, x in enumerate(dataset_error_types) if x == err_type or x == "Normal"]
            
            if len(indices) > 0:
                filtered_targets = all_step_targets[indices]
                filtered_outputs = all_step_outputs[indices]
                
                metrics = calculate_metrics(filtered_targets, filtered_outputs, threshold)
                category_metrics[err_type] = metrics
                print(f"Error: {err_type} -> F1: {metrics[const.F1]:.4f}, AUC: {metrics[const.AUC]:.4f}")
    else:
        print("Warning: Impossible to match error types to predictions. Implement 'get_error_types' in Dataset.")

    return test_losses, {}, step_metrics