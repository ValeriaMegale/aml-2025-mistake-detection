#!/usr/bin/env python3
"""
Script per monitorare lo stato di training e evaluation delle extension tasks.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

def check_processes():
    """Controlla processi di training in background."""
    import subprocess
    processes = []
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        for line in lines:
            if 'train_gnn' in line or 'train_task_graph' in line or 'train_task_verification' in line:
                if 'grep' not in line and 'monitor' not in line:
                    processes.append(line)
    except:
        pass
    return processes

def count_checkpoints(dir_path, pattern):
    """Conta i checkpoint disponibili."""
    if not os.path.exists(dir_path):
        return 0, []
    files = list(Path(dir_path).glob(pattern))
    return len(files), sorted([f.name for f in files])

def get_log_tail(log_path, n_lines=5):
    """Ottiene le ultime righe del log."""
    if not os.path.exists(log_path):
        return "Log non trovato"
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            return ''.join(lines[-n_lines:])
    except:
        return "Errore leggendo log"

def check_results_dir(dir_path):
    """Verifica se esistono risultati."""
    if not os.path.exists(dir_path):
        return False
    files = list(Path(dir_path).glob('*'))
    return len(files) > 0

def main():
    print("="*80)
    print("MONITORAGGIO STATO EXTENSION TASKS")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Processi in background
    print("🔍 PROCESSI IN BACKGROUND")
    print("-" * 80)
    processes = check_processes()
    if processes:
        for proc in processes:
            # Estrai PID e comando
            parts = proc.split()
            if len(parts) >= 11:
                pid = parts[1]
                cpu = parts[2]
                mem = parts[3]
                cmd = ' '.join(parts[10:])
                print(f"  ✅ PID {pid} | CPU: {cpu}% | MEM: {mem}%")
                print(f"     Comando: {cmd[:70]}...")
        print()
    else:
        print("  ⚪ Nessun processo di training in background\n")
    
    # 2. Substep 2: Task Verification Baseline
    print("📊 SUBSTEP 2: Task Verification Baseline")
    print("-" * 80)
    ckpt_dir = "extension/substep2_task_verification/checkpoints_hiero"
    count, files = count_checkpoints(ckpt_dir, "*.pth")
    print(f"  Checkpoint disponibili: {count}/24")
    if count == 24:
        print("  ✅ Completato!")
    elif count > 0:
        print(f"  ⚠️  Parziale ({count}/24)")
    else:
        print("  ❌ Nessun checkpoint")
    
    results_dir = "results/extension/substep2_task_verification"
    if check_results_dir(results_dir):
        print("  ✅ Risultati disponibili")
    else:
        print("  ⚠️  Risultati non trovati (eseguire eval_task_verification_hiero.py)")
    print()
    
    # 3. Substep 3: Task Graph Matching
    print("📊 SUBSTEP 3: Task Graph Matching")
    print("-" * 80)
    ckpt_dir = "extension/substep3_task_graph_matching/checkpoints_graph_test"
    count, files = count_checkpoints(ckpt_dir, "*.pth")
    print(f"  Checkpoint disponibili: {count}/24")
    if count == 24:
        print("  ✅ Completato!")
    elif count > 0:
        print(f"  ⚠️  Parziale ({count}/24)")
        missing = 24 - count
        print(f"  📋 Mancanti: {missing} ricette")
    else:
        print("  ❌ Nessun checkpoint")
    
    results_dir = "results/extension/substep3_task_graph_matching"
    if check_results_dir(results_dir):
        result_files = list(Path(results_dir).glob("*"))
        print(f"  ✅ Risultati disponibili ({len(result_files)} files)")
        # Verifica se c'è il summary
        if (Path(results_dir) / "results_summary.csv").exists():
            print("  ✅ Summary CSV disponibile")
    else:
        print("  ⚠️  Risultati non trovati")
    print()
    
    # 4. Substep 4: GNN Classification
    print("📊 SUBSTEP 4: GNN Classification")
    print("-" * 80)
    ckpt_dir = "extension/substep4_gnn_classification/checkpoints_gnn"
    count, files = count_checkpoints(ckpt_dir, "*.pth")
    print(f"  Checkpoint disponibili: {count}/24")
    if count == 24:
        print("  ✅ Completato!")
    elif count > 0:
        print(f"  ⚠️  In corso ({count}/24)")
        print(f"  📋 Mancanti: {24-count} ricette")
        # Mostra ricette completate
        completed = [f.replace('gnn_classifier_recipe_', '').replace('.pth', '') 
                    for f in files if 'recipe_' in f]
        if completed:
            print(f"  ✅ Ricette completate: {', '.join(sorted(completed)[:10])}")
            if len(completed) > 10:
                print(f"     ... e altre {len(completed)-10}")
    else:
        print("  ⚠️  Training non ancora completato")
    
    # Verifica log e progresso
    log_path = "/tmp/train_gnn_full.log"
    if os.path.exists(log_path):
        print(f"  📝 Log disponibile: {log_path}")
        # Cerca ultima ricetta in training
        with open(log_path, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "Training for Recipe" in line:
                    print(f"  🔄 {line.strip()}")
                    break
            # Cerca ultimo epoch
            for line in reversed(lines[-50:]):
                if "Epoch" in line and "/50" in line:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        epoch_info = parts[1]  # Epoch X/50
                        print(f"  📊 {epoch_info}: {line.strip()[line.find('['):]}")
                        break
    else:
        print("  ⚠️  Log non trovato")
    
    results_dir = "results/extension/substep4_gnn_classification"
    if check_results_dir(results_dir):
        print("  ✅ Risultati disponibili")
    else:
        print("  ⚠️  Risultati non disponibili (eseguire eval_gnn_classification.py dopo training)")
    print()
    
    # 5. Report
    print("📄 REPORT")
    print("-" * 80)
    report_path = "REPORT.md"
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            content = f.read()
            has_substep2 = "Substep 2" in content and "0.5855" in content
            has_substep3 = "Substep 3" in content and "0.5395" in content
            has_substep4 = "Substep 4" in content
            
            print(f"  ✅ Report presente")
            if has_substep2:
                print("  ✅ Substep 2 risultati nel report")
            if has_substep3:
                print("  ✅ Substep 3 risultati nel report")
            if has_substep4:
                print("  ⚠️  Substep 4 sezione presente (risultati da aggiornare)")
    else:
        print("  ❌ Report non trovato")
    print()
    
    # 6. Dati preparati
    print("📦 DATI PREPARATI")
    print("-" * 80)
    data_dirs = {
        "Step embeddings (HiERO)": "extension/substep1_step_localization/data/step_embeddings_perception.npy",
        "Graph data (Substep 4)": "extension/substep4_gnn_classification/graph_classification_substep4/data/metadata.json"
    }
    for name, path in data_dirs.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024*1024)  # MB
            print(f"  ✅ {name}: {size:.1f} MB")
        else:
            print(f"  ❌ {name}: Non trovato")
    print()
    
    # 7. Summary
    print("="*80)
    print("RIEPILOGO")
    print("="*80)
    
    substep2_complete = count_checkpoints("extension_task_verification/checkpoints_hiero", "*.pth")[0] == 24
    substep3_complete = count_checkpoints("substep3_step_detection/checkpoints_graph_test", "*.pth")[0] >= 10
    substep4_complete = count_checkpoints("substep4/checkpoints_gnn", "*.pth")[0] == 24
    
    total_tasks = 3
    completed = sum([substep2_complete, substep3_complete, substep4_complete])
    
    print(f"Substep 2: {'✅ Completato' if substep2_complete else '⚠️  In corso/Parziale'}")
    print(f"Substep 3: {'✅ Completato' if substep3_complete else '⚠️  In corso/Parziale'}")
    print(f"Substep 4: {'✅ Completato' if substep4_complete else '⚠️  In corso/Parziale'}")
    print()
    print(f"Progresso complessivo: {completed}/{total_tasks} completati")
    
    if processes:
        print(f"\n⚠️  ATTENZIONE: {len(processes)} processo(i) di training in background")
        print("   Usa 'ps aux | grep train' per vedere i dettagli")
        print("   Usa 'tail -f /tmp/train_gnn_full.log' per monitorare il progresso")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
