#!/bin/bash

# Script per eseguire i 6 training mancanti con learning rate 1e-3
# MLP, RNN, Transformer - sia perception che omnivore

# Parametri comuni
EPOCHS=100
SPLIT="step"
MODALITY="video"
LR="1e-3"
WEIGHT_DECAY="5e-3"  # Usa lo stesso weight_decay della configurazione NEW
MAX_PARALLEL=2  # Numero massimo di training paralleli

# Arrays di combinazioni
VARIANTS=("MLP" "RNN" "Transformer")
BACKBONES=("perception" "omnivore")

# Funzione per eseguire training
run_training() {
    local variant=$1
    local backbone=$2
    
    echo ""
    echo "=========================================="
    echo "Avvio: $variant | $backbone | LR=$LR"
    echo "=========================================="
    
    # Rimuovi pyenv e attiva conda
    export PATH=$(echo $PATH | tr ":" "\n" | grep -v pyenv | tr "\n" ":" | sed "s/:$//")
    
    cd /home/pierpaolosorbellini/AML/aml-2025-mistake-detection
    
    # Nome file log
    local log_file="nohup_LR1E3_${variant}_${backbone}.out"
    
    # Esegui il training in background
    nohup bash -c "
        export PATH=\$(echo \$PATH | tr ':' '\n' | grep -v pyenv | tr '\n' ':' | sed 's/:$//')
        source \$(conda info --base)/etc/profile.d/conda.sh
        conda activate aml
        python train_er.py \
            --variant '$variant' \
            --backbone '$backbone' \
            --split '$SPLIT' \
            --modality '$MODALITY' \
            --num_epochs $EPOCHS \
            --lr $LR \
            --weight_decay $WEIGHT_DECAY
    " > "$log_file" 2>&1 &
    
    local pid=$!
    echo "✓ Training avviato in background (PID: $pid, log: $log_file)"
    echo $pid
}

# Funzione per attendere che ci siano meno di MAX_PARALLEL processi
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 5
    done
}

# Array per tracciare i PID
declare -a PIDS=()

echo ""
echo "###############################################"
echo "### TRAINING CON LEARNING RATE 1e-3         ###"
echo "###############################################"
echo ""

for variant in "${VARIANTS[@]}"; do
    for backbone in "${BACKBONES[@]}"; do
        wait_for_slot
        pid=$(run_training "$variant" "$backbone")
        PIDS+=($pid)
        sleep 2  # Piccolo delay per evitare race conditions
    done
done

# Attendi che tutti i training siano completati
echo ""
echo "Attendo completamento di tutti i training..."
wait

echo ""
echo "=========================================="
echo "✓✓✓ TUTTI I TRAINING COMPLETATI ✓✓✓"
echo "=========================================="
echo ""
echo "Log files:"
ls -lh nohup_LR1E3_*.out 2>/dev/null
echo ""
echo "Checkpoints salvati in: ./checkpoints/error_recognition/"
