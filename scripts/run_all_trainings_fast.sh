#!/bin/bash

# Script per eseguire tutti i training con parallelizzazione aggressiva
# Esegue max 3 training in parallelo (per GPU potenti come A100)

# Parametri comuni
EPOCHS=50
SPLIT="step"
MODALITY="video"
MAX_PARALLEL=3  # Numero massimo di training paralleli (più aggressivo)

# Arrays di combinazioni
VARIANTS=("MLP" "RNN" "Transformer")
BACKBONES=("omnivore" "perception")

# Configurazioni
CONFIG_OLD_PW="2.5"
CONFIG_OLD_LR="1e-3"
CONFIG_OLD_WD="1e-3"

CONFIG_NEW_PW="1.6"
CONFIG_NEW_LR="5e-4"
CONFIG_NEW_WD="5e-3"

# Funzione per modificare pos_weight in base.py
modify_pos_weight() {
    local pos_weight=$1
    sed -i "s/pos_weight=torch.tensor(\[[0-9.]*\]/pos_weight=torch.tensor([$pos_weight]/" /home/pierpaolosorbellini/AML/aml-2025-mistake-detection/base.py
    echo "✓ pos_weight modificato a $pos_weight in base.py"
}

# Funzione per eseguire training
run_training() {
    local variant=$1
    local backbone=$2
    local lr=$3
    local weight_decay=$4
    local config_name=$5
    
    echo ""
    echo "=========================================="
    echo "Avvio: $variant | $backbone | $config_name"
    echo "=========================================="
    
    export PATH=$(echo $PATH | tr ":" "\n" | grep -v pyenv | tr "\n" ":" | sed "s/:$//")
    
    cd /home/pierpaolosorbellini/AML/aml-2025-mistake-detection
    
    local log_file="nohup_${config_name}_${variant}_${backbone}.out"
    
    nohup bash -c "
        export PATH=\$(echo \$PATH | tr ':' '\n' | grep -v pyenv | tr '\n' ':' | sed 's/:$//')
        conda activate aml
        python train_er.py \
            --variant '$variant' \
            --backbone '$backbone' \
            --split '$SPLIT' \
            --modality '$MODALITY' \
            --num_epochs $EPOCHS \
            --lr $lr \
            --weight_decay $weight_decay
    " > "$log_file" 2>&1 &
    
    local pid=$!
    echo "✓ Training avviato (PID: $pid, log: $log_file)"
    echo $pid
}

# Funzione per attendere che ci siano meno di MAX_PARALLEL processi
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 5
    done
}

echo ""
echo "###############################################"
echo "### MODALITÀ FAST: $MAX_PARALLEL training paralleli ###"
echo "###############################################"

# ===========================================
# CONFIGURAZIONE VECCHIA (pos_weight=2.5)
# ===========================================
echo ""
echo "### CONFIGURAZIONE VECCHIA (pos_weight=2.5) ###"
modify_pos_weight $CONFIG_OLD_PW

for variant in "${VARIANTS[@]}"; do
    for backbone in "${BACKBONES[@]}"; do
        wait_for_slot
        run_training "$variant" "$backbone" "$CONFIG_OLD_LR" "$CONFIG_OLD_WD" "OLD"
        sleep 2
    done
done

echo ""
echo "Attendo completamento di tutti i training OLD..."
wait

# ===========================================
# CONFIGURAZIONE NUOVA (pos_weight=1.6)
# ===========================================
echo ""
echo "### CONFIGURAZIONE NUOVA (pos_weight=1.6) ###"
modify_pos_weight $CONFIG_NEW_PW

for variant in "${VARIANTS[@]}"; do
    for backbone in "${BACKBONES[@]}"; do
        wait_for_slot
        run_training "$variant" "$backbone" "$CONFIG_NEW_LR" "$CONFIG_NEW_WD" "NEW"
        sleep 2
    done
done

echo ""
echo "Attendo completamento di tutti i training NEW..."
wait

# Ripristina pos_weight a default
modify_pos_weight 1.5

echo ""
echo "=========================================="
echo "✓✓✓ TUTTI I TRAINING COMPLETATI ✓✓✓"
echo "=========================================="
echo ""
echo "Log files:"
ls -lh nohup_*_*.out 2>/dev/null


