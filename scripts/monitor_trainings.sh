#!/bin/bash

# Script per monitorare i training in corso

echo "=========================================="
echo "MONITORAGGIO TRAINING IN CORSO"
echo "=========================================="
echo ""

# Processi attivi
echo "=== Processi Python Attivi ==="
ps aux | grep "python train_er.py" | grep -v grep | awk '{print "PID:", $2, "| CPU:", $3"%", "| MEM:", $4"%", "| CMD:", $11, $12, $13, $14, $15, $16}'

echo ""
echo "=== Numero totale processi: $(ps aux | grep 'python train_er.py' | grep -v grep | wc -l) ==="

echo ""
echo "=========================================="
echo "ULTIMI LOG PER OGNI TRAINING"
echo "=========================================="

# Trova tutti i file di log e mostra gli ultimi progressi
for log in nohup_*_*.out; do
    if [ -f "$log" ]; then
        echo ""
        echo "--- $log ---"
        # Mostra l'ultima riga con "Epoch:"
        tail -20 "$log" | grep "Epoch:" | tail -1
    fi
done

echo ""
echo "=========================================="
echo "CHECKPOINTS SALVATI"
echo "=========================================="
find checkpoints/error_recognition/ -name "*.pt" -type f -mmin -120 | sort -r | head -20

echo ""
echo "=========================================="
echo "Per vedere log completo: tail -f nohup_[CONFIG]_[VARIANT]_[BACKBONE].out"
echo "Per killare tutti: pkill -f 'python train_er.py'"
echo "=========================================="


