#!/bin/bash

# Script per monitorare lo status del training task verification
# Usage: ./extension_task_verification/check_training_status.sh

cd "$(dirname "$0")/.." || exit 1

LOG_FILE="extension_task_verification/training_hiero.log"
CKPT_DIR="extension_task_verification/checkpoints_hiero"

echo "=========================================="
echo "TASK VERIFICATION TRAINING STATUS"
echo "=========================================="
echo ""

# Check if process is running
if ps aux | grep -v grep | grep -q "train_task_verification_hiero"; then
    echo "✓ Training process: RUNNING"
    PID=$(ps aux | grep "train_task_verification_hiero" | grep -v grep | awk '{print $2}' | head -1)
    echo "  PID: $PID"
else
    echo "✗ Training process: NOT RUNNING"
fi

echo ""

# Check log file
if [ -f "$LOG_FILE" ]; then
    echo "=== Last 10 lines of log ==="
    tail -10 "$LOG_FILE"
    echo ""
    
    # Count folds completed
    FOLDS_COMPLETED=$(grep -c "Saved checkpoint:" "$LOG_FILE" 2>/dev/null || echo "0")
    TOTAL_FOLDS=$(grep -c "FOLD.*Holding out Recipe" "$LOG_FILE" 2>/dev/null || echo "0")
    
    if [ "$TOTAL_FOLDS" != "0" ] && [ -n "$TOTAL_FOLDS" ]; then
        echo "Progress: $FOLDS_COMPLETED/$TOTAL_FOLDS folds completed"
    elif [ "$FOLDS_COMPLETED" != "0" ]; then
        echo "Folds completed: $FOLDS_COMPLETED"
    fi
else
    echo "Log file not found: $LOG_FILE"
fi

echo ""

# Check checkpoints
if [ -d "$CKPT_DIR" ]; then
    CHECKPOINTS=$(ls -1 "$CKPT_DIR"/*.pth 2>/dev/null | wc -l)
    echo "Checkpoints saved: $CHECKPOINTS"
    if [ "$CHECKPOINTS" -gt 0 ]; then
        echo "Latest checkpoint:"
        ls -lht "$CKPT_DIR"/*.pth 2>/dev/null | head -3
    fi
else
    echo "Checkpoint directory not found: $CKPT_DIR"
fi

echo ""
echo "=========================================="
echo "To see live updates: tail -f $LOG_FILE"
echo "=========================================="


