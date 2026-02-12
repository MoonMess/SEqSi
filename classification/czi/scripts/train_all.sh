#!/bin/bash

# This script automates the training process for multiple seeds and model types.
#
# Usage:
# ./scripts/train_all.sh /path/to/classification_data <train_tomo_type>
#
# Example:
# ./scripts/train_all.sh ./data/classification wbp

# --- Configuration ---
SEEDS=(5 7 42 137 181)
MODES=("standard" "seqsi")
DATA_ROOT=$1
TRAIN_TYPE=$2

# --- Validation ---
if [ -z "$DATA_ROOT" ] || [ -z "$TRAIN_TYPE" ]; then
    echo "Error: Missing arguments."
    echo "Usage: $0 <data_root_path> <train_tomo_type>"
    exit 1
fi

if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data root directory '$DATA_ROOT' not found."
    exit 1
fi

echo "Starting training runs..."
echo "Data Root: $DATA_ROOT"
echo "Training on Tomogram Type: $TRAIN_TYPE"
echo "Seeds: ${SEEDS[*]}"
echo "Models: ${MODES[*]}"
echo "========================================"

# --- Training Loop ---
for seed in "${SEEDS[@]}"; do
    for mode in "${MODES[@]}"; do
        echo ""
        echo "--- Training model: $mode | Seed: $seed ---"

        python train.py \
            --data_root "$DATA_ROOT" \
            --tomo_type_train "$TRAIN_TYPE" \
            --mode "$mode" \
            --seed "$seed" \
            --comment "automated_run"
    done
done

echo "========================================"
echo "All training runs completed."

