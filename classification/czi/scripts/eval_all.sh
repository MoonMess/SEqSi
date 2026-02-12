#!/bin/bash

# This script finds all 'bestmodel.ckpt' files in the checkpoints directory
# and runs evaluation on them.
#
# Usage:
# ./scripts/eval_all.sh /path/to/classification_data
#
# Example:
# ./scripts/eval_all.sh ./data/classification

# --- Configuration ---
CHECKPOINTS_DIR="./checkpoints"
DATA_ROOT=$1

# --- Validation ---
if [ -z "$DATA_ROOT" ]; then
    echo "Error: Missing data root argument."
    echo "Usage: $0 <data_root_path>"
    exit 1
fi

if [ ! -d "$CHECKPOINTS_DIR" ]; then
    echo "Error: Checkpoints directory '$CHECKPOINTS_DIR' not found."
    exit 1
fi

echo "Starting evaluation for all found models..."
echo "Data Root: $DATA_ROOT"
echo "========================================"

# --- Evaluation Loop ---
find "$CHECKPOINTS_DIR" -name "bestmodel.ckpt" | while read -r ckpt_path; do
    # Extract the seed from the checkpoint path.
    # The path is expected to be like: ./checkpoints/SEED/run_name/bestmodel.ckpt
    # We use cut to get the 3rd field using '/' as a delimiter.
    seed=$(echo "$ckpt_path" | cut -d'/' -f3)

    echo ""
    echo "--- Evaluating checkpoint: $ckpt_path (using seed: $seed) ---"

    python eval.py \
        --checkpoint "$ckpt_path" \
        --data_root "$DATA_ROOT" \
        --seed "$seed"
done

echo "========================================"
echo "All evaluations completed."

