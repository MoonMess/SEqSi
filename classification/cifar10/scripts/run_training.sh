#!/bin/bash

# ===================================================================================
# GENERAL TRAINING SCRIPT FOR REPRODUCIBILITY
#
# This script launches the training for all models and their variants
# (normalization, dropout, data augmentation) across multiple random seeds.
# It is designed for reviewers to reproduce the experiments locally.
#
# Usage:
# 1. Make sure the Conda environment is activated:
#    conda activate <your_env_name>
# 2. Run the script from the project root:
#    bash script/run_training.sh
# ===================================================================================

# Exit the script if any command fails.
set -euo pipefail

# --- Configuration ---

# Random seeds for experiment reproducibility.
SEEDS=(5 7 42 137 181) # Use a subset for faster testing (e.g., SEEDS=(42))

# Base models to be trained with all variants (norm, dropout, aug).
MODELS_BASE=("standard" "se" "seqsi_avg" "affeq_avg")

# Special models (e.g., telescopic) to be trained without additional variants.
MODELS_SPECIAL=("seqsi_telescopic" "affeq_telescopic")

# Normalization layer variants to test.
NORMS=("instance" "batch" "layer" "mix")

# Dropout rates to test.
DROPOUTS=(0.3)

# Data augmentation strategies to test.
AUGS=("affine_aug" "nonaffine_aug" "random_aug")

# Path to the dataset (adjust if necessary).
DATA_PATH="./data"

# Number of epochs. Reduced to 1 for faster execution by reviewers.
# The original value of 500 can be restored for full convergence.
EPOCHS=500

# --- Environment Activation and GPU Check ---

# Make sure your Conda environment is activated before running the script.
echo "Verifying environment and GPU..."
nvidia-smi

# --- Main Training Loop ---

for seed in "${SEEDS[@]}"; do
    echo "================================================="
    echo "===== STARTING TRAINING FOR SEED: $seed ====="
    echo "================================================="

    # --- 1. Train base models with their variants ---
    for model in "${MODELS_BASE[@]}"; do
        # a) Base training (no specific norm/dropout/aug)
        echo "--- Training: Model=${model}, Variant=Base ---"
        python train.py --model "$model" --seed "$seed" --data_path "$DATA_PATH" --epochs "$EPOCHS"

        # b) Variants with normalization
        for norm in "${NORMS[@]}"; do
            echo "--- Training: Model=${model}, Variant=Norm(${norm}) ---"
            python train.py --model "$model" --seed "$seed" --data_path "$DATA_PATH" --epochs "$EPOCHS" --norm_type "$norm"
        done

        # c) Variant with dropout
        for dp in "${DROPOUTS[@]}"; do
            echo "--- Training: Model=${model}, Variant=Dropout(${dp}) ---"
            python train.py --model "$model" --seed "$seed" --data_path "$DATA_PATH" --epochs "$EPOCHS" --dropout_rate "$dp"
        done

        # d) Variants with data augmentation
        for aug in "${AUGS[@]}"; do
            echo "--- Training: Model=${model}, Variant=Aug(${aug}) ---"
            python train.py --model "$model" --seed "$seed" --data_path "$DATA_PATH" --epochs "$EPOCHS" --"$aug"
        done
    done

    # --- 2. Train special models (without variants) ---
    for model in "${MODELS_SPECIAL[@]}"; do
        echo "--- Training: Special Model=${model} ---"
        python train.py --model "$model" --seed "$seed" --data_path "$DATA_PATH" --epochs "$EPOCHS"
    done
done

echo "All training runs are complete."

