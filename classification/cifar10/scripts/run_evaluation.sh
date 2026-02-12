#!/bin/bash

# ===================================================================================
# GENERAL EVALUATION SCRIPT FOR REPRODUCIBILITY
#
# This script runs two types of evaluations on the trained models:
# 1. Robustness evaluation: tests accuracy on noisy data using `test.py`.
# 2. Equivariance evaluation: measures the stability of model logits using `evaluate_invariance.py`.
#
# Usage:
# 1. Ensure that models have been trained using `run_training.sh`.
# 2. Activate your Conda environment.
# 3. Run the script from the project root:
#    bash script/run_evaluation.sh
# ===================================================================================

# Exit the script if any command fails.
set -euo pipefail

# --- Configuration ---

# Must match the seeds used for training.
SEEDS=(5 7 42 137 181)

# Output directories for results.
ROBUSTNESS_RESULTS_DIR="results/robustness_evaluation"
EQUIVARIANCE_RESULTS_DIR="results/equivariance_evaluation"

# --- Environment Activation and GPU Check ---

echo "Verifying environment and GPU..."
nvidia-smi

# --- Part 1: Robustness Evaluation (test.py) ---

echo "========================================================"
echo "===== STARTING ROBUSTNESS EVALUATION (test.py) ====="
echo "========================================================"

for seed in "${SEEDS[@]}"; do
    CHECKPOINT_DIR="checkpoint/${seed}"
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "Warning: Checkpoint directory not found for seed $seed. Skipping."
        continue
    fi

    echo "--- Processing seed: $seed ---"
    # Find all checkpoints (.pth) for the current seed and evaluate them.
    find "$CHECKPOINT_DIR" -name "*.pth" | while read -r checkpoint_file; do
        echo "  -> Evaluating: $(basename "$checkpoint_file")"
        # Evaluation 
        python test.py -c "$checkpoint_file" --output_dir "${ROBUSTNESS_RESULTS_DIR}/${seed}" --seed "$seed"
    done
done

# --- Part 2: Equivariance/Invariance Evaluation (evaluate_invariance.py) ---

echo ""
echo "======================================================================="
echo "===== STARTING EQUIVARIANCE EVALUATION (evaluate_invariance.py) ====="
echo "======================================================================="

# This evaluation is specific to telescopic models.
for seed in "${SEEDS[@]}"; do
    CHECKPOINT_DIR="checkpoint/${seed}"
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "Warning: Checkpoint directory not found for seed $seed. Skipping."
        continue
    fi

    # Evaluate all models except those trained with data augmentation.
    # This evaluation is specific to non-augmented telescopic models.
    find "$CHECKPOINT_DIR" -name "*_telescopic_*.pth" | grep -vE "_affine_aug|_nonaffine_aug|_random_aug" | while read -r checkpoint_file; do
        echo "--- Processing seed $seed for equivariance: $(basename "$checkpoint_file") ---"

        # Evaluation in float32 precision (standard).
        echo "  -> Evaluating in standard precision (float32)..."
        python evaluate_invariance.py -c "$checkpoint_file" --output_dir "${EQUIVARIANCE_RESULTS_DIR}/${seed}" --seed "$seed"

        # Evaluation in float64 precision for a stricter numerical check.
        echo "  -> Evaluating for stricter numerical check (intended as float64)..."
        python evaluate_invariance.py -c "$checkpoint_file" --output_dir "${EQUIVARIANCE_RESULTS_DIR}/${seed}" --seed "$seed" --float64
    done
done

echo ""
echo "All evaluations are complete."
