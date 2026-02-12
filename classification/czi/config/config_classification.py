import torch
import os

# --- 1. Data and Paths Configuration ---
# Path to the root directory for a specific tomogram type (e.g., 'denoised').
# This directory should contain 'train', 'val', and 'test' subfolders.
# Example: "/path/to/your/classification_data/denoised/"
DATASET_DIR = "./data/" # Please adjust this path

# --- 2. Model Parameters ---
NUM_CLASSES = 6
IN_CHANNELS = 1

MODEL_MODE = 'standard' # Options: 'standard', 'seqsi'

# Class names for logging. ORDER MUST MATCH the IDs in particle_config.py
CLASS_NAMES = [
    'apo-ferritin', 'beta-amylase', 'beta-galactosidase',
    'ribosome', 'thyroglobulin', 'virus-like-particle'
]

# --- 3. Training Parameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
USE_SCHEDULER = True
USE_AUGMENTATION = True
BATCH_SIZE = 8
NUM_WORKERS = 8
EPOCHS = 300

# The target size to which all patches will be padded or cropped.
# e.g., (64, 64, 64)
TARGET_PATCH_SIZE = (64, 64, 64)

# --- 4. Logging and Checkpointing ---
# Frequency of validation. A full validation will be run every N epochs.
VALIDATE_EVERY_N_EPOCHS = 1
# Frequency for logging visualization images to W&B.
LOG_VIS_EVERY_N_EPOCHS = 5
