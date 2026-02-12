# CIFAR-10 Classification

This repository contains the code for training and evaluating 2D Convolutional Neural Networks for classifying images from CIFAR-10 dataset.

## Project Structure

```
.
├── scripts/                 # Utility and automation scripts
│   ├── run_training.sh
│   └── run_evaluation.sh
├── transformation/         # Transformation configurations and implementations
│   ├── config.py
│   └── photometric_extensions.py
├── data/                   # Placeholder for data (downloaded by script)
├── output/                 # Default directory for evaluation results and visualizations
├── checkpoint/             # Default directory for saved model checkpoints
├── model.py                # Model definitions (ResNet, ResidualBlock)
├── equiv_blocks.py         # Equivariant building blocks
├── dataloader.py           # Data loading and augmentation
├── utils.py                # Utility functions (seed, model loading, checkpoint naming)
├── train.py                # Main training script
├── evaluate_invariance.py  # Script for evaluating photometric equivariance
└── README.md
```
## Prerequisites & Reproducibility

To reproduce the experiments for the conference submission, please follow these steps. We recommend using Conda to manage the environment.

### 1. Setup Environment

First, create and activate the Conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate <your_env_name> # Replace <your_env_name> with the name specified in environment.yml (e.g., 'NE')
```
## Workflow
The workflow is divided into two main steps: model training, and evaluation. (data are automatically downloaded during training)

### 1. Run Training

To train all models and their variants, run the following script from the project root:

```
./scripts/run_training.sh
```

### 3. Run Evaluation

After training is complete, you can evaluate the models to generate the results tables:

```
./scripts/run_evaluation.sh
```
Remark for fast check run: You can modify the file script/run_training.sh to get a faster training, for example, set 1 epoch and/or only 1 seed.
