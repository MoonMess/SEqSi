# Cryo-ET Particle Classification

This repository contains the code for training and evaluating 3D Convolutional Neural Networks for classifying macromolecule patches from Cryo-Electron Tomography data.

## Project Structure

```
.
├── config/                 # Configuration files
│   ├── config_classification.py
│   └── particle_config.py
├── data/                   # Placeholder for data (not included)
├── models/                 # Model definitions
│   ├── resnet3d.py
│   └── equiv_blocks.py
├── output/                 # Default directory for evaluation results and visualizations
├── checkpoints/            # Default directory for saved model checkpoints
├── scripts/                # Utility and automation scripts
│   ├── train_all.sh
│   └── eval_all.sh
├── train.py                # Main training script
├── eval.py                 # Script for evaluating trained models
├── prepare_classification_data.py # Script to preprocess data
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

The workflow is divided into three main steps: data preparation, model training, and evaluation.

### 1. Data Preparation

The raw CZI Challenge dataset can be downloaded from Kaggle:
[https://www.kaggle.com/competitions/czii-cryo-et-object-identification/data](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/data)

First, you need to generate the classification dataset from the raw CZI Challenge data. This script extracts patches for each annotated particle and splits them into `train`, `val`, and `test` sets based on their tomogram of origin.

Run the `prepare_classification_data.py` script. You need to specify the root directory of the CZI dataset and the desired output directory.

**Example:**
This command will process all tomogram types (`wbp`, `denoised`, etc.) and create patches of size 64x64x64.

```bash
python prepare_classification_data.py \
    --data_root /path/to/czi/dataset \
    --output_dir /path/to/your/classification_data \
    --patch_size 64 64 64 \
    --tomo_type wbp denoised isonetcorrected ctfdeconvolved
```
Remark: you can use "data/" as output_dir, as it is the default directory in the config file.

After running, your `classification_data` directory will look like this:

```
/path/to/your/classification_data/
├── wbp/
│   ├── train/
│   ├── val/
│   └── test/
├── denoised/
│   ├── train/
│   ├── val/
│   └── test/
└── ...
```

### 2. Model Training

The `scripts/train_all.sh` script automates the training process for multiple seeds for both the `standard` and `seqsi` models.

Before running, make the script executable:
```bash
chmod u+x scripts/train_all.sh
```

**Usage:**
The script takes two arguments:
1.  The root directory of the classification data (the one created in the previous step).
2.  The tomogram type to use for training (e.g., `wbp`).

**Example:**
To train models on the `wbp` data over 5 different seeds:

```bash
./scripts/train_all.sh /path/to/your/classification_data wbp
```

This will run `train.py` for each seed and model type, saving checkpoints to the `./checkpoints` directory.  

### 3. Model Evaluation

After training, you can evaluate the models' performance.

#### a) General Performance Evaluation

The `scripts/eval_all.sh` script finds all `bestmodel.ckpt` files in the `./checkpoints` directory and evaluates each one on all available test sets.

Make the script executable:
```bash
chmod u+x scripts/eval_all.sh
```

**Usage:**
The script takes one argument: the root directory of the classification data.

**Example:**

```bash
./scripts/eval_all.sh /path/to/your/classification_data
```

The results will be printed to the console, and summary `.csv` files will be saved in the `./output/evaluation` directory.

