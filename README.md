# SEqSI Neural Network

This repository contains the source code for the experiments presented in our paper : **Designing Affine-Invariant Neural Networks for Photometric Corruption Robustness and Generalization**.

The code is organized into two main components, corresponding to the two major tasks explored in the paper: **Classification** and **Object Localization**.

## Repository Structure

The repository is structured as follows:

-   `./classification/`: Contains all code related to the classification experiments. (CIFAR-10 and CZI)
-   `./object_localization/`: Contains all code related to the localization experiments. (Science Bowl 2018 and 3D fluerescence microscopy)


Each of these directories contains its own set of experiments, models, and instructions, as detailed below.

---

## Classification Experiments (`../classification/`)

This directory contains the code for training and evaluating models on image and volume classification tasks. It is further divided into two sub-projects:

-   `cifar10/`: For 2D image classification on the **CIFAR-10** dataset.
-   `czi/`: For 3D volume classification on a **Cryo-Electron Tomography (Cryo-ET)** dataset.

### Common Project Structure

Both classification projects share a similar organization to facilitate reproducibility:

-   **`scripts/`**: Contains shell scripts (`.sh`) to automate the process of training and evaluating all model variants across multiple random seeds.
-   **`model.py`**, **`models/`**, or **`equiv_blocks.py`**: These files define the neural network architectures, including the baseline ResNet and our proposed SEqSI models.
-   **`train.py`**: The main script for running a single training instance. It handles model initialization, data loading, the training loop, and checkpointing.
-   **`eval.py`**, **`test.py`** or **`evaluate_invariance.py`**: Scripts used for evaluating the performance of trained models on various test sets.
-   **`dataloader.py`** or **`prepare_classification_data.py`**: Utility scripts for data handling, including downloading, preprocessing, and setting up data loaders for training and testing.

### How to Run

For detailed, step-by-step instructions on setting up the environment, preparing the data, and running the experiments, please refer to the `README.md` file located within each respective project directory (`cifar10/README.md` and `czi/README.md`).

---

## Object Localization Experiments (`./object_localization/`)

This directory contains the experiments for object localization, focusing on both 2D and 3D scenarios. It is further divided into two sub-projects:

-   `2D/`: For 2D object localization on the **Science Bowl 2018** dataset.
-   `3D/`: For 3D object localization on a synthetic **3D fluorescence microscopy** dataset.

While both projects share a similar goal, their implementation details and script names differ slightly. Below is a general overview, but please refer to the specific `README.md` in each subdirectory for exact commands.

### General Project Structure

-   **`configs/`**: Contains configuration files (`.yaml`) for training and testing.
-   **`models/`** or similar: Defines the neural network architectures.
-   **`training.py`**: The main script for running a training instance based on a configuration file.
-   **Testing Scripts**: Various scripts for evaluation, such as `testing.py` (3D), `inv_experiments.py` (2D), or `artifacts_exp.py` (3D).
-   **Data-related Scripts**: Utilities for data handling. For example, `data/pre-process/generate_score.py` in the 2D project is used to create ground truth maps.
-   **Result Aggregation Scripts**: Scripts like `generate_res_table.py` and `merge_tabs.py` to process experiment outputs and create summary tables.

### How to Run

For detailed, step-by-step instructions on setting up the environment, preparing the data, and running the experiments, please refer to the `README.md` file located within each respective project directory (`2D/README.md` and `3D/README.md`).

**Specifics for each case:**

-   **2D Case (`./object_localization/2D/`)**: We provide the link to download the public dataset (Science Bowl 2018) and the code to generate the Ground Truth (GT) score maps. This allows for complete retraining of models and reproduction of experiments.

-   **3D Case (`./object_localization/3D/`)**: Our synthetic dataset for the 3D case (3D fluorescence microscopy) is too large to be directly included in the repository. However, we provide the code and one example of an image + GT score map (within a `tar.gz` archive) to demonstrate the data format and processing. If full reproduction of the 3D experiments is required, we can arrange anonymous data provision during the reviewing process.

## Environment

This repository requires different environments for its main components to manage dependencies correctly.

-   **CIFAR-10 Classification (`classification/cifar10/`)**: This project requires its own environment. Please install the packages from `classification/cifar10/environment.yml`.

-   **CZI Classification (`classification/czi/`)**: This project also requires a separate environment. Please install the packages from `classification/czi/environment.yml`.

-   **Object Localization (`object_localization/`)**: Both 2D and 3D localization experiments share a single environment. Please install the packages from `object_localization/requirements.txt`.

The repository is structured as follows:

-   `./classification/`: Contains all code related to the classification experiments. (CIFAR-10 and CZI)
-   `./object_localization/`: Contains all code related to the localization experiments. (Science Bowl 2018 and 3D fluerescence microscopy)

## Acknowledgements
This work was support by Agence Nationale de la Recherche (ANR-23-CE45-0012-02). Access to the HPC resources of IDRIS was granted under the allocation 2025-AD011015932R1 made by GENCI.

## Citation
@inproceedings{
messaoudi2026designing,
title={Designing Affine-Invariant Neural Networks for Photometric Corruption Robustness and Generalization},
author={Mounir Messaoudi and Quentin Rapilly and S{\'e}bastien Herbreteau and Ana{\"\i}s Badoual and Charles Kervrann},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=fhEwTOLYNZ}
}

