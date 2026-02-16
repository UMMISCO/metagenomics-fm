# Metagenomics Foundation Models

**Author:** Giulia Perciballi 
**Advisor** Ahmad Fall; Edi PRifti and Jean-Daniel Zucker
**Date:** February 2026
**Affiliation:** UMMISCO

## Overview

This project develops foundation models for classification tasks on microbiome abundance data. It combines transformer-based architectures (TabPFN, MotherNet) with synthetic data generation techniques tailored to the sparse, high-dimensional nature of metagenomic datasets.

The goal is to enable few-shot learning on microbiome data by training general-purpose classifiers on synthetic priors, then evaluating them on real-world metagenomic classification benchmarks.

## Project Structure

```text
metagenomics-fm/
├── foundation_models/
│   ├── mothernet/              # Core model system
│   │   ├── models/             # Neural network architectures (TabPFN, encoders, decoders)
│   │   ├── priors/             # Synthetic data priors (MLP, GP, boolean, zero-inflated)
│   │   ├── config/             # TOML-based configuration system
│   │   ├── training/           # Training infrastructure and tests
│   │   ├── prediction/         # Inference code
│   │   ├── train.py            # Training loop (distributed, mixed precision)
│   │   ├── fit_model.py        # Main training entry point
│   │   ├── model_builder.py    # Model factory and checkpoint loading
│   │   └── dataloader.py       # Prior-based data loader
│   └── src/
│       └── tabicl_original/    # TabICL classifier implementation
├── data_transformations/
│   ├── sparse_data_generator.py      # Sparsity-based data augmentation
│   └── perturbation_generator.py     # Feature perturbation methods
├── testing/
│   ├── test_models.py                # Cross-validation evaluation script
│   └── testing_data/
│       ├── pasolli/                  # Pasolli microbiome datasets
│       ├── metacardis/               # MetaCARDIS cohort data
│       ├── open_ml/                  # OpenML benchmark datasets
│       └── preprocessing/            # Feature filtering (presence/abundance)
└── README.md
```

## Key Components

### Foundation Models

- **TabPFN**: Transformer-based architecture for tabular data with few-shot capabilities. Uses encoder-transformer-decoder pipeline with support for missing value handling.
- **MotherNet**: Extended model architecture built on top of TabPFN.
- **Prior System**: Synthetic data generation via learned priors (MLP, Gaussian Process, boolean conjunctions, zero-inflated distributions) to train the models without requiring large labeled datasets.

### Data Transformations

- **SparseDataGenerator**: Applies sparsity transformations (gamma exponent zeroing, threshold-based zeroing, random feature removal/addition) to simulate realistic microbiome data characteristics.
- **PerturbationGenerator**: Discovers informative features via Random Forest importance or LASSO, then applies perturbations (masking, scaling, shuffling) to evaluate model robustness.

### Testing & Evaluation

Cross-validation evaluation on multiple microbiome datasets:

- Cirrhosis, Obesity, IBD, T2D, WT2D (Pasolli)
- MetaCARDIS cohort

Benchmarked against: TabPFNv2, ContextTab, TabICL
Metrics: Accuracy, ROC-AUC

## Dependencies

Key libraries:

- PyTorch (+ distributed training via DDP)
- Scikit-learn
- Pandas / NumPy
- MLflow / Weights & Biases (experiment tracking)
- TabPFN, TabICL, ContextTab (external model packages)

## Usage

### Training

```python
from foundation_models.mothernet.fit_model import main
main(config_path="path/to/config.toml")
```

### Evaluation

```python
from testing.test_models import cross_val_results
cross_val_results(dataset_name, model, n_splits=10)
```

### Data Augmentation

```python
from data_transformations.sparse_data_generator import DataGenerator
generator = DataGenerator(X, y)
X_augmented = generator.generate()
```

## License

All rights reserved.
