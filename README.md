# Microbiome Perturbation Analysis

> **Accepted paper - [ECML PKDD 2026], Applied Data Science track.** 

**Author of the code:** Giulia Perciballi\
**Authors:** Giulia Perciballi, Ahmad Fall, Federica Granese, Edi Prifti and Jean-Daniel Zucker\
**Date:** June 2026\
**Affiliation:** IRD, Sorbonne Université, Unité de Modélisation Mathématique et Informatique des Systèmes Complexes (UMMISCO)

Pipeline for evaluating TFM robustness on microbiome compositional data
under controlled perturbations.

---

## Repository Structure

```
data_generation/
    select_protected_features.py  # Step 0 — select protected features per dataset
    generate_perturbed_datasets.py  # Step 1 — pre-compute all perturbed parquets
    data_generator.py             # DataGenerator — load, perturb, visualize
    perturbation_core.py          # Perturbation classes + PerturbationStats

benchmarking/
    benchmarking.py               # Benchmarker — OOD cross-validation
    test_perturbations.py         # Entry point for one (dataset, pert_type) job
    run_all.sh                    # Parallel launcher across 4 GPUs
    merge_tidy.py                 # Merge result parquets into flat CSVs
    statistical_test.py           # DeLong test 

analyses/
    1_make_plots.Rmd              # R Markdown — figures and plots from results

ablation_protected.csv            # Per-dataset n_features_protect (output of Step 0)
perturbed_datasets_final/         # Output of Step 1
benchmark_results_new_final/      # Output of benchmarking
```

---

## Installation

Requires [pip](https://pip.pypa.io/) (used to install [uv](https://docs.astral.sh/uv/),
which manages the environment and dependencies).

```bash
git clone https://github.com/UMMISCO/metagenomics-fm/
cd metagenomics-fm

# Install uv
pip install uv

# Create the virtual environment with Python 3.12.2
uv venv --python 3.12.2

# Activate the environment
source .venv/bin/activate

# Install dependencies
uv sync
```

---

## Datasets

Six binary classification datasets from the Pasolli et al. gut microbiome collection.
Features are microbial species relative abundances, renormalized to sum to 1 per sample.

```python
DATASETS = [
    'abundance_cirrhosis--stagediscovery',    # Liver cirrhosis (discovery)
    'abundance_cirrhosis--stagevalidation',   # Liver cirrhosis (validation)
    'abundance_obesity',                      # Obesity
    'abundance_ibd',                          # Inflammatory Bowel Disease
    'abundance_t2d',                          # Type 2 Diabetes
    'abundance_WT2D',                         # Women Type 2 Diabetes
]
```

---

## Step 0 — Select Protected Features

For each dataset, determines how many top-ranked features (by ANOVA F-score) must
be excluded from perturbation to preserve predictive signal.
Runs RF cross-validation iteratively, removing one more top feature each time,
until AUROC drops by >= 3%. Saves results to `ablation_protected.csv`.

**Must be run before Step 1.**

```bash
python data_generation/select_protected_features.py
```

---

## Part 1 — Data Generation

### Overview

Pre-computes all perturbed versions of each dataset and saves them to disk.
One parquet file per (dataset, perturbation_type, param_value).
Protected features (from `ablation_protected.csv`) are never modified.

### Perturbation types

**remove features**
Removes the k highest-abundance non-protected features and renormalizes.
Simulates incomplete sequencing panels or feature dropout.
- 10 levels: k from 1 to n_features // 2, evenly spaced
- Selection method: highest_abundance

**zero inflation**
Increases the fraction of zeros by gamma-scaling non-protected feature values.
Simulates underdetection due to low sequencing depth.
- 5 levels: target sparsity from current_sparsity to 0.99

**zero imputation**
Decreases the fraction of zeros by filling them with values sampled from
each feature's empirical non-zero distribution.
Simulates improved detection recovering previously undetected species.
- 5 levels: target sparsity from current_sparsity to 0.01

### Protected features

For each dataset, n_features_protect features (from ablation_protected.csv) are
selected by ANOVA F-score (SelectKBest, f_classif) and excluded from all
perturbations. This is model-agnostic and introduces no bias toward any classifier.

### Output structure

```
perturbed_datasets_final/
    {dataset}/
        remove_features/
            original.parquet
            k=1.parquet
            k=25.parquet
            ...
        zero_inflation/
            original.parquet
            0.812.parquet
            ...
        zero_imputation/
            original.parquet
            0.649.parquet
            ...
```

Each parquet contains all feature columns + a `label` column.

### Run

```bash
python data_generation/generate_perturbed_datasets.py
```

Already-existing files are skipped.

---

## Part 2 — Benchmarking

### Evaluation protocol

Robustness evaluation: train (context for TFMs) on original data, test on perturbed data.
Both baseline and benchmarking use 5-fold stratified cross-validation with the same fold splits,
enabling paired statistical comparison.

```
Baseline : TRAIN original  | TEST original   → measures in-distribution performance
Benchmark      : TRAIN original  | TEST perturbed  → measures robustness to perturbation
```

### Models (require separate package installation)

| Key           | Model                  | 
|---------------|------------------------|
| rf            | Random Forest          | 
| xgb           | XGBoost                | 
| tabdpt        | TabDPT                 | 
| original_v2   | TabPFN v2              |
| tabicl        | TabICL                 |
| contextab     | ContextTab (SAP_RPT)   |

### Metrics

- **AUROC** — primary metric, reported per fold and aggregated
- **F1, Precision, Recall** — secondary metrics
- **delta_auroc** = baseline_auroc - ood_auroc

### Run - Baseline (No perturbations)
```bash
python benchmarking/test_perturbations.py --models original_v2,tabicl --baseline_only
```

### Run — single job
**Options:**

--pert remove_features, zero_inflation, zero_imputation

--models rf, xgb, tabdpt, original_v2, tabicl, contextab

```bash
python benchmarking/test_perturbations.py --dataset abundance_ibd --pert remove_features
python benchmarking/test_perturbations.py --dataset abundance_obesity --pert sparsity --models rf,tabicl
```

```bash on ALL 6 DATASETS (without --dataset)
python benchmarking/test_perturbations.py  --pert remove_features
```

### Run — Full reproduction (HPC, multi-GPU)

```bash
bash benchmarking/run_all.sh
```

Launches 36 parallel jobs (6 datasets × 3 perturbations × 2 environments)
distributed across 4 GPUs and 2 different environments due to libraries conflicts. Logs written to `logs_final/`.

Monitor progress:
```bash
tail -f logs_final/*.log
grep -rl 'Error\|Traceback' logs_final/   # check for failures
```

### Output structure

```
benchmark_results_new_final/
    {dataset}/
        {pert_type}__{env_tag}.parquet              # metrics per fold
        {pert_type}__predictions__{env_tag}.parquet # per-sample probabilities
```

---

## Part 3 — Post-processing

### Merge results

```bash
python benchmarking/merge_tidy.py
```

Scans all parquets under `benchmark_results_new_final/`, deduplicates,
remaps model keys to display names, and writes:
- `metrics_all_final.csv`     — one row per (dataset, model, perturbation, param_value, fold)
- `predictions_all_final.csv` — one row per (dataset, model, perturbation, param_value, fold, sample_idx)

Model name mapping:
```python
MODEL_DISPLAY = {
    'rf':          'RF',
    'tabicl':      'TabICL',
    'original_v2': 'TabPFN',
    'xgb':         'XGBoost',
    'tabdpt':      'TabDPT',
    'contextab':   'ContextTab',
}
```

### Statistical testing

```bash
python benchmarking/statistical_test.py
```

Performs paired DeLong tests comparing baseline AUROC vs OOD AUROC for every
(dataset, model, perturbation, param_value) combination, using per-sample
predicted probabilities aligned across folds.

- Test: DeLong (1988) — paired comparison of correlated AUROCs
- Significance threshold: p < 0.05

Output: `statistical_results_final.csv`

```
columns: dataset, model, perturbation, param_value,
         auroc_baseline, auroc_ood, delta_auroc, p_value, significant
```

---

## Reproducibility

All scripts use seed 42.

## Funding

This work was supported by a grant from the French Agence Nationale de la Recherche (ANR) for the DeepIntegrOmics project (number ANR-21-CE45-0030).
