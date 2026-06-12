"""
generate_perturbed_datasets.py

Pre-computes all perturbed versions of each dataset and saves them to disk.
One parquet file per (dataset, perturbation_type, param_value).

Protected features are determined per-dataset from ablation_protected.csv,
and selected using ANOVA F-score (SelectKBest) — model-agnostic, no bias
toward any of the benchmarked models.

Output structure:
    save_dir/
        abundance_cirrhosis--stagediscovery/
            remove_features/
                k=1.parquet
                ...
            sparsity/
                0.812.parquet
                ...
            densification/
                0.649.parquet
                ...
        ...
"""

import sys
import os
import pathlib
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
fromdata_generation.data_generator import DataGenerator


def set_seed(n_seed: int = 42):
    import random
    import torch
    random.seed(n_seed)
    np.random.seed(n_seed)
    torch.manual_seed(n_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(n_seed)


# ── CONFIG ────────────────────────────────────────────────────────────────────

DATASETS = [
    'abundance_cirrhosis--stagediscovery',
    'abundance_cirrhosis--stagevalidation',
    'abundance_obesity',
    'abundance_ibd',
    'abundance_t2d',
    'abundance_WT2D',
]

PERTURBATION_TYPES = ['remove_features', 'sparsity', 'densification']

SEED             = 42
SELECTION_METHOD = 'highest_abundance'  # for remove_features

_DT_DIR      = pathlib.Path(__file__).resolve().parents[1]   # data_transformations/
ABLATION_CSV = str(_DT_DIR / 'ablation_protected.csv')
SAVE_DIR     = str(_DT_DIR / 'perturbed_datasets_final')


# ── LOAD PER-DATASET N_FEATURES_PROTECT ──────────────────────────────────────

def load_n_features_protect(ablation_csv: str) -> dict:
    """
    Legge ablation_protected.csv e ritorna un dizionario
    {dataset_name: n_features_protect}.
    """
    df = pd.read_csv(ablation_csv)
    return dict(zip(df['dataset'], df['n_features_protect'].astype(int)))


# ── ANOVA FEATURE SELECTION ───────────────────────────────────────────────────

def get_anova_top_k_features(X: np.ndarray, y: np.ndarray, k: int, col_names: list) -> list:
    """
    Seleziona le top-k feature per ANOVA F-score.
    Ritorna la lista dei nomi delle feature protette.
    """
    if k == 0:
        return []
    k = min(k, X.shape[1])
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X, y)
    idx = selector.get_support(indices=True).tolist()
    return [col_names[i] for i in idx]


# ── ADAPTIVE PARAM GENERATION ─────────────────────────────────────────────────

def get_params(pert_type: str, n_features: int, actual_sparsity: float):
    if pert_type == 'remove_features':
        k_max    = max(1, n_features // 2)
        k_values = [int(k) for k in np.linspace(1, k_max, 10)]
        return [{'k': k, 'selection_method': SELECTION_METHOD, 'seed': SEED} for k in k_values]
    elif pert_type == 'sparsity':
        sparsity_values = [round(s, 3) for s in np.linspace(actual_sparsity, 0.99, 7)[1:-1]]
        return [{'target_sparsity': s, 'seed': SEED} for s in sparsity_values]
    elif pert_type == 'densification':
        sparsity_values = [round(s, 3) for s in np.linspace(actual_sparsity, 0.01, 7)[1:-1]]
        return [{'target_sparsity': s, 'seed': SEED} for s in sparsity_values]
    return []


def param_to_filename(pert_type: str, params: dict) -> str:
    if pert_type == 'remove_features':
        return f"k={params['k']}.parquet"
    elif pert_type in ('sparsity', 'densification'):
        return f"{params['target_sparsity']}.parquet"
    return "params.parquet"


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # carica n_features_protect per dataset
    n_protect_dict = load_n_features_protect(ABLATION_CSV)
    print("Per-dataset n_features_protect:")
    for ds, k in n_protect_dict.items():
        print(f"  {ds}: {k}")

    for dataset in DATASETS:
        print(f"\n{'='*60}\nDataset: {dataset}\n{'='*60}")

        if dataset not in n_protect_dict:
            raise ValueError(f"Dataset '{dataset}' not found in {ABLATION_CSV}")
        n_protect = n_protect_dict[dataset]
        print(f"  n_features_protect = {n_protect} (from ablation)")

        for pert_type in PERTURBATION_TYPES:
            gen = DataGenerator(generator_type=pert_type, data_source='pasolli')
            gen.load_data(dataset)

            X         = gen.X_original.values
            y         = gen.y_original.values
            col_names = gen.X_original.columns.tolist()

            # seleziona le feature protette con ANOVA
            protected_cols = get_anova_top_k_features(X, y, k=n_protect, col_names=col_names)
            gen.protected_features = protected_cols
            print(f"  [{pert_type}] Protected {len(protected_cols)} features via ANOVA")

            n_features      = gen.X_original.shape[1]
            actual_sparsity = float((X == 0).mean())
            param_list      = get_params(pert_type, n_features, actual_sparsity)

            out_dir = os.path.join(SAVE_DIR, dataset, pert_type)
            os.makedirs(out_dir, exist_ok=True)

            # salva originale
            orig_path = os.path.join(out_dir, 'original.parquet')
            if not os.path.exists(orig_path):
                df_orig          = gen.X_original.copy()
                df_orig['label'] = y
                df_orig.to_parquet(orig_path)
                print(f"  Saved original -> {orig_path}")

            print(f"  Perturbation: {pert_type} ({len(param_list)} levels)")

            for params in param_list:
                fname    = param_to_filename(pert_type, params)
                out_path = os.path.join(out_dir, fname)

                if os.path.exists(out_path):
                    print(f"    [SKIP] {fname} already exists")
                    continue

                X_pert           = gen.generate(**params)
                df_pert          = X_pert.copy()
                df_pert['label'] = y
                df_pert.to_parquet(out_path)
                sparsity = float((X_pert.values == 0).mean())
                print(f"    Saved {fname}  shape={X_pert.shape}  sparsity={sparsity:.3f}")

    print(f"\nDone. All perturbed datasets saved to:\n   {SAVE_DIR}")


if __name__ == '__main__':
    set_seed(42)
    main()
