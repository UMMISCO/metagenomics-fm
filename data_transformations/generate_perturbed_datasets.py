"""
generate_perturbed_datasets.py

Pre-computes all perturbed versions of each dataset and saves them to disk.
One parquet file per (dataset, perturbation_type, param_value).

Output structure:
    save_dir/
        abundance_cirrhosis--stagediscovery/
            remove_features/
                k=1.parquet
                k=29.parquet
                ...
            sparsity/
                0.812.parquet
                ...
            densification/
                0.649.parquet
                ...
        abundance_obesity/
            ...
        labels/
            abundance_cirrhosis--stagediscovery.parquet  # y series
            ...
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/')
from data_transformations.data_generator import DataGenerator

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

N_FEATURES_PROTECT = 2
SEED               = 42
SELECTION_METHOD   = 'highest_abundance'  # for remove_features

SAVE_DIR = '/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/data_transformations/perturbed_datasets/'

# ── ADAPTIVE PARAM GENERATION (same logic as Benchmarker) ────────────────────

def get_params(pert_type: str, n_features: int, actual_sparsity: float):
    if pert_type == 'remove_features':
        k_max = max(1, n_features // 2)
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
    labels_dir = os.path.join(SAVE_DIR, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    for dataset in DATASETS:
        print(f"\n{'='*60}\nDataset: {dataset}\n{'='*60}")

        for pert_type in PERTURBATION_TYPES:
            gen = DataGenerator(generator_type=pert_type, data_source='pasolli')
            gen.load_data(dataset)
            gen.discover_and_protect(method='random_forest', n_features=N_FEATURES_PROTECT, verbose=False)

            n_features      = gen.X_original.shape[1]
            actual_sparsity = float((gen.X_original.values == 0).mean())
            param_list      = get_params(pert_type, n_features, actual_sparsity)

            # Save labels once per dataset
            labels_path = os.path.join(labels_dir, f"{dataset}.parquet")
            if not os.path.exists(labels_path):
                gen.y_original.to_frame(name='label').to_parquet(labels_path)
                print(f"  Saved labels → {labels_path}")

            # Save original
            out_dir = os.path.join(SAVE_DIR, dataset, pert_type)
            os.makedirs(out_dir, exist_ok=True)
            orig_path = os.path.join(out_dir, 'original.parquet')
            if not os.path.exists(orig_path):
                gen.X_original.to_parquet(orig_path)

            print(f"  Perturbation: {pert_type} ({len(param_list)} levels)")

            for params in param_list:
                fname = param_to_filename(pert_type, params)
                out_path = os.path.join(out_dir, fname)

                if os.path.exists(out_path):
                    print(f"    [SKIP] {fname} already exists")
                    continue

                X_pert = gen.generate(**params)
                X_pert.to_parquet(out_path)
                sparsity = float((X_pert.values == 0).mean())
                print(f"    Saved {fname}  shape={X_pert.shape}  sparsity={sparsity:.3f}")

    print(f"\n✅ Done. All perturbed datasets saved to:\n   {SAVE_DIR}")


if __name__ == '__main__':
    main()
