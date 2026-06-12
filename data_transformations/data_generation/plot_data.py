# %% Imports & config

import os
import sys
import pathlib
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from data_transformations.data_generation.data_generator import DataGenerator

_DT_DIR      = pathlib.Path(__file__).resolve().parents[1]   # data_transformations/
SAVE_DIR     = str(_DT_DIR / 'perturbed_datasets_final')
ABLATION_CSV = str(_DT_DIR / 'ablation_protected.csv')
PLOT_DIR     = str(_DT_DIR / 'plot_paper')

DATASETS = [
    'abundance_cirrhosis--stagediscovery',
    'abundance_cirrhosis--stagevalidation',
    'abundance_obesity',
    'abundance_ibd',
    'abundance_t2d',
    'abundance_WT2D',
]

PERTURBATION_TYPES = ['remove_features', 'sparsity', 'densification']


# %% Helper — ANOVA protected features

def get_protected_features(X: pd.DataFrame, y: pd.Series, k: int) -> list:
    if k == 0:
        return []
    k = min(k, X.shape[1])
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X.values, y.values)
    return [X.columns[i] for i in selector.get_support(indices=True)]

_n_protect = dict(zip(
    *[pd.read_csv(ABLATION_CSV)[c] for c in ['dataset', 'n_features_protect']]
))


# %% Helper — load parquets

def _sort_key(p):
    name = os.path.splitext(os.path.basename(p))[0]
    return float(name.split('=')[-1])

def load_perturbations(dataset, pert_type):
    pert_dir  = os.path.join(SAVE_DIR, dataset, pert_type)
    df_orig   = pd.read_parquet(os.path.join(pert_dir, 'original.parquet'))
    y         = df_orig['label']
    X_orig    = df_orig.drop(columns=['label'])
    perturbed_paths = sorted(
        (p for p in glob.glob(os.path.join(pert_dir, '*.parquet'))
         if not p.endswith('original.parquet')),
        key=_sort_key
    )
    perturbations = []
    for path in perturbed_paths:
        df_pert = pd.read_parquet(path)
        X_pert  = df_pert.drop(columns=['label'])
        label   = os.path.splitext(os.path.basename(path))[0]
        perturbations.append((label, X_pert))
    return X_orig, y, perturbations


# %% Helper — visualize and save two PNGs per (dataset, pert_type)

def visualize(dataset, pert_type):
    X_orig, y, perturbations = load_perturbations(dataset, pert_type)

    n_protect       = int(_n_protect.get(dataset, 0))
    protected_feats = get_protected_features(X_orig, y, k=n_protect)
    label_map       = {0: 'Control', 1: 'Cases'}
    y_named         = y.map(lambda v: label_map.get(v, str(v)))

    print(f"\n{'='*60}")
    print(f"  {dataset}  |  {pert_type}  ({len(perturbations)} levels)")
    print(f"  protected: {len(protected_feats)} features")
    print(f"{'='*60}")

    gen = DataGenerator(generator_type=pert_type, data_source='pasolli')
    gen.X_original         = X_orig
    gen.y_original         = y
    gen.protected_features = protected_feats

    save_dir = os.path.join(PLOT_DIR, dataset, pert_type)
    os.makedirs(save_dir, exist_ok=True)

    base = f'{dataset}__{pert_type}'

    # ── Plot 1: scatter per perturbation ──────────────────────────────
    gen.visualizer.plot_per_perturbation(
        original=X_orig,
        perturbations=perturbations,
        title='',
        save_path=os.path.join(save_dir, f'{base}__scatter.png'),
    )
    plt.close('all')

    # ── Plot 2: feature trajectories ──────────────────────────────────
    gen.visualizer.plot_feature_trajectories(
        original=X_orig,
        perturbations=perturbations,
        y_labels=y_named,
        protected_features=protected_feats,
        figsize=(16, 8),
        title='',
        save_path=os.path.join(save_dir, f'{base}__trajectories.png'),
    )
    plt.close('all')

    print(f"  Saved → {save_dir}/")


# %% Run — single dataset/perturbation (edit as needed)

visualize('abundance_WT2D', 'sparsity')


# %% Run — loop over all datasets and perturbation types

for dataset in DATASETS:
    for pert_type in PERTURBATION_TYPES:
        visualize(dataset, pert_type)