#%%
import sys
import numpy as np
import pandas as pd

sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/')
from data_transformations.benchmarking import Benchmarker

def set_seed(n_seed: int = 42):
    import random
    import torch
    random.seed(n_seed)
    np.random.seed(n_seed)
    torch.manual_seed(n_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(n_seed)

#%%

DATASETS = [
    'abundance_cirrhosis--stagediscovery',
    'abundance_cirrhosis--stagevalidation',
    'abundance_obesity',
    'abundance_ibd',
    'abundance_t2d',
    'abundance_WT2D',
]

PERTURBATION_TYPES = ['remove_features', 'sparsity', 'densification']

MODEL_LIST = ['rf', 'tabicl', 'original_v2']

PRECOMPUTED_DIR = '/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/data_transformations/perturbed_datasets/'
SAVE_DIR        = '/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/data_transformations/benchmark_results/'

set_seed(42)

bench = Benchmarker(data_source='pasolli')

#%%
# --- Run benchmark: one (dataset, perturbation) at a time ---
all_results = []

for dataset in DATASETS:
    for pert_type in PERTURBATION_TYPES:
        df = bench.run_one(
            dataset         = dataset,
            pert_type       = pert_type,
            model_names     = MODEL_LIST,
            cv              = 5,
            n_features_protect = 2,
            n_features_max  = 100000,
            device          = 'cpu',
            seed            = 42,
            precomputed_dir = PRECOMPUTED_DIR,
            save_dir        = SAVE_DIR,
        )
        all_results.append(df)

#%%
# --- Merge all results and plot ---
results_df = pd.concat(all_results, ignore_index=True)

bench.plot(
    results_df,
    figsize   = (8, 5),
    save_dir  = SAVE_DIR,
)
