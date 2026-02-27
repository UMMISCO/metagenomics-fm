#%%

import sys
sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/')
import numpy as np

from data_transformations.benchmarking import Benchmarker

def set_seed(n_seed: int = 6274):
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
    # 'abundance_cirrhosis--stagevalidation',
    # 'abundance_obesity',
    # 'abundance_ibd',
    # 'abundance_t2d',
    # 'abundance_WT2D',
]

# PERTURBATION_CONFIGS serves as template only:
# - k values are ignored (computed adaptively per dataset)
# - sparsity values are ignored (computed adaptively per dataset)
# - selection_method and seed are used
PERTURBATION_CONFIGS = {
    'remove_features':     [{'k': 1, 'selection_method': 'highest_abundance', 'seed': 42}],
    'sparsity':            [{'target_sparsity': 0.9, 'seed': 42}],
    'add_random_features': [{'k': 1, 'seed': 42}],
}

set_seed(42)

bench = Benchmarker(data_source='pasolli')

# model_list = ['rf', 'original_v2', 'tabicl']
model_list = ['rf']
models_str = "_".join(model_list)

results = bench.run(
    datasets=DATASETS,
    perturbation_configs=PERTURBATION_CONFIGS,
    model_names=model_list,
    cv=5,
    n_features_protect=2,
    n_features_max=100000,
    device='cpu',
    # save_dir=f'/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/data_transformations/benchmark_results_2f_{models_str}/',
)