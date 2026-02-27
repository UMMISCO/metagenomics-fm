#%%

import sys
sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/')

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
    'abundance_cirrhosis--stagevalidation',
    'abundance_obesity',
    'abundance_ibd',
    'abundance_t2d',
    'abundance_WT2D',
]

PERTURBATION_CONFIGS = {
    'remove_features': [
        # K range —> max size(dataset)/2,
        # then take a step such that size(dataset)/2 is divided by 10 (round up)
        {'k': 10,  'selection_method': 'highest_abundance', 'seed': 42},
        {'k': 50,  'selection_method': 'highest_abundance', 'seed': 42},
        {'k': 100, 'selection_method': 'highest_abundance', 'seed': 42},
        {'k': 200, 'selection_method': 'highest_abundance', 'seed': 42},
        # {'k': 10,  'selection_method': 'random', 'seed': 42},
        # {'k': 50,  'selection_method': 'random', 'seed': 42},
        # {'k': 100, 'selection_method': 'random', 'seed': 42},
        # {'k': 200, 'selection_method': 'random', 'seed': 42},
    ],
    'sparsity': [
        #linspace from actual sparsity to 1 --> step=5
        {'target_sparsity': 0.50, 'seed': 42},
        {'target_sparsity': 0.60, 'seed': 42},
        {'target_sparsity': 0.90, 'seed': 42},
        {'target_sparsity': 0.95, 'seed': 42},
    ],
    'add_random_features': [
        # K range —> max size(dataset)/2,
        # then take a step such that size(dataset)/2 is divided by 10 (round up)
        {'k': 10,  'seed': 42},
        {'k': 50,  'seed': 42},
        {'k': 100, 'seed': 42},
        {'k': 200, 'seed': 42},
    ],
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
    n_features_protect=5,
    n_features_max=100000,
    device='cpu',
    # save_dir=f'/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/data_transformations/benchmark_results_2f_{models_str}/',
)