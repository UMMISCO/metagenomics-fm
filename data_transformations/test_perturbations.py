#%%

import sys
sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/')

from data_transformations.benchmarking import Benchmarker

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
        {'k': 10,  'selection_method': 'highest_abundance', 'seed': 42},
        {'k': 50,  'selection_method': 'highest_abundance', 'seed': 42},
        {'k': 100, 'selection_method': 'highest_abundance', 'seed': 42},
        {'k': 200, 'selection_method': 'highest_abundance', 'seed': 42},
        {'k': 10,  'selection_method': 'random', 'seed': 42},
        {'k': 50,  'selection_method': 'random', 'seed': 42},
        {'k': 100, 'selection_method': 'random', 'seed': 42},
        {'k': 200, 'selection_method': 'random', 'seed': 42},
    ],
    'sparsity': [
        {'target_sparsity': 0.50, 'seed': 42},
        {'target_sparsity': 0.85, 'seed': 42},
        {'target_sparsity': 0.90, 'seed': 42},
        {'target_sparsity': 0.95, 'seed': 42},
    ],
    'add_random_features': [
        {'k': 10,  'seed': 42},
        {'k': 50,  'seed': 42},
        {'k': 100, 'seed': 42},
        {'k': 200, 'seed': 42},
    ],
}

bench = Benchmarker(data_source='pasolli')

results = bench.run(
    datasets=DATASETS,
    perturbation_configs=PERTURBATION_CONFIGS,
    # model_names=['rf', 'original_v2', 'tabicl'],
    model_names=['rf'],
    cv=5,
    n_features_protect=15,
    n_features_max=10000,
    device='cpu',
    save_dir='/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/data_transformations/benchmark_results/',
)