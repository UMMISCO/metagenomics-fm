#%%
import os
import sys
import argparse
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

MODEL_LIST = ['tabicl', 'rf', 'tabicl', 'original_v2']

PRECOMPUTED_DIR = '/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/data_transformations/perturbed_datasets/'
SAVE_DIR        = '/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/data_transformations/benchmark_results/'

#%%

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=None, choices=DATASETS)
parser.add_argument('--pert',    type=str, default=None, choices=PERTURBATION_TYPES)
args, _ = parser.parse_known_args()  # _ evita errori se lanciato da Jupyter

# Se passati da CLI usa quelli, altrimenti gira tutto
datasets   = [args.dataset]   if args.dataset else DATASETS
pert_types = [args.pert]      if args.pert    else PERTURBATION_TYPES

set_seed(42)
bench = Benchmarker(data_source='pasolli')

#%%
# --- Run benchmark ---
all_results = []

for dataset in datasets:
    for pert_type in pert_types:
        df = bench.run_one(
            dataset            = dataset,
            pert_type          = pert_type,
            model_names        = MODEL_LIST,
            cv                 = 5,
            n_features_protect = 2,
            n_features_max     = 100000,
            device             = 'cuda',
            seed               = 42,
            precomputed_dir    = PRECOMPUTED_DIR,
            save_dir           = SAVE_DIR,
        )
        all_results.append(df)

        # Salva subito il CSV per questa combinazione
        out_dir = os.path.join(SAVE_DIR, dataset)
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, f"{pert_type}.csv"), index=False)
        print(f"Saved -> {out_dir}/{pert_type}.csv")


# #%%
# # --- plot_results.py (da lanciare alla fine) ---
# import os
# import glob
# import pandas as pd
# import sys
#
# sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/')
# from data_transformations.benchmarking import Benchmarker
#
# SAVE_DIR = '/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/data_transformations/benchmark_results/'
#
# all_csv = glob.glob(os.path.join(SAVE_DIR, '*', '*.csv'))
# results_df = pd.concat([pd.read_csv(f) for f in all_csv], ignore_index=True)
# print(f"Loaded {len(all_csv)} files, {len(results_df)} rows total")
#
# bench = Benchmarker()
# bench.plot(results_df, figsize=(8, 5), save_dir=SAVE_DIR)

####################

'''To run the test:
python test_perturbations.py --dataset abundance_cirrhosis--stagediscovery --pert remove_features
python test_perturbations.py --dataset abundance_obesity --pert sparsity
...
...
...

'''