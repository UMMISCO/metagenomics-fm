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

# Default completo — viene overridato da --models se passato da CLI
MODEL_LIST_DEFAULT = ['rf', 'tabicl', 'original_v2', 'xgb', 'tabdpt', 'contextab']

SAVE_DIR = '/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/data_transformations/benchmark_results_new/'

#%%
'''
To run the test:
python test_perturbations.py --dataset abundance_cirrhosis--stagediscovery --pert remove_features
python test_perturbations.py --dataset abundance_obesity --pert sparsity --models rf,tabicl
...
    '''

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  type=str, default=None, choices=DATASETS)
parser.add_argument('--pert',     type=str, default=None, choices=PERTURBATION_TYPES)
parser.add_argument('--models',   type=str, default=None,
                    help='Comma-separated list of models, e.g. rf,tabicl,original_v2')
parser.add_argument('--save_dir', type=str, default=SAVE_DIR,
                    help='Directory where CSV results are saved')
args, _ = parser.parse_known_args()

datasets   = [args.dataset]         if args.dataset else DATASETS
pert_types = [args.pert]            if args.pert    else PERTURBATION_TYPES
MODEL_LIST = args.models.split(',') if args.models  else MODEL_LIST_DEFAULT
SAVE_DIR   = args.save_dir

set_seed(42)
bench = Benchmarker(data_source='pasolli')

#%%
# --- Run benchmark ---
for dataset in datasets:
    for pert_type in pert_types:

        results_df, oof_df = bench.run_one(
            dataset            = dataset,
            pert_type          = pert_type,
            model_names        = MODEL_LIST,
            cv                 = 5,
            n_features_protect = 5,
            n_features_max     = 100000,
            device             = 'cuda',
            seed               = 42,
        )

        # Ogni env scrive su un file separato — no race condition
        env_tag  = '_'.join(MODEL_LIST)
        out_dir  = os.path.join(SAVE_DIR, dataset)
        os.makedirs(out_dir, exist_ok=True)

        # metriche aggregate
        metrics_path = os.path.join(out_dir, f"{pert_type}__{env_tag}.csv")
        results_df.to_csv(metrics_path, index=False)
        print(f"Saved metrics    -> {metrics_path}")

        # predizioni OOF per sample
        preds_path = os.path.join(out_dir, f"{pert_type}__predictions__{env_tag}.csv")
        oof_df.to_csv(preds_path, index=False)
        print(f"Saved predictions-> {preds_path}")
