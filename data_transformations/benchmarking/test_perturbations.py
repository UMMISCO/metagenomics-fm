#%%
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/')
from benchmarking import Benchmarker

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
# PERTURBATION_TYPES =[]

# Default completo — viene overridato da --models se passato da CLI
MODEL_LIST_DEFAULT = ['rf', 'tabicl', 'original_v2', 'xgb', 'tabdpt', 'contextab']

SAVE_DIR        = '/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/data_transformations/benchmark_results_new_final/'
PRECOMPUTED_DIR = '/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/data_transformations/perturbed_datasets_final/'

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
parser.add_argument('--no_save',       action='store_true', help='Skip saving results to disk')
parser.add_argument('--baseline_only', action='store_true', help='Run only baseline CV, no perturbations, no saving')
args, _ = parser.parse_known_args()

datasets   = [args.dataset]         if args.dataset else DATASETS
pert_types = [args.pert]            if args.pert    else PERTURBATION_TYPES
MODEL_LIST = args.models.split(',') if args.models  else MODEL_LIST_DEFAULT
SAVE_DIR   = args.save_dir

set_seed(42)
bench = Benchmarker(data_source='pasolli')

#%%
# --- Run benchmark ---
if args.baseline_only:
    for dataset in datasets:
        results_df, _ = bench.run_one(
            dataset        = dataset,
            pert_type      = 'remove_features',
            model_names    = MODEL_LIST,
            cv             = 5,
            n_features_max = 100000,
            device         = 'cuda',
            seed           = 42,
            baseline_only  = True,
        )
        print(results_df.to_string(index=False))

else:
    for dataset in datasets:
        for pert_type in pert_types:

            results_df, predictions_df = bench.run_one(
                dataset            = dataset,
                pert_type          = pert_type,
                model_names        = MODEL_LIST,
                cv                 = 5,
                # n_features_protect = 5,
                n_features_max     = 100000,
                device             = 'cuda',
                seed               = 42,
                precomputed_dir    = PRECOMPUTED_DIR,
            )

            if not args.no_save:
                # Ogni env scrive su un file separato — no race condition
                env_tag = '_'.join(MODEL_LIST)
                out_dir = os.path.join(SAVE_DIR, dataset)
                os.makedirs(out_dir, exist_ok=True)

                metrics_path = os.path.join(out_dir, f"{pert_type}__{env_tag}.parquet")
                results_df.to_parquet(metrics_path, index=False)
                print(f"Saved metrics      -> {metrics_path}")

                preds_path = os.path.join(out_dir, f"{pert_type}__predictions__{env_tag}.parquet")
                predictions_df.to_parquet(preds_path, index=False)
                print(f"Saved predictions  -> {preds_path}")
