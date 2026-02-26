import sys
sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/')

from data_transformations.data_generator import DataGenerator

#%%
#REMOVE FEATURES

gen = DataGenerator(
    generator_type='remove_features',
    data_source='pasolli',
)
X, y = gen.load_data('abundance_cirrhosis--stagediscovery')

protected = gen.discover_and_protect(method='random_forest', n_features=20)

SELECTION_METHODS = [
    'random',
    'lowest_abundance',
    'highest_abundance',
    # 'lowest_prevalence',
    # 'highest_prevalence',
]

K_VALUES = [10, 50, 100, 200]

for method in SELECTION_METHODS:
    print(f"\n{'=' * 70}")
    print(f"SELECTION METHOD: {method}")
    print(f"{'=' * 70}")

    perturbation_params = [
        {'k': k, 'selection_method': method, 'seed': 42}
        for k in K_VALUES
    ]

    gen.visualize_perturbations(perturbation_params=perturbation_params,subplot_size=(5, 5))
    gen.evaluate_classifier_performance(perturbation_params=perturbation_params, cv=5)

#%%
# SPARSITY

gen = DataGenerator(generator_type='sparsity', data_source='pasolli')
X, y = gen.load_data('abundance_cirrhosis--stagediscovery')
protected = gen.discover_and_protect(method='random_forest', n_features=20)

perturbation_params = [
    {'target_sparsity': 0.60, 'seed': 42},
    {'target_sparsity': 0.70, 'seed': 42},
    {'target_sparsity': 0.85, 'seed': 42},
    {'target_sparsity': 0.95, 'seed': 42},
]

gen.visualize_perturbations(perturbation_params=perturbation_params, subplot_size=(5, 5))

results = gen.evaluate_classifier_performance(perturbation_params=perturbation_params, cv=5)

#%%
# ADD_FEATURES perturbation

gen_add = DataGenerator(generator_type='add_random_features', data_source='pasolli')
X, y = gen_add.load_data('abundance_cirrhosis--stagediscovery')
gen_add.discover_and_protect(method='random_forest', n_features=20)

add_params = [
    {'k': 10,  'seed': 42},
    {'k': 50,  'seed': 42},
    {'k': 100, 'seed': 42},
    {'k': 200, 'seed': 42},
]

gen_add.visualize_perturbations(perturbation_params=add_params, subplot_size=(5, 5))
gen_add.evaluate_classifier_performance(perturbation_params=add_params, cv=5)

#%%
# WILCOXON TEST — degradation significance across perturbation levels
import sys
sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/')

from data_transformations.data_generator import DataGenerator

#%%
#REMOVE FEATURES

gen = DataGenerator(
    generator_type='remove_features',
    data_source='pasolli',
)
X, y = gen.load_data('abundance_cirrhosis--stagediscovery')

protected = gen.discover_and_protect(method='random_forest', n_features=20)

SELECTION_METHODS = [
    'random',
    'lowest_abundance',
    'highest_abundance',
    # 'lowest_prevalence',
    # 'highest_prevalence',
]

K_VALUES = [10, 50, 100, 200]

for method in SELECTION_METHODS:
    print(f"\n{'=' * 70}")
    print(f"SELECTION METHOD: {method}")
    print(f"{'=' * 70}")

    perturbation_params = [
        {'k': k, 'selection_method': method, 'seed': 42}
        for k in K_VALUES
    ]

    gen.visualize_perturbations(perturbation_params=perturbation_params,subplot_size=(5, 5))
    gen.evaluate_classifier_performance(perturbation_params=perturbation_params, cv=5)

#%%
# SPARSITY

gen = DataGenerator(generator_type='sparsity', data_source='pasolli')
X, y = gen.load_data('abundance_cirrhosis--stagediscovery')
protected = gen.discover_and_protect(method='random_forest', n_features=20)

perturbation_params = [
    {'target_sparsity': 0.60, 'seed': 42},
    {'target_sparsity': 0.70, 'seed': 42},
    {'target_sparsity': 0.85, 'seed': 42},
    {'target_sparsity': 0.95, 'seed': 42},
]

gen.visualize_perturbations(perturbation_params=perturbation_params, subplot_size=(5, 5))

results = gen.evaluate_classifier_performance(perturbation_params=perturbation_params, cv=5)

#%%
# ADD_FEATURES perturbation

gen_add = DataGenerator(generator_type='add_random_features', data_source='pasolli')
X, y = gen_add.load_data('abundance_cirrhosis--stagediscovery')
gen_add.discover_and_protect(method='random_forest', n_features=20)

add_params = [
    {'k': 10,  'seed': 42},
    {'k': 50,  'seed': 42},
    {'k': 100, 'seed': 42},
    {'k': 200, 'seed': 42},
]

gen_add.visualize_perturbations(perturbation_params=add_params, subplot_size=(5, 5))
gen_add.evaluate_classifier_performance(perturbation_params=add_params, cv=5)

#%%
# WILCOXON TEST — degradation significance across perturbation levels

from scipy.stats import wilcoxon
import pandas as pd
import numpy as np

from scipy.stats import wilcoxon
import numpy as np
import pandas as pd


def wilcoxon_degradation_single_dataset(results_df: pd.DataFrame, metric_folds: str = 'roc_auc_folds'):
    """
    Confronta Original vs ogni livello di perturbazione usando i valori AUROC per-fold,
    applicando Wilcoxon signed-rank test (alternative='greater').

    Ritorna:
    - original_mean
    - perturbed_mean
    - delta
    - p_value
    - significant (p<0.05)
    """

    rows = []

    # Recupera i valori per-fold dell'originale
    original_folds = results_df.loc[results_df['label'] == 'Original', metric_folds].values[0]
    original_folds = np.array(original_folds)

    for label in results_df[results_df['label'] != 'Original']['label'].unique():

        pert_folds = results_df.loc[results_df['label'] == label, metric_folds].values[0]
        pert_folds = np.array(pert_folds)

        # Devono avere lo stesso numero di fold
        if len(pert_folds) != len(original_folds):
            print(f"⚠️ Skip {label}: numero di fold diverso.")
            continue

        # Wilcoxon: test (Original > Perturbed)
        stat, p = wilcoxon(original_folds, pert_folds, alternative='greater')

        rows.append({
            'perturbation': label,
            'original_mean': round(float(original_folds.mean()), 4),
            'perturbed_mean': round(float(pert_folds.mean()), 4),
            'delta': round(float(original_folds.mean() - pert_folds.mean()), 4),
            'p_value': round(float(p), 4),
            'significant': p < 0.05
        })

    return pd.DataFrame(rows).sort_values('p_value')

gen_w = DataGenerator(generator_type='remove_features', data_source='pasolli')
gen_w.load_data('abundance_cirrhosis--stagediscovery')
gen_w.discover_and_protect(method='random_forest', n_features=20)
results_remove = gen_w.evaluate_classifier_performance(perturbation_params=[
    {'k': 10,  'selection_method': 'highest_abundance', 'seed': 42},
    {'k': 50,  'selection_method': 'highest_abundance', 'seed': 42},
    {'k': 100, 'selection_method': 'highest_abundance', 'seed': 42},
    {'k': 200, 'selection_method': 'highest_abundance', 'seed': 42},
], cv=5)
print("\n--- remove_features Wilcoxon ---")
print(wilcoxon_degradation(results_remove).to_string(index=False))


gen_w = DataGenerator(generator_type='sparsity', data_source='pasolli')
gen_w.load_data('abundance_cirrhosis--stagediscovery')
gen_w.discover_and_protect(method='random_forest', n_features=20)
results_sparsity = gen_w.evaluate_classifier_performance(perturbation_params=[
    {'target_sparsity': 0.60, 'seed': 42},
    {'target_sparsity': 0.70, 'seed': 42},
    {'target_sparsity': 0.85, 'seed': 42},
    {'target_sparsity': 0.95, 'seed': 42},
], cv=5)
print("\n--- sparsity Wilcoxon ---")
print(wilcoxon_degradation(results_sparsity).to_string(index=False))


gen_w = DataGenerator(generator_type='add_random_features', data_source='pasolli')
gen_w.load_data('abundance_cirrhosis--stagediscovery')
gen_w.discover_and_protect(method='random_forest', n_features=20)
results_add = gen_w.evaluate_classifier_performance(perturbation_params=[
    {'k': 10,  'seed': 42},
    {'k': 50,  'seed': 42},
    {'k': 100, 'seed': 42},
    {'k': 200, 'seed': 42},
], cv=5)
print("\n--- add_random_features Wilcoxon ---")
print(wilcoxon_degradation(results_add).to_string(index=False))