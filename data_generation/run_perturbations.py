import sys
import pathlib
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

fromdata_generation.data_generator import DataGenerator

#%%
#REMOVE FEATURES

gen = DataGenerator(
    generator_type='remove_features',
    data_source='pasolli',
)
X, y = gen.load_data('abundance_t2d')

protected = gen.discover_and_protect(method='random_forest', n_features=2)

SELECTION_METHODS = [
    # 'random',
    # 'lowest_abundance',
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
X, y = gen.load_data('abundance_t2d')
protected = gen.discover_and_protect(method='random_forest', n_features=2)

perturbation_params = [
    {'target_sparsity': 0.50, 'seed': 42},
    {'target_sparsity': 0.60, 'seed': 42},
    {'target_sparsity': 0.85, 'seed': 42},
    {'target_sparsity': 0.95, 'seed': 42},
]

gen.visualize_perturbations(perturbation_params=perturbation_params, subplot_size=(5, 5))

results = gen.evaluate_classifier_performance(perturbation_params=perturbation_params, cv=5)

#%%
# ADD_FEATURES perturbation

gen_add = DataGenerator(generator_type='add_random_features', data_source='pasolli')
X, y = gen_add.load_data('abundance_t2d')
gen_add.discover_and_protect(method='random_forest', n_features=2)

add_params = [
    {'k': 10,  'seed': 42},
    {'k': 50,  'seed': 42},
    {'k': 100, 'seed': 42},
    {'k': 200, 'seed': 42},
]

gen_add.visualize_perturbations(perturbation_params=add_params, subplot_size=(5, 5))
gen_add.evaluate_classifier_performance(perturbation_params=add_params, cv=5)

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

