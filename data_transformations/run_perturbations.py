from perturbation_pkg.data_generator import DataGenerator

# %%
# 1. Initialise
gen = DataGenerator(
    generator_type='remove_features',
    data_source='pasolli',
)

# 2. Load dataset
X, y = gen.load_data('abundance_ibd')

# 3. Discover and protect top informative features
protected = gen.discover_and_protect(method='random_forest', n_features=20)

# 4. One subplot per perturbation: x = original value, y = perturbed value
#    Points on the diagonal = no change
gen.visualize_perturbations(
    perturbation_params=[
        {'k': 10,  'selection_method': 'highest_abundance', 'seed': 42},
        {'k': 50,  'selection_method': 'highest_abundance', 'seed': 42},
        {'k': 100, 'selection_method': 'highest_abundance', 'seed': 42},
        {'k': 200, 'selection_method': 'highest_abundance', 'seed': 42},
    ],
    subplot_size=(5, 5),      # size of each individual subplot
    save_path='perturbation_scatter.png',  # optional
)

# 5. Summary statistics across the same perturbation levels
stats = gen.compare_perturbation_statistics(
    perturbation_params=[
        {'k': 10,  'seed': 42},
        {'k': 50,  'seed': 42},
        {'k': 100, 'seed': 42},
        {'k': 200, 'seed': 42},
    ],
    metrics=['sparsity', 'n_features', 'mean_abundance', 'diversity'],
)
print(stats)