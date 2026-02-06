# %%
import os
import sys
import numpy as np
import pandas as pd
from openTSNE import TSNE
from lets_plot import *
from typing import Tuple, Optional

sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/tests')
from testing_data.pasolli.pasolli import open_pasolli
from testing_data.metacardis.metacardis import open_metacardis
from testing_data.preprocessing.filter_or_logic import open_and_filter


class DataGenerator:
    """
    A flexible data generator class for microbiome compositional data.
    Currently supports sparsity transformation.
    """

    # Available dataset names
    PASOLLI_DATASETS = [
        'abundance_cirrhosis--stagediscovery',
        'abundance_cirrhosis--stagevalidation',
        'abundance_obesity',
        'abundance_ibd',
        'abundance_t2d',
        'abundance_WT2D'
    ]

    def __init__(
            self,
            generator_type: str = 'sparsity',
            data_source: str = 'pasolli',
            dataset_name: Optional[str] = None,
            filter_params: Optional[Tuple[float, float]] = None,
            **kwargs
    ):
        """
        Initialize DataGenerator.

        Parameters
        ----------
        generator_type : str
            Type of data generation (default = 'sparsity')
        data_source : str
            Data source ('pasolli', 'metacardis', etc.)
        dataset_name : str, optional
            Name of the dataset to load (e.g., 'abundance_WT2D')
        filter_params : tuple of (float, float), optional
            Parameters for open_and_filter (default: (0.0, 0.0))
        **kwargs : dict
            Additional parameters for specific generators:
            - gamma : float, exponent increase factor for sparsity (default: 1.5)
            - threshold : float, values below this are zeroed (default: 1e-6)
            - verbose : bool, print sparsity statistics (default: True)
        """
        self.generator_type = generator_type
        self.data_source = data_source
        self.dataset_name = dataset_name
        self.filter_params = filter_params or (0.0, 0.0)
        self.params = kwargs

        # Store original data
        self.X_original = None
        self.y_original = None
        self.X_generated = None

    def load_data(
            self,
            dataset_name: Optional[str] = None,
            filter_params: Optional[Tuple[float, float]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load data from the specified source.

        Parameters
        ----------
        dataset_name : str, optional
            Override the dataset name from __init__
        filter_params : tuple of (float, float), optional
            Override the filter params from __init__

        Returns
        -------
        X, y : Tuple[pd.DataFrame, pd.Series]
            Features and labels
        """
        dataset_name = dataset_name or self.dataset_name
        filter_params = filter_params or self.filter_params

        if dataset_name is None:
            raise ValueError("dataset_name must be provided")

        # Load data based on source
        if self.data_source == 'pasolli':
            X, y = open_pasolli(dataset_name)
        elif self.data_source == 'metacardis':
            X, y = open_metacardis(dataset_name)
        else:
            raise ValueError(f"Unknown data source: {self.data_source}")

        # Apply filtering
        X, y = open_and_filter(X, *filter_params)

        self.X_original = X
        self.y_original = y

        return X, y

    def generate(
            self,
            X: Optional[pd.DataFrame] = None,
            **override_params
    ) -> pd.DataFrame:
        """
        Generate data based on the specified generator type.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Input data. If None, uses self.X_original
        **override_params : dict
            Parameters to override for this specific generation

        Returns
        -------
        pd.DataFrame
            Generated/transformed data
        """
        if X is None:
            if self.X_original is None:
                raise ValueError("No data available. Call load_data() first or provide X.")
            X = self.X_original

        if self.generator_type == 'sparsity':
            self.X_generated = self._increase_zeros(X, **override_params)
        elif self.generator_type == 'remove_features':
            self.X_generated = self._remove_features(X, **override_params)
        elif self.generator_type == 'add_random_features':
            self.X_generated = self._add_random_features(X, **override_params)
        elif self.generator_type == 'identity':
            # No transformation
            self.X_generated = X.copy()
        else:
            raise ValueError(f"Unknown generator type: {self.generator_type}")

        return self.X_generated

    def _remove_features(
            self,
            X: pd.DataFrame,
            k: Optional[int] = None,
            features_to_remove: Optional[List[str]] = None,
            selection_method: str = 'random',
            seed: Optional[int] = None,
            verbose: Optional[bool] = None) -> pd.DataFrame:
        """
        Remove k features from compositional data and renormalize.

        This preserves compositionality by removing features and
        renormalizing the remaining features so each sample sums to 1.

        Parameters
        ----------
        X : pd.DataFrame
            Compositional data where rows sum to 1
        k : int, optional
            Number of features to remove (required if features_to_remove not provided)
        features_to_remove : List[str], optional
            Specific feature names to remove. If provided, overrides k and selection_method
        selection_method : str, optional
            Method to select features to remove (default: 'random')
            Options:
            - 'random': randomly select k features
            - 'lowest_abundance': remove k features with lowest mean abundance
            - 'highest_abundance': remove k features with highest mean abundance
            - 'lowest_prevalence': remove k features present in fewest samples
            - 'highest_prevalence': remove k features present in most samples
        seed : int, optional
            Random seed for reproducibility (default: None)
        verbose : bool, optional
            Print information about removed features (default: True)

        Returns
        -------
        pd.DataFrame
            Compositional dataset with k features removed and renormalized

        Examples
        --------
        # Remove 10 random features
        X_reduced = gen._remove_features(X, k=10, selection_method='random', seed=42)

        # Remove 50 lowest abundance features
        X_reduced = gen._remove_features(X, k=50, selection_method='lowest_abundance')

        # Remove specific features
        X_reduced = gen._remove_features(X, features_to_remove=['taxon1', 'taxon2'])
        """
        # Get parameters from init or use defaults
        k = k if k is not None else self.params.get('k', None)
        selection_method = self.params.get('selection_method', selection_method)
        seed = seed if seed is not None else self.params.get('seed', None)
        verbose = verbose if verbose is not None else self.params.get('verbose', True)

        if features_to_remove is None and k is None:
            raise ValueError("Either 'k' or 'features_to_remove' must be provided")

        X = X.copy()

        # If specific features provided, use those
        if features_to_remove is not None:
            return self._remove_and_renormalize(X, features_to_remove, verbose)

        # Validate k
        if k > X.shape[1]:
            raise ValueError(f"k={k} is larger than number of features ({X.shape[1]})")

        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Select features based on method
        if selection_method == 'random':
            features_to_remove = np.random.choice(X.columns, size=k, replace=False)

        elif selection_method == 'lowest_abundance':
            mean_abundance = X.mean(axis=0)
            features_to_remove = mean_abundance.nsmallest(k).index.tolist()

        elif selection_method == 'highest_abundance':
            mean_abundance = X.mean(axis=0)
            features_to_remove = mean_abundance.nlargest(k).index.tolist()

        elif selection_method == 'lowest_prevalence':
            prevalence = (X > 0).sum(axis=0)  # Count non-zero samples per feature
            features_to_remove = prevalence.nsmallest(k).index.tolist()

        elif selection_method == 'highest_prevalence':
            prevalence = (X > 0).sum(axis=0)
            features_to_remove = prevalence.nlargest(k).index.tolist()

        else:
            raise ValueError(f"Unknown selection_method: {selection_method}")

        return self._remove_and_renormalize(X, features_to_remove, verbose)

    def _remove_and_renormalize(
            self,
            X: pd.DataFrame,
            features_to_remove: List[str],
            verbose: bool = True
    ) -> pd.DataFrame:
        """
        Helper function to remove specified features and renormalize.
        """

        features_to_keep = [f for f in X.columns if f not in features_to_remove]

        if verbose:
            print(f"Removing {len(features_to_remove)} features")
            print(f"Remaining features: {len(features_to_keep)}")

        # Remove features
        X_reduced = X[features_to_keep].copy()

        # Renormalize each sample to sum to 1
        row_sums = X_reduced.sum(axis=1)

        # Check for samples that would become all zeros
        zero_samples = row_sums == 0
        if zero_samples.any():
            n_zero = zero_samples.sum()
            if verbose:
                print(f"Warning: {n_zero} samples have no abundance in remaining features")
                print(f"These samples will remain as zeros")

        # Renormalize non-zero samples
        X_reduced[~zero_samples] = X_reduced[~zero_samples].div(row_sums[~zero_samples], axis=0)

        if verbose:
            print(f"Original shape: {X.shape}")
            print(f"New shape: {X_reduced.shape}")
            if not zero_samples.any():
                # Verify renormalization
                new_sums = X_reduced.sum(axis=1)
                print(f"Row sums after renormalization: min={new_sums.min():.6f}, max={new_sums.max():.6f}")

        return X_reduced

    def _add_random_features(
            self,
            X: pd.DataFrame,
            k: Optional[int] = None,
            min_abundance: float = 1e-04,
            max_abundance: float = 1e-03,
            feature_prefix: str = 'random_feature',
            seed: Optional[int] = None,
            verbose: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Add k random features sampled from a log-normal distribution and renormalize.

        This adds synthetic features with abundances drawn from a log-normal distribution
        within a specified range, then renormalizes each sample to sum to 1.

        Parameters
        ----------
        X : pd.DataFrame
            Compositional data where rows sum to 1
        k : int, optional
            Number of random features to add (required)
        min_abundance : float, optional
            Minimum abundance for random features (default: 1e-04)
        max_abundance : float, optional
            Maximum abundance for random features (default: 1e-03)
        feature_prefix : str, optional
            Prefix for naming new features (default: 'random_feature')
        seed : int, optional
            Random seed for reproducibility (default: None)
        verbose : bool, optional
            Print information about added features (default: True)

        Returns
        -------
        pd.DataFrame
            Compositional data with k random features added and renormalized (rows sum to 1)

        Notes
        -----
        The function uses a log-normal distribution to generate abundances.
        Parameters are automatically calculated so that approximately 95% of values
        fall within [min_abundance, max_abundance].

        For log-normal distribution:
        - μ (log_mean) is set to the midpoint in log-space
        - σ (log_std) is calculated so ±2σ covers the range

        Examples
        --------
        # Add 10 random features with default range [1e-04, 1e-03]
        X_augmented = gen._add_random_features(X, k=10, seed=42)

        # Add 50 random features with custom range
        X_augmented = gen._add_random_features(
            X, k=50,
            min_abundance=5e-05,
            max_abundance=5e-04,
            seed=42
        )
        """
        # Get parameters from init or use defaults
        k = k if k is not None else self.params.get('k', None)
        min_abundance = self.params.get('min_abundance', min_abundance)
        max_abundance = self.params.get('max_abundance', max_abundance)
        seed = seed if seed is not None else self.params.get('seed', None)
        verbose = verbose if verbose is not None else self.params.get('verbose', True)
        feature_prefix = self.params.get('feature_prefix', feature_prefix)

        if k is None:
            raise ValueError("'k' (number of features to add) must be provided")

        if min_abundance >= max_abundance:
            raise ValueError(f"min_abundance ({min_abundance}) must be < max_abundance ({max_abundance})")

        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        X = X.copy()
        n_samples = X.shape[0]

        # Calculate log-normal parameters to cover [min_abundance, max_abundance]
        log_min = np.log(min_abundance)
        log_max = np.log(max_abundance)

        log_mean = (log_min + log_max) / 2  # Midpoint in log-space
        log_std = (log_max - log_min) / 4  # So ±2σ covers the range

        if verbose:
            print(f"Adding {k} random features to {n_samples} samples")
            print(f"Target range: [{min_abundance:.2e}, {max_abundance:.2e}]")
            print(f"Log-normal parameters: μ={log_mean:.3f}, σ={log_std:.3f}")
            print(f"  Theoretical median: {np.exp(log_mean):.2e}")

        # Sample from log-normal distribution
        # Shape: (n_samples, k)
        random_abundances = np.random.lognormal(
            mean=log_mean,
            sigma=log_std,
            size=(n_samples, k)
        )

        # Create feature names
        new_feature_names = [f"{feature_prefix}_{i + 1}" for i in range(k)]

        # Create DataFrame with new features
        random_features_df = pd.DataFrame(
            random_abundances,
            index=X.index,
            columns=new_feature_names
        )

        # Concatenate original and random features
        X_augmented = pd.concat([X, random_features_df], axis=1)

        # Renormalize each sample to sum to 1
        row_sums = X_augmented.sum(axis=1)
        X_augmented = X_augmented.div(row_sums, axis=0)

        if verbose:
            print(f"\nOriginal shape: {X.shape}")
            print(f"Augmented shape: {X_augmented.shape}")
            print(f"\nRandom features statistics (before renormalization):")
            print(f"  Mean: {random_abundances.mean():.2e}")
            print(f"  Median: {np.median(random_abundances):.2e}")
            print(f"  Std: {random_abundances.std():.2e}")
            print(f"  Min: {random_abundances.min():.2e}")
            print(f"  Max: {random_abundances.max():.2e}")
            print(f"  2.5th percentile: {np.percentile(random_abundances, 2.5):.2e}")
            print(f"  97.5th percentile: {np.percentile(random_abundances, 97.5):.2e}")

            # Check coverage in target range
            in_range = ((random_abundances >= min_abundance) &
                        (random_abundances <= max_abundance)).mean() * 100
            print(f"  % in target range: {in_range:.1f}%")

            print(f"\nRandom features statistics (after renormalization):")
            print(f"  Mean: {X_augmented[new_feature_names].mean().mean():.2e}")
            print(f"  Proportion of total abundance: {X_augmented[new_feature_names].sum(axis=1).mean():.4f}")

            # Verify renormalization
            new_sums = X_augmented.sum(axis=1)
            print(f"\nRow sums after renormalization: min={new_sums.min():.6f}, max={new_sums.max():.6f}")

        return X_augmented

    def _increase_zeros(
            self,
            X: pd.DataFrame,
            gamma: Optional[float] = None,
            threshold: Optional[float] = None,
            verbose: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Increase sparsity of compositional data (rows sum to 1)
        without destroying the per-sample distribution.

        This method:
        - Preserves relative structure among non-zero dominant features
        - Removes weak, low-abundance taxa first
        - Pushes small values toward zero in a realistic compositional way

        Steps:
        1. Raise each value to power (1 + gamma) — shrinks small values more strongly
        2. Zero out values below `threshold`
        3. Renormalize each sample back to sum=1

        Parameters
        ----------
        X : pd.DataFrame
            Compositional data where rows sum to 1
        gamma : float, optional
            Exponent increase factor (default: 1.5)
            Higher values → more aggressive sparsification
        threshold : float, optional
            Values below this are set to zero (default: 1e-6)
        verbose : bool, optional
            Print sparsity statistics (default: True)

        Returns
        -------
        pd.DataFrame
            Sparsified compositional data (rows still sum to 1)
        """
        # Get parameters from init or use defaults
        gamma = gamma if gamma is not None else self.params.get('gamma', 1.5)
        threshold = threshold if threshold is not None else self.params.get('threshold', 1e-6)
        verbose = verbose if verbose is not None else self.params.get('verbose', True)

        X = X.copy().astype(float)
        X_sparse = pd.DataFrame(index=X.index, columns=X.columns)

        for i in X.index:
            x = X.loc[i].values

            # Step 1 — shrink small values strongly
            x_shrunk = x ** (1 + gamma)

            # Step 2 — thresholding to inject zeros
            x_shrunk[x_shrunk < threshold] = 0.0

            # Step 3 — renormalize (if any nonzero values remain)
            s = x_shrunk.sum()
            if s > 0:
                x_shrunk = x_shrunk / s
            else:
                # Extremely sparse sample: fallback to original vector
                x_shrunk = x

            X_sparse.loc[i] = x_shrunk

        if verbose:
            print(f"Original sparsity: {(X == 0).mean(axis=1).mean():.4f}")
            print(f"New sparsity: {(X_sparse == 0).mean(axis=1).mean():.4f}")

        return X_sparse

    def get_stats(self) -> dict:
        """
        Get statistics about the generated data.

        Returns
        -------
        dict
            Statistics dictionary
        """
        if self.X_generated is None:
            raise ValueError("No generated data available. Call generate() first.")

        stats = {
            'dataset_name': self.dataset_name,
            'generator_type': self.generator_type,
            'original_shape': self.X_original.shape if self.X_original is not None else None,
            'generated_shape': self.X_generated.shape,
            'sparsity': (self.X_generated == 0).mean(axis=1).mean(),
            'mean_abundance': self.X_generated.mean(axis=1).mean(),
            'n_samples': len(self.X_generated),
            'n_features': self.X_generated.shape[1]
        }

        if self.X_original is not None:
            stats['original_sparsity'] = (self.X_original == 0).mean(axis=1).mean()
            stats['sparsity_increase'] = (
                    (self.X_generated == 0).mean(axis=1).mean() -
                    (self.X_original == 0).mean(axis=1).mean()
            )

        if self.y_original is not None:
            stats['n_classes'] = len(np.unique(self.y_original))
            stats['class_distribution'] = dict(pd.Series(self.y_original).value_counts())

        return stats

    def visualize_tsne(
            self,
            X: Optional[pd.DataFrame] = None,
            y: Optional[pd.Series] = None,
            perplexity: int = 30,
            n_iter: int = 500
    ):
        """
        Create t-SNE visualization of the data.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Data to visualize. If None, uses X_generated or X_original
        y : pd.Series, optional
            Labels for coloring. If None, uses y_original
        perplexity : int
            t-SNE perplexity parameter
        n_iter : int
            Number of iterations

        Returns
        -------
        lets_plot figure
        """
        if X is None:
            X = self.X_generated if self.X_generated is not None else self.X_original
        if y is None:
            y = self.y_original

        if X is None:
            raise ValueError("No data available for visualization")

        # Run t-SNE
        tsne = TSNE(perplexity=perplexity, n_iter=n_iter, random_state=42)
        embedding = tsne.fit(X.values)

        # Create dataframe for plotting
        plot_df = pd.DataFrame({
            'TSNE1': embedding[:, 0],
            'TSNE2': embedding[:, 1],
            'label': y.values if y is not None else 'unknown'
        })

        # Create plot
        return (
                ggplot(plot_df, aes(x='TSNE1', y='TSNE2', color='label')) +
                geom_point(size=2, alpha=0.7) +
                labs(title=f't-SNE: {self.dataset_name} ({self.generator_type})',
                     x='t-SNE 1', y='t-SNE 2') +
                theme_minimal()
        )

# %% TEST - Using DataGenerator workflow
# Initialize with dataset
gen = DataGenerator(
    generator_type='sparsity',
    data_source='pasolli',
    dataset_name='abundance_WT2D',
    filter_params=(0.0, 0.0),
    gamma=3,
    threshold=1e-6,
    verbose=True
)

# Load and generate in one go
X, y = gen.load_data()
X_sparse = gen.generate()

# Get statistics
stats = gen.get_stats()
print("\nDataset Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")

# TEST - Visualize with t-SNE
plot = gen.visualize_tsne()
plot.show()

# %% TEST - Compare multiple datasets
datasets = [
    'abundance_cirrhosis--stagediscovery',
    'abundance_cirrhosis--stagevalidation',
    'abundance_obesity',
    'abundance_ibd',
    'abundance_t2d',
    'abundance_WT2D'
]
for dataset in datasets:
    gen = DataGenerator(
        generator_type='sparsity',
        data_source='pasolli',
        dataset_name=dataset,
        gamma=3,
        threshold=1e-6
    )
    gen.load_data()
    gen.generate()
    stats = gen.get_stats()
    print(f"\n{dataset}:")
    print(f"  Samples: {stats['n_samples']}")
    print(f"  Features: {stats['n_features']}")
    print(f"  Sparsity increase: {stats['sparsity_increase']:.4f}")

    plot = gen.visualize_tsne()
    plot.show()

# %% Experiment: Test all selection methods with proper train/test split
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Load data
gen = DataGenerator(
    generator_type='remove_features',
    data_source='pasolli',
    dataset_name='abundance_WT2D',
    verbose=False
)
gen.load_data()

# IMPORTANT: Split data FIRST, before any transformations
X_train, X_test, y_train, y_test = train_test_split(
    gen.X_original,
    gen.y_original,
    test_size=0.2,  # 80% train, 20% test
    random_state=42,
    stratify=gen.y_original  # Maintain class proportions
)

print(f"Original data shape: {gen.X_original.shape}")
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Train class distribution: {y_train.value_counts().to_dict()}")
print(f"Test class distribution: {y_test.value_counts().to_dict()}")

# Experimental parameters
k_values = [1, 5, 10, 20, 50]
selection_methods = [
    'random',
    'lowest_abundance',
    'highest_abundance',
    'lowest_prevalence',
    'highest_prevalence'
]

# Store results
results = []

print("\n" + "=" * 80)
print("Running experiments with proper train/test split...")
print("=" * 80)

for method in selection_methods:
    print(f"\nTesting method: {method}")
    print("-" * 80)

    for k in k_values:
        # CRITICAL: Feature selection based ONLY on training data
        gen_temp = DataGenerator(
            generator_type='remove_features',
            verbose=False
        )

        # Apply transformation to training data
        X_train_reduced = gen_temp._remove_features(
            X_train,
            k=k,
            selection_method=method,
            seed=42
        )

        # Get the features that were kept
        features_kept = X_train_reduced.columns.tolist()

        # Apply SAME transformation to test data (use same features)
        X_test_reduced = gen_temp._remove_and_renormalize(
            X_test,
            features_to_remove=[f for f in X_test.columns if f not in features_kept],
            verbose=False
        )

        # Calculate statistics
        train_sparsity = (X_train_reduced == 0).mean().mean()
        test_sparsity = (X_test_reduced == 0).mean().mean()
        n_features_remaining = X_train_reduced.shape[1]

        # Train model on training data only
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_train_reduced, y_train)

        # Evaluate on training data (with CV for robustness)
        cv_scores = cross_val_score(rf, X_train_reduced, y_train, cv=5)
        train_accuracy = cv_scores.mean()
        train_std = cv_scores.std()

        # Evaluate on held-out test data (TRUE performance)
        test_predictions = rf.predict(X_test_reduced)
        test_accuracy = accuracy_score(y_test, test_predictions)

        # Store results
        results.append({
            'method': method,
            'k': k,
            'n_features_remaining': n_features_remaining,
            'train_sparsity': train_sparsity,
            'test_sparsity': test_sparsity,
            'train_accuracy_mean': train_accuracy,
            'train_accuracy_std': train_std,
            'test_accuracy': test_accuracy,
            'overfitting': train_accuracy - test_accuracy  # Gap indicates overfitting
        })

        print(f"  k={k:2d}: features={n_features_remaining:4d}, "
              f"train_acc={train_accuracy:.4f}±{train_std:.4f}, "
              f"test_acc={test_accuracy:.4f}, "
              f"gap={train_accuracy - test_accuracy:.4f}")

# Convert to DataFrame
results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("SUMMARY OF RESULTS")
print("=" * 80)
print(results_df.to_string(index=False))

# %% Visualize results with train/test comparison
from lets_plot import *

# Plot 1: Train vs Test Accuracy
plot_data = results_df.melt(
    id_vars=['method', 'k'],
    value_vars=['train_accuracy_mean', 'test_accuracy'],
    var_name='dataset',
    value_name='accuracy'
)

plot1 = (
        ggplot(plot_data, aes(x='k', y='accuracy', color='method', linetype='dataset')) +
        geom_line(size=1) +
        geom_point(size=2) +
        labs(title='Train vs Test Performance: Feature Removal',
             x='Number of features removed (k)',
             y='Accuracy',
             color='Selection method',
             linetype='Dataset') +
        theme_minimal() +
        facet_wrap('method', ncol=3)
)
plot1.show()
