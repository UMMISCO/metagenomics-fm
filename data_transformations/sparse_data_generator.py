# %%
import os
import sys
import numpy as np
import pandas as pd
from openTSNE import TSNE
from lets_plot import *
from typing import Tuple, Optional, List

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

        # if verbose:
        #     print(f"Original shape: {X.shape}")
        #     print(f"New shape: {X_reduced.shape}")
        #     if not zero_samples.any():
        #         # Verify renormalization
        #         new_sums = X_reduced.sum(axis=1)
        #         print(f"Row sums after renormalization: min={new_sums.min():.6f}, max={new_sums.max():.6f}")

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

    def _adjust_sparsity(
            self,
            X: pd.DataFrame,
            target_sparsity: Optional[float] = None,
            noise_range: Optional[Tuple[float, float]] = None,
            verbose: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Adjust sparsity by adding or removing the EXACT number of zeros needed.

        - If current sparsity < target → Add zeros to reach target
        - If current sparsity > target → Fill zeros to reach target

        Parameters
        ----------
        X : pd.DataFrame
            Compositional data where rows sum to 1
        target_sparsity : float, optional
            Desired sparsity level (default: 0.5)
        noise_range : tuple, optional
            (min, max) for random noise added to zeros (default: (1e-6, 1e-4))
        verbose : bool, optional
            Print sparsity statistics (default: True)

        Returns
        -------
        pd.DataFrame
            Adjusted compositional data (rows sum to 1)

        Examples
        --------
        # Increase sparsity to 70%
        X_sparse = gen._adjust_sparsity(X, target_sparsity=0.7)

        # Decrease sparsity to 30%
        X_dense = gen._adjust_sparsity(X, target_sparsity=0.3)
        """
        # Get parameters
        target_sparsity = target_sparsity if target_sparsity is not None else self.params.get('target_sparsity', 0.5)
        noise_range = noise_range if noise_range is not None else self.params.get('noise_range', (1e-6, 1e-4))
        verbose = verbose if verbose is not None else self.params.get('verbose', True)

        X = X.copy().astype(float)
        current_sparsity = (X == 0).sum().sum() / X.size
        total_elements = X.size

        current_zeros = int(current_sparsity * total_elements)
        target_zeros = int(target_sparsity * total_elements)
        zeros_to_adjust = target_zeros - current_zeros

        if verbose:
            print("=" * 60)
            print("EXACT SPARSITY ADJUSTMENT")
            print("=" * 60)
            print(f"Current sparsity: {current_sparsity:.4f} ({current_sparsity * 100:.1f}%)")
            print(f"Target sparsity:  {target_sparsity:.4f} ({target_sparsity * 100:.1f}%)")
            print(f"Current zeros:    {current_zeros}")
            print(f"Target zeros:     {target_zeros}")
            print(f"Zeros to adjust:  {zeros_to_adjust:+d}")
            print()

        if zeros_to_adjust == 0:
            if verbose:
                print("✓ Already at target sparsity")
            return X

        if zeros_to_adjust > 0:
            # Need to ADD zeros (sparsify)
            if verbose:
                print(f"STRATEGY: Add {zeros_to_adjust} zeros")
            X_adjusted = self._add_exact_zeros(X, zeros_to_adjust, verbose)
        else:
            # Need to REMOVE zeros (densify)
            if verbose:
                print(f"STRATEGY: Fill {-zeros_to_adjust} zeros")
            X_adjusted = self._fill_exact_zeros(X, -zeros_to_adjust, noise_range, verbose)

        # Final statistics
        final_sparsity = (X_adjusted == 0).sum().sum() / X_adjusted.size
        final_zeros = int(final_sparsity * total_elements)

        if verbose:
            print()
            print("=" * 60)
            print("FINAL RESULTS")
            print("=" * 60)
            print(f"Original sparsity: {current_sparsity:.4f} ({current_zeros} zeros)")
            print(f"Target sparsity:   {target_sparsity:.4f} ({target_zeros} zeros)")
            print(f"Final sparsity:    {final_sparsity:.4f} ({final_zeros} zeros)")
            print(f"Deviation:         {abs(final_zeros - target_zeros)} zeros")
            print("=" * 60)

        return X_adjusted

    def _add_exact_zeros(
            self,
            X: pd.DataFrame,
            num_zeros_to_add: int,
            threshold: float = 1e-6,
            verbose: bool = False
    ) -> pd.DataFrame:
        """
        Add exactly N zeros using power transformation.

        Strategy:
        1. Use binary search to find the right gamma
        2. Apply power transformation: x^(1+gamma)
        3. Set values below fixed threshold (1e-6) to zero
        4. Adjust gamma until exactly N new zeros are created
        5. Renormalize

        Parameters
        ----------
        X : pd.DataFrame
            Compositional data
        num_zeros_to_add : int
            Exact number of zeros to add
        threshold : float
            Fixed threshold below which values become zero (default: 1e-6)
        verbose : bool
            Print progress

        Returns
        -------
        pd.DataFrame
            Sparsified data with exactly N more zeros
        """
        X_sparse = X.copy()

        # Count current non-zero values
        non_zero_mask = X_sparse > 0
        num_non_zero = non_zero_mask.sum().sum()
        current_zeros = (X_sparse == 0).sum().sum()

        if num_non_zero < num_zeros_to_add:
            raise ValueError(
                f"Cannot add {num_zeros_to_add} zeros. "
                f"Only {num_non_zero} non-zero values available."
            )

        if verbose:
            print(f"  Using power transformation to add {num_zeros_to_add} zeros")
            print(f"  Current zeros: {current_zeros}")
            print(f"  Non-zero values available: {num_non_zero}")
            print(f"  Fixed threshold: {threshold:.2e}")

        # Binary search to find the right gamma
        gamma_min, gamma_max = 0.0, 10.0
        tolerance = 1  # Allow ±1 zero deviation
        max_iterations = 50

        best_gamma = None
        best_X_sparse = None
        best_diff = float('inf')

        for iteration in range(max_iterations):
            gamma = (gamma_min + gamma_max) / 2

            # Apply power transformation
            X_transformed = X_sparse ** (1 + gamma)

            # Apply fixed threshold
            X_thresholded = X_transformed.copy()
            X_thresholded[X_thresholded < threshold] = 0.0

            # Count new zeros created
            new_zeros = (X_thresholded == 0).sum().sum() - current_zeros
            diff = abs(new_zeros - num_zeros_to_add)

            if verbose and iteration % 10 == 0:
                print(f"    Iteration {iteration}: gamma={gamma:.3f}, new_zeros={new_zeros}, diff={diff}")

            # Track best result
            if diff < best_diff:
                best_diff = diff
                best_gamma = gamma
                best_X_sparse = X_thresholded.copy()

            # Check convergence
            if diff <= tolerance:
                if verbose:
                    print(f"  ✓ Found gamma={gamma:.3f} creating {new_zeros} new zeros (target: {num_zeros_to_add})")
                X_final = X_thresholded
                break

            # Adjust search range based on result
            if new_zeros < num_zeros_to_add:
                # Need higher gamma (more aggressive shrinkage)
                gamma_min = gamma
            else:
                # Need lower gamma (less aggressive shrinkage)
                gamma_max = gamma

        else:
            # Max iterations reached, use best result
            if verbose:
                print(f"  ⚠ Max iterations reached. Using gamma={best_gamma:.3f}")
                print(
                    f"    Created {(best_X_sparse == 0).sum().sum() - current_zeros} new zeros (target: {num_zeros_to_add})")
            X_final = best_X_sparse

        # Renormalize each row to sum to 1
        row_sums = X_final.sum(axis=1)
        zero_rows = row_sums == 0

        if zero_rows.any():
            if verbose:
                print(f"  Warning: {zero_rows.sum()} rows became all zeros")
            X_final[~zero_rows] = X_final[~zero_rows].div(row_sums[~zero_rows], axis=0)
        else:
            X_final = X_final.div(row_sums, axis=0)

        if verbose:
            final_zeros = (X_final == 0).sum().sum()
            print(f"  Final zero count: {final_zeros} (added {final_zeros - current_zeros})")

        return X_final

    def _fill_exact_zeros(
            self,
            X: pd.DataFrame,
            num_zeros_to_fill: int,
            noise_range: Tuple[float, float] = (1e-6, 1e-4),
            verbose: bool = False
    ) -> pd.DataFrame:
        """
        Fill exactly N zeros with small random values.

        Strategy:
        1. Find all zero positions
        2. Randomly select N zeros to fill
        3. Replace them with random noise
        4. Renormalize each row to sum to 1

        Parameters
        ----------
        X : pd.DataFrame
            Compositional data
        num_zeros_to_fill : int
            Exact number of zeros to fill
        noise_range : tuple
            (min, max) for random noise
        verbose : bool
            Print progress

        Returns
        -------
        pd.DataFrame
            Densified data with exactly N fewer zeros
        """
        X_dense = X.copy()

        # Get all zero positions
        zero_mask = (X_dense == 0)
        num_total_zeros = zero_mask.sum().sum()

        if num_total_zeros < num_zeros_to_fill:
            raise ValueError(
                f"Cannot fill {num_zeros_to_fill} zeros. "
                f"Only {num_total_zeros} zeros available."
            )

        # Get positions of all zeros
        zero_positions = np.argwhere(zero_mask.values)

        # Randomly select which zeros to fill
        np.random.seed(None)  # Ensure randomness
        indices_to_fill = np.random.choice(
            len(zero_positions),
            size=num_zeros_to_fill,
            replace=False
        )
        selected_positions = zero_positions[indices_to_fill]

        # Generate random noise
        noise = np.random.uniform(noise_range[0], noise_range[1], size=num_zeros_to_fill)

        if verbose:
            print(f"  Filling {num_zeros_to_fill} randomly selected zeros")
            print(f"  Noise range: [{noise_range[0]:.2e}, {noise_range[1]:.2e}]")
            print(f"  Actual noise: [{noise.min():.2e}, {noise.max():.2e}]")

        # Fill selected zeros
        for idx, (i, j) in enumerate(selected_positions):
            X_dense.iloc[i, j] = noise[idx]

        # Renormalize rows to sum to 1
        row_sums = X_dense.sum(axis=1)
        X_dense = X_dense.div(row_sums, axis=0)

        return X_dense

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

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

# --------------------------
# Normalization
# --------------------------
def normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    row_sums = df.sum(axis=1)
    return df.div(row_sums, axis=0)

# --------------------------
# Heatmap Comparison
# --------------------------
def plot_heatmap_comparison(
        X_original: pd.DataFrame,
        X_transformed: pd.DataFrame,
        title_original: str = "Original Data",
        title_transformed: str = "Transformed Data",
        figsize: Tuple[int, int] = (16, 6),
        cmap: str = "YlOrRd",
        vmin: float = 0
):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Separate vmax for each heatmap
    vmax_orig = np.percentile(X_original.values, 99)
    vmax_trans = np.percentile(X_transformed.values, 99)

    sns.heatmap(
        X_original.T,
        ax=axes[0],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax_orig,
        cbar_kws={'label': 'Abundance'},
        xticklabels=False,
        yticklabels=False
    )
    axes[0].set_title(title_original)

    sns.heatmap(
        X_transformed.T,
        ax=axes[1],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax_trans,
        cbar_kws={'label': 'Abundance'},
        xticklabels=False,
        yticklabels=False
    )
    axes[1].set_title(title_transformed)

    plt.tight_layout()
    plt.show()



# ================================================================
# LOAD ORIGINAL DATA (UNNORMALIZED)
# ================================================================
X_original, y = open_pasolli('abundance_cirrhosis--stagediscovery')
X_original, y = open_and_filter(X_original, 0.0, 0.0)

# Ensure float
X_original = X_original.astype(float)

print(f"Original data shape: {X_original.shape}")
print(f"Original sparsity: {(X_original == 0).sum().sum() / X_original.size:.2%}")
print("Example original row sums:", X_original.sum(axis=1).head())
print("=" * 60)

# Normalized version for FAIR COMPARISON
X_norm = normalize_rows(X_original)



# ================================================================
# EXAMPLE 1: Remove Features - Random Selection
# ================================================================
print("\n" + "=" * 60)
print("EXAMPLE 1: Remove Features - Random Selection")
print("=" * 60)

gen1 = DataGenerator(
    generator_type='remove_features',
    k=50,
    selection_method='random',
    seed=42,
    verbose=True
)

X_reduced_random = gen1.generate(X_original).astype(float)
X_reduced_random_norm = normalize_rows(X_reduced_random)

plot_heatmap_comparison(
    X_norm,
    X_reduced_random_norm,
    title_original="Original Data (Normalized)",
    title_transformed="After Removing 50 Random Features (Normalized)",
    figsize=(16, 6)
)



# ================================================================
# EXAMPLE 2: Remove 100 Lowest Abundance Features
# ================================================================
print("\n" + "=" * 60)
print("EXAMPLE 2: Remove 100 Lowest Abundance Features")
print("=" * 60)

gen2 = DataGenerator(
    generator_type='remove_features',
    k=100,
    selection_method='lowest_abundance',
    seed=42,
    verbose=True
)

X_reduced_low = gen2.generate(X_original).astype(float)
X_reduced_low_norm = normalize_rows(X_reduced_low)

plot_heatmap_comparison(
    X_norm,
    X_reduced_low_norm,
    title_original="Original Data (Normalized)",
    title_transformed="After Removing 100 Lowest Abundance Features (Normalized)",
    figsize=(16, 6)
)



# ================================================================
# EXAMPLE 3: Increase Sparsity to 85%
# ================================================================
print("\n" + "=" * 60)
print("EXAMPLE 3: Adjust Sparsity → 85%")
print("=" * 60)

gen3 = DataGenerator(
    generator_type='sparsity',
    target_sparsity=0.85,
    verbose=True
)

X_sparse_85 = gen3._adjust_sparsity(X_original).astype(float)
X_sparse_85_norm = normalize_rows(X_sparse_85)

plot_heatmap_comparison(
    X_norm,
    X_sparse_85_norm,
    title_original=f"Original Data (Normalized)",
    title_transformed="After Increasing Sparsity to 85% (Normalized)",
    figsize=(16, 6),
    cmap="Blues"
)



# ================================================================
# EXAMPLE 4: Decrease Sparsity to 60%
# ================================================================
print("\n" + "=" * 60)
print("EXAMPLE 4: Adjust Sparsity → 60%")
print("=" * 60)

gen4 = DataGenerator(
    generator_type='sparsity',
    target_sparsity=0.60,
    noise_range=(1e-7, 1e-5),
    verbose=True
)

X_sparse_60 = gen4._adjust_sparsity(X_original).astype(float)
X_sparse_60_norm = normalize_rows(X_sparse_60)

plot_heatmap_comparison(
    X_norm,
    X_sparse_60_norm,
    title_original="Original Data (Normalized)",
    title_transformed="After Decreasing Sparsity to 60% (Normalized)",
    figsize=(16, 6),
    cmap="Greens"
)



# ================================================================
# EXAMPLE 5: Add Random Features (50)
# ================================================================
print("\n" + "=" * 60)
print("EXAMPLE 5: Add 50 Random Low-Abundance Features")
print("=" * 60)

gen5 = DataGenerator(
    generator_type='add_random_features',
    k=50,
    min_abundance=1e-5,
    max_abundance=1e-4,
    seed=42,
    verbose=True
)

X_aug_50 = gen5.generate(X_original).astype(float)
X_aug_50_norm = normalize_rows(X_aug_50)

plot_heatmap_comparison(
    X_norm,
    X_aug_50_norm,
    title_original="Original Data (Normalized)",
    title_transformed="After Adding 50 Random Features (Normalized)",
    figsize=(16, 6),
    cmap="Purples"
)



# ================================================================
# EXAMPLE 6: Add Random Features (100)
# ================================================================
print("\n" + "=" * 60)
print("EXAMPLE 6: Add 100 Random Medium-Abundance Features")
print("=" * 60)

gen6 = DataGenerator(
    generator_type='add_random_features',
    k=100,
    min_abundance=1e-4,
    max_abundance=1e-3,
    seed=123,
    verbose=True
)

X_aug_100 = gen6.generate(X_original).astype(float)
X_aug_100_norm = normalize_rows(X_aug_100)

plot_heatmap_comparison(
    X_norm,
    X_aug_100_norm,
    title_original="Original Data (Normalized)",
    title_transformed="After Adding 100 Random Features (Normalized)",
    figsize=(16, 6),
    cmap="Oranges"
)



# ================================================================
# SUMMARY TABLE
# ================================================================
print("\n" + "=" * 60)
print("SUMMARY OF ALL TRANSFORMATIONS")
print("=" * 60)

transformations = [
    ("Original (Norm)", X_norm),
    ("Remove 50 Random", X_reduced_random_norm),
    ("Remove 100 Low", X_reduced_low_norm),
    ("Sparsity 85%", X_sparse_85_norm),
    ("Sparsity 60%", X_sparse_60_norm),
    ("Add 50 Random", X_aug_50_norm),
    ("Add 100 Random", X_aug_100_norm),
]

summary_df = pd.DataFrame([
    {
        'Transformation': name,
        'Rows': data.shape[0],
        'Features': data.shape[1],
        'Sparsity': f"{(data == 0).sum().sum() / data.size:.2%}",
        'Mean Abundance': f"{data.mean().mean():.3e}"
    }
    for name, data in transformations
])

print(summary_df.to_string(index=False))



# ================================================================
# COMPARATIVE GRID (optional)
# ================================================================
print("\n" + "=" * 60)
print("Creating comparative visualization...")
print("=" * 60)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

# Remove last empty subplot
fig.delaxes(axes[-1])

for idx, (name, data) in enumerate(transformations):
    vmax_val = np.percentile(data.values, 99)

    sns.heatmap(
        data.T,
        ax=axes[idx],
        cmap="viridis",
        vmin=0,
        vmax=vmax_val,
        cbar=True,
        xticklabels=False,
        yticklabels=False
    )

    sparsity = (data == 0).sum().sum() / data.size
    axes[idx].set_title(
        f"{name}\n{data.shape[1]} features, Sparsity: {sparsity:.1%}",
        fontsize=11,
        fontweight='bold'
    )

plt.tight_layout()
plt.show()
