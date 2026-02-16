#%%

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Union


class DataGenerator:
    """
    A flexible data generator class for microbiome compositional data.
    Currently supports sparsity transformation with feature protection.
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
            protected_features: Optional[List[str]] = None,
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
        protected_features : list of str, optional
            Feature names that should NOT be modified during perturbations
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
        self.protected_features = protected_features or []

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

    def discover_informative_features(
            self,
            method: str = 'random_forest',
            n_features: Optional[int] = None,
            threshold: Optional[float] = None,
            return_scores: bool = False,
            verbose: bool = True,
            **method_kwargs
    ) -> Union[List[str], Tuple[List[str], pd.Series]]:
        """
        Discover informative features using Random Forest or LASSO.

        Parameters
        ----------
        method : str
            Feature selection method:
            - 'random_forest' or 'rf': Feature importance from Random Forest
            - 'lasso' or 'l1': L1-regularized logistic regression coefficients
        n_features : int, optional
            Number of top features to select
        threshold : float, optional
            Importance threshold (alternative to n_features)
            Features with importance > threshold are selected
        return_scores : bool
            If True, return (features, scores) tuple
        verbose : bool
            Print detailed information
        **method_kwargs : dict
            Additional parameters:
            - For RF: n_estimators, max_depth, random_state, etc.
            - For LASSO: C, max_iter, random_state, etc.

        Returns
        -------
        features : list of str
            Names of informative features
        scores : pd.Series (if return_scores=True)
            Importance scores for all features

        Examples
        --------
        # Get top 20 features by Random Forest
        informative = gen.discover_informative_features(
            method='random_forest',
            n_features=20
        )

        # Get features above importance threshold with LASSO
        informative = gen.discover_informative_features(
            method='lasso',
            threshold=0.01,
            C=0.1
        )

        # Get both features and their scores
        features, scores = gen.discover_informative_features(
            method='rf',
            n_features=15,
            return_scores=True
        )
        """
        if self.X_original is None or self.y_original is None:
            raise ValueError("Data must be loaded first. Call load_data().")

        if n_features is None and threshold is None:
            raise ValueError("Either n_features or threshold must be specified")

        # Normalize method name
        method = method.lower()
        if method in ['random_forest', 'rf']:
            scores = self._importance_random_forest(verbose, **method_kwargs)
            method_name = 'Random Forest'
        elif method in ['lasso', 'l1']:
            scores = self._importance_lasso(verbose, **method_kwargs)
            method_name = 'LASSO'
        else:
            raise ValueError(f"Unknown method: {method}. Use 'random_forest' or 'lasso'")

        # Select features based on n_features or threshold
        if n_features is not None:
            selected_features = scores.nlargest(n_features).index.tolist()
        else:
            selected_features = scores[scores > threshold].index.tolist()

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"INFORMATIVE FEATURES DISCOVERED: {len(selected_features)}")
            print(f"{'=' * 70}")
            print(f"Method: {method_name}")
            if n_features:
                print(f"Selection: Top {n_features} features")
            else:
                print(f"Selection: Threshold > {threshold}")

            print(f"\nTop 10 features:")
            for i, (feat, score) in enumerate(scores.nlargest(10).items(), 1):
                feat_display = feat[:50] + '...' if len(feat) > 50 else feat
                print(f"  {i:2d}. {feat_display:53s} {score:.4f}")

        if return_scores:
            return selected_features, scores
        return selected_features

    def _importance_random_forest(
            self,
            verbose: bool = True,
            n_estimators: int = 100,
            max_depth: Optional[int] = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            random_state: int = 42,
            **rf_kwargs
    ) -> pd.Series:
        """Random Forest feature importance."""
        from sklearn.ensemble import RandomForestClassifier

        if verbose:
            print(f"Computing Random Forest importance (n_estimators={n_estimators})...")

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            **rf_kwargs
        )
        rf.fit(self.X_original, self.y_original)

        importances = pd.Series(
            rf.feature_importances_,
            index=self.X_original.columns
        ).sort_values(ascending=False)

        return importances

    def _importance_lasso(
            self,
            verbose: bool = True,
            C: float = 1.0,
            max_iter: int = 1000,
            random_state: int = 42,
            **lasso_kwargs
    ) -> pd.Series:
        """L1-regularized logistic regression coefficients."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        if verbose:
            print(f"Computing LASSO importance (C={C})...")

        # Standardize features for LASSO
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_original)

        lasso = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            **lasso_kwargs
        )
        lasso.fit(X_scaled, self.y_original)

        # Use absolute coefficients as importance
        if len(np.unique(self.y_original)) == 2:
            importances = pd.Series(
                np.abs(lasso.coef_[0]),
                index=self.X_original.columns
            ).sort_values(ascending=False)
        else:
            # Multi-class: use max absolute coefficient across classes
            importances = pd.Series(
                np.abs(lasso.coef_).max(axis=0),
                index=self.X_original.columns
            ).sort_values(ascending=False)

        return importances

    def discover_and_protect(
            self,
            method: str = 'random_forest',
            n_features: int = 20,
            **kwargs
    ) -> List[str]:
        """
        Discover informative features and automatically set them as protected.

        Parameters
        ----------
        method : str
            'random_forest' or 'lasso'
        n_features : int
            Number of features to protect
        **kwargs : dict
            Additional parameters for the discovery method

        Returns
        -------
        list of str
            Protected feature names

        Examples
        --------
        # Discover and protect top 20 features using Random Forest
        protected = gen.discover_and_protect(
            method='random_forest',
            n_features=20,
            n_estimators=200
        )

        # Discover and protect using LASSO
        protected = gen.discover_and_protect(
            method='lasso',
            n_features=15,
            C=0.1
        )
        """
        informative_features = self.discover_informative_features(
            method=method,
            n_features=n_features,
            **kwargs
        )

        self.protected_features = informative_features

        print(f"\n✅ {len(informative_features)} features marked as PROTECTED")
        print("These will NOT be modified during perturbations\n")

        return informative_features

    def _get_modifiable_features(self, X: pd.DataFrame) -> List[str]:
        """Get list of features that can be perturbed (not protected)."""
        all_features = X.columns.tolist()
        modifiable = [f for f in all_features if f not in self.protected_features]
        return modifiable

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
            self.X_generated = self._adjust_sparsity(X, **override_params)
        elif self.generator_type == 'remove_features':
            self.X_generated = self._remove_features(X, **override_params)
        elif self.generator_type == 'add_random_features':
            self.X_generated = self._add_random_features(X, **override_params)
        elif self.generator_type == 'identity':
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
            verbose: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Remove k features from compositional data and renormalize.
        PROTECTED FEATURES ARE NEVER REMOVED.

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

        # Get modifiable features only
        modifiable_features = self._get_modifiable_features(X)

        if verbose and self.protected_features:
            print(f"\n🛡️  Protected features: {len(self.protected_features)}")
            print(f"   Modifiable features: {len(modifiable_features)}")

        # If specific features provided, check they're not protected
        if features_to_remove is not None:
            protected_in_removal = set(features_to_remove) & set(self.protected_features)
            if protected_in_removal:
                raise ValueError(
                    f"Cannot remove protected features: {protected_in_removal}"
                )
            return self._remove_and_renormalize(X, features_to_remove, verbose)

        # Validate k
        if k > len(modifiable_features):
            raise ValueError(
                f"k={k} is larger than number of modifiable features ({len(modifiable_features)}). "
                f"{len(self.protected_features)} features are protected."
            )

        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Select features based on method (from MODIFIABLE features only)
        if selection_method == 'random':
            features_to_remove = np.random.choice(modifiable_features, size=k, replace=False)

        elif selection_method == 'lowest_abundance':
            mean_abundance = X[modifiable_features].mean(axis=0)
            features_to_remove = mean_abundance.nsmallest(k).index.tolist()

        elif selection_method == 'highest_abundance':
            mean_abundance = X[modifiable_features].mean(axis=0)
            features_to_remove = mean_abundance.nlargest(k).index.tolist()

        elif selection_method == 'lowest_prevalence':
            prevalence = (X[modifiable_features] > 0).sum(axis=0)
            features_to_remove = prevalence.nsmallest(k).index.tolist()

        elif selection_method == 'highest_prevalence':
            prevalence = (X[modifiable_features] > 0).sum(axis=0)
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
            if self.protected_features:
                protected_remaining = len(set(features_to_keep) & set(self.protected_features))
                print(f"Protected features still present: {protected_remaining}/{len(self.protected_features)}")

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
        PROTECTED FEATURES ARE NEVER ZEROED.

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

        # Get modifiable features
        modifiable_features = self._get_modifiable_features(X)

        # Count current non-zero values in MODIFIABLE features only
        non_zero_mask = X_sparse[modifiable_features] > 0
        num_non_zero = non_zero_mask.sum().sum()
        current_zeros = (X_sparse[modifiable_features] == 0).sum().sum()

        if num_non_zero < num_zeros_to_add:
            raise ValueError(
                f"Cannot add {num_zeros_to_add} zeros. "
                f"Only {num_non_zero} non-zero values available in modifiable features."
            )

        if verbose:
            print(f"  Using power transformation to add {num_zeros_to_add} zeros")
            if self.protected_features:
                print(f"  🛡️  Protected features: {len(self.protected_features)} (will not be zeroed)")
            print(f"  Current zeros in modifiable features: {current_zeros}")
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

            # Apply power transformation ONLY to modifiable features
            X_transformed = X_sparse.copy()
            X_transformed[modifiable_features] = X_sparse[modifiable_features] ** (1 + gamma)

            # Apply fixed threshold ONLY to modifiable features
            X_thresholded = X_transformed.copy()
            modifiable_mask = X_thresholded[modifiable_features] < threshold
            X_thresholded.loc[:, modifiable_features] = X_thresholded[modifiable_features].where(
                ~modifiable_mask, 0.0
            )

            # Count new zeros created in modifiable features
            new_zeros = (X_thresholded[modifiable_features] == 0).sum().sum() - current_zeros
            diff = abs(new_zeros - num_zeros_to_add)

            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: gamma={gamma:.3f}, new_zeros={new_zeros}, diff={diff}")

            # Track best result
            if diff < best_diff:
                best_diff = diff
                best_gamma = gamma
                best_X_sparse = X_thresholded.copy()

            # Check convergence
            if diff <= tolerance:
                if verbose:
                    print(f"✓ Found gamma={gamma:.3f} creating {new_zeros} new zeros (target: {num_zeros_to_add})")
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
                    f"    Created {(best_X_sparse[modifiable_features] == 0).sum().sum() - current_zeros} new zeros (target: {num_zeros_to_add})")
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
            final_zeros = (X_final[modifiable_features] == 0).sum().sum()
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

    #%%
    def visualize_perturbations(
            self,
            perturbation_params: List[dict],
            method: str = 'pca',
            figsize: Tuple[int, int] = (15, 10),
            save_path: Optional[str] = None,
            random_state: int = 42
    ) -> None:
        """
        Create scatter plots comparing original data with multiple perturbations.

        Parameters
        ----------
        perturbation_params : list of dict
            List of parameter dictionaries for each perturbation.
            Each dict should contain parameters for the generate() method.
            Example: [{'k': 10}, {'k': 50}, {'k': 100}]
        method : str
            Dimensionality reduction method:
            - 'pca': Principal Component Analysis (default)
            - 'tsne': t-SNE
            - 'umap': UMAP (requires umap-learn package)
        figsize : tuple
            Figure size (width, height)
        save_path : str, optional
            Path to save the figure. If None, display only.
        random_state : int
            Random seed for reproducibility

        Examples
        --------
        # Visualize feature removal with different k values
        gen = DataGenerator(generator_type='remove_features')
        X, y = gen.load_data('abundance_ibd')
        gen.discover_and_protect(method='rf', n_features=20)

        gen.visualize_perturbations(
            perturbation_params=[
                {'k': 10},
                {'k': 50},
                {'k': 100},
                {'k': 200}
            ],
            method='pca'
        )

        # Visualize sparsity with different targets
        gen = DataGenerator(generator_type='sparsity')
        X, y = gen.load_data('abundance_ibd')
        gen.discover_and_protect(method='rf', n_features=20)

        gen.visualize_perturbations(
            perturbation_params=[
                {'target_sparsity': 0.3},
                {'target_sparsity': 0.5},
                {'target_sparsity': 0.7},
                {'target_sparsity': 0.9}
            ],
            method='tsne'
        )
        """
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        import seaborn as sns

        if self.X_original is None or self.y_original is None:
            raise ValueError("Data must be loaded first. Call load_data().")

        # Store original generator type to restore later
        original_gen_type = self.generator_type

        # Prepare data: original + all perturbations
        datasets = []
        labels = []

        # Add original data
        datasets.append(self.X_original)
        labels.append('Original')

        # Generate perturbed versions
        print(f"Generating {len(perturbation_params)} perturbations...")
        for i, params in enumerate(perturbation_params):
            print(f"  Perturbation {i + 1}/{len(perturbation_params)}: {params}")
            X_pert = self.generate(**params)
            datasets.append(X_pert)

            # Create label from parameters
            param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
            labels.append(f"Pert {i + 1}: {param_str}")

        # Combine all datasets
        X_combined = pd.concat(datasets, axis=0, ignore_index=True)

        # Create dataset labels for coloring
        dataset_labels = []
        for i, (X, label) in enumerate(zip(datasets, labels)):
            dataset_labels.extend([label] * len(X))

        # Apply dimensionality reduction
        print(f"\nApplying {method.upper()} dimensionality reduction...")

        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=random_state)
            X_reduced = reducer.fit_transform(X_combined)
            var_explained = reducer.explained_variance_ratio_
            xlabel = f'PC1 ({var_explained[0] * 100:.1f}%)'
            ylabel = f'PC2 ({var_explained[1] * 100:.1f}%)'

        elif method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=random_state, perplexity=30)
            X_reduced = reducer.fit_transform(X_combined)
            xlabel = 't-SNE 1'
            ylabel = 't-SNE 2'

        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=random_state)
                X_reduced = reducer.fit_transform(X_combined)
                xlabel = 'UMAP 1'
                ylabel = 'UMAP 2'
            except ImportError:
                raise ImportError("UMAP requires umap-learn: pip install umap-learn")
        else:
            raise ValueError(f"Unknown method: {method}. Use 'pca', 'tsne', or 'umap'")

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Define colors for each perturbation level
        colors = sns.color_palette("husl", len(labels))

        # Plot each dataset
        start_idx = 0
        for i, (X, label, color) in enumerate(zip(datasets, labels, colors)):
            end_idx = start_idx + len(X)

            if i == 0:
                # Original data: larger markers, different shape
                ax.scatter(
                    X_reduced[start_idx:end_idx, 0],
                    X_reduced[start_idx:end_idx, 1],
                    c=[color],
                    label=label,
                    alpha=0.7,
                    s=100,
                    marker='o',
                    edgecolors='black',
                    linewidths=1.5
                )
            else:
                # Perturbed data
                ax.scatter(
                    X_reduced[start_idx:end_idx, 0],
                    X_reduced[start_idx:end_idx, 1],
                    c=[color],
                    label=label,
                    alpha=0.5,
                    s=50,
                    marker='x'
                )

            start_idx = end_idx

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(
            f'Data Perturbation Comparison ({method.upper()})\n'
            f'Generator: {self.generator_type}, Protected features: {len(self.protected_features)}',
            fontsize=14,
            fontweight='bold'
        )
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")

        plt.show()

        # Restore original generator type
        self.generator_type = original_gen_type

    def compare_perturbation_statistics(
            self,
            perturbation_params: List[dict],
            metrics: List[str] = ['sparsity', 'n_features', 'mean_abundance'],
            figsize: Tuple[int, int] = (12, 4)
    ) -> pd.DataFrame:
        """
        Compare statistics across different perturbation levels.

        Parameters
        ----------
        perturbation_params : list of dict
            List of parameter dictionaries for each perturbation
        metrics : list of str
            Metrics to compute:
            - 'sparsity': Percentage of zeros
            - 'n_features': Number of features remaining
            - 'mean_abundance': Mean feature abundance
            - 'median_abundance': Median feature abundance
            - 'diversity': Shannon diversity
        figsize : tuple
            Figure size

        Returns
        -------
        pd.DataFrame
            Statistics for each perturbation level

        Examples
        --------
        stats = gen.compare_perturbation_statistics(
            perturbation_params=[
                {'k': 10},
                {'k': 50},
                {'k': 100}
            ],
            metrics=['sparsity', 'n_features', 'diversity']
        )
        print(stats)
        """
        import matplotlib.pyplot as plt

        if self.X_original is None:
            raise ValueError("Data must be loaded first. Call load_data().")

        stats_list = []

        # Compute stats for original
        stats_list.append(self._compute_stats(self.X_original, 'Original', metrics))

        # Compute stats for each perturbation
        print(f"Computing statistics for {len(perturbation_params)} perturbations...")
        for i, params in enumerate(perturbation_params):
            X_pert = self.generate(**params)
            param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
            label = f"Pert {i + 1}"
            stats_list.append(self._compute_stats(X_pert, label, metrics, params))

        # Create DataFrame
        stats_df = pd.DataFrame(stats_list)

        # Plot statistics
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            ax.plot(range(len(stats_df)), stats_df[metric], marker='o', linewidth=2, markersize=8)
            ax.set_xticks(range(len(stats_df)))
            ax.set_xticklabels(stats_df['label'], rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return stats_df

    def _compute_stats(
            self,
            X: pd.DataFrame,
            label: str,
            metrics: List[str],
            params: Optional[dict] = None
    ) -> dict:
        """Helper function to compute statistics for a dataset."""
        from scipy.stats import entropy

        stats = {'label': label}

        if params:
            stats.update(params)

        if 'sparsity' in metrics:
            sparsity = (X == 0).sum().sum() / X.size
            stats['sparsity'] = sparsity

        if 'n_features' in metrics:
            stats['n_features'] = X.shape[1]

        if 'mean_abundance' in metrics:
            stats['mean_abundance'] = X.mean().mean()

        if 'median_abundance' in metrics:
            stats['median_abundance'] = X.median().median()

        if 'diversity' in metrics:
            # Shannon diversity per sample, then average
            diversities = []
            for i in range(len(X)):
                row = X.iloc[i]
                row_nonzero = row[row > 0]
                if len(row_nonzero) > 0:
                    diversities.append(entropy(row_nonzero))
            stats['diversity'] = np.mean(diversities) if diversities else 0

        return stats

    #%%

    # 1. Setup
    from data_generator import DataGenerator

    gen = DataGenerator(
        generator_type='remove_features',
        data_source='pasolli'
    )
    X, y = gen.load_data('abundance_ibd')

    # 2. Protect important features
    protected = gen.discover_and_protect(method='random_forest', n_features=20)

    # 3. Visualize perturbations with different k values
    gen.visualize_perturbations(
        perturbation_params=[
            {'k': 10, 'selection_method': 'random', 'seed': 42},
            {'k': 50, 'selection_method': 'random', 'seed': 42},
            {'k': 100, 'selection_method': 'random', 'seed': 42},
            {'k': 200, 'selection_method': 'random', 'seed': 42}
        ],
        method='pca',
        save_path='perturbation_comparison_pca.png'
    )

    # 4. Also try with t-SNE
    gen.visualize_perturbations(
        perturbation_params=[
            {'k': 10, 'selection_method': 'random', 'seed': 42},
            {'k': 50, 'selection_method': 'random', 'seed': 42},
            {'k': 100, 'selection_method': 'random', 'seed': 42},
            {'k': 200, 'selection_method': 'random', 'seed': 42}
        ],
        method='tsne',
        save_path='perturbation_comparison_tsne.png'
    )

    # 5. Compare statistics
    stats = gen.compare_perturbation_statistics(
        perturbation_params=[
            {'k': 10, 'seed': 42},
            {'k': 50, 'seed': 42},
            {'k': 100, 'seed': 42},
            {'k': 200, 'seed': 42}
        ],
        metrics=['sparsity', 'n_features', 'mean_abundance', 'diversity']
    )
    print(stats)