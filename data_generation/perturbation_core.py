import sys
import pathlib
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# =============================================================================
# FEATURE SELECTION MODULE
# =============================================================================

class FeatureSelector:
    """
    Discovers informative features using Random Forest or LASSO.
    Operates on a fixed (X, y) pair provided at construction.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

    def random_forest_importance(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        verbose: bool = True,
        **kwargs,
    ) -> pd.Series:
        """Return feature importances from a Random Forest classifier."""
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
            **kwargs,
        )
        rf.fit(self.X, self.y)

        return pd.Series(
            rf.feature_importances_, index=self.X.columns
        ).sort_values(ascending=False)

    def lasso_importance(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        verbose: bool = True,
        **kwargs,
    ) -> pd.Series:
        """Return absolute LASSO coefficients as feature importances."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        if verbose:
            print(f"Computing LASSO importance (C={C})...")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        lasso = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs,
        )
        lasso.fit(X_scaled, self.y)

        # Binary: coef_[0]; multi-class: max absolute coef across classes
        if len(np.unique(self.y)) == 2:
            coefs = np.abs(lasso.coef_[0])
        else:
            coefs = np.abs(lasso.coef_).max(axis=0)

        return pd.Series(coefs, index=self.X.columns).sort_values(ascending=False)

    def select(
        self,
        method: str = 'random_forest',
        n_features: Optional[int] = None,
        threshold: Optional[float] = None,
        return_scores: bool = False,
        verbose: bool = True,
        **method_kwargs,
    ) -> Union[List[str], Tuple[List[str], pd.Series]]:
        """
        Select features by importance.

        Parameters
        ----------
        method : str
            'random_forest' / 'rf'  or  'lasso' / 'l1'
        n_features : int, optional
            Return the top-N features.
        threshold : float, optional
            Return features whose importance exceeds this value.
        return_scores : bool
            If True, also return the full importance Series.
        verbose : bool
        **method_kwargs
            Forwarded to the underlying importance method.

        Returns
        -------
        List[str]  or  (List[str], pd.Series)
        """
        if n_features is None and threshold is None:
            raise ValueError("Specify either n_features or threshold.")

        method = method.lower()
        if method in ('random_forest', 'rf'):
            scores = self.random_forest_importance(verbose=verbose, **method_kwargs)
            method_name = 'Random Forest'
        elif method in ('lasso', 'l1'):
            scores = self.lasso_importance(verbose=verbose, **method_kwargs)
            method_name = 'LASSO'
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'random_forest' or 'lasso'.")

        if n_features is not None:
            selected = scores.nlargest(n_features).index.tolist()
        else:
            selected = scores[scores > threshold].index.tolist()

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"INFORMATIVE FEATURES DISCOVERED: {len(selected)}")
            print(f"{'=' * 70}")
            print(f"Method: {method_name}")
            criterion = f"Top {n_features}" if n_features is not None else f"Threshold > {threshold}"
            print(f"Selection: {criterion}")
            print(f"\nTop 10 features:")
            for i, (feat, score) in enumerate(scores.nlargest(10).items(), 1):
                feat_display = feat[:50] + '...' if len(feat) > 50 else feat
                print(f"  {i:2d}. {feat_display:53s} {score:.4f}")

        return (selected, scores) if return_scores else selected


# =============================================================================
# PERTURBATION MODULE
# =============================================================================

class Perturbation:
    """
    Base class for individual perturbation strategies.
    Subclasses implement `apply(X, **kwargs) -> pd.DataFrame`.
    Protected features are never modified.
    """

    def __init__(self, protected_features: Optional[List[str]] = None):
        self.protected_features: List[str] = protected_features or []

    def _modifiable(self, X: pd.DataFrame) -> List[str]:
        return [c for c in X.columns if c not in self.protected_features]

    @staticmethod
    def _renormalize(X: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """Divide each row by its sum so rows sum to 1. Rows of all-zeros are left as-is."""
        X = X.astype(float)
        row_sums = X.sum(axis=1)
        zero_rows = row_sums == 0
        if verbose and zero_rows.any():
            print(f"Warning: {zero_rows.sum()} samples are all-zero after perturbation.")
        result = X.copy()
        result.loc[~zero_rows] = X.loc[~zero_rows].div(row_sums[~zero_rows], axis=0)
        return result

    def apply(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError


class RemoveFeaturesPerturbation(Perturbation):
    """
    Remove k features from compositional data and renormalize.
    Protected features are never removed.

    Selection methods
    -----------------
    'random'             – uniform random sample
    'lowest_abundance'   – k features with lowest mean abundance
    'highest_abundance'  – k features with highest mean abundance
    'lowest_prevalence'  – k features present in fewest samples
    'highest_prevalence' – k features present in most samples
    """

    VALID_METHODS = {
        'random', 'lowest_abundance', 'highest_abundance',
        'lowest_prevalence', 'highest_prevalence',
    }

    def apply(
        self,
        X: pd.DataFrame,
        k: Optional[int] = None,
        features_to_remove: Optional[List[str]] = None,
        selection_method: str = 'random',
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        X : pd.DataFrame  – compositional data (rows sum to 1)
        k : int           – number of features to remove
        features_to_remove : list of str, optional
            Explicit list of features to remove (overrides k / selection_method)
        selection_method : str
        seed : int, optional
        verbose : bool
        """
        X = X.copy()
        modifiable = self._modifiable(X)

        if verbose and self.protected_features:
            print(f"  Protected features : {len(self.protected_features)}")
            print(f"  Modifiable features: {len(modifiable)}")

        # Explicit list provided
        if features_to_remove is not None:
            bad = set(features_to_remove) & set(self.protected_features)
            if bad:
                raise ValueError(f"Cannot remove protected features: {bad}")
            return self._remove_and_renormalize(X, list(features_to_remove), verbose)

        # Validate k
        if k is None:
            raise ValueError("Provide either 'k' or 'features_to_remove'.")
        if k > len(modifiable):
            raise ValueError(
                f"k={k} exceeds modifiable feature count ({len(modifiable)}); "
                f"{len(self.protected_features)} features are protected."
            )
        if selection_method not in self.VALID_METHODS:
            raise ValueError(
                f"Unknown selection_method '{selection_method}'. "
                f"Choose from {self.VALID_METHODS}."
            )

        rng = np.random.default_rng(seed)

        X_mod = X[modifiable].astype(float)

        if selection_method == 'random':
            to_remove = rng.choice(modifiable, size=k, replace=False).tolist()
        elif selection_method == 'lowest_abundance':
            to_remove = X_mod.mean().nsmallest(k).index.tolist()
        elif selection_method == 'highest_abundance':
            to_remove = X_mod.mean().nlargest(k).index.tolist()
        elif selection_method == 'lowest_prevalence':
            to_remove = (X_mod > 0).sum().nsmallest(k).index.tolist()
        elif selection_method == 'highest_prevalence':
            to_remove = (X_mod > 0).sum().nlargest(k).index.tolist()

        return self._remove_and_renormalize(X, to_remove, verbose)

    def _remove_and_renormalize(
        self,
        X: pd.DataFrame,
        features_to_remove: List[str],
        verbose: bool,
    ) -> pd.DataFrame:
        keep = [c for c in X.columns if c not in features_to_remove]
        if verbose:
            print(f"  Removing {len(features_to_remove)} features → {len(keep)} remaining.")
            if self.protected_features:
                n_protected_kept = len(set(keep) & set(self.protected_features))
                print(f"  Protected features kept: {n_protected_kept}/{len(self.protected_features)}")
        return self._renormalize(X[keep], verbose)


import numpy as np
import pandas as pd
from typing import Optional, Tuple


class SparsityPerturbation(Perturbation):
    """
    Adjust sparsity by adding or filling an exact number of zeros.
    Protected features are never modified.
    """

    # ------------------------------------------------------------------
    # APPLY: main entry point
    # ------------------------------------------------------------------
    def apply(
        self,
        X: pd.DataFrame,
        target_sparsity: float = 0.5,
        noise_range: Tuple[float, float] = (1e-6, 1e-4),
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:

        X = X.copy().astype(float)
        current_sparsity = (X == 0).sum().sum() / X.size
        total_elements = X.size

        current_zeros = int(current_sparsity * total_elements)
        target_zeros = int(target_sparsity * total_elements)
        delta = target_zeros - current_zeros

        if verbose:
            print(f"  Sparsity: {current_sparsity:.3f} → target {target_sparsity:.3f} (Δ {delta:+d} zeros)")

        modifiable = self._modifiable(X)  # uninformative features per your definition

        # Count modifiable entries
        mod_nonzeros = (X[modifiable] > 0).sum().sum()
        mod_zeros = (X[modifiable] == 0).sum().sum()

        # ===============================================================
        # CASE 0: No change
        # ===============================================================
        if delta == 0:
            if verbose:
                print("  Already at target sparsity.")
            return X

        # ===============================================================
        # CASE 1: Need MORE zeros → ADD zeros → sparsify modifiable features
        # ===============================================================
        if delta > 0:
            max_addable = mod_nonzeros
            if delta > max_addable:
                if verbose:
                    print(f"  [WARN] Cannot add {delta} zeros; clipping to {max_addable}.")
                delta = max_addable

            X = self._add_exact_zeros(X, delta, seed=seed, verbose=verbose)

        # ===============================================================
        # CASE 2: Need FEWER zeros → REMOVE zeros → densify modifiable features
        # NOTE: This case should never occur in our benchmark since target_sparsity
        # is always > current_sparsity for SparsityPerturbation.
        # Densification is handled by DensificationPerturbation instead.
        # ===============================================================
        else:
            if verbose:
                print("  [WARN] target_sparsity < current_sparsity — use DensificationPerturbation instead. Returning X unchanged.")
            return X

        # need_to_fill = -delta
        # max_fillable = mod_zeros
        # if need_to_fill > max_fillable:
        #     if verbose:
        #         print(f"  [WARN] Cannot fill {need_to_fill} zeros; clipping to {max_fillable}.")
        #     need_to_fill = max_fillable
        # X = self._fill_exact_zeros(
        #     X, need_to_fill,
        #     noise_range=noise_range,
        #     seed=seed,
        #     verbose=verbose,
        # )

        # ===============================================================
        # Final report
        # ===============================================================
        final_sparsity = (X == 0).sum().sum() / X.size
        if verbose:
            print(f"  Final sparsity: {final_sparsity:.4f}")

        return X

    # ------------------------------------------------------------------
    # ADD ZEROS by power transform (binary search)
    # ------------------------------------------------------------------
    def _add_exact_zeros(
        self,
        X: pd.DataFrame,
        num_zeros: int,
        threshold: float = 1e-6,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:

        modifiable = self._modifiable(X)
        non_zero_count = (X[modifiable] > 0).sum().sum()
        current_zeros = (X[modifiable] == 0).sum().sum()

        # ---- Clipping instead of failing ----
        if non_zero_count < num_zeros:
            if verbose:
                print(f"[WARN] Cannot add {num_zeros} zeros; clipping to {non_zero_count}.")
            num_zeros = non_zero_count

        gamma_min, gamma_max = 0.0, 10.0
        best_gamma, best_X, best_diff = None, None, float('inf')

        for iteration in range(50):
            gamma = (gamma_min + gamma_max) / 2

            X_t = X.copy()
            X_t[modifiable] = X[modifiable] ** (1 + gamma)
            X_t.loc[:, modifiable] = X_t[modifiable].where(
                X_t[modifiable] >= threshold, 0.0
            )

            new_zeros = (X_t[modifiable] == 0).sum().sum() - current_zeros
            diff = abs(new_zeros - num_zeros)

            # Track best solution
            if diff < best_diff:
                best_diff, best_gamma, best_X = diff, gamma, X_t.copy()

            if diff <= 1:
                if verbose:
                    print(f"  Power transform γ={gamma:.4f} → {new_zeros} new zeros (target {num_zeros})")
                break

            if new_zeros < num_zeros:
                gamma_min = gamma
            else:
                gamma_max = gamma

        else:
            if verbose:
                actual = (best_X[modifiable] == 0).sum().sum() - current_zeros
                print(f"  Max iterations reached. γ={best_gamma:.4f} → {actual} new zeros (target {num_zeros})")
            X_t = best_X

        return self._renormalize(X_t, verbose)

    # ------------------------------------------------------------------
    # FILL ZEROS with random noise
    # ------------------------------------------------------------------
    def _fill_exact_zeros(
        self,
        X: pd.DataFrame,
        num_zeros: int,
        noise_range: Tuple[float, float] = (1e-6, 1e-4),
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:

        modifiable = self._modifiable(X)
        rng = np.random.default_rng(seed)

        zero_mask = X[modifiable] == 0
        zero_positions = np.argwhere(zero_mask.values)

        # ---- Clipping instead of failing ----
        if len(zero_positions) < num_zeros:
            if verbose:
                print(f"[WARN] Cannot fill {num_zeros} zeros; clipping to {len(zero_positions)}.")
            num_zeros = len(zero_positions)

        chosen = rng.choice(len(zero_positions), size=num_zeros, replace=False)
        selected = zero_positions[chosen]
        noise = rng.uniform(noise_range[0], noise_range[1], size=num_zeros)

        X = X.copy()
        mod_cols = list(modifiable)

        for idx, (row_i, col_i) in enumerate(selected):
            col_name = mod_cols[col_i]
            X.at[X.index[row_i], col_name] = noise[idx]

        if verbose:
            print(f"  Filled {num_zeros} zeros with noise in [{noise_range[0]:.1e}, {noise_range[1]:.1e}]")

        return self._renormalize(X, verbose)

# =============================================================================
# STATISTICS MODULE
# =============================================================================

# =============================================================================
# DENSIFICATION PERTURBATION
# =============================================================================

class DensificationPerturbation(Perturbation):
    """
    Reduce sparsity by filling zeros in non-protected features with real values
    sampled from the same non-protected features (cross-sample).

    Biologically motivated: simulates species already present in the dataset
    but undetected in certain samples due to sequencing depth limitations.
    The filled values are drawn from the empirical distribution of non-zero
    abundances of non-protected features, preserving realistic abundance ranges.

    Parameters
    ----------
    target_sparsity : float
        Target sparsity LOWER than current. If >= current, returns X unchanged.
    seed : int, optional
    verbose : bool
    """

    def apply(
        self,
        X: pd.DataFrame,
        target_sparsity: float = 0.5,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:

        X = X.copy().astype(float)
        modifiable = self._modifiable(X)
        rng = np.random.default_rng(seed)

        current_sparsity = float((X == 0).sum().sum() / X.size)
        total_elements = X.size
        current_zeros = int(current_sparsity * total_elements)
        target_zeros = int(target_sparsity * total_elements)
        need_to_fill = current_zeros - target_zeros

        if verbose:
            print(f"  Densification: sparsity {current_sparsity:.3f} → target {target_sparsity:.3f} (fill {need_to_fill} zeros)")

        if need_to_fill <= 0:
            if verbose:
                print("  target_sparsity >= current_sparsity — no densification needed.")
            return X

        # Positions of zeros in modifiable features
        mod_cols = list(modifiable)
        zero_mask = X[modifiable] == 0
        zero_positions = np.argwhere(zero_mask.values)

        if len(zero_positions) < need_to_fill:
            if verbose:
                print(f"  [WARN] Only {len(zero_positions)} zeros available; clipping to that.")
            need_to_fill = len(zero_positions)

        # OLD APPROACH (global pool — sampling from all non-zero values across all features):
        # nonzero_vals = X[modifiable].values.flatten()
        # nonzero_vals = nonzero_vals[nonzero_vals > 0]
        # chosen_pos = rng.choice(len(zero_positions), size=need_to_fill, replace=False)
        # fill_values = rng.choice(nonzero_vals, size=need_to_fill, replace=True)
        # for idx, pos_idx in enumerate(chosen_pos):
        #     row_i, col_i = zero_positions[pos_idx]
        #     col_name = mod_cols[col_i]
        #     X.at[X.index[row_i], col_name] = fill_values[idx]

        # Beginning NEW APPROACH — per-feature distribution:
        # For each modifiable feature, collect its non-zero values
        feature_nonzero = {}
        for col in mod_cols:
            vals = X[col].values
            nz = vals[vals > 0]
            if len(nz) > 0:
                feature_nonzero[col] = nz

        ##### End new approach

        if len(feature_nonzero) == 0:
            if verbose:
                print("  [WARN] No non-zero values in non-protected features — cannot densify.")
            return X

        # Sample positions to fill
        chosen_pos = rng.choice(len(zero_positions), size=need_to_fill, replace=False)

        for pos_idx in chosen_pos:
            row_i, col_i = zero_positions[pos_idx]
            col_name = mod_cols[col_i]

            # Sample from the feature-specific distribution
            # If this feature has no non-zero values, fall back to a random feature's distribution
            if col_name in feature_nonzero:
                pool = feature_nonzero[col_name]
            else:
                fallback = rng.choice(list(feature_nonzero.keys()))
                pool = feature_nonzero[fallback]

            fill_val = rng.choice(pool)
            X.at[X.index[row_i], col_name] = fill_val

        X = self._renormalize(X, verbose)
        final_sparsity = float((X == 0).sum().sum() / X.size)
        if verbose:
            print(f"  Final sparsity: {final_sparsity:.4f}")

        return X



class PerturbationStats:
    """
    Computes descriptive statistics for one or more (label, DataFrame) pairs.
    """

    VALID_METRICS = {'sparsity', 'n_features', 'mean_abundance', 'median_abundance', 'diversity'}

    def __init__(self, metrics: Optional[List[str]] = None):
        metrics = metrics or ['sparsity', 'n_features', 'mean_abundance', 'diversity']
        unknown = set(metrics) - self.VALID_METRICS
        if unknown:
            raise ValueError(f"Unknown metrics: {unknown}. Choose from {self.VALID_METRICS}.")
        self.metrics = metrics

    def compute(self, X: pd.DataFrame, label: str, extra: Optional[dict] = None) -> dict:
        """Return a stats dict for a single dataset."""
        from scipy.stats import entropy

        row = {'label': label}
        if extra:
            row.update(extra)

        if 'sparsity' in self.metrics:
            row['sparsity'] = (X == 0).sum().sum() / X.size

        if 'n_features' in self.metrics:
            # Number of features with at least one non-zero value across samples
            row['n_features'] = (X > 0).any(axis=0).sum()

        if 'mean_abundance' in self.metrics:
            row['mean_abundance'] = X.mean().mean()

        if 'median_abundance' in self.metrics:
            row['median_abundance'] = X.median().median()

        if 'diversity' in self.metrics:
            diversities = []
            for _, row_ in X.iterrows():
                # Ensure numeric dtype and filter positives
                row_vals = pd.to_numeric(row_, errors='coerce').fillna(0.0).values
                row_nonzero = row_vals[row_vals > 0]
                if len(row_nonzero) > 0:
                    diversities.append(entropy(row_nonzero))
            row['diversity'] = float(np.mean(diversities)) if diversities else 0.0

        return row

    def compute_all(
        self,
        datasets: List[Tuple[str, pd.DataFrame, Optional[dict]]],
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        datasets : list of (label, X, extra_params_dict)
        """
        return pd.DataFrame([self.compute(X, label, extra) for label, X, extra in datasets])