# %%
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, List, Union

sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/')

from testing.testing_data.pasolli.pasolli import open_pasolli
from testing.testing_data.metacardis.metacardis import open_metacardis
from testing.testing_data.preprocessing.filter_or_logic import open_and_filter


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


class AddRandomFeaturesPerturbation(Perturbation):
    """
    Add k synthetic features drawn from a log-normal distribution and renormalize.
    """

    def apply(
        self,
        X: pd.DataFrame,
        k: Optional[int] = None,
        min_abundance: float = 1e-4,
        max_abundance: float = 1e-3,
        feature_prefix: str = 'random_feature',
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        X : pd.DataFrame  – compositional data
        k : int           – number of synthetic features to add
        min_abundance, max_abundance : float
            95 % of samples will fall within this range (log-normal parametrisation).
        feature_prefix : str
        seed : int, optional
        verbose : bool
        """
        if k is None:
            raise ValueError("'k' (number of features to add) must be provided.")
        if min_abundance >= max_abundance:
            raise ValueError("min_abundance must be strictly less than max_abundance.")

        rng = np.random.default_rng(seed)
        X = X.copy()

        log_min, log_max = np.log(min_abundance), np.log(max_abundance)
        log_mean = (log_min + log_max) / 2
        log_std = (log_max - log_min) / 4  # ±2σ covers the range

        random_abundances = rng.lognormal(
            mean=log_mean, sigma=log_std, size=(X.shape[0], k)
        )
        new_cols = [f"{feature_prefix}_{i + 1}" for i in range(k)]
        new_df = pd.DataFrame(random_abundances, index=X.index, columns=new_cols)

        X_aug = pd.concat([X, new_df], axis=1)
        X_aug = self._renormalize(X_aug, verbose)

        if verbose:
            print(f"  Added {k} random features. Shape: {X.shape} → {X_aug.shape}")
            in_range = (
                (random_abundances >= min_abundance) & (random_abundances <= max_abundance)
            ).mean() * 100
            print(f"  Values in target range [{min_abundance:.1e}, {max_abundance:.1e}]: {in_range:.1f}%")

        return X_aug


class SparsityPerturbation(Perturbation):
    """
    Adjust sparsity by adding or filling an exact number of zeros.
    Protected features are never modified.
    """

    def apply(
        self,
        X: pd.DataFrame,
        target_sparsity: float = 0.5,
        noise_range: Tuple[float, float] = (1e-6, 1e-4),
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        X : pd.DataFrame     – compositional data
        target_sparsity : float   – desired fraction of zeros (0–1)
        noise_range : (float, float)  – range for fill noise when densifying
        seed : int, optional
        verbose : bool
        """
        X = X.copy().astype(float)
        current_sparsity = (X == 0).sum().sum() / X.size
        total_elements = X.size

        current_zeros = int(current_sparsity * total_elements)
        target_zeros = int(target_sparsity * total_elements)
        delta = target_zeros - current_zeros

        if verbose:
            print(f"  Sparsity: {current_sparsity:.3f} → target {target_sparsity:.3f} (Δ {delta:+d} zeros)")

        if delta == 0:
            if verbose:
                print("  Already at target sparsity.")
            return X
        elif delta > 0:
            X = self._add_exact_zeros(X, delta, seed=seed, verbose=verbose)
        else:
            X = self._fill_exact_zeros(X, -delta, noise_range=noise_range, seed=seed, verbose=verbose)

        final_sparsity = (X == 0).sum().sum() / X.size
        if verbose:
            print(f"  Final sparsity: {final_sparsity:.4f}")

        return X

    def _add_exact_zeros(
        self,
        X: pd.DataFrame,
        num_zeros: int,
        threshold: float = 1e-6,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Binary-search for a power-transform gamma that creates exactly `num_zeros` new zeros."""
        modifiable = self._modifiable(X)
        non_zero_count = (X[modifiable] > 0).sum().sum()
        current_zeros = (X[modifiable] == 0).sum().sum()

        if non_zero_count < num_zeros:
            raise ValueError(
                f"Cannot add {num_zeros} zeros; only {non_zero_count} non-zero values "
                "available in modifiable features."
            )

        gamma_min, gamma_max = 0.0, 10.0
        best_gamma, best_X, best_diff = None, None, float('inf')

        for iteration in range(50):
            gamma = (gamma_min + gamma_max) / 2

            X_t = X.copy()
            X_t[modifiable] = X[modifiable] ** (1 + gamma)
            X_t.loc[:, modifiable] = X_t[modifiable].where(X_t[modifiable] >= threshold, 0.0)

            new_zeros = (X_t[modifiable] == 0).sum().sum() - current_zeros
            diff = abs(new_zeros - num_zeros)

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

    def _fill_exact_zeros(
        self,
        X: pd.DataFrame,
        num_zeros: int,
        noise_range: Tuple[float, float] = (1e-6, 1e-4),
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Fill exactly `num_zeros` zeros in modifiable features with small random noise."""
        modifiable = self._modifiable(X)
        rng = np.random.default_rng(seed)  # Respects seed; does not reset global state

        # Only consider zeros in modifiable columns
        zero_mask = X[modifiable] == 0
        zero_positions = np.argwhere(zero_mask.values)  # (row_idx, col_idx) within modifiable slice

        if len(zero_positions) < num_zeros:
            raise ValueError(
                f"Cannot fill {num_zeros} zeros; only {len(zero_positions)} "
                "available in modifiable features."
            )

        chosen = rng.choice(len(zero_positions), size=num_zeros, replace=False)
        selected = zero_positions[chosen]
        noise = rng.uniform(noise_range[0], noise_range[1], size=num_zeros)

        X = X.copy()
        mod_col_positions = {col: i for i, col in enumerate(modifiable)}
        for idx, (row_i, col_i) in enumerate(selected):
            col_name = modifiable[col_i]
            X.at[X.index[row_i], col_name] = noise[idx]

        if verbose:
            print(f"  Filled {num_zeros} zeros with noise in [{noise_range[0]:.1e}, {noise_range[1]:.1e}]")

        return self._renormalize(X, verbose)


# =============================================================================
# STATISTICS MODULE
# =============================================================================

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


# =============================================================================
# VISUALIZATION MODULE
# =============================================================================

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, List, Union

sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/')

from testing.testing_data.pasolli.pasolli import open_pasolli
from testing.testing_data.metacardis.metacardis import open_metacardis
from testing.testing_data.preprocessing.filter_or_logic import open_and_filter

# =============================================================================
# VISUALIZATION MODULE
# =============================================================================

class PerturbationVisualizer:
    """
    Three scatter plots for perturbation analysis.
    Each point is always one (sample, feature) pair → n_samples × n_features points total.
    Only shared features between original and perturbed are plotted (missing ≠ zero).
    """

    @staticmethod
    def _paired_vals(
        original: pd.DataFrame,
        X_pert: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Return (x_vals, y_vals, shared_cols) aligned by sorted shared columns."""
        shared_cols = sorted(original.columns.intersection(X_pert.columns))
        x_vals = original[shared_cols].values.flatten().astype(float)
        y_vals = X_pert[shared_cols].values.flatten().astype(float)
        return x_vals, y_vals, shared_cols

    # ------------------------------------------------------------------
    # Plot 1: one subplot per perturbation
    # ------------------------------------------------------------------
    def plot_per_perturbation(
        self,
        original: pd.DataFrame,
        perturbations: List[Tuple[str, pd.DataFrame]],
        subplot_size: Tuple[int, int] = (5, 5),
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """One subplot per perturbation. x = original value, y = perturbed value.
        Total points per panel = n_samples × n_shared_features."""
        n = len(perturbations)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(subplot_size[0] * ncols, subplot_size[1] * nrows),
            squeeze=False,
        )
        palette = sns.color_palette("husl", n)

        for idx, (label, X_pert) in enumerate(perturbations):
            ax = axes[idx // ncols][idx % ncols]
            x_vals, y_vals, shared_cols = self._paired_vals(original, X_pert)
            n_pts = len(x_vals)  # n_samples × n_shared_features

            ax.scatter(x_vals, y_vals, alpha=0.3, s=8,
                       color=palette[idx], edgecolors='none',
                       label=f"n_pts={n_pts} (s={X_pert.shape[0]}, f={X_pert.shape[1]})")
            lim_max = max(x_vals.max(), y_vals.max())
            ax.plot([0, lim_max], [0, lim_max], color='black',
                    linewidth=1, linestyle='--', label='y = x')
            ax.set_xlabel("Original value", fontsize=9)
            ax.set_ylabel("Perturbed value", fontsize=9)
            ax.set_title(label, fontsize=9, fontweight='bold')
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(fontsize=7, frameon=True)

        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(title or "Original vs. Perturbations (per perturbation)",
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    # ------------------------------------------------------------------
    # Plot 2: all perturbations overlaid, colour = perturbation level
    # ------------------------------------------------------------------
    def plot_overlay(
        self,
        original: pd.DataFrame,
        perturbations: List[Tuple[str, pd.DataFrame]],
        figsize: Tuple[int, int] = (8, 7),
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """All perturbations overlaid. Colour = perturbation level.
        Total points per layer = n_samples × n_shared_features."""
        fig, ax = plt.subplots(figsize=figsize)
        palette = sns.color_palette("husl", len(perturbations))

        for idx, (label, X_pert) in enumerate(perturbations):
            x_vals, y_vals, _ = self._paired_vals(original, X_pert)
            n_pts = len(x_vals)
            ax.scatter(x_vals, y_vals, alpha=0.3, s=8,
                       color=palette[idx], edgecolors='none',
                       label=f"{label} — {n_pts} pts (f={X_pert.shape[1]})")

        all_max = max(
            original.values.astype(float).max(),
            max(X.values.astype(float).max() for _, X in perturbations),
        )
        ax.plot([0, all_max], [0, all_max], color='black',
                linewidth=1.2, linestyle='--', label='y = x (no change)')
        ax.set_xlabel("Original value", fontsize=11)
        ax.set_ylabel("Perturbed value", fontsize=11)
        ax.set_title(title or "Original vs. Perturbations (overlay)",
                     fontsize=12, fontweight='bold')
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8, frameon=True, markerscale=2)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    # ------------------------------------------------------------------
    # Plot 3: colour = sample class, black ring = protected feature
    # ------------------------------------------------------------------
    def plot_class_and_protected(
        self,
        original: pd.DataFrame,
        perturbations: List[Tuple[str, pd.DataFrame]],
        y_labels: pd.Series,
        protected_features: List[str],
        subplot_size: Tuple[int, int] = (6, 6),
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """One subplot per perturbation.
        Colour = sample class label (repeated per feature).
        Black ring = protected feature."""
        n = len(perturbations)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(subplot_size[0] * ncols, subplot_size[1] * nrows),
            squeeze=False,
        )

        classes = sorted(y_labels.unique())
        class_palette = dict(zip(classes, sns.color_palette("Set2", len(classes))))

        for idx, (label, X_pert) in enumerate(perturbations):
            ax = axes[idx // ncols][idx % ncols]
            shared_cols = sorted(original.columns.intersection(X_pert.columns))
            n_samples = original.shape[0]

            x_vals = original[shared_cols].values.flatten().astype(float)
            y_vals = X_pert[shared_cols].values.flatten().astype(float)

            # Each (sample, feature) pair gets the colour of that sample's class
            colours = np.array([class_palette[c] for c in y_labels for _ in shared_cols])
            # Protected mask: True where the feature is protected
            is_protected = np.tile([f in protected_features for f in shared_cols], n_samples)

            # Non-protected behind
            ax.scatter(x_vals[~is_protected], y_vals[~is_protected],
                       c=colours[~is_protected], alpha=0.2, s=8, edgecolors='none')
            # Protected on top with black ring
            ax.scatter(x_vals[is_protected], y_vals[is_protected],
                       c=colours[is_protected], alpha=0.7, s=25,
                       edgecolors='black', linewidths=0.6)

            lim_max = max(x_vals.max(), y_vals.max())
            ax.plot([0, lim_max], [0, lim_max], color='black',
                    linewidth=1, linestyle='--')

            if idx == 0:
                for cls, col in class_palette.items():
                    ax.scatter([], [], color=col, label=str(cls), s=20)
                ax.scatter([], [], edgecolors='black', facecolors='grey',
                           s=25, linewidths=0.6, label='protected feature')
                ax.legend(fontsize=7, frameon=True, title='Class')

            ax.set_xlabel("Original value", fontsize=9)
            ax.set_ylabel("Perturbed value", fontsize=9)
            ax.set_title(label, fontsize=9, fontweight='bold')
            ax.grid(True, linestyle="--", alpha=0.3)

        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(title or "Original vs. Perturbations (by class & protected features)",
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    # ------------------------------------------------------------------
    # Plot 4: distribution of (perturbed - original) split by class
    # ------------------------------------------------------------------
    def plot_delta_by_class(
        self,
        original: pd.DataFrame,
        perturbations: List[Tuple[str, pd.DataFrame]],
        y_labels: pd.Series,
        figsize: Tuple[int, int] = (12, 5),
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """
        Boxplot of (perturbed - original) per (perturbation level x class).
        Each observation is one (sample, feature) delta value.
        If boxes overlap -> perturbation is class-agnostic.
        If boxes separate -> perturbation introduces class bias.
        """
        classes = sorted(y_labels.unique())
        class_palette = dict(zip(classes, sns.color_palette("Set2", len(classes))))

        # Build long-form: one row per (sample, feature, perturbation)
        records = []
        for label, X_pert in perturbations:
            shared_cols = sorted(original.columns.intersection(X_pert.columns))
            for sample_idx, cls in zip(original.index, y_labels):
                orig_row = original.loc[sample_idx, shared_cols].values.astype(float)
                pert_row = X_pert.loc[sample_idx, shared_cols].values.astype(float)
                # Exclude only (0, 0) pairs — keep if non-zero in either original or perturbed
                nonzero_mask = (orig_row != 0) | (pert_row != 0)
                for delta in (pert_row[nonzero_mask] - orig_row[nonzero_mask]):
                    records.append({'perturbation': label, 'class': str(cls), 'delta': delta})

        df_long = pd.DataFrame(records)
        pert_labels = [label for label, _ in perturbations]
        n_classes = len(classes)
        width = 0.8 / n_classes
        offsets = np.linspace(-(0.8 - width) / 2, (0.8 - width) / 2, n_classes)

        fig, ax = plt.subplots(figsize=figsize)

        for cls_idx, cls in enumerate(classes):
            df_cls = df_long[df_long['class'] == str(cls)]
            positions = np.arange(len(pert_labels)) + offsets[cls_idx]
            data_per_pert = [
                df_cls[df_cls['perturbation'] == lbl]['delta'].values
                for lbl in pert_labels
            ]
            ax.boxplot(
                data_per_pert,
                positions=positions,
                widths=width * 0.9,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color='black', linewidth=1.5),
                boxprops=dict(facecolor=class_palette[cls], alpha=0.7),
                whiskerprops=dict(linewidth=1),
                capprops=dict(linewidth=1),
            )
            ax.scatter([], [], color=class_palette[cls], label=f'Class {cls}', s=40)

        ax.axhline(0, color='black', linewidth=1.2, linestyle='--', label='no change')
        ax.set_xticks(np.arange(len(pert_labels)))
        ax.set_xticklabels(pert_labels, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel("Perturbed - Original (excl. double-zero pairs)", fontsize=11)
        ax.set_xlabel("Perturbation level", fontsize=11)
        ax.set_title(title or "Delta abundance by class", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.3, axis='y')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


    # ------------------------------------------------------------------
    # Plot 5: per-class feature trajectory across perturbation levels
    # ------------------------------------------------------------------
    def plot_feature_trajectories(
        self,
        original: pd.DataFrame,
        perturbations: List[Tuple[str, pd.DataFrame]],
        y_labels: pd.Series,
        protected_features: List[str],
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """
        One subplot per class.
        X-axis = perturbation level (k=0, k=10, ...).
        Y-axis = mean abundance of each feature for samples of that class.
        Each dot = one feature at one k level, placed inside the boxplot with jitter.
        Dots of the same feature are connected by a line across k levels.
        Protected features in red, others in grey.
        Y-axis capped at 0.02 to show low-abundance features clearly.
        """
        classes = sorted(y_labels.unique())
        all_datasets = [("k=0", original)] + [(label, X) for label, X in perturbations]
        shared_cols = sorted(set.intersection(*[set(X.columns) for _, X in all_datasets]))
        n_k = len(all_datasets)

        fig, axes = plt.subplots(1, len(classes), figsize=figsize, sharey=True)
        if len(classes) == 1:
            axes = [axes]

        x_positions = np.arange(n_k)
        x_labels = [label for label, _ in all_datasets]
        rng = np.random.default_rng(0)

        for ax, cls in zip(axes, classes):
            sample_mask = y_labels == cls

            # mean abundance per feature per k level
            feat_matrix = np.array([
                [X.loc[sample_mask, f].values.astype(float).mean()
                 for _, X in all_datasets]
                for f in shared_cols
            ])  # shape: (n_features, n_k)

            # --- Boxplot at each k ---
            ax.boxplot(
                [feat_matrix[:, k] for k in range(n_k)],
                positions=x_positions,
                widths=0.5,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color='black', linewidth=2, zorder=5),
                boxprops=dict(facecolor='#d5e8f7', alpha=0.5, zorder=1),
                whiskerprops=dict(linewidth=1, zorder=1),
                capprops=dict(linewidth=1, zorder=1),
            )

            # --- Lines + jittered dots per feature ---
            for f_idx, feat in enumerate(shared_cols):
                is_protected = feat in protected_features
                colour = '#c0392b' if is_protected else '#555555'
                alpha  = 0.9 if is_protected else 0.6
                lw     = 1.8 if is_protected else 0.7
                ms     = 25  if is_protected else 14
                zorder = 4   if is_protected else 3

                y_vals = feat_matrix[f_idx]
                jitter = rng.uniform(-0.18, 0.18, size=n_k)

                # line on exact x positions
                ax.plot(x_positions, y_vals, color=colour,
                        alpha=alpha * 0.5, linewidth=lw, zorder=zorder)
                # dots with jitter inside box
                ax.scatter(x_positions + jitter, y_vals, color=colour,
                           alpha=alpha, s=ms, zorder=zorder + 1, edgecolors='none')

            ax.set_yscale('symlog', linthresh=1e-4)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=25, ha='right', fontsize=7)
            ax.set_title(f"Class {cls}", fontsize=11, fontweight='bold')
            ax.set_xlabel("Perturbation level", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.3, axis='y')

            if ax == axes[0]:
                ax.set_ylabel("Mean feature abundance (symlog scale)", fontsize=10)
                ax.scatter([], [], color='#c0392b', s=25, label='protected feature')
                ax.scatter([], [], color='#555555', s=14, alpha=0.7, label='other feature')
                ax.legend(fontsize=8, frameon=True)

        fig.suptitle(title or "Feature trajectories across perturbation levels by class",
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

# =============================================================================
# DATA LOADER
# =============================================================================

class DataLoader:
    """
    Thin wrapper around dataset-specific loading functions.
    """

    PASOLLI_DATASETS = [
        'abundance_cirrhosis--stagediscovery',
        'abundance_cirrhosis--stagevalidation',
        'abundance_obesity',
        'abundance_ibd',
        'abundance_t2d',
        'abundance_WT2D',
    ]

    def load(
        self,
        dataset_name: str,
        data_source: str = 'pasolli',
        filter_params: Tuple[float, float] = (0.0, 0.0),
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and filter a dataset.

        Parameters
        ----------
        dataset_name : str
        data_source : 'pasolli' or 'metacardis'
        filter_params : (float, float)

        Returns
        -------
        X : pd.DataFrame
        y : pd.Series
        """
        if data_source == 'pasolli':
            X, y = open_pasolli(dataset_name)
        elif data_source == 'metacardis':
            X, y = open_metacardis(dataset_name)
        else:
            raise ValueError(f"Unknown data_source '{data_source}'.")

        # open_and_filter signature: open_and_filter(X, p1, p2) -> (X, y)
        # It derives y internally from X, so we do not pass y in.
        X, y = open_and_filter(X, *filter_params)
        return X, y


# =============================================================================
# ORCHESTRATOR (DataGenerator)
# =============================================================================

class DataGenerator:
    """
    Orchestrates loading, feature selection, perturbation, and visualisation
    of microbiome compositional data.

    Modules
    -------
    loader      : DataLoader
    selector    : FeatureSelector  (set after load_data)
    perturbation: one of RemoveFeaturesPerturbation |
                          AddRandomFeaturesPerturbation |
                          SparsityPerturbation
    stats       : PerturbationStats
    visualizer  : PerturbationVisualizer

    Quick start
    -----------
    gen = DataGenerator(generator_type='remove_features', data_source='pasolli')
    X, y = gen.load_data('abundance_ibd')
    gen.discover_and_protect(method='random_forest', n_features=20)
    gen.visualize_perturbations(
        perturbation_params=[{'k': 10}, {'k': 50}, {'k': 100}],
        method='pca',
    )
    """

    _GENERATOR_MAP = {
        'remove_features':     RemoveFeaturesPerturbation,
        'add_random_features': AddRandomFeaturesPerturbation,
        'sparsity':            SparsityPerturbation,
    }

    def __init__(
        self,
        generator_type: str = 'sparsity',
        data_source: str = 'pasolli',
        dataset_name: Optional[str] = None,
        filter_params: Optional[Tuple[float, float]] = None,
        protected_features: Optional[List[str]] = None,
    ):
        if generator_type not in self._GENERATOR_MAP and generator_type != 'identity':
            raise ValueError(
                f"Unknown generator_type '{generator_type}'. "
                f"Choose from {list(self._GENERATOR_MAP.keys()) + ['identity']}."
            )

        self.generator_type = generator_type
        self.data_source = data_source
        self.dataset_name = dataset_name
        self.filter_params = filter_params or (0.0, 0.0)
        self.protected_features: List[str] = protected_features or []

        # Populated after load_data
        self.X_original: Optional[pd.DataFrame] = None
        self.y_original: Optional[pd.Series] = None

        # Sub-modules
        self.loader = DataLoader()
        self.selector: Optional[FeatureSelector] = None
        self.stats_module = PerturbationStats()
        self.visualizer = PerturbationVisualizer()

    # ------------------------------------------------------------------
    # Internal helper: current perturbation object (respects protected_features)
    # ------------------------------------------------------------------
    def _make_perturbation(self) -> Perturbation:
        cls = self._GENERATOR_MAP.get(self.generator_type)
        if cls is None:
            raise ValueError(f"No perturbation class for '{self.generator_type}'.")
        return cls(protected_features=self.protected_features)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_data(
        self,
        dataset_name: Optional[str] = None,
        filter_params: Optional[Tuple[float, float]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and store the dataset. Returns (X, y)."""
        dataset_name = dataset_name or self.dataset_name
        filter_params = filter_params or self.filter_params

        if dataset_name is None:
            raise ValueError("Provide dataset_name either here or at __init__.")

        X, y = self.loader.load(dataset_name, self.data_source, filter_params)
        X = X.astype(float)
        X = X.div(X.sum(axis=1), axis=0)  # renormalise rows to sum to 1
        self.X_original = X
        self.y_original = y
        self.selector = FeatureSelector(X, y)

        print(f"Loaded '{dataset_name}': {X.shape[0]} samples, {X.shape[1]} features (rows renormalised to sum to 1).")
        return X, y

    # ------------------------------------------------------------------
    # Feature discovery / protection
    # ------------------------------------------------------------------
    def discover_informative_features(
        self,
        method: str = 'random_forest',
        n_features: Optional[int] = None,
        threshold: Optional[float] = None,
        return_scores: bool = False,
        verbose: bool = True,
        **method_kwargs,
    ) -> Union[List[str], Tuple[List[str], pd.Series]]:
        """Delegate to FeatureSelector.select()."""
        self._require_data()
        return self.selector.select(
            method=method,
            n_features=n_features,
            threshold=threshold,
            return_scores=return_scores,
            verbose=verbose,
            **method_kwargs,
        )

    def discover_and_protect(
        self,
        method: str = 'random_forest',
        n_features: int = 20,
        verbose: bool = True,
        **kwargs,
    ) -> List[str]:
        """Discover top features and mark them as protected from perturbation."""
        self._require_data()
        features = self.selector.select(
            method=method, n_features=n_features, verbose=verbose, **kwargs
        )
        self.protected_features = features
        print(f"\n✅ {len(features)} features marked as PROTECTED (will not be modified).\n")
        return features

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        X: Optional[pd.DataFrame] = None,
        **params,
    ) -> pd.DataFrame:
        """
        Apply the configured perturbation to X (defaults to X_original).

        Parameters forwarded to the perturbation's apply() method.
        """
        self._require_data()
        X = X if X is not None else self.X_original

        if self.generator_type == 'identity':
            return X.copy()

        return self._make_perturbation().apply(X, **params)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def visualize_perturbations(
        self,
        perturbation_params: List[dict],
        subplot_size: Tuple[int, int] = (5, 5),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Produces three plots:
          1. One subplot per perturbation (x=original, y=perturbed)
          2. All perturbations overlaid, colour = perturbation level
          3. One subplot per perturbation, colour = class label,
             black ring = protected feature

        Parameters
        ----------
        perturbation_params : list of dict, each forwarded to generate()
        subplot_size – size of each individual subplot (width, height)
        save_path    – optional path to save the figures
        """
        self._require_data()

        perturbations: List[Tuple[str, pd.DataFrame]] = []
        print(f"Generating {len(perturbation_params)} perturbations...")
        for i, params in enumerate(perturbation_params, 1):
            param_str = ', '.join(f"{k}={v}" for k, v in params.items())
            print(f"  [{i}/{len(perturbation_params)}] {param_str}")
            X_pert = self.generate(**params)
            perturbations.append((f"Pert {i}: {param_str}", X_pert))

        base_title = (
            f'Original vs. Perturbations — {self.generator_type}, '
            f'{len(self.protected_features)} protected features'
        )

        # Plot 1: one subplot per perturbation
        self.visualizer.plot_per_perturbation(
            original=self.X_original,
            perturbations=perturbations,
            subplot_size=subplot_size,
            save_path=save_path,
            title=base_title,
        )

        # Plot 2: all perturbations overlaid
        self.visualizer.plot_overlay(
            original=self.X_original,
            perturbations=perturbations,
            figsize=(8, 7),
            title=base_title + " (overlay)",
        )

        # Plot 3: colour by class, highlight protected features
        self.visualizer.plot_class_and_protected(
            original=self.X_original,
            perturbations=perturbations,
            y_labels=self.y_original,
            protected_features=self.protected_features,
            subplot_size=subplot_size,
            title=base_title + " (by class & protected)",
        )

        # Plot 4: delta distribution by class
        self.visualizer.plot_delta_by_class(
            original=self.X_original,
            perturbations=perturbations,
            y_labels=self.y_original,
            title=base_title + " - Delta by class",
        )

        # Plot 5: per-class feature trajectories across k levels
        self.visualizer.plot_feature_trajectories(
            original=self.X_original,
            perturbations=perturbations,
            y_labels=self.y_original,
            protected_features=self.protected_features,
            title=base_title + " - Feature trajectories by class",
        )

    # ------------------------------------------------------------------
    # Statistics comparison
    # ------------------------------------------------------------------
    def compare_perturbation_statistics(
        self,
        perturbation_params: List[dict],
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 4),
    ) -> pd.DataFrame:
        """
        Compute and plot statistics across perturbation levels.

        Parameters
        ----------
        perturbation_params : list of dict
        metrics : list of str (default: ['sparsity', 'n_features', 'mean_abundance', 'diversity'])
        figsize : (int, int)

        Returns
        -------
        pd.DataFrame with one row per perturbation level (plus the original).
        """
        self._require_data()

        self.stats_module = PerturbationStats(metrics=metrics)
        datasets_for_stats = [('Original', self.X_original, None)]

        print(f"Computing statistics for {len(perturbation_params)} perturbations...")
        for i, params in enumerate(perturbation_params, 1):
            X_pert = self.generate(**params)
            label = f"Pert {i}"
            datasets_for_stats.append((label, X_pert, params))

        stats_df = self.stats_module.compute_all(datasets_for_stats)

        # Plot
        plot_metrics = [m for m in self.stats_module.metrics if m in stats_df.columns]
        fig, axes = plt.subplots(1, len(plot_metrics), figsize=figsize)
        if len(plot_metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, plot_metrics):
            ax.plot(range(len(stats_df)), stats_df[metric], marker='o', linewidth=2, markersize=8)
            ax.set_xticks(range(len(stats_df)))
            ax.set_xticklabels(stats_df['label'], rotation=45, ha='right', fontsize=8)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return stats_df

    # ------------------------------------------------------------------
    # Internal guards
    # ------------------------------------------------------------------
    def _require_data(self) -> None:
        if self.X_original is None or self.y_original is None:
            raise RuntimeError("No data loaded. Call load_data() first.")


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

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