import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Union

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sets.pasolli.pasolli import open_pasolli
from sets.metacardis.metacardis import open_metacardis
from sets.preprocessing.filter_or_logic import open_and_filter

from data_generation.perturbation_core import FeatureSelector, Perturbation, RemoveFeaturesPerturbation, \
    SparsityPerturbation, DensificationPerturbation, PerturbationStats
from data_generation.visualizer import PerturbationVisualizer

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
        'remove_features':   RemoveFeaturesPerturbation,
        'sparsity':          SparsityPerturbation,
        'densification':     DensificationPerturbation,
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
    # Classification performance vs perturbation level
    # ------------------------------------------------------------------
    def evaluate_classifier_performance(
            self,
            perturbation_params: List[dict],
            cv: int = 5,
            random_state: int = 42,
            figsize: Tuple[int, int] = (10, 5),
            save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, roc_auc_score

        self._require_data()

        def _evaluate(X, label):
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
            acc_scores, auc_scores = [], []
            X_vals = X.values if hasattr(X, 'values') else X
            y_vals = self.y_original.values if hasattr(self.y_original, 'values') else self.y_original

            for train_idx, test_idx in skf.split(X_vals, y_vals):
                clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
                clf.fit(X_vals[train_idx], y_vals[train_idx])
                proba = clf.predict_proba(X_vals[test_idx])
                pred = clf.predict(X_vals[test_idx])
                acc_scores.append(accuracy_score(y_vals[test_idx], pred))
                auc_scores.append(roc_auc_score(y_vals[test_idx], proba[:, 1]))

            sparsity = (X_vals == 0).sum() / X_vals.size
            return {
                'label': label,
                'sparsity': round(float(sparsity), 4),
                'accuracy_mean': round(float(np.mean(acc_scores)), 4),
                'accuracy_std': round(float(np.std(acc_scores)), 4),
                'roc_auc_mean': round(float(np.mean(auc_scores)), 4),
                'roc_auc_std': round(float(np.std(auc_scores)), 4),
            }

        rows = [_evaluate(self.X_original, 'Original')]
        print(f"Evaluating classifier on original + {len(perturbation_params)} perturbations...")

        for i, params in enumerate(perturbation_params, 1):
            param_str = ', '.join(f"{k}={v}" for k, v in params.items())
            print(f"  [{i}/{len(perturbation_params)}] {param_str}")
            X_pert = self.generate(**params)
            rows.append(_evaluate(X_pert, f"Pert {i}: {param_str}"))

        results_df = pd.DataFrame(rows)

        # --- Plot ---
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        x = np.arange(len(results_df))

        for ax, metric, colour in zip(
                axes,
                ['accuracy', 'roc_auc'],
                ['#2ecc71', '#3498db'],
        ):
            means = results_df[f'{metric}_mean'].values
            stds = results_df[f'{metric}_std'].values

            ax.plot(x, means, marker='o', linewidth=2, color=colour)
            ax.fill_between(x, means - stds, means + stds,
                            alpha=0.2, color=colour, label='±1 std')
            ax.axhline(means[0], color='black', linewidth=1,
                       linestyle='--', label='original baseline')

            ax.set_xticks(x)
            ax.set_xticklabels(results_df['label'], rotation=25, ha='right', fontsize=7)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
            ax.set_title(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1.05)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(fontsize=8)

            for xi, row in zip(x, results_df.itertuples()):
                ax.annotate(f"s={row.sparsity:.2f}",
                            xy=(xi, 0.01), xycoords=('data', 'axes fraction'),
                            ha='center', fontsize=6, color='grey')

        fig.suptitle(
            f'Classifier performance vs perturbation — {self.generator_type}, RF, {cv}-fold CV',
            fontsize=12, fontweight='bold',
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print("\nResults:")
        print(results_df.to_string(index=False))
        return results_df

    # ------------------------------------------------------------------
    # Internal guards
    # ------------------------------------------------------------------
    def _require_data(self) -> None:
        if self.X_original is None or self.y_original is None:
            raise RuntimeError("No data loaded. Call load_data() first.")