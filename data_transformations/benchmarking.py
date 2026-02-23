import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import copy

from typing import List, Optional, Tuple, Dict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif

sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/')
from data_transformations.data_generator import DataGenerator

warnings.filterwarnings('ignore')


def load_tabfn_model(model_name: str, device: str = 'cpu'):
    if model_name == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_name == 'tabdpt':
        from tabdpt import TabDPTClassifier
        return TabDPTClassifier(device='cpu')
    elif model_name == 'original_v2':
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device=device)
    elif model_name == 'tabicl':
        from tabicl import TabICLClassifier
        return TabICLClassifier(device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _reduce_features(X_train: np.ndarray, y_train: np.ndarray, n_features_max: int) -> np.ndarray:
    """F-statistic feature selection — same as original cross_val_results."""
    selector = SelectKBest(f_classif, k=n_features_max)
    selector.fit(X_train, y_train)
    return selector.get_support(indices=True)


def _cv_auroc(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: int,
    n_features_max: int,
    device: str,
    random_state: int,
) -> Tuple[float, float]:
    """
    Stratified k-fold AUROC. Mirrors original cross_val_results logic:
    - Feature selection on train set only
    - Model reinstantiated per fold
    - Handles multiclass with ovr
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = []

    for train_idx, test_idx in skf.split(X, y):
        X_train = X.iloc[train_idx].values
        X_test  = X.iloc[test_idx].values
        y_train = y[train_idx]
        y_test  = y[test_idx]

        # Feature selection on train only
        if X_train.shape[1] > n_features_max:
            selected = _reduce_features(X_train, y_train, n_features_max)
            X_train  = X_train[:, selected]
            X_test   = X_test[:, selected]

        model = load_tabfn_model(model_name, device=device)

        if model_name == 'tabdpt':
            y_train = np.asarray(y_train)

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)

        if proba.shape[1] > 2:
            auc = roc_auc_score(y_test, proba, multi_class='ovr')
        else:
            auc = roc_auc_score(y_test, proba[:, 1])

        scores.append(auc)

    return float(np.mean(scores)), float(np.std(scores))


class Benchmarker:
    """
    Evaluates classifier AUROC across datasets, perturbation types, and models.
    """

    def __init__(self, data_source: str = 'pasolli'):
        self.data_source = data_source

    def run(
        self,
        datasets: List[str],
        perturbation_configs: Dict[str, List[dict]],
        model_names: List[str] = None,
        cv: int = 10,
        n_features_protect: int = 20,
        n_features_max: int = 10000,
        random_state: int = 42,
        device: str = 'cpu',
        figsize: Tuple[int, int] = (7, 5),
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        datasets             : list of dataset names
        perturbation_configs : dict mapping perturbation_type -> list of param dicts
        model_names          : list of model names: 'rf', 'original_v2', 'tabicl', 'tabdpt'
        cv                   : number of CV folds
        n_features_protect   : features to protect per dataset
        n_features_max       : max features after F-statistic selection (per fold)
        random_state         : random seed
        device               : 'cpu' or 'cuda'
        figsize              : figure size per plot
        save_path            : optional path prefix to save figures

        Returns
        -------
        pd.DataFrame with all results
        """
        if model_names is None:
            model_names = ['rf']

        all_rows = []

        for pert_type, param_list in perturbation_configs.items():
            print(f"\n{'='*60}\nPerturbation: {pert_type}\n{'='*60}")

            for dataset in datasets:
                print(f"  Dataset: {dataset}")

                gen = DataGenerator(
                    generator_type=pert_type,
                    data_source=self.data_source,
                )
                gen.load_data(dataset)
                gen.discover_and_protect(
                    method='random_forest',
                    n_features=n_features_protect,
                    verbose=False,
                )
                y = gen.y_original.values

                for model_name in model_names:
                    print(f"    Model: {model_name}")

                    # Baseline on original
                    baseline_mean, _ = _cv_auroc(
                        model_name, gen.X_original, y,
                        cv, n_features_max, device, random_state,
                    )

                    for params in param_list:
                        param_val = next(v for k, v in params.items() if k != 'seed')
                        param_key = next(k for k in params.keys() if k != 'seed')

                        X_pert = gen.generate(**params)
                        auroc_mean, auroc_std = _cv_auroc(
                            model_name, X_pert, y,
                            cv, n_features_max, device, random_state,
                        )

                        all_rows.append({
                            'perturbation':   pert_type,
                            'dataset':        dataset,
                            'model':          model_name,
                            'param_key':      param_key,
                            'param_value':    param_val,
                            'auroc_mean':     round(auroc_mean, 4),
                            'auroc_std':      round(auroc_std, 4),
                            'baseline_auroc': round(baseline_mean, 4),
                        })
                        print(f"      {param_key}={param_val} → AUROC={auroc_mean:.3f}±{auroc_std:.3f}")

        results_df = pd.DataFrame(all_rows)
        self._plot(results_df, perturbation_configs, figsize, save_path)
        return results_df

    def _plot(
        self,
        results_df: pd.DataFrame,
        perturbation_configs: dict,
        figsize: Tuple[int, int],
        save_path: Optional[str],
    ) -> None:
        """One plot per perturbation type. Colour = dataset, linestyle = model."""
        dataset_names = results_df['dataset'].unique()
        model_names   = results_df['model'].unique()
        palette       = sns.color_palette('tab10', len(dataset_names))
        linestyles    = ['-', '--', ':', '-.']

        for pert_type in perturbation_configs:
            df_pert = results_df[results_df['perturbation'] == pert_type]
            param_key = df_pert['param_key'].iloc[0]

            fig, ax = plt.subplots(figsize=figsize)

            for d_idx, dataset in enumerate(dataset_names):
                for m_idx, model_name in enumerate(model_names):
                    sub = df_pert[
                        (df_pert['dataset'] == dataset) &
                        (df_pert['model'] == model_name)
                    ].sort_values('param_value')

                    if sub.empty:
                        continue

                    short_name = dataset.replace('abundance_', '')
                    label = f"{short_name} / {model_name}"

                    ax.plot(
                        sub['param_value'], sub['auroc_mean'],
                        marker='o', linewidth=1.5,
                        color=palette[d_idx],
                        linestyle=linestyles[m_idx % len(linestyles)],
                        label=label,
                    )
                    ax.fill_between(
                        sub['param_value'],
                        sub['auroc_mean'] - sub['auroc_std'],
                        sub['auroc_mean'] + sub['auroc_std'],
                        alpha=0.1, color=palette[d_idx],
                    )
                    # Per-dataset baseline (dotted, same colour)
                    ax.axhline(
                        sub['baseline_auroc'].iloc[0],
                        color=palette[d_idx], linewidth=0.8,
                        linestyle=':', alpha=0.5,
                    )

            ax.axhline(0.5, color='grey', linewidth=1, linestyle='--', label='random baseline')
            ax.set_xlabel(param_key, fontsize=10)
            ax.set_ylabel("AUROC", fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{pert_type} — AUROC vs {param_key}", fontsize=11, fontweight='bold')
            ax.legend(fontsize=7, frameon=True, bbox_to_anchor=(1.01, 1), loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()

            if save_path:
                plt.savefig(f"{save_path}_{pert_type}.png", dpi=300, bbox_inches='tight')
            plt.show()