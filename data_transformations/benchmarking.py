import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from typing import List, Optional, Tuple, Dict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
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
    selector = SelectKBest(f_classif, k=n_features_max)
    selector.fit(X_train, y_train)
    return selector.get_support(indices=True)


def _param_label(params: dict) -> Tuple[str, str]:
    """
    Returns (param_key, param_val_str) for use as x-axis label.
    For remove_features: combines k + selection_method.
    For others: uses the first non-seed key.
    """
    filtered = {k: v for k, v in params.items() if k != 'seed'}
    if 'k' in filtered and 'selection_method' in filtered:
        return 'k / method', f"k={filtered['k']} / {filtered['selection_method']}"
    key = next(iter(filtered))
    return key, filtered[key]


def _cv_scores(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: int,
    n_features_max: int,
    device: str,
    random_state: int,
) -> dict:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    auroc_scores, f1_scores, prec_scores, rec_scores = [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train = X.iloc[train_idx].values
        X_test  = X.iloc[test_idx].values
        y_train = y[train_idx]
        y_test  = y[test_idx]

        if X_train.shape[1] > n_features_max:
            selected = _reduce_features(X_train, y_train, n_features_max)
            X_train  = X_train[:, selected]
            X_test   = X_test[:, selected]

        model = load_tabfn_model(model_name, device=device)
        if model_name == 'tabdpt':
            y_train = np.asarray(y_train)

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        pred  = model.predict(X_test)

        avg = 'macro' if len(np.unique(y)) > 2 else 'binary'
        auroc_scores.append(roc_auc_score(y_test, proba, multi_class='ovr') if proba.shape[1] > 2
                            else roc_auc_score(y_test, proba[:, 1]))
        f1_scores.append(f1_score(y_test, pred, average=avg, zero_division=0))
        prec_scores.append(precision_score(y_test, pred, average=avg, zero_division=0))
        rec_scores.append(recall_score(y_test, pred, average=avg, zero_division=0))

    return {
        'auroc_mean': round(float(np.mean(auroc_scores)), 4),
        'auroc_std':  round(float(np.std(auroc_scores)), 4),
        'f1_mean':    round(float(np.mean(f1_scores)), 4),
        'f1_std':     round(float(np.std(f1_scores)), 4),
        'prec_mean':  round(float(np.mean(prec_scores)), 4),
        'prec_std':   round(float(np.std(prec_scores)), 4),
        'rec_mean':   round(float(np.mean(rec_scores)), 4),
        'rec_std':    round(float(np.std(rec_scores)), 4),
    }


class Benchmarker:

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
        figsize: Tuple[int, int] = (8, 5),
        save_dir: Optional[str] = None,
    ) -> pd.DataFrame:

        if model_names is None:
            model_names = ['rf']
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        all_rows = []

        for pert_type, param_list in perturbation_configs.items():
            print(f"\n{'='*60}\nPerturbation: {pert_type}\n{'='*60}")

            for dataset in datasets:
                print(f"  Dataset: {dataset}")

                gen = DataGenerator(generator_type=pert_type, data_source=self.data_source)
                gen.load_data(dataset)
                gen.discover_and_protect(method='random_forest', n_features=n_features_protect, verbose=False)
                y = gen.y_original.values

                for model_name in model_names:
                    print(f"    Model: {model_name}")

                    baseline = _cv_scores(model_name, gen.X_original, y, cv, n_features_max, device, random_state)

                    for params in param_list:
                        param_key, param_val = _param_label(params)
                        X_pert = gen.generate(**params)
                        scores = _cv_scores(model_name, X_pert, y, cv, n_features_max, device, random_state)

                        row = {
                            'perturbation':    pert_type,
                            'dataset':         dataset,
                            'model':           model_name,
                            'param_key':       param_key,
                            'param_value':     param_val,
                            'baseline_auroc':  baseline['auroc_mean'],
                        }
                        row.update(scores)
                        all_rows.append(row)
                        print(f"      {param_val} → AUROC={scores['auroc_mean']:.3f}±{scores['auroc_std']:.3f}  F1={scores['f1_mean']:.3f}")

        results_df = pd.DataFrame(all_rows)
        self._plot_and_save(results_df, perturbation_configs, figsize, save_dir)

        if save_dir:
            results_df.to_csv(os.path.join(save_dir, 'benchmark_results.csv'), index=False)

        return results_df

    def _plot_and_save(self, results_df, perturbation_configs, figsize, save_dir):
        metrics = [
            ('auroc_mean', 'auroc_std', 'AUROC'),
            ('f1_mean',    'f1_std',    'F1'),
            ('prec_mean',  'prec_std',  'Precision'),
            ('rec_mean',   'rec_std',   'Recall'),
        ]

        dataset_names = results_df['dataset'].unique()
        model_names   = results_df['model'].unique()
        palette       = sns.color_palette('tab10', len(dataset_names))
        linestyles    = ['-', '--', ':', '-.']

        for pert_type in perturbation_configs:
            df_pert = results_df[results_df['perturbation'] == pert_type]

            fig, axes = plt.subplots(1, len(metrics), figsize=(figsize[0] * len(metrics), figsize[1]))

            for ax, (mean_col, std_col, title) in zip(axes, metrics):
                for d_idx, dataset in enumerate(dataset_names):
                    for m_idx, model_name in enumerate(model_names):
                        sub = df_pert[
                            (df_pert['dataset'] == dataset) &
                            (df_pert['model'] == model_name)
                        ]
                        if sub.empty:
                            continue

                        x = range(len(sub))
                        label = f"{dataset.replace('abundance_', '')} / {model_name}"
                        ax.plot(x, sub[mean_col], marker='o', linewidth=1.5,
                                color=palette[d_idx], linestyle=linestyles[m_idx % len(linestyles)],
                                label=label)
                        ax.fill_between(x,
                                        sub[mean_col] - sub[std_col],
                                        sub[mean_col] + sub[std_col],
                                        alpha=0.1, color=palette[d_idx])

                ax.set_xticks(range(len(df_pert['param_value'].unique())))
                ax.set_xticklabels(df_pert['param_value'].unique(), rotation=30, ha='right', fontsize=7)
                ax.set_ylabel(title, fontsize=10)
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.set_ylim(0, 1.05)
                ax.grid(True, linestyle='--', alpha=0.3)

            axes[0].legend(fontsize=7, frameon=True, bbox_to_anchor=(0, -0.3), loc='upper left', ncol=2)
            fig.suptitle(f"{pert_type} — performance vs perturbation", fontsize=12, fontweight='bold')
            plt.tight_layout()

            if save_dir:
                plt.savefig(os.path.join(save_dir, f'{pert_type}_performance.png'), dpi=200, bbox_inches='tight')
            plt.show()