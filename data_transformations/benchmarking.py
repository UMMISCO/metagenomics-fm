import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from typing import List, Optional, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, f_classif

sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/')
from data_transformations.check_data_on_fly.data_generator import DataGenerator

warnings.filterwarnings('ignore')


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_tabfn_model(model_name: str, device: str = 'cuda'):
    print(f"  [DEBUG] Loading {model_name} on device={device}")

    if model_name == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_name == 'tabdpt':
        from tabdpt import TabDPTClassifier
        return TabDPTClassifier(device=device)
    elif model_name == 'original_v2':
        from tabpfn import TabPFNClassifier
        import torch
        return TabPFNClassifier(device=device, inference_precision=torch.float32)
    elif model_name == 'tabicl':
        from tabicl import TabICLClassifier
        return TabICLClassifier(device=device, use_amp=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# HELPERS
# =============================================================================

def _reduce_features(X_train: np.ndarray, y_train: np.ndarray, n_features_max: int) -> np.ndarray:
    selector = SelectKBest(f_classif, k=n_features_max)
    selector.fit(X_train, y_train)
    return selector.get_support(indices=True)


def _param_label(params: dict) -> Tuple[str, str]:
    filtered = {k: v for k, v in params.items() if k != 'seed'}
    if 'k' in filtered and 'selection_method' in filtered:
        return 'k / method', f"k={filtered['k']} / {filtered['selection_method']}"
    key = next(iter(filtered))
    return key, str(filtered[key])


def _params_to_filename(pert_type: str, params: dict) -> str:
    if pert_type == 'remove_features':
        return f"k={params['k']}.parquet"
    elif pert_type in ('sparsity', 'densification'):
        return f"{params['target_sparsity']}.parquet"
    return "params.parquet"


def _adaptive_params(pert_type: str, n_features: int, actual_sparsity: float, seed: int = 42) -> List[dict]:
    if pert_type == 'remove_features':
        k_max = max(1, n_features // 2)
        k_values = [int(k) for k in np.linspace(1, k_max, 10)]
        return [{'k': k, 'selection_method': 'highest_abundance', 'seed': seed} for k in k_values]
    elif pert_type == 'sparsity':
        sparsity_values = [round(s, 3) for s in np.linspace(actual_sparsity, 0.99, 7)[1:-1]]
        return [{'target_sparsity': s, 'seed': seed} for s in sparsity_values]
    elif pert_type == 'densification':
        sparsity_values = [round(s, 3) for s in np.linspace(actual_sparsity, 0.01, 7)[1:-1]]
        return [{'target_sparsity': s, 'seed': seed} for s in sparsity_values]
    return []


def _cv_scores(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: int,
    n_features_max: int,
    device: str,
    random_state: int,
) -> dict:
    """Baseline CV: TRAIN e TEST entrambi su X_original."""
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


def _cv_scores_ood(
    model_name: str,
    X_original: pd.DataFrame,
    X_perturbed: pd.DataFrame,
    y: np.ndarray,
    cv: int,
    n_features_max: int,
    device: str,
    random_state: int,
) -> dict:
    """
    OOD evaluation:
      - Split calcolato su X_original (stratificazione consistente)
      - TRAIN su X_original[train_idx]   <- dati puliti
      - TEST  su X_perturbed[test_idx]   <- stessi campioni, perturbati

    Feature selection fittata solo sul train originale (no leakage).
    """
    common_cols = X_perturbed.columns
    X_original = X_original[common_cols]

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    auroc_scores, f1_scores, prec_scores, rec_scores = [], [], [], []

    for train_idx, test_idx in skf.split(X_original, y):
        X_train = X_original.iloc[train_idx].values   # <- ORIGINALE
        X_test  = X_perturbed.iloc[test_idx].values   # <- PERTURBATO
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


# =============================================================================
# BENCHMARKER
# =============================================================================

class Benchmarker:

    def __init__(self, data_source: str = 'pasolli'):
        self.data_source = data_source

    def run_one(
        self,
        dataset: str,
        pert_type: str,
        model_names: List[str] = None,
        cv: int = 5,
        n_features_protect: int = 2,
        n_features_max: int = 100000,
        random_state: int = 42,
        device: str = 'cuda',
        seed: int = 42,
        precomputed_dir: Optional[str] = None,
        save_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        OOD benchmark per una singola combinazione (dataset, perturbation_type).

        Protocollo per fold
        -------------------
        baseline : train X_original[train_idx] -> test X_original[test_idx]
        OOD      : train X_original[train_idx] -> test X_perturbed[test_idx]
        """
        if model_names is None:
            model_names = ['rf']

        print(f"\n{'='*60}")
        print(f"Dataset : {dataset}  |  Perturbation: {pert_type}")
        print(f"Protocol: TRAIN on original -> TEST on perturbed (OOD)")
        print(f"{'='*60}")

        gen = DataGenerator(generator_type=pert_type, data_source=self.data_source)
        gen.load_data(dataset)
        gen.discover_and_protect(method='random_forest', n_features=n_features_protect, verbose=False)
        y = gen.y_original.values

        actual_sparsity = float((gen.X_original.values == 0).mean())
        n_features      = gen.X_original.shape[1]
        dataset_params  = _adaptive_params(pert_type, n_features, actual_sparsity, seed=seed)
        print(f"  Adaptive params: {len(dataset_params)} levels  (sparsity={actual_sparsity:.3f}, n_features={n_features})")

        perturbed_cache: dict = {}
        if precomputed_dir:
            orig_path = os.path.join(precomputed_dir, dataset, pert_type, 'original.parquet')
            if os.path.exists(orig_path):
                df_orig    = pd.read_parquet(orig_path)
                y          = df_orig['label'].values
                X_original = df_orig.drop(columns=['label'])
            else:
                X_original = gen.X_original

            for params in dataset_params:
                _, param_val = _param_label(params)
                fname = _params_to_filename(pert_type, params)
                fpath = os.path.join(precomputed_dir, dataset, pert_type, fname)
                if os.path.exists(fpath):
                    df_pert = pd.read_parquet(fpath)
                    perturbed_cache[param_val] = df_pert.drop(columns=['label'])
                else:
                    print(f"  [WARN] Non trovato: {fpath} — generazione on-the-fly")
        else:
            X_original = gen.X_original

        rows = []
        for model_name in model_names:
            print(f"\n  Model: {model_name}")

            # Baseline: orig -> orig
            baseline = _cv_scores(model_name, X_original, y, cv, n_features_max, device, random_state)
            print(f"    [baseline] orig->orig  AUROC={baseline['auroc_mean']:.3f}+-{baseline['auroc_std']:.3f}")

            for params in dataset_params:
                param_key, param_val = _param_label(params)

                X_pert = perturbed_cache.get(param_val)
                if X_pert is None:
                    X_pert = gen.generate(**params)

                # OOD: train originale -> test perturbato
                scores = _cv_scores_ood(
                    model_name, X_original, X_pert, y, cv, n_features_max, device, random_state
                )
                print(f"    [OOD] {param_val:35s}  AUROC={scores['auroc_mean']:.3f}+-{scores['auroc_std']:.3f}  F1={scores['f1_mean']:.3f}")

                rows.append({
                    'perturbation':   pert_type,
                    'dataset':        dataset,
                    'model':          model_name,
                    'param_key':      param_key,
                    'param_value':    param_val,
                    'baseline_auroc': baseline['auroc_mean'],
                    'baseline_f1':    baseline['f1_mean'],
                    'baseline_prec':  baseline['prec_mean'],
                    'baseline_rec':   baseline['rec_mean'],
                    **scores,
                })

        results_df = pd.DataFrame(rows)

        if save_dir:
            out_dir = os.path.join(save_dir, dataset)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{pert_type}.parquet")
            results_df.to_parquet(out_path, index=False)
            print(f"\n  Saved -> {out_path}")

        return results_df

    def plot(
        self,
        results_df: pd.DataFrame,
        figsize: Tuple[int, int] = (8, 5),
        save_dir: Optional[str] = None,
    ) -> None:
        """
        Linea continua    = score OOD  (train originale / test perturbato)
        Linea tratteggiata = baseline  (train originale / test originale)
        """
        metrics = [
            ('auroc_mean', 'auroc_std', 'AUROC',    'baseline_auroc'),
            ('f1_mean',    'f1_std',    'F1',        'baseline_f1'),
            ('prec_mean',  'prec_std',  'Precision', 'baseline_prec'),
            ('rec_mean',   'rec_std',   'Recall',    'baseline_rec'),
        ]

        dataset_names = results_df['dataset'].unique()
        model_names   = results_df['model'].unique()
        palette       = sns.color_palette('tab10', len(dataset_names))
        linestyles    = ['-', '--', ':', '-.']

        for pert_type in results_df['perturbation'].unique():
            df_pert = results_df[results_df['perturbation'] == pert_type]

            fig, axes = plt.subplots(1, len(metrics), figsize=(figsize[0] * len(metrics), figsize[1]))

            for ax, (mean_col, std_col, title, base_col) in zip(axes, metrics):
                for d_idx, dataset in enumerate(dataset_names):
                    for m_idx, model_name in enumerate(model_names):
                        sub = df_pert[
                            (df_pert['dataset'] == dataset) &
                            (df_pert['model']   == model_name)
                        ]
                        if sub.empty:
                            continue

                        x     = range(len(sub))
                        label = f"{dataset.replace('abundance_', '')} / {model_name}"
                        color = palette[d_idx]
                        ls    = linestyles[m_idx % len(linestyles)]

                        # Curva OOD
                        ax.plot(x, sub[mean_col], marker='o', linewidth=1.5,
                                color=color, linestyle=ls, label=label)
                        ax.fill_between(
                            x,
                            sub[mean_col] - sub[std_col],
                            sub[mean_col] + sub[std_col],
                            alpha=0.1, color=color,
                        )

                        # Baseline orizzontale tratteggiata
                        if base_col in sub.columns:
                            ax.axhline(sub[base_col].iloc[0], color=color,
                                       linestyle='--', linewidth=0.9, alpha=0.55)

                seen = set()
                ordered_params = [v for v in df_pert['param_value'].tolist()
                                  if not (v in seen or seen.add(v))]
                ax.set_xticks(range(len(ordered_params)))
                ax.set_xticklabels(ordered_params, rotation=30, ha='right', fontsize=7)
                ax.set_ylabel(title, fontsize=10)
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.set_ylim(0, 1.05)
                ax.grid(True, linestyle='--', alpha=0.3)

            axes[0].legend(fontsize=7, frameon=True,
                           bbox_to_anchor=(0, -0.3), loc='upper left', ncol=2)
            fig.suptitle(
                f"{pert_type}  —  OOD: train=original | test=perturbed\n"
                f"(tratteggiato = baseline: train=original | test=original)",
                fontsize=11, fontweight='bold',
            )
            plt.tight_layout()

            if save_dir:
                os.makedirs(os.path.join(save_dir, pert_type), exist_ok=True)
                plt.savefig(
                    os.path.join(save_dir, pert_type, f'{pert_type}_ood_performance.png'),
                    dpi=200, bbox_inches='tight',
                )
            plt.show()