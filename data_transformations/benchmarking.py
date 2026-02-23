import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict

from data_transformations.data_generator import DataGenerator


class Benchmarker:
    """
    Evaluates classifier AUROC across multiple datasets,
    perturbation types, and models.
    """

    def __init__(self, data_source: str = 'pasolli'):
        self.data_source = data_source

    def run(
        self,
        datasets: List[str],
        perturbation_configs: Dict[str, List[dict]],
        models: Optional[dict] = None,
        cv: int = 5,
        n_features_protect: int = 20,
        random_state: int = 42,
        figsize: Tuple[int, int] = (7, 5),
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        For each dataset and model, evaluate AUROC on perturbed data.
        Produces one plot per perturbation type.

        Parameters
        ----------
        datasets             : list of dataset names
        perturbation_configs : dict mapping perturbation_type -> list of param dicts
        models               : dict of {name: sklearn estimator}, default RF + LR
        cv                   : cross-validation folds
        n_features_protect   : features to protect per dataset
        random_state         : random seed
        figsize              : size of each subplot
        save_path            : optional path prefix to save figures

        Returns
        -------
        pd.DataFrame with all results
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import cross_val_score

        if models is None:
            models = {
                'RandomForest': RandomForestClassifier(
                    n_estimators=100, random_state=random_state, n_jobs=-1),
                'LogisticReg': Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', LogisticRegression(max_iter=1000, random_state=random_state)),
                ]),
            }

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

                for model_name, clf in models.items():
                    for params in param_list:
                        param_val = next(v for k, v in params.items() if k != 'seed')
                        param_key = next(k for k in params.keys() if k != 'seed')

                        X_pert = gen.generate(**params)
                        scores = cross_val_score(
                            clf, X_pert, gen.y_original,
                            cv=cv, scoring='roc_auc',
                        )
                        all_rows.append({
                            'perturbation': pert_type,
                            'dataset':      dataset,
                            'model':        model_name,
                            'param_key':    param_key,
                            'param_value':  param_val,
                            'auroc_mean':   round(scores.mean(), 4),
                            'auroc_std':    round(scores.std(), 4),
                        })

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
        """One plot per perturbation type."""
        palette = sns.color_palette('tab10', len(results_df['dataset'].unique()))
        linestyles = ['-', '--', ':', '-.']

        for pert_type in perturbation_configs:
            df_pert = results_df[results_df['perturbation'] == pert_type]
            param_key = df_pert['param_key'].iloc[0]
            dataset_names = df_pert['dataset'].unique()
            model_names = df_pert['model'].unique()

            fig, ax = plt.subplots(figsize=figsize)

            for d_idx, dataset in enumerate(dataset_names):
                for m_idx, model_name in enumerate(model_names):
                    sub = df_pert[
                        (df_pert['dataset'] == dataset) &
                        (df_pert['model'] == model_name)
                    ].sort_values('param_value')

                    label = f"{dataset.replace('abundance_', '')} / {model_name}"
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

            ax.set_xlabel(param_key, fontsize=10)
            ax.set_ylabel("AUROC", fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.axhline(0.5, color='grey', linewidth=1, linestyle='--', label='random baseline')
            ax.set_title(f"{pert_type} — AUROC vs {param_key}", fontsize=11, fontweight='bold')
            ax.legend(fontsize=7, frameon=True, bbox_to_anchor=(1.01, 1), loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()

            if save_path:
                plt.savefig(f"{save_path}_{pert_type}.png", dpi=300, bbox_inches='tight')
            plt.show()
