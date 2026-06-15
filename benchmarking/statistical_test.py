import numpy as np
import pandas as pd
from scipy import stats
from MLstatkit import Delong_test

"""
DeLong test for every combination (dataset, model, perturbation, param_value).
Compares baseline AUROC vs OOD AUROC using per-sample predictions.

For each (dataset, model, perturbation, param_value):
  - merges baseline and OOD predicted probabilities for the same
    (fold, sample_idx, y_true) triples
  - runs the DeLong test on the pooled out-of-fold samples

Output: statistical_results_final.csv
"""

PREDICTIONS_CSV = "results/predictions_all_final.csv"
OUTPUT_CSV      = "results/statistical_results_final.csv"
ALPHA           = 0.05

def main():
    print(f"Caricamento {PREDICTIONS_CSV}...")
    df = pd.read_csv(PREDICTIONS_CSV)
    print(f"  {len(df)} righe, colonne: {df.columns.tolist()}")

    df_base = df[df['split'] == 'baseline']
    df_ood  = df[df['split'] == 'ood']

    groups = df_ood.groupby(['dataset', 'model', 'perturbation', 'param_value'])

    rows = []
    total = len(groups)
    print(f"\nCalcolo DeLong test per {total} combinazioni...")

    for i, ((dataset, model, perturbation, param_value), ood_group) in enumerate(groups):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{total}")

        # baseline per questo (dataset, model, perturbation) — tutti i fold
        base_all = df_base[
            (df_base['dataset']      == dataset) &
            (df_base['model']        == model) &
            (df_base['perturbation'] == perturbation)
        ]

        # join su (sample_idx, fold, y_true) → 96 campioni OOF
        merged = ood_group.merge(
            base_all[['fold', 'sample_idx', 'y_true', 'proba_class1']],
            on=['fold', 'sample_idx', 'y_true'],
            suffixes=('_ood', '_base')
        )

        if len(merged) < 10:
            rows.append({
                'dataset': dataset, 'model': model,
                'perturbation': perturbation, 'param_value': param_value,
                'auroc_baseline': np.nan, 'auroc_ood': np.nan,
                'delta_auroc': np.nan, 'p_value': np.nan,
            })
            continue

        y_true     = merged['y_true'].values
        score_base = merged['proba_class1_base'].values
        score_ood  = merged['proba_class1_ood'].values

        z, pval, ci_A, ci_B, auc_base, auc_ood, info = Delong_test(
            y_true, score_base, score_ood,
            alpha=0.95, return_ci=True, return_auc=True, verbose=0
        )
        delta = auc_base - auc_ood if not np.isnan(auc_base) else np.nan

        rows.append({
            'dataset':        dataset,
            'model':          model,
            'perturbation':   perturbation,
            'param_value':    param_value,
            'auroc_baseline': round(float(auc_base), 4) if not np.isnan(auc_base) else np.nan,
            'auroc_ood':      round(float(auc_ood),  4) if not np.isnan(auc_ood)  else np.nan,
            'delta_auroc':    round(float(delta),    4) if not np.isnan(delta)    else np.nan,
            'p_value':        round(float(pval),     4) if not np.isnan(pval)     else np.nan,
        })

    results = pd.DataFrame(rows)

    valid_mask = ~results['p_value'].isna()
    pvals = results.loc[valid_mask, 'p_value'].values
    results['significant'] = results['p_value'] < ALPHA

    results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSalvato → {OUTPUT_CSV}")
    print(f"  Totale test     : {len(results)}")
    print(f"  Significativi   : {results['significant'].sum()} (p < {ALPHA})")
    print(results[results['significant']].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
