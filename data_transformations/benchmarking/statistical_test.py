import os
import numpy as np
import pandas as pd
from scipy import stats
from MLstatkit import Delong_test

from itertools import combinations

"""
DeLong test per ogni combinazione (dataset, model, perturbation, param_value).
Confronta AUROC baseline vs AUROC OOD usando le predizioni per sample.

Per ogni fold:
  - prende le proba_class1 baseline e OOD per gli stessi sample_idx
  - calcola il DeLong test statistic

Poi aggrega i p-value dei 5 fold con il metodo di Fisher.
Corregge per test multipli con FDR (Benjamini-Hochberg).

Output: statistical_results.csv
"""

# =============================================================================
# CONFIG
# =============================================================================
PREDICTIONS_CSV = "results/predictions_all_final.csv"
OUTPUT_CSV      = "results/statistical_results_final.csv"
ALPHA           = 0.05

# =============================================================================
# DELONG TEST
# =============================================================================

def compute_auc_and_variance(y_true, y_score):
    """
    Calcola AUC e varianza con il metodo DeLong.
    Ritorna (auc, variance).
    """
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan

    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]

    # kernel di Mann-Whitney
    V10 = np.array([np.mean(p > neg_scores) + 0.5 * np.mean(p == neg_scores)
                    for p in pos_scores])
    V01 = np.array([np.mean(n < pos_scores) + 0.5 * np.mean(n == pos_scores)
                    for n in neg_scores])

    auc = np.mean(V10)

    var = (np.var(V10, ddof=1) / n_pos + np.var(V01, ddof=1) / n_neg)
    return auc, var


def delong_test(y_true, y_score_baseline, y_score_ood):
    """
    DeLong test per confrontare due AUROC sugli stessi campioni.
    Ritorna p-value (two-sided).
    """
    auc1, var1 = compute_auc_and_variance(y_true, y_score_baseline)
    auc2, var2 = compute_auc_and_variance(y_true, y_score_ood)

    if np.isnan(var1) or np.isnan(var2):
        return np.nan, np.nan, np.nan

    # covarianza tra i due AUC (DeLong et al. 1988)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    pos_base = y_score_baseline[y_true == 1]
    neg_base = y_score_baseline[y_true == 0]
    pos_ood  = y_score_ood[y_true == 1]
    neg_ood  = y_score_ood[y_true == 0]

    V10_base = np.array([np.mean(p > neg_base) + 0.5 * np.mean(p == neg_base) for p in pos_base])
    V10_ood  = np.array([np.mean(p > neg_ood)  + 0.5 * np.mean(p == neg_ood)  for p in pos_ood])
    V01_base = np.array([np.mean(n < pos_base) + 0.5 * np.mean(n == pos_base) for n in neg_base])
    V01_ood  = np.array([np.mean(n < pos_ood)  + 0.5 * np.mean(n == pos_ood)  for n in neg_ood])

    cov = (np.cov(V10_base, V10_ood)[0, 1] / n_pos +
           np.cov(V01_base, V01_ood)[0, 1] / n_neg)

    var_diff = var1 + var2 - 2 * cov
    if var_diff <= 0:
        return auc1, auc2, np.nan

    z = (auc1 - auc2) / np.sqrt(var_diff)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc1, auc2, p


# =============================================================================
# MAIN
# =============================================================================

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

        # auc_base, auc_ood, pval = delong_test(y_true, score_base, score_ood)
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

    # Uncomment if you want  FDR correction
    valid_mask = ~results['p_value'].isna()
    pvals = results.loc[valid_mask, 'p_value'].values
    # corrected = fdr_correction(pvals, alpha=ALPHA)
    # results.loc[valid_mask, 'p_value_fdr'] = corrected
    # results['significant'] = results['p_value_fdr'] < ALPHA
    results['significant'] = results['p_value'] < ALPHA

    results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSalvato → {OUTPUT_CSV}")
    print(f"  Totale test     : {len(results)}")
    print(f"  Significativi   : {results['significant'].sum()} (p < {ALPHA})")
    print(results[results['significant']].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
