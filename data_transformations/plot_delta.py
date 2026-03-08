import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

"""
Plot delta (baseline - OOD) per fold con std dei delta.
Un plot per (dataset, perturbazione, metrica).
X = livelli del parametro di perturbazione (ordinati)
Y = mean(delta_fold_i) ± std(delta_fold_i)
Colore = modello
"""

# =============================================================================
# CONFIG
# =============================================================================
METRICS_CSV = "metrics_all.csv"
SAVE_DIR    = "plots_delta/"
METRICS     = ["auroc", "f1", "rec"]

MODEL_COLORS = {
    "RF":         "#2196F3",
    "TabICL":     "#FF5722",
    "TabPFN":     "#4CAF50",
    "XGBoost":    "#9C27B0",
    "TabDPT":     "#F44336",
    "ContextTab": "#FF9800",
}

DATASET_LABELS = {
    "abundance_WT2D":                          "WT2D",
    "abundance_cirrhosis--stagediscovery":     "Cirrhosis discovery",
    "abundance_cirrhosis--stagevalidation":    "Cirrhosis validation",
    "abundance_ibd":                           "IBD",
    "abundance_obesity":                       "Obesity",
    "abundance_t2d":                           "T2D",
}

# =============================================================================
# HELPERS
# =============================================================================

def sort_param_values(values: list) -> list:
    try:
        return sorted(values, key=lambda x: float(str(x).split("=")[-1].split("/")[0].strip()))
    except ValueError:
        return sorted(values, key=str)


def compute_delta_stats(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Per ogni (dataset, model, perturbation, param_value) calcola
    delta_fold_i = baseline_{metric}_fold_i - {metric}_fold_i
    poi mean e std dei delta sui 5 fold.
    """
    df = df.copy()
    df[f'delta_{metric}'] = df[f'baseline_{metric}'] - df[metric]

    stats = df.groupby(
        ['dataset', 'model', 'perturbation', 'param_key', 'param_value']
    )[f'delta_{metric}'].agg(['mean', 'std']).reset_index()
    stats.columns = ['dataset', 'model', 'perturbation', 'param_key', 'param_value',
                     f'delta_{metric}_mean', f'delta_{metric}_std']
    return stats


# =============================================================================
# PLOT
# =============================================================================

def plot_delta(df_stats: pd.DataFrame, dataset: str, pert_type: str,
               metric: str, save_dir: str) -> None:

    df_sub = df_stats[
        (df_stats['dataset']      == dataset) &
        (df_stats['perturbation'] == pert_type)
    ].copy()

    if df_sub.empty:
        return

    models = [m for m in MODEL_COLORS if m in df_sub['model'].unique()]

    param_values = sort_param_values(df_sub['param_value'].unique().tolist())
    x_positions  = {v: i for i, v in enumerate(param_values)}

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for model in models:
        color = MODEL_COLORS[model]
        sub   = df_sub[df_sub['model'] == model].copy()
        if sub.empty:
            continue

        sub['x'] = sub['param_value'].map(x_positions)
        sub = sub.sort_values('x')

        x     = sub['x'].values
        mean  = sub[f'delta_{metric}_mean'].values
        std   = sub[f'delta_{metric}_std'].values

        ax.plot(x, mean, color=color, linestyle='-', marker='o',
                linewidth=1.8, markersize=5, label=model, alpha=0.9)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.1)

    ax.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.5)

    ax.set_xticks(range(len(param_values)))
    ax.set_xticklabels([str(v) for v in param_values],
                       rotation=35, ha='right', fontsize=8)
    ax.set_xlabel("Perturbation parameter value", fontsize=9)
    ax.set_ylabel(f"Δ {metric.upper()}  (baseline − OOD)", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)

    dataset_label = DATASET_LABELS.get(dataset, dataset)
    ax.set_title(f"{dataset_label}  |  {pert_type}  |  Δ {metric.upper()}",
                 fontsize=11, fontweight='bold')

    model_handles = [
        mlines.Line2D([], [], color=MODEL_COLORS[m], linewidth=2,
                      marker='o', markersize=4, label=m)
        for m in models if m in df_sub['model'].unique()
    ]
    zero_handle = mlines.Line2D([], [], color='black', linestyle='--',
                                linewidth=1.0, label='no degradation (Δ=0)')
    ax.legend(handles=model_handles + [zero_handle], fontsize=8, frameon=True, loc='best')

    plt.tight_layout()

    if save_dir:
        subdir = os.path.join(save_dir, dataset)
        os.makedirs(subdir, exist_ok=True)
        out = os.path.join(subdir, f"{pert_type}__{metric}__delta.png")
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"  Salvato → {out}")

    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    df = pd.read_csv(METRICS_CSV)
    print(f"Caricato {METRICS_CSV}  —  {len(df)} righe")

    datasets   = df['dataset'].unique()
    pert_types = df['perturbation'].unique()

    for metric in METRICS:
        print(f"\nCalcolo delta per metrica: {metric}")
        df_stats = compute_delta_stats(df, metric)

        for dataset in datasets:
            for pert_type in pert_types:
                plot_delta(df_stats, dataset, pert_type, metric, SAVE_DIR)

    print(f"\nDone — plot salvati in {SAVE_DIR}")


if __name__ == "__main__":
    main()
