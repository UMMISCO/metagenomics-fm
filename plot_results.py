import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

"""
Un plot per (dataset, perturbazione, metrica).
X      = livelli del parametro di perturbazione (ordinati)
Y      = metrica (mean ± std)
Colore = modello
Linea nera tratteggiata = baseline di ogni modello
"""

# =============================================================================
# CONFIG
# =============================================================================
TIDY_CSV = "benchmark_tidy.csv"
SAVE_DIR = "plots/"
METRICS  = ["AUROC", "F1"]

MODEL_COLORS = {
    "RF":         "#2196F3",   # blu
    "TabICL":     "#FF5722",   # arancio
    "TabPFN":     "#4CAF50",   # verde
    "XGBoost":    "#9C27B0",   # viola
    "TabDPT":     "#F44336",   # rosso
    "ContextTab": "#FF9800",   # ambra
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


# =============================================================================
# PLOT
# =============================================================================

def plot_one(df: pd.DataFrame, dataset: str, pert_type: str, metric: str, save_dir: str) -> None:
    df_sub = df[
        (df["Dataset name"]      == dataset) &
        (df["Perturbation name"] == pert_type) &
        (df["Metric name"]       == metric)
    ].copy()

    if df_sub.empty:
        return

    models = [m for m in MODEL_COLORS if m in df_sub["Model name"].unique()]

    # parametri OOD sull'asse x (esclude baseline che ha std NaN)
    df_ood  = df_sub.dropna(subset=["Metric std"])
    param_values = sort_param_values(df_ood["Perturbation parameter value"].unique().tolist())
    x_positions  = {v: i for i, v in enumerate(param_values)}

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for model in models:
        color = MODEL_COLORS[model]

        sub_ood = df_ood[df_ood["Model name"] == model].copy()
        if sub_ood.empty:
            continue

        sub_ood["x"] = sub_ood["Perturbation parameter value"].map(x_positions)
        sub_ood = sub_ood.sort_values("x")

        x    = sub_ood["x"].values
        mean = sub_ood["Metric mean"].values
        std  = sub_ood["Metric std"].values

        # linea OOD colorata
        ax.plot(x, mean,
                color=color, linestyle='-', marker='o',
                linewidth=1.8, markersize=5, label=model, alpha=0.9)
        ax.fill_between(x, mean - std, mean + std,
                        color=color, alpha=0.1)

        # baseline: linea nera tratteggiata
        sub_base = df_sub[
            (df_sub["Model name"]  == model) &
            (df_sub["Metric std"].isna())
        ]
        if not sub_base.empty:
            baseline_val = sub_base["Metric mean"].values[0]
            ax.axhline(baseline_val, color=color, linestyle='--',
                       linewidth=1.2, alpha=0.5)

    ax.set_xticks(range(len(param_values)))
    ax.set_xticklabels([str(v) for v in param_values],
                       rotation=35, ha='right', fontsize=8)
    ax.set_xlabel("Perturbation parameter value", fontsize=9)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.3)

    dataset_label = DATASET_LABELS.get(dataset, dataset)
    ax.set_title(
        f"{dataset_label}  |  {pert_type}  |  {metric}",
        fontsize=11, fontweight='bold'
    )

    # legenda modelli
    model_handles = [
        mlines.Line2D([], [], color=MODEL_COLORS[m], linewidth=2,
                      marker='o', markersize=4, label=m)
        for m in models
    ]
    # entry baseline
    baseline_handle = mlines.Line2D([], [], color='gray', linestyle='--',
                                    linewidth=1.2, label='baseline (dashed)')
    ax.legend(handles=model_handles + [baseline_handle],
              fontsize=8, frameon=True, loc='best')

    plt.tight_layout()

    if save_dir:
        subdir = os.path.join(save_dir, dataset)
        os.makedirs(subdir, exist_ok=True)
        fname = f"{pert_type}__{metric}.png"
        out   = os.path.join(subdir, fname)
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"  Salvato → {out}")

    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    df = pd.read_csv(TIDY_CSV)
    print(f"Caricato {TIDY_CSV}  —  {len(df)} righe")

    datasets  = df["Dataset name"].unique()
    pert_types = df["Perturbation name"].unique()

    for dataset in datasets:
        for pert_type in pert_types:
            for metric in METRICS:
                plot_one(df, dataset, pert_type, metric, SAVE_DIR)

    print(f"\nDone — plot salvati in {SAVE_DIR}")


if __name__ == "__main__":
    main()
