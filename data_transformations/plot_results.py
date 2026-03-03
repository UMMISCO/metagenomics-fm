import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

# =============================================================================
# CONFIG
# =============================================================================
TIDY_CSV  = "benchmark_tidy.csv"
SAVE_DIR  = "plots/"
METRICS   = ["AUROC", "F1"]

MODEL_COLORS = {
    "RF":     "#2196F3",   # blu
    "TabICL": "#FF5722",   # arancio
    "TabPFN": "#4CAF50",   # verde
}

DATASET_LINESTYLES = {
    "abundance_WT2D":                          "-",
    "abundance_cirrhosis--stagediscovery":     "--",
    "abundance_cirrhosis--stagevalidation":    "-.",
    "abundance_ibd":                           ":",
    "abundance_obesity":                       (0, (3, 1, 1, 1)),      # trattino-punto
    "abundance_t2d":                           (0, (5, 1)),             # trattino lungo
}

DATASET_MARKERS = {
    "abundance_WT2D":                          "o",
    "abundance_cirrhosis--stagediscovery":     "s",
    "abundance_cirrhosis--stagevalidation":    "^",
    "abundance_ibd":                           "D",
    "abundance_obesity":                       "P",
    "abundance_t2d":                           "*",
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
    """Ordina i parametri: numerici in modo crescente, stringhe alfabeticamente."""
    try:
        return sorted(values, key=lambda x: float(str(x).split("=")[-1].split("/")[0].strip()))
    except ValueError:
        return sorted(values, key=str)


def plot_perturbation(df: pd.DataFrame, pert_type: str, save_dir: str) -> None:
    """
    Un plot per (perturbazione, metrica) — 2 plot per perturbazione.
    X      = livelli del parametro di perturbazione (ordinati)
    Y      = metrica (mean ± std)
    Colore = modello
    Linea  = dataset
    """
    df_pert = df[df["Perturbation name"] == pert_type].copy()
    if df_pert.empty:
        print(f"[SKIP] Nessun dato per perturbazione: {pert_type}")
        return

    datasets = [d for d in DATASET_LINESTYLES if d in df_pert["Dataset name"].unique()]
    models   = [m for m in MODEL_COLORS       if m in df_pert["Model name"].unique()]

    # Ordina i parametri sull'asse x
    param_values = sort_param_values(df_pert["Perturbation parameter value"].unique().tolist())
    x_positions  = {v: i for i, v in enumerate(param_values)}

    for metric in METRICS:
        df_m = df_pert[df_pert["Metric name"] == metric]

        fig, ax = plt.subplots(figsize=(11, 5))

        for dataset in datasets:
            ls     = DATASET_LINESTYLES[dataset]
            marker = DATASET_MARKERS[dataset]
            label  = DATASET_LABELS[dataset]

            for model in models:
                color = MODEL_COLORS[model]

                sub = df_m[
                    (df_m["Dataset name"] == dataset) &
                    (df_m["Model name"]   == model)
                ].copy()

                if sub.empty:
                    continue

                # Rimuovi baseline (std = NaN) dal plot OOD
                sub_ood = sub.dropna(subset=["Metric std"])
                if sub_ood.empty:
                    continue

                sub_ood = sub_ood.copy()
                sub_ood["x"] = sub_ood["Perturbation parameter value"].map(x_positions)
                sub_ood = sub_ood.sort_values("x")

                x    = sub_ood["x"].values
                mean = sub_ood["Metric mean"].values
                std  = sub_ood["Metric std"].values

                ax.plot(x, mean,
                        color=color, linestyle=ls, marker=marker,
                        linewidth=1.6, markersize=5, alpha=0.85)
                ax.fill_between(x, mean - std, mean + std,
                                color=color, alpha=0.07)

                # Baseline: linea orizzontale tratteggiata sottile
                sub_base = sub[sub["Metric std"].isna()]
                if not sub_base.empty:
                    baseline_val = sub_base["Metric mean"].values[0]
                    ax.axhline(baseline_val, color=color, linestyle=ls,
                               linewidth=0.8, alpha=0.35)

        # --- Asse X ---
        ax.set_xticks(range(len(param_values)))
        ax.set_xticklabels([str(v) for v in param_values],
                           rotation=35, ha='right', fontsize=8)
        ax.set_xlabel("Perturbation parameter value", fontsize=10)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title(
            f"{pert_type}  —  {metric}\n"
            f"color = model  |  line style = dataset  |  dashed = baseline",
            fontsize=11, fontweight='bold'
        )

        # --- Legenda modelli (colore) ---
        model_handles = [
            mlines.Line2D([], [], color=MODEL_COLORS[m], linewidth=2, label=m)
            for m in models
        ]

        # --- Legenda dataset (linestyle + marker) ---
        dataset_handles = [
            mlines.Line2D([], [], color='gray',
                          linestyle=DATASET_LINESTYLES[d],
                          marker=DATASET_MARKERS[d],
                          markersize=5, linewidth=1.5,
                          label=DATASET_LABELS[d])
            for d in datasets
        ]

        leg1 = ax.legend(handles=model_handles,
                         title="Model", fontsize=8, title_fontsize=9,
                         loc='upper right', frameon=True)
        ax.add_artist(leg1)
        ax.legend(handles=dataset_handles,
                  title="Dataset", fontsize=8, title_fontsize=9,
                  loc='lower left', frameon=True, ncol=2)

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out = os.path.join(save_dir, f"{pert_type}_{metric}.png")
            plt.savefig(out, dpi=200, bbox_inches='tight')
            print(f"  Salvato → {out}")

        plt.show()
        plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    df = pd.read_csv(TIDY_CSV)
    print(f"Caricato {TIDY_CSV}  —  {len(df)} righe")

    for pert_type in df["Perturbation name"].unique():
        print(f"\nPlot: {pert_type}")
        plot_perturbation(df, pert_type, SAVE_DIR)


if __name__ == "__main__":
    main()
