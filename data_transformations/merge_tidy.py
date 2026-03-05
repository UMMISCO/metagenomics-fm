import os
import glob
import argparse
import pandas as pd

'''
Merges the csv files for each perturbation and dataset in one big csv file in
order to be able to study and process it easily.
'''
# =============================================================================
# CONFIG
# =============================================================================
RESULTS_DIR = (
    "/data/projects/deepintegromics/analyses/3.tabpfn/"
    "metagen_foundation_models/data_transformations/benchmark_results_v2/"
)
OUTPUT_FILE = "benchmark_tidy.csv"

MODEL_DISPLAY = {
    "rf":          "RF",
    "tabicl":      "TabICL",
    "original_v2": "TabPFN",
    "xgb":         "XGBoost",
    "tabdpt":      "TabDPT",
    "contextab":   "ContextTab",
}

METRICS = [
    ("auroc_mean", "auroc_std", "AUROC"),
    ("f1_mean",    "f1_std",    "F1"),
    ("prec_mean",  "prec_std",  "Precision"),
    ("rec_mean",   "rec_std",   "Recall"),
]

REQUIRED_COLS = {
    "dataset", "model", "perturbation",
    "param_key", "param_value",
    "auroc_mean", "auroc_std",
    "f1_mean",    "f1_std",
    "prec_mean",  "prec_std",
    "rec_mean",   "rec_std",
}

# =============================================================================
# HELPERS
# =============================================================================

def load_all_csvs(results_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(results_dir, "**", "*.csv"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    return files


def wide_to_tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Converte una riga wide in 4 righe tidy (una per metrica)."""
    rows = []
    for _, row in df.iterrows():
        model_name = MODEL_DISPLAY.get(str(row["model"]), str(row["model"]))
        for mean_col, std_col, metric_name in METRICS:
            rows.append({
                "Dataset name":                 row["dataset"],
                "Model name":                   model_name,
                "Perturbation name":            row["perturbation"],
                "Perturbation parameter name":  row["param_key"],
                "Perturbation parameter value": row["param_value"],
                "Metric name":                  metric_name,
                "Metric mean":                  row[mean_col],
                "Metric std":                   row[std_col],
            })
    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--out", default=OUTPUT_FILE)
    args = parser.parse_args()

    csv_files = load_all_csvs(args.results_dir)
    if not csv_files:
        print(f"[ERROR] Nessun CSV trovato in: {args.results_dir}")
        return

    print(f"Trovati {len(csv_files)} file CSV\n")
    print(f"{'FILE':<65} {'WIDE':>6}  {'MODELLI':<30}  {'PARAMS':>6}  {'TIDY':>6}")
    print("-" * 120)

    if os.path.exists(args.out):
        os.remove(args.out)
        print(f"[INFO] Output esistente rimosso: {args.out}\n")

    total_wide = 0
    total_tidy = 0

    for i, fpath in enumerate(csv_files):
        fname = os.path.relpath(fpath, args.results_dir)

        try:
            df_wide = pd.read_csv(fpath)
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")
            continue

        if df_wide.empty:
            print(f"  [SKIP]  {fname} — vuoto")
            continue

        missing = REQUIRED_COLS - set(df_wide.columns)
        if missing:
            print(f"  [SKIP]  {fname} — colonne mancanti: {missing}")
            continue

        before_dedup = len(df_wide)
        df_wide = df_wide.drop_duplicates(
            subset=["dataset", "model", "perturbation", "param_key", "param_value"]
        )
        after_dedup = len(df_wide)
        dedup_note = f" (rimossi {before_dedup - after_dedup} duplicati)" if before_dedup != after_dedup else ""

        models_found = df_wide["model"].value_counts().to_dict()
        models_str   = ", ".join(f"{MODEL_DISPLAY.get(k,k)}:{v}" for k, v in models_found.items())
        n_params     = df_wide["param_value"].nunique()

        df_tidy = wide_to_tidy(df_wide)
        n_tidy  = len(df_tidy)

        write_header = (i == 0) or (not os.path.exists(args.out))
        df_tidy.to_csv(args.out, mode='a', header=write_header, index=False)

        total_wide += after_dedup
        total_tidy += n_tidy

        print(f"  [{i+1:02d}] {fname:<62} {after_dedup:>6}  {models_str:<30}  {n_params:>6}  {n_tidy:>6}{dedup_note}")

    print("-" * 120)
    print(f"\n{'TOTALE':<65} {total_wide:>6}  {'':30}  {'':>6}  {total_tidy:>6}")
    print(f"\nSalvato → {args.out}")

    df_final = pd.read_csv(args.out)
    n_total    = len(df_final)
    n_std_nan  = df_final["Metric std"].isna().sum()
    n_models   = df_final["Model name"].nunique()
    print(f"\nVerifica CSV finale:")
    print(f"  Righe totali  : {n_total}")
    print(f"  Modelli unici : {n_models} → {sorted(df_final['Model name'].unique())}")
    print(f"  Metric std NaN: {n_std_nan} (attesi: n_modelli × 6 dataset × 3 pert × 4 metriche = {n_models*6*3*4})")


if __name__ == "__main__":
    main()
