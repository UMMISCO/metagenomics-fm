import os
import glob
import argparse
import pandas as pd

"""
Merges all parquet files produced by benchmarking.py into two big CSVs:
  - metrics_all.csv     : one row per (dataset, model, perturbation, param_value, fold)
  - predictions_all.csv : one row per (dataset, model, perturbation, param_value, fold, sample_idx)
"""

# =============================================================================
# CONFIG
# =============================================================================
import pathlib as _pl
RESULTS_DIR = str(_pl.Path(__file__).resolve().parents[1] / 'benchmark_results_new_final')
METRICS_OUT     = "metrics_all_final.csv"
PREDICTIONS_OUT = "predictions_all_final.csv"

MODEL_DISPLAY = {
    "rf":          "RF",
    "tabicl":      "TabICL",
    "original_v2": "TabPFN",
    "xgb":         "XGBoost",
    "tabdpt":      "TabDPT",
    "contextab":   "ContextTab",
}

# =============================================================================
# HELPERS
# =============================================================================

def load_parquets(results_dir: str, kind: str) -> pd.DataFrame:
    """
    kind = 'metrics'     → carica tutti i {pert_type}__{env}.parquet
    kind = 'predictions' → carica tutti i {pert_type}__predictions__{env}.parquet
    """
    if kind == 'metrics':
        pattern = os.path.join(results_dir, "**", "*.parquet")
        files   = [f for f in sorted(glob.glob(pattern, recursive=True))
                   if '__predictions__' not in os.path.basename(f)]
    else:
        pattern = os.path.join(results_dir, "**", "*__predictions__*.parquet")
        files   = sorted(glob.glob(pattern, recursive=True))

    if not files:
        print(f"[ERROR] Nessun file trovato per kind='{kind}' in: {results_dir}")
        return pd.DataFrame()

    dfs = []
    for fpath in files:
        fname = os.path.relpath(fpath, results_dir)
        try:
            df = pd.read_parquet(fpath)
            if df.empty:
                print(f"  [SKIP] {fname} — vuoto")
                continue
            dfs.append(df)
            print(f"  [OK]   {fname:<70}  {len(df):>6} righe")
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def remap_models(df: pd.DataFrame) -> pd.DataFrame:
    if 'model' in df.columns:
        df['model'] = df['model'].map(lambda x: MODEL_DISPLAY.get(x, x))
    return df


def dedup(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=keys)
    after = len(df)
    if before != after:
        print(f"  [INFO] Rimossi {before - after} duplicati")
    return df

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",     default=RESULTS_DIR)
    parser.add_argument("--metrics_out",     default=METRICS_OUT)
    parser.add_argument("--predictions_out", default=PREDICTIONS_OUT)
    args = parser.parse_args()

    # --- METRICS ---
    print(f"\n{'='*60}")
    print(f"Caricamento metriche da: {args.results_dir}")
    print(f"{'='*60}")
    metrics = load_parquets(args.results_dir, kind='metrics')

    if not metrics.empty:
        metrics = remap_models(metrics)
        metrics = dedup(metrics, keys=['dataset', 'model', 'perturbation', 'param_value', 'fold'])
        metrics.to_csv(args.metrics_out, index=False)
        print(f"\nSalvato → {args.metrics_out}  ({len(metrics)} righe)")
        print(f"  Modelli  : {sorted(metrics['model'].unique())}")
        print(f"  Dataset  : {sorted(metrics['dataset'].unique())}")
        print(f"  Fold unici: {sorted(metrics['fold'].unique())}")
    else:
        print("[ERROR] Nessuna metrica trovata.")

    # --- PREDICTIONS ---
    print(f"\n{'='*60}")
    print(f"Caricamento predizioni da: {args.results_dir}")
    print(f"{'='*60}")
    predictions = load_parquets(args.results_dir, kind='predictions')

    if not predictions.empty:
        predictions = remap_models(predictions)
        predictions = dedup(predictions, keys=['dataset', 'model', 'perturbation', 'param_value', 'fold', 'sample_idx'])
        predictions.to_csv(args.predictions_out, index=False)
        print(f"\nSalvato → {args.predictions_out}  ({len(predictions)} righe)")
        print(f"  Modelli  : {sorted(predictions['model'].unique())}")
        print(f"  Split    : {sorted(predictions['split'].unique())}")
    else:
        print("[ERROR] Nessuna predizione trovata.")


if __name__ == "__main__":
    main()
