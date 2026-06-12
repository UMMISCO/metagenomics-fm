"""
make_latex_tables.py
Upload the 18 CSV and generates Latex tables.

"""

import os
import glob
import argparse
import pandas as pd
import numpy as np

import pathlib as _pl
RESULTS_DIR = str(_pl.Path(__file__).resolve().parent / 'benchmark_results')
OUTPUT_FILE = 'latex_tables.tex'

MODEL_DISPLAY = {
    'rf':         'RF',
    'tabicl':     'TabICL',
    'original_v2': 'TabPFN',
}

METRICS = [
    ('auroc_mean', 'auroc_std', 'AUROC'),
    ('f1_mean',    'f1_std',    'F1'),
    ('prec_mean',  'prec_std',  'Precision'),
    ('rec_mean',   'rec_std',   'Recall'),
]

BASELINE_COLS = {
    'auroc_mean': 'baseline_auroc',
    'f1_mean':    'baseline_f1',
    'prec_mean':  'baseline_prec',
    'rec_mean':   'baseline_rec',
}


def fmt(mean, std=None):
    """Format mean +- std for LaTeX."""
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return '--'
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return f'{float(mean):.3f}'
    return f'{float(mean):.3f} \\pm {float(std):.3f}'


def make_table(df, dataset, pert_type):
    ds_label = dataset.replace('abundance_', '').replace('--', '-')
    pert_label = pert_type.replace('_', ' ')

    models = [m for m in MODEL_DISPLAY if m in df['model'].unique()]
    param_values = list(dict.fromkeys(df['param_value'].tolist()))  # preserve order, deduplicate

    n_metric_cols = len(METRICS)
    col_spec = 'l|l|' + 'c' * n_metric_cols

    lines = []
    lines.append(f'\\begin{{table}}[h]')
    lines.append(f'\\caption{{\\textbf{{{ds_label} --- {pert_label} perturbation}}. '
                 f'Mean $\\pm$ std over 5-fold CV. '
                 f'\\textit{{Baseline}}: model trained and tested on original data. '
                 f'Each block shows all models at one perturbation level.}}')
    lines.append(f'\\centering')
    lines.append(f'\\small')
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append(f'\\toprule')

    metric_headers = ' & '.join(m[2] for m in METRICS)
    lines.append(f'Model & Param & {metric_headers} \\\\')
    lines.append(f'\\midrule')

    # --- Baseline block ---
    lines.append(f'\\multicolumn{{{2 + n_metric_cols}}}{{l}}{{\\textit{{Baseline (original data)}}}} \\\\')
    lines.append(f'\\midrule')

    for model in models:
        row = df[df['model'] == model].iloc[0]
        vals = []
        for mean_col, std_col, _ in METRICS:
            base_col = BASELINE_COLS[mean_col]
            v = row.get(base_col, None)
            vals.append(f'${fmt(v)}$' if v is not None else '--')
        model_label = MODEL_DISPLAY.get(model, model)
        lines.append(f'{model_label} & original & {" & ".join(vals)} \\\\')

    # --- Perturbation blocks ---
    for pv in param_values:
        lines.append(f'\\midrule')
        sub = df[df['param_value'] == pv]
        for model in models:
            row = sub[sub['model'] == model]
            if row.empty:
                continue
            row = row.iloc[0]
            vals = [f'${fmt(row[mc], row[sc])}$' for mc, sc, _ in METRICS]
            pv_clean = str(pv).replace('_', '\\_')
            model_label = MODEL_DISPLAY.get(model, model)
            lines.append(f'{model_label} & {pv_clean} & {" & ".join(vals)} \\\\')

    lines.append(f'\\bottomrule')
    lines.append(f'\\end{{tabular}}')
    label = f'{ds_label.replace("-", "_")}_{pert_type}'
    lines.append(f'\\label{{tab:{label}}}')
    lines.append(f'\\end{{table}}')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR)
    parser.add_argument('--out', type=str, default=OUTPUT_FILE)
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.results_dir, '*', '*.csv')))
    print(f'Found {len(csv_files)} CSV files')

    if not csv_files:
        print(f'No CSV files found in {args.results_dir}')
        return

    all_tables = []
    for fpath in csv_files:
        df = pd.read_csv(fpath)
        if df.empty:
            print(f'  [SKIP] Empty: {fpath}')
            continue

        dataset  = df['dataset'].iloc[0]
        pert_type = df['perturbation'].iloc[0]
        ds_short = dataset.replace('abundance_', '').replace('--', '-')
        print(f'  Generating table: {ds_short} / {pert_type}  ({len(df)} rows)')

        table = make_table(df, dataset, pert_type)
        all_tables.append(f'% ===== {ds_short} | {pert_type} =====\n{table}')

    output = '\n\n'.join(all_tables)
    with open(args.out, 'w') as f:
        f.write(output)

    print(f'\nSaved {len(all_tables)} tables to: {args.out}')


if __name__ == '__main__':
    main()