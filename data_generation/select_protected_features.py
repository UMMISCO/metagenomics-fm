"""
select_protected_features.py

For each dataset, determines how many top-ranked features (by ANOVA F-score)
must be protected from perturbation so that removing them degrades CV AUROC by
at least THRESHOLD.

Algorithm
---------
1. Rank all features by ANOVA F-score (descending).
2. Compute baseline AUROC with all features (RF, 5-fold CV).
3. Iteratively remove the top-k ranked features and re-evaluate.
4. The first k at which AUROC drops by >= THRESHOLD defines the boundary;
   n_features_protect = k - 1.

Output
------
ablation_protected.csv   (written to data_transformations/)
    dataset              : dataset name
    auroc_baseline       : CV AUROC with all features
    first_k_degraded     : first k at which degradation >= THRESHOLD
    n_features_protect   : features to protect (first_k_degraded - 1)

Run
---
    python data_generation/select_protected_features.py
"""

import sys
import pathlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_generation.data_generator import DataGenerator

# ── CONFIG ────────────────────────────────────────────────────────────────────

DATASETS = [
    'abundance_cirrhosis--stagediscovery',
    'abundance_cirrhosis--stagevalidation',
    'abundance_obesity',
    'abundance_ibd',
    'abundance_t2d',
    'abundance_WT2D',
]

CV        = 5
SEED      = 42
THRESHOLD = 0.03   # minimum AUROC drop that triggers protection

_DT_DIR    = pathlib.Path(__file__).resolve().parents[1]   # data_transformations/
OUTPUT_CSV = _DT_DIR / 'ablation_protected.csv'

# ── HELPERS ───────────────────────────────────────────────────────────────────

def anova_ranking(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    selector = SelectKBest(f_classif, k='all')
    selector.fit(X, y)
    return np.argsort(np.nan_to_num(selector.scores_))[::-1]


def cv_auroc(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    skf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=SEED)
    scores = [
        roc_auc_score(
            y[test_idx],
            RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
            .fit(X[train_idx], y[train_idx])
            .predict_proba(X[test_idx])[:, 1],
        )
        for train_idx, test_idx in skf.split(X, y)
    ]
    return float(np.mean(scores)), float(np.std(scores))


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    rows = []

    for dataset in DATASETS:
        print(f"\n{'=' * 52}")
        print(f"  {dataset}")
        print(f"{'=' * 52}")

        gen = DataGenerator(generator_type='zero_imputation', data_source='pasolli')
        gen.load_data(dataset)
        X = gen.X_original.values
        y = gen.y_original.values
        print(f"  shape: {X.shape}")

        ranked_idx        = anova_ranking(X, y)
        auroc_baseline, _ = cv_auroc(X, y)

        print(f"  baseline AUROC : {auroc_baseline * 100:.1f}%")
        print(f"  threshold      : {(auroc_baseline - THRESHOLD) * 100:.1f}%  "
              f"(baseline - {THRESHOLD * 100:.0f}%)")

        n_protect = 0
        first_k   = X.shape[1]

        for k in range(1, X.shape[1] + 1):
            mask = np.ones(X.shape[1], dtype=bool)
            mask[ranked_idx[:k]] = False
            mean_auc, std_auc = cv_auroc(X[:, mask], y)
            print(f"  k={k:3d} removed  AUROC = {mean_auc * 100:.1f} ± {std_auc * 100:.1f}")

            if (auroc_baseline - mean_auc) >= THRESHOLD:
                n_protect = max(0, k - 1)
                first_k   = k
                print(f"\n  degradation >= {THRESHOLD * 100:.0f}% at k={k}")
                print(f"  -> n_features_protect = {n_protect}")
                break

        rows.append({
            'dataset':            dataset,
            'auroc_baseline':     round(auroc_baseline, 4),
            'first_k_degraded':   first_k,
            'n_features_protect': n_protect,
        })

    protect_df = pd.DataFrame(rows)
    protect_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved -> {OUTPUT_CSV}")
    print(protect_df.to_string(index=False))
    return protect_df


if __name__ == '__main__':
    run()
