#!/bin/bash
# =============================================================================
# run_all.sh — lancia 18 job × 2 env in parallelo su 4 GPU
#
# Environments:
#   new_env   → rf, tabicl, original_v2, xgb, tabdpt (cpu)
#   contextab → contextab
#
# Usage: bash run_all.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/test_perturbations.py"
LOG_DIR="$SCRIPT_DIR/logs_final"
SAVE_DIR="$SCRIPT_DIR/../benchmark_results_new_final"

mkdir -p "$LOG_DIR"
mkdir -p "$SAVE_DIR"

# Override these with env vars if needed, e.g.:
#   export PYTHON_NEW_ENV=/path/to/env/bin/python
#   export PYTHON_CONTEXTAB=/path/to/contextab/bin/python
PYTHON_NEW_ENV="${PYTHON_NEW_ENV:-python3}"
PYTHON_CONTEXTAB="${PYTHON_CONTEXTAB:-python3}"

DATASETS=(
    "abundance_cirrhosis--stagediscovery"
    "abundance_cirrhosis--stagevalidation"
    "abundance_obesity"
    "abundance_ibd"
    "abundance_t2d"
    "abundance_WT2D"
)
PERTS=("remove_features" "sparsity" "densification")

GPU=0

for ds in "${DATASETS[@]}"; do
    for pert in "${PERTS[@]}"; do

        # --- env 1: new_env → rf, tabicl, tabpfn, xgb, tabdpt ---
        log="${LOG_DIR}/${ds}__${pert}__new_env.log"
        echo "GPU $GPU -> [new_env]   $ds | $pert"
        CUDA_VISIBLE_DEVICES=$GPU \
            "$PYTHON_NEW_ENV" "$SCRIPT" \
            --dataset "$ds" --pert "$pert" \
            --models "rf,tabicl,original_v2,xgb,tabdpt" \
            --save_dir "$SAVE_DIR" \
            > "$log" 2>&1 &
        GPU=$(( (GPU + 1) % 4 ))
        sleep 2

        # --- env 2: contextab → contextab ---
        log="${LOG_DIR}/${ds}__${pert}__contextab.log"
        echo "GPU $GPU -> [contextab] $ds | $pert"
        CUDA_VISIBLE_DEVICES=$GPU \
            "$PYTHON_CONTEXTAB" "$SCRIPT" \
            --dataset "$ds" --pert "$pert" \
            --models "contextab" \
            --save_dir "$SAVE_DIR" \
            > "$log" 2>&1 &
        GPU=$(( (GPU + 1) % 4 ))
        sleep 2

    done
done

echo ""
echo "Tutti i job lanciati (18 dataset×pert × 2 env = 36 processi)"
echo "Monitora con:  tail -f logs/*.log"
echo "Controlla KO:  grep -rl 'Error\|Traceback' logs/"
wait
echo ""
echo "Done! Verifica risultati:"
echo "  python merge_tidy.py --results_dir $SAVE_DIR"
