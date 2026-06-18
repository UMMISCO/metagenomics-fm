#!/bin/bash
# =============================================================================
# run_all.sh — launches 18 jobs in parallel on 4 GPU
#
# Usage: bash run_all.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/test_perturbations.py"
LOG_DIR="$SCRIPT_DIR/logs_final"
SAVE_DIR="$SCRIPT_DIR/../benchmark_results_new_final"

mkdir -p "$LOG_DIR"
mkdir -p "$SAVE_DIR"

DATASETS=(
    "abundance_cirrhosis--stagediscovery"
    "abundance_cirrhosis--stagevalidation"
    "abundance_obesity"
    "abundance_ibd"
    "abundance_t2d"
    "abundance_WT2D"
)
PERTS=("remove_features" "zero_inflation" "zero_imputation")

GPU=0

for ds in "${DATASETS[@]}"; do
    for pert in "${PERTS[@]}"; do

        log="${LOG_DIR}/${ds}__${pert}.log"
        echo "GPU $GPU -> $ds | $pert"
        CUDA_VISIBLE_DEVICES=$GPU \
            python "$SCRIPT" \
            --dataset "$ds" --pert "$pert" \
            --models "rf,tabicl,original_v2,xgb,tabdpt,contextab" \
            --save_dir "$SAVE_DIR" \
            > "$log" 2>&1 &
        GPU=$(( (GPU + 1) % 4 ))
        sleep 2

    done
done

echo ""
echo "All jobs launched (6 datasets × 3 perts = 18 processes)"
echo "Monitor with:  tail -f logs_final/*.log"
echo "Check for failures:  grep -rl 'Error\|Traceback' logs_final/"
wait
echo ""
echo "Done! Check results:"
echo "  python merge_tidy.py --results_dir $SAVE_DIR"