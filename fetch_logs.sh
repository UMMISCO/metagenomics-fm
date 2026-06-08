#!/bin/zsh

# Usage:
#   ./fetch_logs.sh logs_folder [--live]

if [ -z "$1" ]; then
  echo "Missing logs source; usage: ./fetch_logs.sh logs_folder [--live]"
  exit 1
fi

LIVE_MODE=false
if [ "$2" = "--live" ]; then
  LIVE_MODE=true
fi

REMOTE_PATH="uzt44fk@jean-zay.idris.fr:/lustre/fswork/projects/rech/hyd/uzt44fk/Tab_ICL/tabicl/src/tabicl_original/checkpoints/checkpoints_tabicl_similarity"
#$1
LOCAL_PATH="/Users/giuliaperciballi/Documents/Lab/Predomics/Tab_ICL/tabicl/src/tabicl_original/checkpoints"

function sync_logs() {
  echo "Syncing logs from remote at $(date)..."
  rsync -hPzra --delete --partial "$REMOTE_PATH" "$LOCAL_PATH"

  echo "Syncing WandB runs..."
  find "$LOCAL_PATH" -type l -name "latest-run" | while read -r symlink_path; do
    dir_path="$(dirname "$symlink_path")"
    echo "Syncing in $dir_path"
    (cd "$dir_path" && wandb sync latest-run)
  done
  echo "Sync complete at $(date)"
}

# Do the first sync
sync_logs

if $LIVE_MODE; then
  echo "Live mode enabled: periodically syncing from remote every 60s."
  while true; do
    sleep 60
    sync_logs
  done
fi


# FLuctuations
/pools/apollon/ummisco/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src/tabicl_original/train/run.py --wandb_log True --wandb_project TabICL --wandb_name Stage1 --wandb_dir tabicl/checkpoints/checkpoints_tabpfn_prior/wandb --wandb_mode online --device cuda --dtype float32 --np_seed 42 --torch_seed 42 --max_steps 100000 --batch_size 512 --micro_batch_size 4 --lr 1e-4 --scheduler cosine_warmup --warmup_proportion 0.02 --gradient_clipping 1.0 --prior_type mix_scm --prior_device cuda --batch_size_per_gp 4 --min_features 2 --max_features 100 --max_classes 10 --max_seq_len 1024 --min_train_size 0.1 --max_train_size 0.9 --embed_dim 128 --col_num_blocks 3 --col_nhead 4 --col_num_inds 128 --row_num_blocks 3 --row_nhead 8 --row_num_cls 4 --row_rope_base 100000 --icl_num_blocks 12 --icl_nhead 4 --ff_factor 2 --norm_first True --checkpoint_dir tabicl/checkpoints/checkpoints_tabpfn_prior --save_temp_every 50 --save_perm_every 5000

#Smooth
