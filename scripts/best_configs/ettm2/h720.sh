#!/usr/bin/env bash
# Matched best fine-tuning stage; called with this run's pretrained checkpoint.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../../.." && pwd)"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
root="${TD_FINETUNE_ROOT:-search_ettm2_720_r5}"
mkdir -p "$root"
pre="${PRETRAINED_MODEL:-decoder_only_NTP/saved_models/patch_vqvae/ettm2_sota_tuned_20260901_055045/ettm2/patch_vqvae_ps8_cb256_cd128_l3_in672_step6_model1_rvq2_grp0.pth}"
if [ ! -f "$pre" ]; then echo "Missing historical checkpoint: $pre" >&2; exit 2; fi
run_one() {
 local ctx=$1 delta=$2 lr=$3 temp=$4 name=$5
 python decoder_only_NTP/patch_vqvae_finetune.py \
  --dset ettm2 --context_points "$ctx" --target_points 720 --batch_size 32 --num_workers 0 \
  --scaler standard --features M --channel_indices 2,5,6,4,0,3,1 --channel_group_id 0 \
  --pretrained_model "$pre" --n_epochs 50 --lr "$lr" --weight_decay 1e-4 --revin 1 --amp 1 \
  --seed 42 --train_loss huber --huber_delta "$delta" --selection_metric mse \
  --use_gumbel_softmax 1 --gumbel_temperature "$temp" --gumbel_hard 0 \
  --ar_step_size 4 --pred_len 8 --use_group_channel_experts 0 --use_multiscale_residual 0 \
  --save_path "$root/${name}" --model_id 1 > "$root/${name}.log" 2>&1
}
run_one 96  1.2 1e-5 .7 c96d12t7



wait
