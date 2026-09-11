#!/usr/bin/env bash
# Historical fine-tuning configuration only. Full retraining provenance is pending.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../../.." && pwd)"

export PYTHONUNBUFFERED=1
root=search_ettm2_full_r1
mkdir -p "$root"
pre_short=decoder_only_NTP/saved_models/patch_vqvae/ettm2_sota_tuned_20260901_051444/ettm2/patch_vqvae_ps8_cb256_cd128_l3_in336_step6_model1_rvq2_dlp_timefilterlitek4_grp0.pth
pre_long=decoder_only_NTP/saved_models/patch_vqvae/ettm2_sota_tuned_20260901_055045/ettm2/patch_vqvae_ps8_cb256_cd128_l3_in672_step6_model1_rvq2_grp0.pth
run_one() {
  local h=$1 delta=$2 lr=$3 step=$4 pred=$5 loss=$6 mw=$7 name=$8 pre=$9
  python decoder_only_NTP/patch_vqvae_finetune.py \
    --dset ettm2 --context_points 96 --target_points "$h" --batch_size 128 --num_workers 0 \
    --scaler standard --features M --channel_indices 2,5,6,4,0,3,1 --channel_group_id 0 \
    --pretrained_model "$pre" --n_epochs 50 --lr "$lr" --weight_decay 1e-4 \
    --revin 1 --amp 1 --seed 42 --train_loss "$loss" --huber_delta "$delta" --mae_weight "$mw" \
    --selection_metric score --use_gumbel_softmax 1 --gumbel_temperature .9 --gumbel_hard 0 \
    --ar_step_size "$step" --pred_len "$pred" --use_group_channel_experts 0 \
    --use_multiscale_residual 0 --save_path "$root/h${h}_${name}" --model_id 1 > "$root/h${h}_${name}.log" 2>&1
}
run_one 192 .4 2e-5   8  12 huber .5 s8 "$pre_short"
wait
