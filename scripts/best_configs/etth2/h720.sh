#!/usr/bin/env bash
# Historical fine-tuning configuration only. Full retraining provenance is pending.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../../.." && pwd)"

export PYTHONUNBUFFERED=1
root="${TD_FINETUNE_ROOT:-search_etth2_mae_finish_r1}"; mkdir -p "$root"
common="--dset etth2 --context_points 96 --batch_size 32 --num_workers 0 --scaler standard --features M --channel_indices 6,5,4,2,1,0,3 --channel_group_id 0 --revin 1 --amp 1 --seed 42 --weight_decay 1e-4 --use_gumbel_softmax 1 --gumbel_temperature .6 --gumbel_hard 0 --selection_metric score --model_id 1"
pexp=search_etth2_expert_pretrain_r1/e3d32/etth2/patch_vqvae_ps8_cb256_cd128_l3_in296_step3_model1_rvq2_grp0.pth
pbase="${PRETRAINED_MODEL:-decoder_only_NTP/saved_models/patch_vqvae/etth2_freq_m7_base1_20260831_011724/etth2/patch_vqvae_ps8_cb256_cd128_l3_in296_step3_model1_rvq2_grp0.pth}"
test -f "$pbase"
for spec in "d5lr5p6 .5 5e-5 6"; do
 read -r name delta lr pred <<<"$spec"
 python decoder_only_NTP/patch_vqvae_finetune.py $common --target_points 720 --pretrained_model "$pbase" \
  --n_epochs 50 --lr "$lr" --train_loss huber --huber_delta "$delta" --ar_step_size 4 --pred_len "$pred" \
  --use_group_channel_experts 0 --use_multiscale_residual 0 \
  --save_path "$root/h720_$name" > "$root/h720_$name.log" 2>&1
done
wait
