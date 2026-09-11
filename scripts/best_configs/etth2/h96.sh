#!/usr/bin/env bash
# Historical fine-tuning configuration only. Full retraining provenance is pending.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../../.." && pwd)"

export PYTHONUNBUFFERED=1
root="${TD_FINETUNE_ROOT:-search_etth2_residual_only_96_r1}"
mkdir -p "$root"
pre="${PRETRAINED_MODEL:-search_etth2_cb128_96_r2/h/etth2/patch_vqvae_finetune_cw96_tw96_model1_grp0.pth}"
if [ ! -f "$pre" ]; then echo "Missing historical checkpoint: $pre" >&2; exit 2; fi
# name kernels gate lr loss mae_weight
specs=("k31224g3 3,12,24 -3 1e-3 mse .5")
for spec in "${specs[@]}"; do
 read -r name kernels gate lr loss mw <<<"$spec"
 python decoder_only_NTP/patch_vqvae_finetune.py \
  --dset etth2 --context_points 96 --target_points 96 --batch_size 32 --num_workers 0 \
  --scaler standard --features M --channel_indices 6,5,4,2,1,0,3 --channel_group_id 0 \
  --pretrained_model "$pre" --n_epochs 30 --lr "$lr" --weight_decay 1e-4 \
  --revin 1 --amp 1 --seed 42 --train_loss "$loss" --huber_delta .75 --mae_weight "$mw" \
  --use_gumbel_softmax 1 --gumbel_temperature .6 --gumbel_hard 0 \
  --ar_step_size 3 --pred_len 6 --use_group_channel_experts 0 \
  --use_multiscale_residual 1 --residual_only 1 --residual_mode forecast_smooth \
  --residual_context_len 96 --residual_kernels "$kernels" \
  --residual_gate_init "$gate" --residual_dropout 0 \
  --save_path "$root/$name" --model_id 1 > "$root/$name.log" 2>&1
done
wait
