#!/usr/bin/env bash
# Historical fine-tuning configuration only. Full retraining provenance is pending.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../../.." && pwd)"

export PYTHONUNBUFFERED=1
root=search_weather_sparse_experts_fixed96_r21
mkdir -p "$root"
pre="${PRETRAINED_MODEL:-search_weather_expert_pretrain_ctx672_r5/e4d32/weather/patch_vqvae_ps8_cb512_cd128_l3_in672_step6_model1_rvq2_dlp_timefilterlitek8_snk20a0p3t0p5_grp0.pth}"
if [ ! -f "$pre" ]; then echo "Missing historical checkpoint: $pre" >&2; exit 2; fi
channels=9,0,10,5,2,19,1,8,6,3,7,4,16,17,20,12,11,18,14,13,15
run_one(){
 local name=$1 target=$2 batch=$3 lr=$4 topk=$5 step=$6 pred=$7
 python decoder_only_NTP/patch_vqvae_finetune.py --dset weather --context_points 96 --target_points "$target" \
  --batch_size "$batch" --num_workers 0 --scaler standard --features M --channel_indices "$channels" --channel_group_id 0 \
  --pretrained_model "$pre" --n_epochs 50 --lr "$lr" --weight_decay 1e-4 --revin 1 --amp 1 --seed 42 \
  --train_loss mse --selection_metric mse --use_gumbel_softmax 1 --gumbel_temperature .9 --gumbel_hard 0 \
  --ar_step_size "$step" --pred_len "$pred" --use_group_channel_experts 1 --group_expert_count 4 \
  --group_expert_dim 32 --group_expert_dropout .1 --group_expert_temperature 1 --group_expert_topk "$topk" \
  --group_expert_gate_init -2 --use_multiscale_residual 0 --save_path "$root/$name" --model_id 1 > "$root/$name.log" 2>&1
}
run_one w336_top2 336 20 5e-5 2 8 12


wait
