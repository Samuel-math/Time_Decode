#!/usr/bin/env bash
# Historical fine-tuning configuration only. Full retraining provenance is pending.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../../.." && pwd)"

export PYTHONUNBUFFERED=1
root="${TD_FINETUNE_ROOT:-search_etth2_expert_finetune_r1}"
mkdir -p "$root"
pre="${PRETRAINED_MODEL:-search_etth2_expert_pretrain_r1/e3d32/etth2/patch_vqvae_ps8_cb256_cd128_l3_in296_step3_model1_rvq2_grp0.pth}"
if [ ! -f "$pre" ]; then echo "Missing historical checkpoint: $pre" >&2; exit 2; fi
run_one() {
 local h=$1 step=$2 pred=$3 lr=$4 delta=$5
 python decoder_only_NTP/patch_vqvae_finetune.py \
  --dset etth2 --context_points 96 --target_points "$h" --batch_size 32 --num_workers 0 \
  --scaler standard --features M --channel_indices 6,5,4,2,1,0,3 --channel_group_id 0 \
  --pretrained_model "$pre" --n_epochs 50 --lr "$lr" --weight_decay 1e-4 \
  --revin 1 --amp 1 --seed 42 --train_loss huber --huber_delta "$delta" \
  --use_gumbel_softmax 1 --gumbel_temperature .6 --gumbel_hard 0 \
  --ar_step_size "$step" --pred_len "$pred" \
  --use_group_channel_experts 1 --group_expert_count 3 --group_expert_dim 32 \
  --group_expert_dropout .1 --group_expert_temperature 1 --group_expert_topk 0 --group_expert_gate_init -2 \
  --use_multiscale_residual 0 --save_path "$root/h$h" --model_id 1 > "$root/h$h.log" 2>&1
}

run_one 192 6 12 7.5e-5 .75


wait
