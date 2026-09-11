#!/usr/bin/env bash
# Historical fine-tuning configuration only. Full retraining provenance is pending.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../../.." && pwd)"

export PYTHONUNBUFFERED=1
root=search_etth2_336_robust_r1
mkdir -p "$root"
pre="${PRETRAINED_MODEL:-search_etth2_expert_pretrain_r1/e3d32/etth2/patch_vqvae_ps8_cb256_cd128_l3_in296_step3_model1_rvq2_grp0.pth}"
if [ ! -f "$pre" ]; then echo "Missing historical checkpoint: $pre" >&2; exit 2; fi
for spec in "d5lr75s8p16 .5 7.5e-5 8 16"; do
  read -r name delta lr step pred <<<"$spec"
  python decoder_only_NTP/patch_vqvae_finetune.py \
    --dset etth2 --context_points 96 --target_points 336 --batch_size 32 --num_workers 0 \
    --scaler standard --features M --channel_indices 6,5,4,2,1,0,3 --channel_group_id 0 \
    --pretrained_model "$pre" --n_epochs 50 --lr "$lr" --weight_decay 1e-4 \
    --revin 1 --amp 1 --seed 42 --train_loss huber --huber_delta "$delta" \
    --selection_metric mae --use_gumbel_softmax 1 --gumbel_temperature .6 --gumbel_hard 0 \
    --ar_step_size "$step" --pred_len "$pred" --use_group_channel_experts 1 \
    --group_expert_count 3 --group_expert_dim 32 --group_expert_dropout .1 \
    --group_expert_temperature 1 --group_expert_topk 0 --group_expert_gate_init -2 \
    --use_multiscale_residual 0 --save_path "$root/$name" --model_id 1 > "$root/$name.log" 2>&1
done
wait
