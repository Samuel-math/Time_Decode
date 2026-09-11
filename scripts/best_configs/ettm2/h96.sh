#!/usr/bin/env bash
# Historical fine-tuning configuration only. Full retraining provenance is pending.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../../.." && pwd)"

export PYTHONUNBUFFERED=1
root=search_ettm2_fixed96_recovery_r16
mkdir -p "$root"
run_one(){
 local h=$1 tag=$2 experts=$3 dim=$4
 local pre=search_ettm2_expert_pretrain_fixed96_r15/$tag/ettm2/patch_vqvae_ps8_cb256_cd128_l3_in336_step6_model1_rvq2_dlp_timefilterlitek4_grp0.pth
 local step=6 pred=12 lr=1.02e-5 delta=.55; [ "$h" = 336 ] && step=10 && pred=10 && lr=1e-5 && delta=.074
 python decoder_only_NTP/patch_vqvae_finetune.py --dset ettm2 --context_points 96 --target_points "$h" --batch_size 128 --num_workers 0 \
  --scaler standard --features M --channel_indices 2,5,6,4,0,3,1 --channel_group_id 0 --pretrained_model "$pre" --n_epochs 50 \
  --lr "$lr" --weight_decay 1e-4 --revin 1 --amp 1 --seed 42 --train_loss huber --huber_delta "$delta" --selection_metric mse \
  --unfreeze_decoder 1 --decoder_lr_ratio .001 --decoder_wd_ratio 1 --use_gumbel_softmax 1 --gumbel_temperature 1 --gumbel_hard 0 \
  --ar_step_size "$step" --pred_len "$pred" --use_group_channel_experts 1 --group_expert_count "$experts" --group_expert_dim "$dim" \
  --group_expert_dropout .1 --group_expert_temperature 1 --group_expert_topk 0 --group_expert_gate_init -2 --use_multiscale_residual 0 \
  --save_path "$root/h${h}_${tag}" --model_id 1 > "$root/h${h}_${tag}.log" 2>&1
}
for spec in "e4d32 4 32"; do
 read -r tag experts dim <<<"$spec"
 run_one 96 "$tag" "$experts" "$dim"

done
wait
