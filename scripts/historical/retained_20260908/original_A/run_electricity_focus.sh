#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/Time_Decode
export PATH=/root/miniconda3/envs/time_decode/bin:/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONUNBUFFERED=1
root=search_electricity_context_r1
mkdir -p "$root"
pre=decoder_only_NTP/saved_models/patch_vqvae/electricity_freq_m321_base1_20260829_021235/electricity/patch_vqvae_ps4_cb256_cd128_l3_in128_step6_model1_rvq2_timefilterlitek128_snk20a0p5t1p0_grp0.pth
channels=$(python -c 'import json; print(",".join(map(str,json.load(open("scripts/channel_groups/electricity_freq321.json"))["groups"][0])))')
run_one() {
 local h=$1 ctx=$2 delta=$3 lr=$4 step=$5 pred=$6 batch=$7 name=$8
 python decoder_only_NTP/patch_vqvae_finetune.py \
  --dset electricity --context_points "$ctx" --target_points "$h" --batch_size "$batch" --num_workers 0 \
  --scaler standard --features M --channel_indices "$channels" --channel_group_id 0 \
  --pretrained_model "$pre" --n_epochs 50 --lr "$lr" --weight_decay 1e-4 --revin 1 --amp 1 \
  --seed 42 --train_loss huber --huber_delta "$delta" --selection_metric mse \
  --use_gumbel_softmax 1 --gumbel_temperature .8 --gumbel_hard 0 \
  --ar_step_size "$step" --pred_len "$pred" --use_group_channel_experts 0 \
  --use_multiscale_residual 0 --save_path "$root/$name" --model_id 1 > "$root/$name.log" 2>&1
}
# Sequential to keep the 321-channel runs within GPU memory.
run_one 96  192 1.0 2e-4 8  10 16 h96_c192
run_one 336 192 1.5 2e-4 20 24 8  h336_c192
run_one 720 192 2.0 2e-4 24 28 4  h720_c192
run_one 720 336 2.0 1e-4 24 28 2  h720_c336
