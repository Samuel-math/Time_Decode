#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Time_Decode
export PATH=/root/miniconda3/envs/time_decode/bin:/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
root=search_electricity_720_fixed96_r4
mkdir -p "$root"
pre=decoder_only_NTP/saved_models/patch_vqvae/electricity_freq_m321_base1_20260831_020025/electricity/patch_vqvae_ps4_cb256_cd128_l3_in128_step6_model1_rvq2_timefilterlitek128_snk20a0p5t1p0_grp0.pth
channels=$(python -c 'import json; d=json.load(open("scripts/channel_groups/electricity_freq321.json")); g=d["groups"][0] if isinstance(d,dict) else d[0]; print(",".join(map(str,g["channel_indices"] if isinstance(g,dict) else g)))')
python -u decoder_only_NTP/patch_vqvae_finetune.py \
 --dset electricity --context_points 96 --target_points 720 --batch_size 8 --num_workers 0 \
 --scaler standard --features M --channel_indices "$channels" --channel_group_id 0 --pretrained_model "$pre" \
 --n_epochs 50 --lr 2e-4 --weight_decay 1e-4 --revin 1 --amp 1 --seed 42 \
 --train_loss huber --huber_delta 1.5 --selection_metric mse \
 --use_gumbel_softmax 1 --gumbel_temperature .8 --gumbel_hard 0 --ar_step_size 24 --pred_len 28 \
 --use_group_channel_experts 1 --group_expert_count 4 --group_expert_dim 32 --group_expert_dropout .1 \
 --group_expert_temperature 1 --group_expert_topk 0 --group_expert_gate_init -2 --use_multiscale_residual 0 \
 --save_path "$root" --model_id 1 2>&1 | tee "$root/train.log"
