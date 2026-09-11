#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/Time_Decode
export PATH=/root/miniconda3/envs/time_decode/bin:/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONUNBUFFERED=1
root=search_ettm2_expert_pretrain_fixed96_r15
mkdir -p "$root"
vq=vqvae-only/saved_models/vqvae_only/ettm2_sota_tuned_20260901_051444/ettm2/codebook_ps8_cb256_cd128_rvq2_dlp_model1_grp0.pth
for spec in "e2d32 2 32" "e4d32 4 32" "e3d48 3 48"; do
 read -r name experts dim <<<"$spec"
 python decoder_only_NTP/patch_vqvae_pretrain.py \
  --dset ettm2 --context_points 336 --progressive_step_size 6 --batch_size 128 --num_workers 0 \
  --scaler standard --features M --channel_indices 2,5,6,4,0,3,1 --channel_group_id 0 \
  --patch_size 8 --embedding_dim 64 --compression_factor 4 --codebook_size 256 \
  --n_layers 3 --n_heads 4 --d_ff 128 --dropout .15 --temporal_backbone timefilter_lite \
  --timefilter_topk 4 --timefilter_temperature 1 --decoder_lowpass 1 --decoder_lowpass_kernel binomial3 \
  --commitment_cost .25 --codebook_ema 1 --disable_ema_update 1 --ema_decay .95 --ema_eps 1e-5 \
  --num_hiddens 128 --num_residual_layers 2 --num_residual_hiddens 128 --vqvae_backbone mlp --vqvae_tcn_kernel_size 5 \
  --vqvae_checkpoint "$vq" --freeze_vqvae 1 --load_vq_weights 1 --per_channel_codebook 0 \
  --n_rq_layers 2 --rq_layer_weights 1 1 --use_raw_input 0 --pred_len 6 \
  --n_epochs 100 --lr 3e-4 --weight_decay 1e-4 --seed 42 --revin 1 --vq_weight 0 --recon_weight 0 \
  --early_stop_patience 5 --early_stop_warmup 5 --early_stop_min_delta 1e-4 \
  --use_group_channel_experts 1 --group_expert_count "$experts" --group_expert_dim "$dim" \
  --group_expert_dropout .1 --group_expert_temperature 1 --group_expert_topk 0 --group_expert_gate_init -2 \
  --save_path "$root/$name" --model_id 1 > "$root/$name.log" 2>&1 &
done
wait
