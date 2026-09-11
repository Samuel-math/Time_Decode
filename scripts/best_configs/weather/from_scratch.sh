#!/usr/bin/env bash
# Recovered from saved checkpoint args and run_weather_expert_pretrain.sh.
# Historical result reproduction is pending an actual complete rerun.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."
unset TD_ABLATION PRETRAINED_MODEL
export PYTHONHASHSEED=42 PYTHONUNBUFFERED=1 TD_CB_SAVE_START=2
h="${1:?forecast horizon required}"
case "$h" in
  96) ctx=96; batch=64; experts=4; dim=32;;
  192|720) ctx=672; batch=24; experts=3; dim=48;;
  336) ctx=672; batch=24; experts=4; dim=32;;
  *) echo "Invalid horizon: $h" >&2; exit 2;;
esac
mkdir -p scratch_runs/weather
run="$(mktemp -d "$PWD/scratch_runs/weather/h${h}_XXXXXXXX")"
exec > >(tee "$run/pipeline.log") 2>&1
printf 'RUN_DIR=%s\n' "$run"
cp "$SCRIPT_DIR/from_scratch.sh" "$SCRIPT_DIR/h$h.sh" "$run/"
python -m pip freeze > "$run/environment.freeze.txt"
if git rev-parse HEAD > "$run/source_commit.txt" 2>/dev/null; then
  git diff --binary > "$run/source_changes.patch"
fi
python vqvae-only/codebook_pretrain.py \
  --dset 'weather' \
  --context_points '672' \
  --target_points '0' \
  --batch_size '64' \
  --num_workers '0' \
  --scaler 'standard' \
  --features 'M' \
  --channel_indices '9,0,10,5,2,19,1,8,6,3,7,4,16,17,20,12,11,18,14,13,15' \
  --patch_size '8' \
  --embedding_dim '64' \
  --codebook_size '512' \
  --compression_factor '4' \
  --num_hiddens '128' \
  --num_residual_layers '2' \
  --num_residual_hiddens '128' \
  --vqvae_backbone 'mlp' \
  --vqvae_tcn_kernel_size '5' \
  --vqvae_chunk_size '2' \
  --decoder_lowpass '1' \
  --decoder_lowpass_kernel 'binomial3' \
  --commitment_cost '0.25' \
  --codebook_ema '1' \
  --ema_decay '0.95' \
  --ema_eps '0.00001' \
  --vq_init_method 'random' \
  --codebook_report_interval '5' \
  --seed '42' \
  --n_epochs '50' \
  --lr '0.0003' \
  --weight_decay '0.0001' \
  --revin '1' \
  --amp '1' \
  --vq_weight '1' \
  --recon_weight '1' \
  --train_sample_ratio '1' \
  --valid_sample_ratio '1' \
  --model_id '1' \
  --channel_group_id '0' \
  --per_channel_codebook '0' \
  --n_rq_layers '2' \
  --codebook_usage_threshold '0.8' \
  --sparse_weight '0.3' \
  --sparse_amplitude '0.05' \
  --lambda_ord '0.01' \
  --order_tau_f '1' \
  --order_eps '0.000001' \
  --layer1_smooth_weight '0' \
  --layer1_smooth_kernel '3' \
  --orth_weight '0.01' \
  --orth_start_epoch '0' \
  --orth_warmup_epochs '2' \
  --save_path "$run/codebook"
cb="$run/codebook/weather/codebook_ps8_cb512_cd128_rvq2_dlp_model1_grp0.pth"
test -f "$cb"
python decoder_only_NTP/patch_vqvae_pretrain.py \
  --dset 'weather' \
  --progressive_step_size '6' \
  --num_workers '0' \
  --scaler 'standard' \
  --features 'M' \
  --channel_indices '9,0,10,5,2,19,1,8,6,3,7,4,16,17,20,12,11,18,14,13,15' \
  --channel_group_id '0' \
  --patch_size '8' \
  --embedding_dim '64' \
  --compression_factor '4' \
  --codebook_size '512' \
  --n_layers '3' \
  --n_heads '4' \
  --d_ff '256' \
  --dropout '0.1' \
  --temporal_backbone 'timefilter_lite' \
  --timefilter_topk '8' \
  --timefilter_temperature '1' \
  --channel_summary_window '4' \
  --channel_summary_gate_init '-4' \
  --use_group_channel_experts '1' \
  --group_expert_dropout '0.1' \
  --group_expert_temperature '1' \
  --group_expert_topk '0' \
  --group_expert_gate_init '-2' \
  --commitment_cost '0.25' \
  --codebook_ema '1' \
  --disable_ema_update '1' \
  --ema_decay '0.95' \
  --ema_eps '0.00001' \
  --num_hiddens '128' \
  --num_residual_layers '2' \
  --num_residual_hiddens '128' \
  --vqvae_backbone 'mlp' \
  --vqvae_tcn_kernel_size '5' \
  --vqvae_chunk_size '2' \
  --decoder_lowpass '1' \
  --decoder_lowpass_kernel 'binomial3' \
  --freeze_vqvae '1' \
  --load_vq_weights '1' \
  --per_channel_codebook '0' \
  --n_rq_layers '2' \
  --rq_layer_weights '1' '1' \
  --soft_neighbor_k '20' \
  --soft_neighbor_alpha '0.3' \
  --soft_neighbor_tau '0.5' \
  --use_raw_input '0' \
  --pred_len '6' \
  --n_epochs '100' \
  --lr '0.0003' \
  --weight_decay '0.0001' \
  --seed '42' \
  --revin '1' \
  --vq_weight '0' \
  --recon_weight '0' \
  --early_stop_patience '5' \
  --early_stop_warmup '5' \
  --early_stop_min_delta '0.0001' \
  --early_stop_smooth_k '1' \
  --model_id '1' \
  --context_points "$ctx" --batch_size "$batch" \
  --group_expert_count "$experts" --group_expert_dim "$dim" \
  --vqvae_checkpoint "$cb" --save_path "$run/pretrain"
pre="$run/pretrain/weather/patch_vqvae_ps8_cb512_cd128_l3_in${ctx}_step6_model1_rvq2_dlp_timefilterlitek8_snk20a0p3t0p5_grp0.pth"
test -f "$pre"
PRETRAINED_MODEL="$pre" TD_FINETUNE_ROOT="$run/finetune" bash "$SCRIPT_DIR/h$h.sh"
printf 'COMPLETED_RUN=%s\n' "$run"
