#!/usr/bin/env bash
# Recovered historical codebook args and archived predictive-pretraining recipe.
# Candidate until complete from-scratch test comparison passes.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."
unset TD_ABLATION PRETRAINED_MODEL
export PYTHONHASHSEED=42 PYTHONUNBUFFERED=1 TD_CB_SAVE_START=5
h="${1:?forecast horizon required}"
case "$h" in
  96|720) expert=0;;
  192|336) expert=1;;
  *) echo "Invalid horizon: $h" >&2; exit 2;;
esac
mkdir -p scratch_runs/etth2
run="$(mktemp -d "$PWD/scratch_runs/etth2/h${h}_XXXXXXXX")"
exec > >(tee "$run/pipeline.log") 2>&1
printf 'RUN_DIR=%s\n' "$run"
cp "$SCRIPT_DIR/from_scratch.sh" "$SCRIPT_DIR/h$h.sh" "$run/"
python -m pip freeze > "$run/environment.freeze.txt"
if git rev-parse HEAD > "$run/source_commit.txt" 2>/dev/null; then
  git diff --binary > "$run/source_changes.patch"
fi
python vqvae-only/codebook_pretrain.py \
  --dset 'etth2' \
  --context_points '128' \
  --target_points '0' \
  --batch_size '64' \
  --num_workers '0' \
  --scaler 'standard' \
  --features 'M' \
  --channel_indices '6,5,4,2,1,0,3' \
  --patch_size '8' \
  --embedding_dim '64' \
  --codebook_size '256' \
  --compression_factor '4' \
  --num_hiddens '128' \
  --num_residual_layers '2' \
  --num_residual_hiddens '128' \
  --vqvae_backbone 'mlp' \
  --vqvae_tcn_kernel_size '5' \
  --vqvae_chunk_size '2' \
  --decoder_lowpass '0' \
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
  --orth_warmup_epochs '5' \
  --save_path "$run/codebook"
cb="$run/codebook/etth2/codebook_ps8_cb256_cd128_rvq2_model1_grp0.pth"
test -f "$cb"
python decoder_only_NTP/patch_vqvae_pretrain.py \
  --dset 'etth2' \
  --context_points '296' \
  --progressive_step_size '3' \
  --batch_size '64' \
  --num_workers '0' \
  --scaler 'standard' \
  --features 'M' \
  --channel_indices '6,5,4,2,1,0,3' \
  --channel_group_id '0' \
  --patch_size '8' \
  --embedding_dim '64' \
  --compression_factor '4' \
  --codebook_size '256' \
  --n_layers '3' \
  --n_heads '4' \
  --d_ff '256' \
  --dropout '0.1' \
  --temporal_backbone 'causal_transformer' \
  --timefilter_topk '8' \
  --timefilter_temperature '1' \
  --channel_summary_window '4' \
  --channel_summary_gate_init '-4' \
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
  --decoder_lowpass '0' \
  --decoder_lowpass_kernel 'binomial3' \
  --freeze_vqvae '1' \
  --load_vq_weights '1' \
  --per_channel_codebook '0' \
  --n_rq_layers '2' \
  --rq_layer_weights '1' '1' \
  --soft_neighbor_k '0' \
  --soft_neighbor_alpha '0.25' \
  --soft_neighbor_tau '0.3' \
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
  --use_group_channel_experts "$expert" --group_expert_count 3 --group_expert_dim 32 \
  --group_expert_dropout .1 --group_expert_temperature 1 --group_expert_topk 0 --group_expert_gate_init -2 \
  --vqvae_checkpoint "$cb" --save_path "$run/pretrain"
pre="$run/pretrain/etth2/patch_vqvae_ps8_cb256_cd128_l3_in296_step3_model1_rvq2_grp0.pth"
test -f "$pre"
if [ "$h" = 96 ]; then
  python decoder_only_NTP/patch_vqvae_finetune.py \
  --dset 'etth2' \
  --context_points '96' \
  --target_points '96' \
  --batch_size '32' \
  --num_workers '0' \
  --scaler 'standard' \
  --features 'M' \
  --channel_indices '6,5,4,2,1,0,3' \
  --channel_group_id '0' \
  --n_epochs '50' \
  --lr '0.000075' \
  --weight_decay '0.0001' \
  --revin '1' \
  --amp '1' \
  --seed '42' \
  --train_loss 'huber' \
  --huber_delta '0.75' \
  --unfreeze_decoder '0' \
  --decoder_lr_ratio '0.02' \
  --decoder_wd_ratio '10' \
  --use_gumbel_softmax '1' \
  --gumbel_temperature '0.6' \
  --gumbel_hard '0' \
  --ar_step_size '3' \
  --pred_len '6' \
  --model_id '1' \
  --pretrained_model "$pre" --save_path "$run/base_finetune"
  pre="$run/base_finetune/etth2/patch_vqvae_finetune_cw96_tw96_model1_grp0.pth"
  test -f "$pre"
fi
PRETRAINED_MODEL="$pre" TD_FINETUNE_ROOT="$run/finetune" bash "$SCRIPT_DIR/h$h.sh"
printf 'COMPLETED_RUN=%s\n' "$run"

