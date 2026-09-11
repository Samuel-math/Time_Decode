#!/usr/bin/env bash
# Matched historical best fine-tuning chains. See PROVENANCE.md.
# Numeric reproduction must be established by rerunning, not by this label.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."
unset TD_ABLATION PRETRAINED_MODEL
export PYTHONHASHSEED=42 PYTHONUNBUFFERED=1 TD_CB_SAVE_START=2
h="${1:?forecast horizon required}"
case "$h" in
  96|336) ctx=336; batch=128; expert=1; ff=128; dropout=.15; backbone=timefilter_lite; lowpass=1; suffix=_dlp_timefilterlitek4;;
  192) ctx=336; batch=64; expert=0; ff=128; dropout=.15; backbone=timefilter_lite; lowpass=1; suffix=_dlp_timefilterlitek4;;
  720) ctx=672; batch=64; expert=0; ff=336; dropout=.1; backbone=causal_transformer; lowpass=0; suffix=;;
  *) echo "Invalid horizon: $h" >&2; exit 2;;
esac
mkdir -p scratch_runs/ettm2
run="$(mktemp -d "$PWD/scratch_runs/ettm2/h${h}_XXXXXXXX")"
exec > >(tee "$run/pipeline.log") 2>&1
printf 'RUN_DIR=%s\n' "$run"
cp "$SCRIPT_DIR/from_scratch.sh" "$SCRIPT_DIR/h$h.sh" "$run/"
python -m pip freeze > "$run/environment.freeze.txt"
if git rev-parse HEAD > "$run/source_commit.txt" 2>/dev/null; then
  git diff --binary > "$run/source_changes.patch"
fi
python vqvae-only/codebook_pretrain.py \
  --dset 'ettm2' \
  --context_points '512' \
  --batch_size '64' \
  --num_workers '0' \
  --scaler 'standard' \
  --features 'M' \
  --patch_size '8' \
  --embedding_dim '64' \
  --compression_factor '4' \
  --codebook_size '256' \
  --num_hiddens '128' \
  --num_residual_layers '2' \
  --num_residual_hiddens '128' \
  --vqvae_backbone 'mlp' \
  --vqvae_tcn_kernel_size '5' \
  --vqvae_chunk_size '2' \
  --codebook_ema '1' \
  --ema_decay '0.95' \
  --n_epochs '50' \
  --lr '3e-4' \
  --weight_decay '1e-4' \
  --revin '1' \
  --per_channel_codebook '0' \
  --n_rq_layers '2' \
  --sparse_weight '0.3' \
  --sparse_amplitude '0.05' \
  --lambda_ord '0.01' \
  --orth_weight '0.01' \
  --orth_start_epoch '0' \
  --orth_warmup_epochs '2' \
  --channel_indices '2,5,6,4,0,3,1' \
  --channel_group_id '0' \
  --model_id '1' \
  --seed 42 --decoder_lowpass "$lowpass" --save_path "$run/codebook"
cb_suffix=""
if [ "$lowpass" = 1 ]; then cb_suffix=_dlp; fi
cb="$run/codebook/ettm2/codebook_ps8_cb256_cd128_rvq2${cb_suffix}_model1_grp0.pth"
test -f "$cb"
python decoder_only_NTP/patch_vqvae_pretrain.py \
  --dset 'ettm2' \
  --progressive_step_size '6' \
  --num_workers '0' \
  --scaler 'standard' \
  --features 'M' \
  --channel_indices '2,5,6,4,0,3,1' \
  --channel_group_id '0' \
  --patch_size '8' \
  --embedding_dim '64' \
  --compression_factor '4' \
  --codebook_size '256' \
  --n_layers '3' \
  --n_heads '4' \
  --timefilter_topk '4' \
  --timefilter_temperature '1' \
  --decoder_lowpass_kernel 'binomial3' \
  --commitment_cost '.25' \
  --codebook_ema '1' \
  --disable_ema_update '1' \
  --ema_decay '.95' \
  --ema_eps '1e-5' \
  --num_hiddens '128' \
  --num_residual_layers '2' \
  --num_residual_hiddens '128' \
  --vqvae_backbone 'mlp' \
  --vqvae_tcn_kernel_size '5' \
  --freeze_vqvae '1' \
  --load_vq_weights '1' \
  --per_channel_codebook '0' \
  --n_rq_layers '2' \
  --rq_layer_weights '1' '1' \
  --use_raw_input '0' \
  --pred_len '6' \
  --n_epochs '100' \
  --lr '3e-4' \
  --weight_decay '1e-4' \
  --seed '42' \
  --revin '1' \
  --vq_weight '0' \
  --recon_weight '0' \
  --early_stop_patience '5' \
  --early_stop_warmup '5' \
  --early_stop_min_delta '1e-4' \
  --group_expert_count '4' \
  --group_expert_dim '32' \
  --group_expert_dropout '.1' \
  --group_expert_temperature '1' \
  --group_expert_topk '0' \
  --group_expert_gate_init '-2' \
  --model_id '1' \
  --context_points "$ctx" --batch_size "$batch" --use_group_channel_experts "$expert" \
  --d_ff "$ff" --dropout "$dropout" --temporal_backbone "$backbone" --decoder_lowpass "$lowpass" \
  --soft_neighbor_k 0 --early_stop_smooth_k 1 \
  --vqvae_checkpoint "$cb" --save_path "$run/pretrain"
pre="$run/pretrain/ettm2/patch_vqvae_ps8_cb256_cd128_l3_in${ctx}_step6_model1_rvq2${suffix}_grp0.pth"
test -f "$pre"
PRETRAINED_MODEL="$pre" TD_FINETUNE_ROOT="$run/finetune" bash "$SCRIPT_DIR/h$h.sh"
printf 'COMPLETED_RUN=%s\n' "$run"
