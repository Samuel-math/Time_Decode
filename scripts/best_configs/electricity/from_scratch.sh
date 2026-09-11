#!/usr/bin/env bash
# Recovered historical checkpoint args; numeric reproduction pending.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."
unset TD_ABLATION PRETRAINED_MODEL
export PYTHONHASHSEED=42 PYTHONUNBUFFERED=1 TD_CB_SAVE_START=2
h="${1:?forecast horizon required}"
case "$h" in
  192) cb_context=512; cb_lr=0.001;;
  96|336|720) cb_context=128; cb_lr=0.0003;;
  *) echo "Invalid horizon: $h" >&2; exit 2;;
esac
mkdir -p scratch_runs/electricity
run="$(mktemp -d "$PWD/scratch_runs/electricity/h${h}_XXXXXXXX")"
exec > >(tee "$run/pipeline.log") 2>&1
cp "$SCRIPT_DIR/from_scratch.sh" "$SCRIPT_DIR/h$h.sh" "$run/"
python -m pip freeze > "$run/environment.freeze.txt"
git rev-parse HEAD > "$run/source_commit.txt"
git diff --binary > "$run/source_changes.patch"
python -u vqvae-only/codebook_pretrain.py \
  --dset 'electricity' \
  --target_points '0' \
  --batch_size '64' \
  --num_workers '0' \
  --scaler 'standard' \
  --features 'M' \
  --channel_indices '273,283,272,180,179,172,169,215,163,281,151,185,173,266,286,287,267,284,154,237,265,138,252,203,315,12,301,164,282,227,229,204,205,256,244,255,235,271,220,208,257,156,259,221,249,213,206,135,262,198,254,279,222,268,144,171,210,233,261,174,187,280,240,165,202,150,285,234,296,297,195,159,191,243,199,170,166,217,211,140,186,167,181,183,175,225,232,161,250,253,176,155,89,314,289,139,158,190,293,290,224,291,93,88,201,269,137,94,162,90,91,192,148,260,153,149,313,242,275,292,141,152,197,178,184,312,145,209,223,212,95,147,309,270,124,177,102,123,308,294,239,307,230,194,216,214,146,142,311,218,277,219,115,245,316,168,110,101,126,278,182,87,114,104,103,6,207,116,81,310,36,264,247,300,111,122,25,57,119,306,160,157,276,46,127,320,305,188,228,108,304,109,288,238,59,118,246,226,74,70,67,196,113,295,302,263,18,120,63,143,241,85,10,98,136,75,303,40,189,231,97,100,3,61,130,96,131,236,15,117,99,319,4,45,125,53,64,24,62,14,44,58,274,128,69,5,80,82,22,26,49,32,35,258,42,34,47,31,72,54,71,48,60,251,200,193,13,92,28,41,21,73,248,20,43,7,134,11,65,51,23,112,55,29,39,33,30,86,76,78,66,38,56,19,17,37,77,68,105,16,132,83,298,27,2,52,8,79,317,129,106,1,107,50,9,84,121,299,0,133,318' \
  --patch_size '4' \
  --embedding_dim '64' \
  --codebook_size '256' \
  --compression_factor '2' \
  --num_hiddens '128' \
  --num_residual_layers '2' \
  --num_residual_hiddens '128' \
  --vqvae_backbone 'mlp' \
  --vqvae_tcn_kernel_size '3' \
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
  --context_points "$cb_context" --lr "$cb_lr" --save_path "$run/codebook"
cb="$run/codebook/electricity/codebook_ps4_cb256_cd128_rvq2_model1_grp0.pth"
test -f "$cb"
python -u decoder_only_NTP/patch_vqvae_pretrain.py \
  --dset 'electricity' \
  --context_points '128' \
  --progressive_step_size '6' \
  --batch_size '32' \
  --num_workers '0' \
  --scaler 'standard' \
  --features 'M' \
  --channel_indices '273,283,272,180,179,172,169,215,163,281,151,185,173,266,286,287,267,284,154,237,265,138,252,203,315,12,301,164,282,227,229,204,205,256,244,255,235,271,220,208,257,156,259,221,249,213,206,135,262,198,254,279,222,268,144,171,210,233,261,174,187,280,240,165,202,150,285,234,296,297,195,159,191,243,199,170,166,217,211,140,186,167,181,183,175,225,232,161,250,253,176,155,89,314,289,139,158,190,293,290,224,291,93,88,201,269,137,94,162,90,91,192,148,260,153,149,313,242,275,292,141,152,197,178,184,312,145,209,223,212,95,147,309,270,124,177,102,123,308,294,239,307,230,194,216,214,146,142,311,218,277,219,115,245,316,168,110,101,126,278,182,87,114,104,103,6,207,116,81,310,36,264,247,300,111,122,25,57,119,306,160,157,276,46,127,320,305,188,228,108,304,109,288,238,59,118,246,226,74,70,67,196,113,295,302,263,18,120,63,143,241,85,10,98,136,75,303,40,189,231,97,100,3,61,130,96,131,236,15,117,99,319,4,45,125,53,64,24,62,14,44,58,274,128,69,5,80,82,22,26,49,32,35,258,42,34,47,31,72,54,71,48,60,251,200,193,13,92,28,41,21,73,248,20,43,7,134,11,65,51,23,112,55,29,39,33,30,86,76,78,66,38,56,19,17,37,77,68,105,16,132,83,298,27,2,52,8,79,317,129,106,1,107,50,9,84,121,299,0,133,318' \
  --channel_group_id '0' \
  --patch_size '4' \
  --embedding_dim '64' \
  --compression_factor '2' \
  --codebook_size '256' \
  --n_layers '3' \
  --n_heads '4' \
  --d_ff '512' \
  --dropout '0.1' \
  --temporal_backbone 'timefilter_lite' \
  --timefilter_topk '128' \
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
  --vqvae_tcn_kernel_size '3' \
  --vqvae_chunk_size '2' \
  --decoder_lowpass '0' \
  --decoder_lowpass_kernel 'binomial3' \
  --freeze_vqvae '1' \
  --load_vq_weights '1' \
  --per_channel_codebook '0' \
  --n_rq_layers '2' \
  --rq_layer_weights '1' '0.5' \
  --soft_neighbor_k '20' \
  --soft_neighbor_alpha '0.5' \
  --soft_neighbor_tau '1' \
  --use_raw_input '0' \
  --pred_len '6' \
  --n_epochs '100' \
  --lr '0.001' \
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
  --use_group_channel_experts 0 \
  --vqvae_checkpoint "$cb" --save_path "$run/pretrain"
pre="$run/pretrain/electricity/patch_vqvae_ps4_cb256_cd128_l3_in128_step6_model1_rvq2_timefilterlitek128_snk20a0p5t1p0_grp0.pth"
test -f "$pre"
PRETRAINED_MODEL="$pre" TD_FINETUNE_ROOT="$run/finetune" bash "$SCRIPT_DIR/h$h.sh"
printf 'COMPLETED_RUN=%s\n' "$run"
