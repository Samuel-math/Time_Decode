#!/usr/bin/env bash
# Historical fine-tuning configuration only. Full retraining provenance is pending.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../../.." && pwd)"

export PYTHONUNBUFFERED=1
tag=e3d48; experts=3; dim=48; ctx=672
export HORIZONS=720
root=search_weather_expert_fixed96_r6_${tag}_ctx${ctx}
mkdir -p "$root"
pre="${PRETRAINED_MODEL:-search_weather_expert_pretrain_ctx${ctx}_r5/$tag/weather/patch_vqvae_ps8_cb512_cd128_l3_in${ctx}_step6_model1_rvq2_dlp_timefilterlitek8_snk20a0p3t0p5_grp0.pth}"
if [ ! -f "$pre" ]; then echo "Missing historical checkpoint: $pre" >&2; exit 2; fi
channels=9,0,10,5,2,19,1,8,6,3,7,4,16,17,20,12,11,18,14,13,15
run_one(){
 local h=$1 step=$2 pred=$3 batch=$4 delta=$5
 python decoder_only_NTP/patch_vqvae_finetune.py --dset weather --context_points 96 --target_points "$h" --batch_size "$batch" --num_workers 0 \
  --scaler standard --features M --channel_indices "$channels" --channel_group_id 0 --pretrained_model "$pre" --n_epochs 50 \
  --lr 1e-4 --weight_decay 1e-4 --revin 1 --amp 1 --seed 42 --train_loss huber --huber_delta "$delta" --selection_metric mse \
  --use_gumbel_softmax 1 --gumbel_temperature 1 --gumbel_hard 0 --ar_step_size "$step" --pred_len "$pred" \
  --use_group_channel_experts 1 --group_expert_count "$experts" --group_expert_dim "$dim" --group_expert_dropout .1 \
  --group_expert_temperature 1 --group_expert_topk 0 --group_expert_gate_init -2 --use_multiscale_residual 0 \
  --save_path "$root/h$h" --model_id 1 > "$root/h$h.log" 2>&1
}
case " ${HORIZONS:-96 192 336 720} " in *" 96 "*) run_one 96 6 6 40 2.0;; esac
case " ${HORIZONS:-96 192 336 720} " in *" 192 "*) run_one 192 8 10 32 2.19;; esac
case " ${HORIZONS:-96 192 336 720} " in *" 336 "*) run_one 336 8 10 24 2.2;; esac
case " ${HORIZONS:-96 192 336 720} " in *" 720 "*) run_one 720 12 14 12 1.5;; esac
wait
