#!/usr/bin/env bash
# Recovered historical checkpoint args; numeric reproduction pending.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."
root="${TD_FINETUNE_ROOT:-search_electricity_192_recovered}"
pre="${PRETRAINED_MODEL:?Use scripts/ECL_best.sh for scratch training}"
test -f "$pre"
mkdir -p "$root"
python -u decoder_only_NTP/patch_vqvae_finetune.py \
  --dset 'electricity' \
  --context_points '96' \
  --target_points '192' \
  --batch_size '32' \
  --num_workers '0' \
  --scaler 'standard' \
  --features 'M' \
  --channel_indices '273,283,272,180,179,172,169,215,163,281,151,185,173,266,286,287,267,284,154,237,265,138,252,203,315,12,301,164,282,227,229,204,205,256,244,255,235,271,220,208,257,156,259,221,249,213,206,135,262,198,254,279,222,268,144,171,210,233,261,174,187,280,240,165,202,150,285,234,296,297,195,159,191,243,199,170,166,217,211,140,186,167,181,183,175,225,232,161,250,253,176,155,89,314,289,139,158,190,293,290,224,291,93,88,201,269,137,94,162,90,91,192,148,260,153,149,313,242,275,292,141,152,197,178,184,312,145,209,223,212,95,147,309,270,124,177,102,123,308,294,239,307,230,194,216,214,146,142,311,218,277,219,115,245,316,168,110,101,126,278,182,87,114,104,103,6,207,116,81,310,36,264,247,300,111,122,25,57,119,306,160,157,276,46,127,320,305,188,228,108,304,109,288,238,59,118,246,226,74,70,67,196,113,295,302,263,18,120,63,143,241,85,10,98,136,75,303,40,189,231,97,100,3,61,130,96,131,236,15,117,99,319,4,45,125,53,64,24,62,14,44,58,274,128,69,5,80,82,22,26,49,32,35,258,42,34,47,31,72,54,71,48,60,251,200,193,13,92,28,41,21,73,248,20,43,7,134,11,65,51,23,112,55,29,39,33,30,86,76,78,66,38,56,19,17,37,77,68,105,16,132,83,298,27,2,52,8,79,317,129,106,1,107,50,9,84,121,299,0,133,318' \
  --channel_group_id '0' \
  --n_epochs '50' \
  --lr '0.001' \
  --weight_decay '0.0001' \
  --revin '1' \
  --amp '1' \
  --seed '42' \
  --train_loss 'huber' \
  --huber_delta '2' \
  --unfreeze_decoder '0' \
  --decoder_lr_ratio '1' \
  --decoder_wd_ratio '1' \
  --use_gumbel_softmax '1' \
  --gumbel_temperature '0.8' \
  --gumbel_hard '0' \
  --ar_step_size '8' \
  --pred_len '10' \
  --model_id '1' \
  --selection_metric mse --use_group_channel_experts 0 --use_multiscale_residual 0 \
  --pretrained_model "$pre" --save_path "$root" 2>&1 | tee "$root/train.log"
