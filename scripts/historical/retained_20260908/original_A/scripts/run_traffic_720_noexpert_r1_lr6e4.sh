#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Time_Decode
export PATH=/root/miniconda3/envs/time_decode/bin:/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1

out=search_traffic_720_noexpert_fixed96_r1_lr6e4
mkdir -p "$out"
pre=imported_checkpoints/traffic_noexpert_20260831.pth
channels=$(python - <<'PY'
import json
d=json.load(open('scripts/channel_groups/traffic_freq862.json'))
g=d['groups'][0] if isinstance(d,dict) and 'groups' in d else d[0]
x=g.get('channel_indices',g.get('channels',g)) if isinstance(g,dict) else g
print(','.join(map(str,x)))
PY
)

python -u decoder_only_NTP/patch_vqvae_finetune.py --dset traffic \
  --context_points 96 --target_points 720 --batch_size 6 --num_workers 0 \
  --scaler standard --features M --channel_indices "$channels" --channel_group_id 0 \
  --pretrained_model "$pre" --n_epochs 50 --lr 6e-4 --weight_decay 1e-4 \
  --revin 1 --seed 42 --train_loss huber --huber_delta 3.0 \
  --selection_metric mse --use_gumbel_softmax 1 --gumbel_temperature .8 --gumbel_hard 0 \
  --ar_step_size 32 --pred_len 40 --use_group_channel_experts 0 \
  --use_multiscale_residual 0 --save_path "$out" --model_id 1 \
  2>&1 | tee "$out/train.log"
