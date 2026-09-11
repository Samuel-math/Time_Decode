#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ETTh2 reproducible per-horizon configuration, tuned on the validation split.
export DSET=etth2
export TOTAL_CHANNELS=7
export MAX_CHANNELS_PER_MODEL=7
export BASE_MODEL_ID=1
export USE_CORR_CHANNEL_GROUPS=1
export CHANNEL_GROUPS_DIR=scripts/channel_groups
export CHANNEL_GROUPS_FILE=scripts/channel_groups/etth2_freq7.json
export RUN_HISTORY_PREFIX=etth2_sota_tuned
export RETAIN_RUNS=20
export FORCE_RETRAIN_ALL="${FORCE_RETRAIN_ALL:-1}"
export FORCE_RETRAIN_PRETRAIN=0
export RESUME_LATEST_RUN="${RESUME_LATEST_RUN:-0}"

export PATCH_SIZE=8
export COMPRESSION_FACTOR=4
export EMBEDDING_DIM=64
export CODEBOOK_SIZE=256
export NUM_HIDDENS=128
export NUM_RESIDUAL_LAYERS=2
export NUM_RESIDUAL_HIDDENS=128
export VQVAE_BACKBONE=mlp
export PER_CHANNEL_CODEBOOK=0
export N_RQ_LAYERS=2
export RQ_LAYER_WEIGHTS="1.0 1.0"

export CB_CONTEXT_POINTS=128
export CB_BATCH_SIZE=64
export CB_EPOCHS=50
export CB_LR=3e-4
export SPARSE_WEIGHT=0.3
export SPARSE_AMPLITUDE=0.05
export LAMBDA_ORD=0.01
export ORTH_WEIGHT=0.01
export ORTH_START_EPOCH=0
export ORTH_WARMUP_EPOCHS=5

export PRETRAIN_CONTEXT_POINTS=296
export PROGRESSIVE_STEP_SIZE=3
export PRETRAIN_PRED_LEN=6
export N_LAYERS=3
export N_HEADS=4
export D_FF=256
export DROPOUT=0.1
export PRETRAIN_EPOCHS=100
export PRETRAIN_BATCH_SIZE=64
export PRETRAIN_LR=3e-4

export FINETUNE_CONTEXT_POINTS=96
export FINETUNE_EPOCHS=50
export FINETUNE_BATCH_SIZE=32
export FINETUNE_LR_LIST="2e-4 2e-4 2e-4 1e-4"
export TARGET_POINTS_LIST="96 192 336 720"
export FORECAST_STEP_SIZE_LIST="3 6 7 4"
export FORECAST_PRED_LEN_LIST="9 12 14 6"
export USE_GUMBEL_SOFTMAX=1
export GUMBEL_TEMPERATURE_LIST="0.8 0.8 0.8 0.6"
export GUMBEL_HARD=0
export TRAIN_LOSS=huber
export HUBER_DELTA_LIST="2.0 2.0 2.0 1.8"

export FEATURES=M
export SCALER=standard
export NUM_WORKERS="${NUM_WORKERS:-0}"
export REVIN=1
export WEIGHT_DECAY=1e-4
export STREAM_LOGS="${STREAM_LOGS:-0}"

python scripts/make_channel_groups.py \
  --dset "${DSET}" --max_channels "${MAX_CHANNELS_PER_MODEL}" \
  --output "${CHANNEL_GROUPS_FILE}"

bash scripts/decoder_only_NTP/channel_group_pipeline.sh
