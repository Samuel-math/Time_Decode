#!/bin/bash
# =====================================================================
# ETTh1 pred96 channel-group wrapper.
#
# Fixed-horizon version of scripts/new_etth1.sh, following the layout of
# scripts/etth2_pred96.sh.
#
# Usage from repo root:
#   bash scripts/etth1_pred96.sh
#
# ETTh1 has 7 variables. With MAX_CHANNELS_PER_MODEL=7 this is equivalent to
# one group; set MAX_CHANNELS_PER_MODEL=3 or 4 to test grouped training.
# =====================================================================

set -euo pipefail
unset TD_ABLATION CB_SAVE_PATH PRETRAIN_SAVE_PATH FINETUNE_SAVE_PATH GROUP_RUN_NAME
export TD_CB_SAVE_START=5
export PYTHONHASHSEED=42

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Dataset / grouping
export DSET=ettm1
export TOTAL_CHANNELS=7
export MAX_CHANNELS_PER_MODEL=7
export BASE_MODEL_ID="${BASE_MODEL_ID:-1}"
export FORCE_RETRAIN_ALL=1
export FORCE_RETRAIN_PRETRAIN=1
export RESUME_LATEST_RUN=0
export USE_CORR_CHANNEL_GROUPS=1
export CHANNEL_GROUPS_DIR="${CHANNEL_GROUPS_DIR:-scripts/channel_groups}"
export RUN_HISTORY_PREFIX="verify_ettm1_96"
export RETAIN_RUNS=999999

# Match new_etth1.sh VQVAE/codebook params
export PATCH_SIZE=16
export COMPRESSION_FACTOR=8
export EMBEDDING_DIM=32
export CODEBOOK_SIZE=256
export NUM_HIDDENS=128
export NUM_RESIDUAL_LAYERS=2
export NUM_RESIDUAL_HIDDENS=128
export PER_CHANNEL_CODEBOOK=0
export N_RQ_LAYERS=2
export RQ_LAYER_WEIGHTS="1.0 1.0"

# Codebook training params
export CB_CONTEXT_POINTS=512
export CB_BATCH_SIZE=64
export CB_EPOCHS=50
export CB_LR=3e-4
export SPARSE_WEIGHT=0.3
export SPARSE_AMPLITUDE=0.1
export LAMBDA_ORD=0.01
export ORTH_WEIGHT=0.003
export ORTH_START_EPOCH="${ORTH_START_EPOCH:-0}"
export ORTH_WARMUP_EPOCHS="${ORTH_WARMUP_EPOCHS:-5}"

# Match new_etth1.sh short-mode NTP pretrain params
export PRETRAIN_CONTEXT_POINTS=512
export PROGRESSIVE_STEP_SIZE=3
export PRETRAIN_PRED_LEN=6
export N_LAYERS=3
export N_HEADS=8
export D_FF=128
export DROPOUT=0.1
export PRETRAIN_EPOCHS=100
export PRETRAIN_BATCH_SIZE=64
export PRETRAIN_LR=3e-4

# Match new_etth1.sh short-mode finetune params, fixed to pred96
export FINETUNE_CONTEXT_POINTS=96
export FINETUNE_EPOCHS=50
export FINETUNE_BATCH_SIZE=64
export FINETUNE_LR=1.5e-4
export UNFREEZE_DECODER=1
export DECODER_LR_RATIO=0.05
export DECODER_WD_RATIO=0.1
export TARGET_POINTS_LIST="96"
export USE_GUMBEL_SOFTMAX=1
export GUMBEL_TEMPERATURE=1.2
export GUMBEL_HARD=0
export TRAIN_LOSS=huber
export HUBER_DELTA=0.75
export FORECAST_STEP_SIZE_LIST="3"
export FORECAST_PRED_LEN_LIST="6"

export FEATURES=M
export SCALER=standard
export NUM_WORKERS="${NUM_WORKERS:-0}"
export REVIN=1
export WEIGHT_DECAY=1e-4
export STREAM_LOGS="${STREAM_LOGS:-1}"

echo "================================================="
echo "ETTh1 pred96 channel-group run"
echo "================================================="
echo "Repo root              : ${REPO_ROOT}"
echo "Max channels per model : ${MAX_CHANNELS_PER_MODEL}"
echo "Target points          : ${TARGET_POINTS_LIST}"
echo "Forecast step list     : ${FORECAST_STEP_SIZE_LIST}"
echo "Forecast pred list     : ${FORECAST_PRED_LEN_LIST}"
echo "Base model id          : ${BASE_MODEL_ID}"
echo "Finetune loss          : ${TRAIN_LOSS} (huber_delta=${HUBER_DELTA})"
echo "Run history prefix     : ${RUN_HISTORY_PREFIX}"
echo "Resume latest run      : ${RESUME_LATEST_RUN:-0}"
echo "Retain runs            : ${RETAIN_RUNS}"
echo "Force rerun all        : ${FORCE_RETRAIN_ALL:-${FORCE_RETRAIN:-0}}"
echo "Force rerun pretrain   : ${FORCE_RETRAIN_PRETRAIN:-0}"
echo "Freq channel grouping  : ${USE_CORR_CHANNEL_GROUPS}"
echo "================================================="

cd "${REPO_ROOT}"

if [ "${USE_CORR_CHANNEL_GROUPS}" = "1" ]; then
    mkdir -p "${CHANNEL_GROUPS_DIR}"
    export CHANNEL_GROUPS_FILE="${CHANNEL_GROUPS_FILE:-${CHANNEL_GROUPS_DIR}/${DSET}_freq${MAX_CHANNELS_PER_MODEL}.json}"
    echo "Generating frequency-feature channel groups: ${CHANNEL_GROUPS_FILE}"
    python scripts/make_channel_groups.py \
        --dset "${DSET}" \
        --max_channels "${MAX_CHANNELS_PER_MODEL}" \
        --output "${CHANNEL_GROUPS_FILE}"
fi

bash scripts/decoder_only_NTP/channel_group_pipeline.sh
