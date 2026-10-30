#!/bin/bash
#
# Stage 2 (baseline): post-train the ORIGINAL LTFv6 planning head (no CLAP-AD)
# on NAVSIM, for comparison against scripts_navsim/posttrain_clapad_navsim.sh.
#
# Requires: `source scripts_navsim/navsim_env.sh` and a stage-1 checkpoint
# from scripts_navsim/pretrain_navsim.sh.
#
# Usage:
#   PRETRAINED_CKPT=outputs/local_training/pretrain_navsim/model_0029.pth \
#     scripts_navsim/posttrain_navsim_baseline.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nproc_per_node=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 50000))

PRETRAINED_CKPT="${PRETRAINED_CKPT:-outputs/local_training/pretrain_navsim/model_0029.pth}"
NAVSIM_CACHE_DIR="${NAVSIM_CACHE_DIR:-${LEAD_PROJECT_ROOT:-.}/data/navsim_training_cache/navtrain}"

echo "Using GPUs: $CUDA_VISIBLE_DEVICES for training"
echo "Pretrained ckpt: $PRETRAINED_CKPT"

export LEAD_TRAINING_CONFIG="logdir=outputs/local_training/posttrain_navsim_baseline \
    log_clearml=false \
    load_file=${PRETRAINED_CKPT} \
    use_planning_decoder=true \
    use_navsim_data=true \
    use_carla_data=false \
    LTF=false \
    use_clapad_head=false \
    navsim_data_root=${NAVSIM_CACHE_DIR} \
    num_route_points_smoothing=20 \
    num_route_points_prediction=10 \
    batch_size=256 \
    epochs=90"

torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node="$nproc_per_node" \
    --max_restarts=0 \
    --rdzv_id="${SLURM_JOB_ID:-posttrain_navsim_baseline}" \
    --rdzv_backend=c10d \
    lead/training/train.py
