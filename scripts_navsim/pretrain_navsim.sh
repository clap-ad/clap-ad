#!/bin/bash
#
# Stage 1: perception backbone pretraining, jointly on CARLA (Bench2Drive) and
# NAVSIM data. Produces the checkpoint that both
# scripts_navsim/posttrain_navsim_baseline.sh (LTFv6 baseline) and
# scripts_navsim/posttrain_clapad_navsim.sh (CLAP-AD) post-train on top of.
#
# Requires: `source scripts_navsim/navsim_env.sh` (NAVSIM venv + env vars) and
# a NAVSIM training cache (scripts_navsim/build_navsim_cache.sh).
#
# Usage:
#   scripts_navsim/pretrain_navsim.sh
#   CUDA_VISIBLE_DEVICES=0,1 scripts_navsim/pretrain_navsim.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=1 # Shuts off numpy multithreading, to avoid threads spawning other threads.
export NCCL_P2P_DISABLE=1 # https://github.com/huggingface/accelerate/issues/314
export NCCL_P2P_LEVEL=NVL # https://github.com/huggingface/accelerate/issues/314
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nproc_per_node=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 50000))

# NAVSIM training cache built by scripts_navsim/build_navsim_cache.sh.
NAVSIM_CACHE_DIR="${NAVSIM_CACHE_DIR:-${LEAD_PROJECT_ROOT:-.}/data/navsim_training_cache/navtrain}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-30}"
TRAINING_LOG_DIR="${TRAINING_LOG_DIR:-outputs/local_training/pretrain_navsim}"

echo "Using GPUs: $CUDA_VISIBLE_DEVICES for training"
echo "NAVSIM cache: $NAVSIM_CACHE_DIR"

export LEAD_TRAINING_CONFIG="logdir=${TRAINING_LOG_DIR} \
    log_clearml=false \
    use_planning_decoder=false \
    use_navsim_data=true \
    use_carla_data=true \
    LTF=true \
    use_clapad_head=false \
    navsim_data_root=${NAVSIM_CACHE_DIR} \
    epochs=${EPOCHS} \
    image_architecture=resnet34 \
    lidar_architecture=resnet34 \
    batch_size=${BATCH_SIZE}"

torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node="$nproc_per_node" \
    --max_restarts=0 \
    --rdzv_id="${SLURM_JOB_ID:-pretrain_navsim}" \
    --rdzv_backend=c10d \
    lead/training/train.py
