#!/bin/bash
#
# Stage 2: post-train CLAP-AD's planning head (path + multi-criteria speed
# scoring) on top of a pretrained TFv6 perception backbone, on CARLA /
# Bench2Drive. This is the CARLA counterpart of
# scripts_navsim/posttrain_clapad_navsim.sh and reproduces the CLAP-AD/TFv6
# numbers reported in the paper.
#
# Requires a stage-1 (perception) checkpoint, see scripts/pretrain_ddp.sh or
# `bash scripts/download_one_checkpoint.sh`.
#
# Usage:
#   PRETRAINED_CKPT=outputs/local_training/pretrain/model_0030.pth \
#     scripts_clapad/posttrain_clapad_b2d.sh
set -euo pipefail

export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=1 # Shuts off numpy multithreading, to avoid threads spawning other threads.
export NCCL_P2P_DISABLE=1 # https://github.com/huggingface/accelerate/issues/314
export NCCL_P2P_LEVEL=NVL # https://github.com/huggingface/accelerate/issues/314
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())")
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 50000))

PRETRAINED_CKPT="${PRETRAINED_CKPT:-outputs/local_training/pretrain/model_0030.pth}"

export LEAD_TRAINING_CONFIG="logdir=outputs/local_training/posttrain_clapad \
    load_file=${PRETRAINED_CKPT} \
    use_planning_decoder=true \
    use_clapad_head=true \
    num_route_points_smoothing=25 \
    num_route_points_prediction=24 \
    target_first_distance=1.0"

torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node="$nproc_per_node" \
    --max_restarts=0 \
    --rdzv_id="${SLURM_JOB_ID:-posttrain_clapad}" \
    --rdzv_backend=c10d \
    lead/training/train.py
