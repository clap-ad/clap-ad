#!/bin/bash
#
# Stage 2: post-train CLAP-AD's planning head (path + multi-criteria speed
# scoring) on top of a pretrained LTFv6 backbone, on NAVSIM.
#
# Extends the planning horizon to 48 m and the collision horizon to the full
# 4 s, matching the real-world generalization setup in the paper (Sec. 5.2).
#
# Requires: `source scripts_navsim/navsim_env.sh` and a stage-1 checkpoint
# from scripts_navsim/pretrain_navsim.sh.
#
# Usage:
#   PRETRAINED_CKPT=outputs/local_training/pretrain_navsim/model_0029.pth \
#     scripts_navsim/posttrain_clapad_navsim.sh
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
echo "NAVSIM cache: $NAVSIM_CACHE_DIR"

export LEAD_TRAINING_CONFIG="logdir=outputs/local_training/posttrain_clapad_navsim \
    load_file=${PRETRAINED_CKPT} \
    use_planning_decoder=true \
    use_navsim_data=true \
    use_carla_data=false \
    LTF=false \
    use_clapad_head=true \
    navsim_data_root=${NAVSIM_CACHE_DIR} \
    num_route_points_smoothing=50 \
    num_route_points_prediction=48 \
    ttc_time_horizon=4.0 \
    ttc_coll_radius=0.5 \
    ttc_safe_distance=1.0 \
    target_first_distance=1.0 \
    use_fde_loss=true \
    lr=1e-4 \
    epochs=90 \
    batch_size=64"

torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node="$nproc_per_node" \
    --max_restarts=0 \
    --rdzv_id="${SLURM_JOB_ID:-posttrain_clapad_navsim}" \
    --rdzv_backend=c10d \
    lead/training/train.py
