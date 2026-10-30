#!/bin/bash
#
# Full closed-loop evaluation of a CLAP-AD/TFv6 checkpoint on all 220
# Bench2Drive routes, using multi-GPU evaluation (one route-shard per GPU).
#
# Requires a running CARLA server per shard (see scripts/start_carla.sh) --
# leaderboard_wrapper.py opens one CARLA client per GPU rank on
# CARLA_PORT + 150 * rank, so make sure that many CARLA servers are listening
# (or lower --nproc-per-node to match how many you started).
#
# Usage:
#   CHECKPOINT=outputs/checkpoints/clapad-tfv6 CUDA_VISIBLE_DEVICES=0,1,2,3 \
#     scripts_clapad/eval_clapad_b2d.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export LEAD_CLOSED_LOOP_CONFIG="${LEAD_CLOSED_LOOP_CONFIG:-produce_debug_video=true}"

IFS=',' read -ra CUDA_DEVICE_LIST <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE="${#CUDA_DEVICE_LIST[@]}"

CHECKPOINT="${CHECKPOINT:-outputs/checkpoints/clapad-tfv6}"
ROUTES="${ROUTES:-3rd_party/Bench2Drive/leaderboard/data/bench2drive220.xml}"

python -u lead/leaderboard_wrapper.py \
    --checkpoint "${CHECKPOINT}" \
    --routes "${ROUTES}" \
    --bench2drive \
    --multigpu \
    --nproc-per-node "${NPROC_PER_NODE}"
