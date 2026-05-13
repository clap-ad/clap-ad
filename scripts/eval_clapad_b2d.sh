#!/bin/bash
 
set -Eeuo pipefail
 
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export LEAD_CLOSED_LOOP_CONFIG="produce_debug_video=true"
 
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra CUDA_DEVICE_LIST <<< "$CUDA_VISIBLE_DEVICES"
    NPROC_PER_NODE="${#CUDA_DEVICE_LIST[@]}"
else
    NPROC_PER_NODE=1
fi
 
cd /workspace/lead
 
/workspace/lead/.venv/bin/python -u lead/leaderboard_wrapper.py \
    --checkpoint /workspace/lead/outputs/local_training/TestByeaghajanzadeh_20260420_120624_7 \
    --routes /workspace/lead/data/benchmark_routes/bench2drive220.xml \
    --bench2drive \
    --multigpu \
    --nproc-per-node ${NPROC_PER_NODE}