#!/bin/bash

# Set which GPUs to use (comma-separated, e.g., "0,1,2,3")
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=1 # Shuts off numpy multithreading, to avoid threads spawning other threads.
export NCCL_P2P_DISABLE=1 # https://github.com/huggingface/accelerate/issues/314
export NCCL_P2P_LEVEL=NVL # https://github.com/huggingface/accelerate/issues/314
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Count number of GPUs from CUDA_VISIBLE_DEVICES
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
fi

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 50000))
export PYTHONPATH=$PYTHONPATH:/workspace/lead:/workspace/lead/3rd_party/Bench2Drive/scenario_runner/srunner:/workspace/lead/3rd_party/CARLA_0915/PythonAPI:/workspace/lead/3rd_party/CARLA_0915/PythonAPI/carla:/workspace/lead/3rd_party/Bench2Drive:/workspace/lead/3rd_party/Bench2Drive/scenario_runner
export PYTHON=/workspace/lead/.venv/bin/python
export USER_NAME="eaghajanzadeh"

echo "Using GPUs: $CUDA_VISIBLE_DEVICES for training"

uv run python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$nproc_per_node \
    --max_restarts=0 \
    --rdzv_id=manual_launch \
    --rdzv_backend=c10d \
    /workspace/lead/lead/training/train.py