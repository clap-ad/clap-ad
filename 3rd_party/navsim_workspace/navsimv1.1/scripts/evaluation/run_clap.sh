#!/bin/bash
# filepath: ./run_clap_eval_navsim.sh

export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export HYDRA_FULL_ERROR=1
export LEAD_PROJECT_ROOT=/workspace/lead
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=/workspace/lead/3rd_party/navsim_workspace/dataset/maps
export NAVSIM_EXP_ROOT=/workspace/lead/3rd_party/navsim_workspace/exp
export NAVSIM_DEVKIT_ROOT=/workspace/lead/3rd_party/navsim_workspace/navsimv1.1
export OPENSCENE_DATA_ROOT=/workspace/lead/3rd_party/navsim_workspace/dataset

/workspace/lead/3rd_party/navsim_workspace/navsimv1.1/venv/bin/python \
    /workspace/lead/3rd_party/navsim_workspace/navsimv1.1/navsim/planning/script/run_pdm_score.py \
    train_test_split=navtest \
    agent=carla_transfuser_agent \
    worker=single_machine_thread_pool \
    worker.max_workers=4 \
    agent.checkpoint_path=/workspace/lead/outputs/local_training/TestByeaghajanzadeh_20260506_155115_v5/model_0060.pth \
    experiment_name=carla_transfuser_agent_eval \
    metric_cache_path=/workspace/lead/3rd_party/navsim_workspace/exp/metric_cache