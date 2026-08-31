#!/usr/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --partition=2080-galvani
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=long.nguyen@student.uni-tuebingen.de
#SBATCH --mem=200gb

source /workspace/lead/3rd_party/navsim_workspace/navsimv1.1/venv/bin/activate
 
export LEAD_PROJECT_ROOT=/workspace/lead
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/navsimv1.1"
export OPENSCENE_DATA_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/dataset"
export HYDRA_FULL_ERROR=1

python "${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/navsimv1.1/navsim/planning/script/run_dataset_caching.py" \
    agent=carla_transfuser_agent \
    train_test_split=navtrain \
    experiment_name=navtrain_cache_transfuser \
    worker=single_machine_thread_pool \
    worker.max_workers=8