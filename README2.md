To be able to run Lead on our Cluster, do the following:

1. Build the Docker Container as usual.
2. Since we work with Python3.10, we need to install its dependencies:
    1. Due to different Python version we need slighlty different torch version for it but still it runs with our drivers and cuda setups: python3.10 -m pip install --no-deps torch==2.1.2+cu118 \
                torchvision==0.16.2+cu118 \
                torchaudio==2.1.2+cu118 \
                --index-url https://download.pytorch.org/whl/cu118
    2. Numpy version and numba are different so instead of normal requirement file we need to install this in python3.10 pyckages: python3.10 -m pip install /workspace/lead/requirements_python3.10.txt
    3. Install these libraries as well: Python3.10 -m pip install diskcache jaxtyping beartype carla omegaconf imgaug ujson timm wandb



# Getting started with Navsim:

cd /workspace/lead

mkdir -p 3rd_party/navsim_workspace/dataset

mkdir -p 3rd_party/navsim_workspace/exp
 
### Symlink NAVSIM dataset
ln -s /data/datasets/AD_datasets/navsim_v2/navsim_logs   3rd_party/navsim_workspace/dataset/navsim_logs
ln -s /data/datasets/AD_datasets/navsim_v2/sensor_blobs   3rd_party/navsim_workspace/dataset/sensor_blobs
ln -s /data/datasets/AD_datasets/navsim_v2/maps            3rd_party/navsim_workspace/dataset/maps
ln -s /data/datasets/AD_datasets/navsim/cache_files/metric_cache_navtest/ metric_cache_navtest
ln -s /data/datasets/AD_datasets/navsim/cache_files/training_cache_backup/ training_cache_backup
 
cd /workspace/lead/3rd_party/navsim_workspace/navsimv1.1
 
sudo apt update
sudo apt install python3.10-venv
 
python3 -m venv venv
source venv/bin/activate
 
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
 
 
### Install LEAD into the same venv
cd /workspace/lead
pip install -e .
 
pip install beartype jaxtyping carla numba
 
 
############
export LEAD_PROJECT_ROOT=/workspace/lead
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/navsimv1.1"
export OPENSCENE_DATA_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/dataset"
export HYDRA_FULL_ERROR=1
 
 
#### build cache for evaluation
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py     train_test_split=navtest     cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache
 
### build cache for training
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py     agent=carla_transfuser_agent     train_test_split=navtrain     experiment_name=navtrain_cache_transfuser     worker=single_machine_thread_pool     worker.max_workers=8
 
 
### download ckpt
 
 
cd /workspace/lead/3rd_party/navsim_workspace/navsimv1.1
 
### run eval
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
    train_test_split=navtest \
    agent=carla_transfuser_agent \
    worker=single_machine_thread_pool \
    agent.checkpoint_path=/workspace/lead/outputs/checkpoints/tfv6_navsim/model_0060.pth \
    experiment_name=carla_transfuser_agent_eval \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache_navtest
 
 
### more workers
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
    train_test_split=navtest \
    agent=carla_transfuser_agent \
    worker=single_machine_thread_pool \
    worker.max_workers=8 \
    agent.checkpoint_path=/workspace/lead/outputs/checkpoints/tfv6_navsim/model_0060.pth \
    experiment_name=carla_transfuser_agent_eval \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache_navtest