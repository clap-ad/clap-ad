To be able to run Lead on our Cluster, do the following:

1. Build the Docker Container as usual.
2. Since we work with Python3.10, we need to install its dependencies:
    1. Due to different Python version we need slighlty different torch version for it but still it runs with our drivers and cuda setups: python3.10 -m pip install --no-deps torch==2.1.2+cu118 \
                torchvision==0.16.2+cu118 \
                torchaudio==2.1.2+cu118 \
                --index-url https://download.pytorch.org/whl/cu118
    2. Numpy version and numba are different so instead of normal requirement file we need to install this in python3.10 pyckages: python3.10 -m pip install /workspace/lead/requirements_python3.10.txt
    3. Install these libraries as well: Python3.10 -m pip install diskcache jaxtyping beartype carla omegaconf imgaug ujson timm wandb