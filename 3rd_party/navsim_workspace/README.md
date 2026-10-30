<h1 align="center"><b>CLAP-AD — NAVSIM Implementation</b></h1>

<p align="center">
  <a href="../../README.md">CARLA README</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://arxiv.org/abs/2610.36079">Paper</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://clap-ad.github.io/">Project page</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://huggingface.co/thomasschwarzfischer/clap-ad-ltfv6-navsim">NAVSIM checkpoint</a>
</p>

This guide documents the supplementary NAVSIM v2 experiments. The model uses
LEAD's LTFv6 backbone with the CLAP-AD planning head and is evaluated with
NAVSIM's PDM protocol on `navhard`. Training uses the LEAD trainer and
NAVSIM-derived feature caches; the vendored NAVSIM agent is an inference
adapter, not a separate trainer. See the [main README](../../README.md) for
the CARLA stack, paper results, license, and citation.

## Overview

- **Backbone:** LTFv6 (Latent TransFuser V6), from
  [LEAD](https://github.com/kesai-labs/lead).
- **Planning head:** CLAP-AD, restricted to the collision/TTC scoring branch
  (traffic-light and stop-sign heads are CARLA-only, since NAVSIM's map/scene
  representation doesn't expose that privileged information the same way).
- **NAVSIM bridge:** `CarlaTransfuserAgent` implements NAVSIM's agent API and
  loads a LEAD checkpoint for inference. Training is performed by
  [`lead/training/train.py`](../../lead/training/train.py), using cached
  NAVSIM features.

### Differences from CARLA

The paper's Section 5.2 describes the CARLA→NAVSIM adaptation:

- NAVSIM evaluates a 4-second trajectory without intermediate replanning. The
  model therefore predicts 48 path points and scores candidates over a 4-second
  collision horizon.
- NAVSIM speed is formed by confidence-weighting the predicted speed bins,
  rather than CARLA's discrete safe-bin selection.
- Only the collision/TTC criterion is trained for NAVSIM; the CARLA
  traffic-light, stop-sign, and overtaking labels are not available in the
  same form.

## Results

Results below report the paper's NAVSIM v2 `navhard` two-stage evaluation.

| Method | Stage | NC ↑ | DAC ↑ | DDC ↑ | TLC ↑ | EP ↑ | TTC ↑ | LK ↑ | HC ↑ | EC ↑ |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TransFuser | 1 | 96.2 | 79.5 | 99.1 | 99.5 | 84.1 | 95.1 | 94.2 | 97.5 | 79.1 |
| TransFuser | 2 | 77.7 | 70.2 | 84.2 | 98.0 | 85.1 | 75.6 | 45.4 | 95.7 | 75.9 |
| LTFv6 (backbone) | 1 | 96.6 | 86.7 | 99.2 | 99.6 | 84.5 | 95.1 | 94.4 | 97.8 | 76.4 |
| LTFv6 (backbone) | 2 | 79.9 | 75.6 | 86.3 | 97.9 | 89.6 | 76.1 | 50.1 | 95.2 | 66.7 |
| **LTFv6 + CLAP-AD (this repo)** | 1 | **98.8** | **87.6** | 99.0 | **99.6** | 78.2 | **98.0** | 93.1 | **98.0** | 37.8 |
| **LTFv6 + CLAP-AD (this repo)** | 2 | **90.3** | **79.1** | **89.6** | **99.2** | 78.8 | **87.7** | **53.6** | **97.2** | 51.6 |

Overall EPDMS ↑: TransFuser 23.1 · LTFv6 31.9 · LTFv6 + CLAP-AD 38.0.

CLAP-AD's collision head drives the largest gains on exactly the metrics it
optimizes for (NC, TTC, DAC), at a cost in Ego Progress (EP) and Comfort
(HC/EC) consistent with a policy that brakes more readily to avoid collisions
— see the paper for the full discussion.

## Setup

Use Linux, Python 3.10, and a CUDA-capable PyTorch device for model training and
inference. The NAVSIM stack has its own virtual environment because it pins
different PyTorch and NumPy versions from the CARLA environment. From the
repository root:

```bash
# Install CARLA 0.9.15 and its Python API as described in the main README.
PYTHON_BIN=python3.10 bash scripts_navsim/setup_navsim.sh
source scripts_navsim/navsim_env.sh
```

The setup script installs NAVSIM v1.1 and its pinned dependencies into
`navsimv1.1/venv`. Use this environment for NAVSIM feature caching and v1.1
evaluation; `navsim_env.sh` activates it and sets dataset, experiment, and
CARLA Python API paths. LEAD training uses the main environment installed by
the [root README](../../README.md#installation), which provides the newer
PyTorch API required by the trainer. The CARLA simulator is not needed for
NAVSIM caching or evaluation. Python 3.10 venv support must be available.

Provide NAVSIM's `navsim_logs/`, `sensor_blobs/`, and `maps/` data tree. To
link an existing complete dataset, set `NAVSIM_DATA_SRC` to its root before
running setup:

```bash
export NAVSIM_DATA_SRC=/path/to/navsim_dataset
NAVSIM_DATA_SRC=/path/to/navsim_dataset \
  PYTHON_BIN=python3.10 bash scripts_navsim/setup_navsim.sh
```

## Training

The full pipeline uses joint CARLA+NAVSIM perception pretraining followed by
NAVSIM planning-head training. Prepare the CARLA caches as described in the
[main README](../../README.md#training), then build the NAVSIM `navtrain`
feature cache in the NAVSIM environment:

```bash
source scripts_navsim/navsim_env.sh
bash scripts_navsim/build_navsim_cache.sh
```

In the main Conda environment from the root README, run training:

```bash
conda activate clapad
export NAVSIM_ACTIVATE_VENV=0
source scripts_navsim/navsim_env.sh
export TRAINING_SESSION_CACHE_PATH="${TMPDIR:-/tmp}/clapad-${USER}"
BATCH_SIZE=64 bash scripts_navsim/pretrain_navsim.sh
PRETRAINED_CKPT=outputs/local_training/pretrain_navsim/model_0029.pth \
  BATCH_SIZE=64 bash scripts_navsim/posttrain_clapad_navsim.sh
```

The baseline planning-head script is
[`scripts_navsim/posttrain_navsim_baseline.sh`](../../scripts_navsim/posttrain_navsim_baseline.sh).
Set `CUDA_VISIBLE_DEVICES`, `BATCH_SIZE`, and `EPOCHS` to choose devices and
training scale. Set `TRAINING_SESSION_CACHE_PATH` to a writable local SSD or
temporary directory if the default `/tmp` location is unsuitable. Checkpoints
and logs are written below `outputs/local_training/`.

## Evaluation

For NAVSIM v1.1 PDM evaluation, build a metric cache for the chosen split and
use the same split and dataset paths when scoring. This evaluates against the
NAVSIM v1.1 `navtest` protocol:

```bash
source scripts_navsim/navsim_env.sh
METRIC_CACHE="${NAVSIM_EXP_ROOT}/metric_cache_navtest"

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
  train_test_split=navtest experiment_name=navtest_metric_cache \
  cache.cache_path="$METRIC_CACHE"

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
  train_test_split=navtest \
  agent=carla_transfuser_agent \
  agent.checkpoint_path="$PWD/outputs/checkpoints/clap-ad-ltfv6-navsim/model_0060.pth" \
  worker=single_machine_thread_pool worker.max_workers=1 \
  experiment_name=clapad_navtest \
  metric_cache_path="$METRIC_CACHE"
```

The paper results use NAVSIM v2.2's `navhard` two-stage protocol. Follow
[NAVSIM v2.2 installation](navsimv2.2/docs/install.md) and its
[submission/evaluation instructions](navsimv2.2/docs/submission.md) to
reproduce those metrics. The NAVSIM v1.1 commands above are for its separate
`navtest` benchmark.

## Checkpoints

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="thomasschwarzfischer/clap-ad-ltfv6-navsim", local_dir="outputs/checkpoints/clap-ad-ltfv6-navsim")'
```

The checkpoint directory includes `config.json` and `model_0060.pth`; NAVSIM's
agent loads both.

## Directory layout

| Path | Contents |
| :--- | :--- |
| `navsimv1.1/` | Vendored [NAVSIM](https://github.com/autonomousvision/navsim) devkit, v1.1 (used for `navtrain`/`navtest` caching + our training bridge agent). |
| `navsimv2.2/` | Vendored NAVSIM devkit, v2.2 (adds the `navhard` two-stage evaluation protocol used for the paper's headline numbers). |
| `dataset/` | Local NAVSIM data mount or symlink; not committed to git. |
| `exp/` | NAVSIM experiment outputs (metric caches, eval results). Not committed to git. |

## License & attribution

`navsimv1.1/` and `navsimv2.2/` contain the upstream
[NAVSIM](https://github.com/autonomousvision/navsim) code and carry their
upstream **Apache License 2.0** terms; see the `LICENSE` files in each
subdirectory. See
[../../THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) for the full
third-party licensing breakdown.

## Citation

See the [main README's Citation section](../../README.md#citation).
