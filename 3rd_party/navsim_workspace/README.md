<h1 align="center"><b>CLAP-AD — NAVSIM Implementation</b></h1>

<p align="center">
  <a href="../../README.md">← Back to the main (CARLA) README</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="#">Paper</a> (TODO)&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://huggingface.co/thomasschwarzfischer/clap-ad">Checkpoints</a> (placeholder)&nbsp;&nbsp;·&nbsp;&nbsp;<a href="../../results/navsim/navsim_v2_navhard.csv">Result CSV</a>
</p>

This directory hosts the **NAVSIM v2** implementation of CLAP-AD, evaluated in
the paper's "Real-World Generalization" section. It uses a **Latent
TransFuser V6 (LTFv6)** backbone with CLAP-AD's planning head (collision/TTC
scoring only — see [§ Differences from CARLA](#differences-from-carla)),
trained on NAVSIM's human-driving demonstrations and evaluated with NAVSIM's
own PDM-based closed-loop simulation (`navhard`, two-stage protocol).

Unlike CARLA, all NAVSIM-side backbones (UniAD, VAD, TransFuser, ...) are
**not** implicated by the Bench2DriveZoo licensing issue described in the main
README — LTFv6 is entirely LEAD/CLAP-AD's own code. This directory is
therefore a complete, self-contained release of the NAVSIM side of CLAP-AD.

## Overview

- **Backbone:** LTFv6 (Latent TransFuser V6), from
  [LEAD](https://github.com/kesai-labs/lead) — camera-only, no LiDAR (NAVSIM's
  `navhard` split doesn't provide it in the same way as CARLA).
- **Planning head:** CLAP-AD, restricted to the collision/TTC scoring branch
  (traffic-light and stop-sign heads are CARLA-only, since NAVSIM's map/scene
  representation doesn't expose that privileged information the same way).
- **Bridge to NAVSIM:** `CarlaTransfuserAgent`
  ([navsimv1.1](navsimv1.1/navsim/agents/carla_transfuser_agent.py) /
  [navsimv2.2](navsimv2.2/navsim/agents/carla_transfuser_agent.py)) implements
  NAVSIM's `AbstractAgent` interface but internally wraps LEAD's
  `OpenLoopInference`, reloading a LEAD-trained checkpoint. It is
  **inference-only** — training happens entirely on the LEAD/CARLA side of the
  repository (`../../lead/training/train.py`), using cached NAVSIM features.

### Differences from CARLA

The paper's Section 5.2 describes the CARLA→NAVSIM adaptation:

- NAVSIM scores a single 4-second trajectory with no intermediate replanning,
  so the planning horizon is extended to **48 m** (`num_route_points_prediction=48`)
  and the collision horizon to the full **4 s** (`ttc_time_horizon=4.0`,
  `ttc_coll_radius=0.5`, `ttc_safe_distance=1.0`).
- CARLA's hard speed thresholding (pick the highest speed bin scored safe) is
  replaced by a **confidence-weighted average over speed bins**
  (`traj_from_clapad` in
  [transfuser_config.py](navsimv1.1/navsim/agents/transfuser/transfuser_config.py) /
  `carla_transfuser_agent.py`), which is then integrated into a full temporal
  trajectory along the predicted path.
- Only the collision/TTC score is supervised (`speed_scoring_loss_weights` is
  `{"TTC_Score": 1.0, "traffic_light": 0.0, "stop_sign": 0.0, "overtake": 0.0}`
  when `use_navsim_data=true`), since traffic-light/stop-sign ground truth
  from privileged CARLA scenario state has no NAVSIM equivalent.

## Results

Full data: [../../results/navsim/navsim_v2_navhard.csv](../../results/navsim/navsim_v2_navhard.csv) (paper Table 3, `navhard` split, two-stage evaluation).

| Method | Stage | NC ↑ | DAC ↑ | DDC ↑ | TL ↑ | EP ↑ | TTC ↑ | LK ↑ | HC ↑ | EC ↑ | EPDMS ↑ |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TransFuser | 1 | 96.2 | 79.5 | 99.1 | 99.5 | 84.1 | 95.1 | 94.2 | 97.5 | 79.1 | 23.1 |
| TransFuser | 2 | 77.7 | 70.2 | 84.2 | 98.0 | 85.1 | 75.6 | 45.4 | 95.7 | 75.9 | |
| LTFv6 (backbone) | 1 | 96.6 | 86.7 | 99.2 | 99.6 | 84.5 | 95.1 | 94.4 | 97.8 | 76.4 | 31.9 |
| LTFv6 (backbone) | 2 | 79.9 | 75.6 | 86.3 | 97.9 | 89.6 | 76.1 | 50.1 | 95.2 | 66.7 | |
| **LTFv6 + CLAP-AD (this repo)** | 1 | **98.8** | **87.6** | 99.0 | **99.6** | 78.2 | **98.0** | 93.1 | **98.0** | 37.8 | **38.0** |
| **LTFv6 + CLAP-AD (this repo)** | 2 | **90.3** | **79.1** | **89.6** | **99.2** | 78.8 | **87.7** | **53.6** | **97.2** | 51.6 | |

CLAP-AD's collision head drives the largest gains on exactly the metrics it
optimizes for (NC, TTC, DAC), at a cost in Ego Progress (EP) and Comfort
(HC/EC) consistent with a policy that brakes more readily to avoid collisions
— see the paper for the full discussion.

## Installation

NAVSIM uses a **separate Python 3.10 virtualenv** from the CARLA side (its
pinned `torch==2.0.1` conflicts with LEAD's CARLA-side `torch==2.6.0`), nested
inside this directory. All commands below assume you're in the repo root
(`/workspace/lead`) inside the
[env-beta devcontainer](../../.devcontainer/env-beta/devcontainer.json) (see
the [main README's Quickstart](../../README.md#quickstart)).

### 1. Get the NAVSIM dataset

Download/mount the NAVSIM v2 dataset (`navsim_logs/`, `sensor_blobs/`,
`maps/`) as described in
[navsimv1.1/docs/install.md](navsimv1.1/docs/install.md) (for `navtrain`/
`navtest`) and
[navsimv2.2/docs/install.md](navsimv2.2/docs/install.md) (for `navhard`). Then
either symlink it, or point `NAVSIM_DATA_SRC` at it and run the setup script
below, which will symlink it for you:

```bash
export NAVSIM_DATA_SRC=/path/to/your/navsim_v2   # contains navsim_logs/, sensor_blobs/, maps/
```

### 2. Create the venv and install dependencies

```bash
cd /workspace/lead
bash scripts_navsim/setup_navsim.sh
```

This is idempotent and will:
1. Symlink `NAVSIM_DATA_SRC` into `3rd_party/navsim_workspace/dataset/`.
2. Create a venv at `3rd_party/navsim_workspace/navsimv1.1/venv`.
3. Install the NAVSIM devkit (`navsimv1.1/requirements.txt` + `pip install -e navsimv1.1`).
4. Install this repo's `lead` package into the same venv (`pip install -e ../..`).

### 3. Activate the environment

In every new shell, before running any NAVSIM/CLAP-AD command:

```bash
source scripts_navsim/navsim_env.sh
```

This activates the venv and exports `NUPLAN_MAPS_ROOT`, `NAVSIM_EXP_ROOT`,
`NAVSIM_DEVKIT_ROOT`, `OPENSCENE_DATA_ROOT`.

## Quickstart

### 1. Build caches

```bash
# Metric cache (needed for evaluation, e.g. navtest split)
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
    train_test_split=navtest \
    cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache_navtest

# Training cache (needed for training)
bash ../../scripts_navsim/build_navsim_cache.sh   # SPLIT=navtrain by default
```

For a quick dry-run, use the small `navmini` split instead:
`SPLIT=navmini bash ../../scripts_navsim/build_navsim_cache.sh`.

### 2. Train

Training runs on the **LEAD side** (not the NAVSIM venv's own training code —
`CarlaTransfuserAgent` is inference-only). From the NAVSIM venv:

```bash
# Stage 1: perception backbone pretraining (joint CARLA + NAVSIM)
bash ../../scripts_navsim/pretrain_navsim.sh

# Stage 2a: baseline LTFv6 planning head (no CLAP-AD), for comparison
PRETRAINED_CKPT=outputs/local_training/pretrain_navsim/model_0029.pth \
  bash ../../scripts_navsim/posttrain_navsim_baseline.sh

# Stage 2b: CLAP-AD planning head
PRETRAINED_CKPT=outputs/local_training/pretrain_navsim/model_0029.pth \
  bash ../../scripts_navsim/posttrain_clapad_navsim.sh
```

### 3. Evaluate

```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
    train_test_split=navtest \
    agent=carla_transfuser_agent \
    worker=single_machine_thread_pool \
    worker.max_workers=8 \
    agent.checkpoint_path=../../outputs/local_training/posttrain_clapad_navsim/model_final.pth \
    experiment_name=clapad_navsim_eval \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache_navtest
```

For the paper's `navhard` two-stage evaluation, follow
[navsimv2.2/docs/install.md](navsimv2.2/docs/install.md) and
[navsimv2.2/docs/submission.md](navsimv2.2/docs/submission.md), pointing
`agent.checkpoint_path` at your trained checkpoint the same way.

### 4. Visualize

```bash
bash ../../scripts_navsim/run_vis_navsim.sh --agent clapad --checkpoint <path-to-checkpoint>
```

Renders per-scene BEV (GT vs. CLAP-AD trajectory + speed) and the predicted
per-bin speed-score bar chart — no metric cache required. See
[vis_navsim_scenes.py](../../scripts_navsim/vis_navsim_scenes.py).

## Checkpoints

```bash
# TODO: placeholder until checkpoints are uploaded.
git clone https://huggingface.co/thomasschwarzfischer/clap-ad outputs/checkpoints/clapad
# Expected layout: outputs/checkpoints/clapad/navsim_ltfv6/model_*.pth
```

## Directory layout

| Path | Contents |
| :--- | :--- |
| `navsimv1.1/` | Vendored [NAVSIM](https://github.com/autonomousvision/navsim) devkit, v1.1 (used for `navtrain`/`navtest` caching + our training bridge agent). |
| `navsimv2.2/` | Vendored NAVSIM devkit, v2.2 (adds the `navhard` two-stage evaluation protocol used for the paper's headline numbers). |
| `dataset/` | Symlinks to the NAVSIM dataset (`navsim_logs/`, `sensor_blobs/`, `maps/`), created by `setup_navsim.sh`. Not committed to git. |
| `exp/` | NAVSIM experiment outputs (metric caches, eval results). Not committed to git. |

## License & attribution

`navsimv1.1/` and `navsimv2.2/` are vendored, unmodified-except-for-agent-bridge
copies of [NAVSIM](https://github.com/autonomousvision/navsim) (Univ. of
Tübingen), licensed under **Apache License 2.0** — see the `LICENSE` file in
each subdirectory. This is compatible with the main repository's MIT license;
attribution is preserved. See
[../../THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) for the full
third-party licensing breakdown, including why UniAD/VAD are not part of this
release.

## Citation

See the [main README's Citation section](../../README.md#citation).
