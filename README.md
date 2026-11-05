<h1 align="center">CLAP-AD</h1>

<p align="center">
  <b>Closed-Loop Aware Planning via Imitation-Free Speed Scoring for End-to-End Autonomous Driving</b><br>
  WACV 2027
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.36079">Paper</a>&nbsp; · &nbsp;
  <a href="https://clap-ad.github.io/">Project page</a>&nbsp; · &nbsp;
  <a href="https://huggingface.co/clap-ad/clap-ad-tfv6">CARLA checkpoint</a>&nbsp; · &nbsp;
  <a href="https://huggingface.co/clap-ad/clap-ad-ltfv6-navsim">NAVSIM checkpoint</a>&nbsp; · &nbsp;
  <a href="#citation">Citation</a>
</p>

<p align="center">
  Emad Aghajanzadeh · Thomas Schwarzfischer · Marc Kaufeld · Moritz Thoma · Ryan Pégoud · Manoj Rohit Vemparala ·
  Pierpaolo Mori · Nael Fasfous · Johannes Betz · Daniel Mueller-Gritschneder · Ulf Schlichtmann
</p>

CLAP-AD separates path prediction from speed selection in end-to-end driving.
It learns the path by imitation and scores candidate speeds against
physically grounded safety criteria, including collision risk, traffic lights,
stop signs, and overtaking. At inference, the planner follows a requested speed
when safe and otherwise selects a safe alternative, without retraining.

This repository contains the CARLA implementation on the LEAD TFv6 stack.
The [supplementary NAVSIM implementation and reproduction guide](3rd_party/navsim_workspace/README.md)
covers the LTFv6 experiments.

<p align="center">
  <img src="https://clap-ad.github.io/static/videos/clapad_animation.gif" alt="CLAP-AD separates path prediction from safety-aware speed selection" width="85%">
  <br><sub>CLAP-AD follows a requested speed when safe and selects a safe alternative otherwise.</sub>
</p>

## Results

### CARLA — Bench2Drive (220 routes)

| Method | Test-time speed control | Driving score ↑ | Δ score | Success rate ↑ |
| :--- | :---: | ---: | ---: | ---: |
| UniAD | No | 45.81 | — | 16.36% |
| **UniAD + CLAP-AD** | Yes | **74.29** | **+28.5** | **49.09%** |
| VAD | No | 42.35 | — | 15.00% |
| **VAD + CLAP-AD** | Yes | **75.04** | **+32.7** | **47.27%** |
| TFv6 | No | **94.70** | — | **82.10%** |
| **TFv6 + CLAP-AD** | Yes | 92.30 | −2.4 | 75.00% |

### CARLA — Bench2Drive-Speed (48 desired-speed routes)

| Method | SA (all) ↑ | Easy ↑ | Medium ↑ | Hard ↑ | Efficiency ↑ | General driving score ↑ | Success rate ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TCP-Speed | 69.23 | 76.18 | 68.71 | 62.81 | 169.84 | 29.66 | 18.75% |
| Hydra-Next | 68.53 | 68.68 | 67.94 | 68.96 | 157.05 | 69.87 | 39.58% |
| SparseDriveV2 | 51.13 | 52.50 | 49.72 | 51.18 | 91.77 | **94.63** | 60.41% |
| UniAD | 54.22 | 56.14 | 52.62 | 53.90 | 108.49 | 18.90 | 4.16% |
| **UniAD + CLAP-AD** | 79.41 | 83.83 | 75.40 | 79.00 | 170.03 | 75.15 | 41.67% |
| VAD | 61.63 | 64.64 | 62.60 | 57.66 | 130.53 | 40.23 | 20.83% |
| **VAD + CLAP-AD** | 75.51 | 80.42 | 72.05 | 74.07 | 159.48 | 90.52 | 52.08% |
| TFv6 | 69.64 | 70.75 | 68.30 | 69.86 | 154.58 | 92.59 | 58.33% |
| **TFv6 + CLAP-AD** | **82.53** | **89.79** | **76.81** | **80.98** | **187.80** | 88.47 | 50.00% |

UniAD/VAD CLAP-AD variants are not released because they depend on the separately licensed
[Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The general Bench2Drive driving score for TFv6 + CLAP-AD is 88.47. The
Bench2Drive-Speed-specific composite (including its overtaking skill) is 80.08;
both values are retained in the released evaluation summaries.

### NAVSIM v2 — navhard

| Method | Stage 2 NC ↑ | Stage 2 TTC ↑ | Overall EPDMS ↑ |
| :--- | ---: | ---: | ---: |
| TransFuser | 77.7 | 75.6 | 23.1 |
| LTFv6 | 79.9 | 76.1 | 31.9 |
| **LTFv6 + CLAP-AD** | **90.3** | **87.7** | **38.0** |

See the [NAVSIM implementation and reproduction notes](3rd_party/navsim_workspace/README.md).

## Installation

Requirements: Linux, Python 3.10, a CUDA-capable NVIDIA driver, and CARLA 0.9.15
for closed-loop evaluation. The Python and CUDA-enabled PyTorch dependencies
are pinned in [uv.lock](uv.lock).

Install Miniconda or Miniforge, then create the environment from the repository
root:

```bash
conda create -n clapad -c conda-forge python=3.10 \
  ffmpeg parallel tree gcc zip unzip git-lfs uv wget -y
conda activate clapad

# Configure uv to install into the active Conda environment.
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
printf 'export VIRTUAL_ENV="$CONDA_PREFIX"\n' \
  > "$CONDA_PREFIX/etc/conda/activate.d/uv.sh"
printf 'unset VIRTUAL_ENV\n' \
  > "$CONDA_PREFIX/etc/conda/deactivate.d/uv.sh"
conda deactivate && conda activate clapad
uv sync --active

export LEAD_PROJECT_ROOT="$PWD"
source scripts/main.sh
```

Install CARLA 0.9.15 with its additional maps:

```bash
bash scripts/setup_carla.sh
```

Alternatively, link an existing CARLA 0.9.15 installation at
`3rd_party/CARLA_0915`. Start the server in a separate terminal after
activating the environment, entering the repository, and setting up its
environment:

```bash
conda activate clapad
cd /path/to/clapad-dev
export LEAD_PROJECT_ROOT="$PWD"
source scripts/main.sh
bash scripts/start_carla.sh 30000
```

## CARLA evaluation

Download the CARLA TFv6 + CLAP-AD checkpoint from
[Hugging Face](https://huggingface.co/clap-ad/clap-ad-tfv6).
The directory must contain both `config.json` and `model_0030.pth`; the config
is required by the inference agent.

```bash
conda activate clapad
cd /path/to/clapad-dev
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="clap-ad/clap-ad-tfv6", local_dir="outputs/checkpoints/clapad-tfv6")'
```

With CARLA running on port `30000`, evaluate Town01 route `24211` from the
vendored Bench2Drive route list. The one-worker distributed mode is used
because the wrapper exposes route filtering through that evaluator:

```bash
conda activate clapad
cd /path/to/clapad-dev
export LEAD_PROJECT_ROOT="$PWD"
source scripts/main.sh

python lead/leaderboard_wrapper.py \
  --checkpoint outputs/checkpoints/clapad-tfv6 \
  --routes 3rd_party/Bench2Drive/leaderboard/data/bench2drive220.xml \
  --bench2drive \
  --multigpu --nproc-per-node 1 --routes-subset 24211 \
  --port 30000 --traffic-manager-port 38000
```

The Bench2Drive-Speed route set carries desired-speed profiles. Evaluate it
with a CARLA server on port `2000`:

```bash
conda activate clapad
cd /path/to/clapad-dev
export LEAD_PROJECT_ROOT="$PWD"
source scripts/main.sh

python lead/leaderboard_wrapper.py \
  --checkpoint outputs/checkpoints/clapad-tfv6 \
  --routes 3rd_party/Bench2Drive-Speed/Bench2Drive/leaderboard/data/b2dspd_eval48.xml \
  --bench2drive-speed \
  --port 2000 --traffic-manager-port 8000
```

For the full 220-route Bench2Drive evaluation, use
[scripts_clapad/eval_clapad_b2d.sh](scripts_clapad/eval_clapad_b2d.sh).
It shards routes across the selected GPUs and requires one CARLA server per
worker. The single-process Bench2Drive-Speed script is
[scripts_clapad/eval_clapad_b2dspeed.sh](scripts_clapad/eval_clapad_b2dspeed.sh).
Evaluation outputs are written under `outputs/local_evaluation/`.

## Training

Download the CARLA expert dataset from
[Hugging Face](https://huggingface.co/datasets/ln2697/lead_carla), then unzip
the route archives and build the feature cache:

```bash
conda activate clapad
cd /path/to/clapad-dev
export LEAD_PROJECT_ROOT="$PWD"
source scripts/main.sh

git clone https://huggingface.co/datasets/ln2697/lead_carla \
  data/carla_leaderboard2/zip
git -C data/carla_leaderboard2/zip lfs pull
bash scripts/unzip_routes.sh

python scripts/build_cache.py \
  carla_root=data/carla_leaderboard2 assigned_cpu_cores=8 workers_per_cpu_cores=1
python scripts/build_buckets_pretrain.py \
  carla_root=data/carla_leaderboard2 assigned_cpu_cores=8 workers_per_cpu_cores=1
```

Train the perception backbone, build planning buckets, then train the CLAP-AD
planning head:

```bash
conda activate clapad
cd /path/to/clapad-dev
export LEAD_PROJECT_ROOT="$PWD"
source scripts/main.sh

bash scripts/pretrain_ddp.sh
python scripts/build_buckets_posttrain.py \
  carla_root=data/carla_leaderboard2 assigned_cpu_cores=8 workers_per_cpu_cores=1
PRETRAINED_CKPT=outputs/local_training/pretrain/model_0030.pth \
  bash scripts_clapad/posttrain_clapad_b2d.sh
```

The training scripts default to batch size 512; set `BATCH_SIZE` to suit the
available GPU memory. Full training is compute- and memory-intensive.
Checkpoints and logs are written under `outputs/local_training/`.

## Upstream LEAD documentation

[README_LEAD.md](README_LEAD.md) is the original README for the underlying
LEAD codebase. It documents upstream LEAD features and is **not** the README
for this CLAP-AD repository.

## Acknowledgements and license

This work builds on [LEAD](https://github.com/kesai-labs/lead) and uses CARLA,
the CARLA Leaderboard, ScenarioRunner, Bench2Drive, and Bench2Drive-Speed.

This repository is released under the [MIT License](LICENSE), subject to the
third-party terms in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@inproceedings{ClapAD2027WACV,
  author    = {Aghajanzadeh, Emad and Schwarzfischer, Thomas and Kaufeld, Marc and 
               Thoma, Moritz and P{\'e}goud, Ryan and Vemparala, Manoj Rohit and
               Mori, Pierpaolo and Fasfous, Nael and Betz, Johannes and
               Mueller-Gritschneder, Daniel and Schlichtmann, Ulf},
  title     = {Closed-Loop Aware Planning via Imitation-Free Speed Scoring for
               End-to-End Autonomous Driving},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision},
  year      = {2027}
}
```

Our implementation builds on TFv6 and LEAD. Please cite it when using the
training and evaluation stack:

```bibtex
@inproceedings{Nguyen2026CVPR,
  author    = {Nguyen, Long and Fauth, Micha and Jaeger, Bernhard and
               Dauner, Daniel and Igl, Maximilian and Geiger, Andreas and
               Chitta, Kashyap},
  title     = {LEAD: Minimizing Learner-Expert Asymmetry in End-to-End Driving},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
```
