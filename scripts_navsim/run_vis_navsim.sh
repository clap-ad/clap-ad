#!/usr/bin/bash
# Launch the CLAP-AD NAVSIM scene visualizer (vis_navsim_scenes.py).
#
# Renders per-scene BEV (GT vs CLAP-AD trajectory + current/target speed) and,
# for --agent clapad, the predicted per-bin speed-score bar chart. Needs no
# metric cache. Runs in the navsim venv (torch 2.0.1) with the lead package on
# PYTHONPATH.
#
# Usage:
#   scripts_clapad/run_vis_navsim.sh                      # default failure/success mix
#   scripts_clapad/run_vis_navsim.sh --tokens <t1> <t2>  # specific tokens
#   scripts_clapad/run_vis_navsim.sh --csv <results.csv> --no-cameras   # all score==0 scenes
#   scripts_clapad/run_vis_navsim.sh --checkpoint <dir-or-.pth>
#
# Any extra args are passed straight through to the Python script.

set -euo pipefail

# --- Paths (edit PROJECT_DIR if the repo lives elsewhere) -------------------
PROJECT_DIR=/workspace/lead
NAVSIM_ROOT="${PROJECT_DIR}/3rd_party/navsim_workspace"
DEVKIT="${NAVSIM_ROOT}/navsimv1.1"
VENV_PY="${DEVKIT}/venv/bin/python"
SCRIPT="${PROJECT_DIR}/scripts_clapad/vis_navsim_scenes.py"

# --- Environment (mirrors scripts/evaluation/run_clap.sh) -------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONDONTWRITEBYTECODE=1
export HYDRA_FULL_ERROR=1
export LEAD_PROJECT_ROOT="${PROJECT_DIR}"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT="${NAVSIM_ROOT}/dataset/maps"
export NAVSIM_EXP_ROOT="${NAVSIM_ROOT}/exp"
export NAVSIM_DEVKIT_ROOT="${DEVKIT}"
export OPENSCENE_DATA_ROOT="${NAVSIM_ROOT}/dataset"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# --- Launch -----------------------------------------------------------------
exec "${VENV_PY}" "${SCRIPT}" "$@"
