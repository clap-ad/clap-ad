#!/usr/bin/bash
# Build a NAVSIM training cache for the CLAP-AD (carla_transfuser) agent.
#
# Stores the precomputed, origin-anchored, 1 m-spaced GT route (from the full
# raw-log future) directly in each target, so training uses it as GT without
# masking or per-sample resampling.
#
# NOTE: this builds the *legacy* cache: //4 camera downsize, JPEG quality 30,
# and a +/-32 m BEV/LiDAR extent. For a higher-quality-image / wider-BEV
# variant (recommended for better curb/road-geometry adherence with a 48 m
# path horizon), use scripts_navsim/build_navsim_cache_hq_wide_bev.sh instead.
#
# Usage:
#   scripts_clapad/build_navsim_cache.sh                     # navtrain (default)
#   SPLIT=navmini scripts_clapad/build_navsim_cache.sh       # quick validation
#   CACHE_DIR=/path/to/cache scripts_clapad/build_navsim_cache.sh
#   MAX_WORKERS=16 scripts_clapad/build_navsim_cache.sh
#   scripts_clapad/build_navsim_cache.sh <extra hydra overrides...>
#
# Any extra args are passed straight through to run_dataset_caching.py.

set -euo pipefail

# --- Paths / config (override via env) --------------------------------------
PROJECT_DIR="${LEAD_PROJECT_ROOT:-/workspace/lead}"
NAVSIM_ROOT="${PROJECT_DIR}/3rd_party/navsim_workspace"
DEVKIT="${NAVSIM_ROOT}/navsimv1.1"
VENV_PY="${DEVKIT}/venv/bin/python"

SPLIT="${SPLIT:-navtrain}"
MAX_WORKERS="${MAX_WORKERS:-8}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_DIR}/data/navsim_training_cache/${SPLIT}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${SPLIT}_cache_transfuser}"

# --- Environment (self-contained; required for the full-future GT) ----------
export LEAD_PROJECT_ROOT="${PROJECT_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export HYDRA_FULL_ERROR=1
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT="${NAVSIM_ROOT}/dataset/maps"
export NAVSIM_EXP_ROOT="${NAVSIM_ROOT}/exp"
export NAVSIM_DEVKIT_ROOT="${DEVKIT}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${NAVSIM_ROOT}/dataset}"   # raw logs for full-future GT
export CARLA_ROOT="${CARLA_ROOT:-${PROJECT_DIR}/3rd_party/CARLA_0915}"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${PROJECT_DIR}:${PYTHONPATH:-}"

echo "Split:     ${SPLIT}"
echo "Cache dir: ${CACHE_DIR}"
echo "Workers:   ${MAX_WORKERS}"

# --- Run --------------------------------------------------------------------
exec "${VENV_PY}" \
  "${DEVKIT}/navsim/planning/script/run_dataset_caching.py" \
  agent=carla_transfuser_agent \
  train_test_split="${SPLIT}" \
  worker=single_machine_thread_pool \
  worker.max_workers="${MAX_WORKERS}" \
  experiment_name="${EXPERIMENT_NAME}" \
  cache_path="${CACHE_DIR}" \
  "$@"
