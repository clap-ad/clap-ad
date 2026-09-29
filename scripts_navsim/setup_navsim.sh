#!/usr/bin/env bash
#
# One-shot installer for NAVSIM dependencies inside the LEAD devcontainer.
#
# Usage:
#   ./scripts/setup_navsim.sh
#
# This script is idempotent: existing symlinks / venv are reused, so it is
# safe to re-run. After it finishes, source the environment file to run
# NAVSIM commands:
#
#   source scripts/navsim_env.sh
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
LEAD_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NAVSIM_WORKSPACE="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace"
NAVSIM_DEVKIT_ROOT="${NAVSIM_WORKSPACE}/navsimv1.1"
DATASET_DIR="${NAVSIM_WORKSPACE}/dataset"
EXP_DIR="${NAVSIM_WORKSPACE}/exp"

# Source of the NAVSIM dataset. Point this at wherever you downloaded/mounted
# the NAVSIM v2 dataset (navsim_logs/, sensor_blobs/, maps/), e.g. via NAVSIM's
# own download scripts (3rd_party/navsim_workspace/navsimv2.2/download/*.sh).
NAVSIM_DATA_SRC="${NAVSIM_DATA_SRC:-/data/datasets/AD_datasets/navsim_v2}"

echo "==> LEAD project root: ${LEAD_PROJECT_ROOT}"
echo "==> NAVSIM devkit:     ${NAVSIM_DEVKIT_ROOT}"

if [ ! -d "${NAVSIM_DEVKIT_ROOT}" ]; then
    echo "ERROR: NAVSIM devkit not found at ${NAVSIM_DEVKIT_ROOT}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Directory layout + dataset symlinks
# ---------------------------------------------------------------------------
echo "==> Creating dataset/exp directories"
mkdir -p "${DATASET_DIR}" "${EXP_DIR}"

echo "==> Linking NAVSIM dataset from ${NAVSIM_DATA_SRC}"
if [ ! -d "${NAVSIM_DATA_SRC}" ]; then
    echo "ERROR: NAVSIM data source not found at ${NAVSIM_DATA_SRC}" >&2
    exit 1
fi
for sub in navsim_logs sensor_blobs maps; do
    if [ ! -d "${NAVSIM_DATA_SRC}/${sub}" ]; then
        echo "WARNING: ${NAVSIM_DATA_SRC}/${sub} does not exist, skipping" >&2
        continue
    fi
    ln -sfn "${NAVSIM_DATA_SRC}/${sub}" "${DATASET_DIR}/${sub}"
    echo "    ${DATASET_DIR}/${sub} -> ${NAVSIM_DATA_SRC}/${sub}"
done

# ---------------------------------------------------------------------------
# 2. System package: python3.10-venv
# ---------------------------------------------------------------------------
if ! dpkg -s python3.10-venv >/dev/null 2>&1; then
    echo "==> Installing python3.10-venv"
    sudo apt-get update
    sudo apt-get install -y python3.10-venv
else
    echo "==> python3.10-venv already installed"
fi

# ---------------------------------------------------------------------------
# 3. Virtualenv
# ---------------------------------------------------------------------------
VENV_DIR="${NAVSIM_DEVKIT_ROOT}/venv"
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "==> Creating virtualenv at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
else
    echo "==> Reusing existing virtualenv at ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip

# ---------------------------------------------------------------------------
# 4. Python dependencies
# ---------------------------------------------------------------------------
echo "==> Installing NAVSIM devkit requirements"
pip install -r "${NAVSIM_DEVKIT_ROOT}/requirements.txt"
pip install -e "${NAVSIM_DEVKIT_ROOT}"

echo "==> Installing LEAD into the NAVSIM venv"
pip install -e "${LEAD_PROJECT_ROOT}"

echo "==> Installing extra NAVSIM/LEAD dependencies"
pip install beartype jaxtyping carla numba

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
cat <<EOF

==============================================================================
NAVSIM setup complete.

To use it in a new shell, activate the environment:

    source scripts/navsim_env.sh

Then, for example, build the metric cache and run evaluation:

    python \$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \\
        train_test_split=navtest \\
        cache.cache_path=\$NAVSIM_EXP_ROOT/metric_cache

    python \$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \\
        train_test_split=navtest \\
        agent=carla_transfuser_agent \\
        worker=single_machine_thread_pool \\
        worker.max_workers=8 \\
        agent.checkpoint_path=/workspace/lead/outputs/checkpoints/tfv6_navsim/model_0060.pth \\
        experiment_name=carla_transfuser_agent_eval \\
        metric_cache_path=\$NAVSIM_EXP_ROOT/metric_cache_navtest
==============================================================================
EOF
