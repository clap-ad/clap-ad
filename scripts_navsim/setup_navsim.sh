#!/usr/bin/env bash
#
# One-shot installer for the NAVSIM virtual environment.
#
# Usage:
#   ./scripts_navsim/setup_navsim.sh
#
# This script is idempotent: existing symlinks / venv are reused, so it is
# safe to re-run. After it finishes, source the environment file to run
# NAVSIM commands:
#
#   source scripts_navsim/navsim_env.sh
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

# Optional source of the NAVSIM dataset. It should contain navsim_logs/,
# sensor_blobs/, and maps/.
NAVSIM_DATA_SRC="${NAVSIM_DATA_SRC:-}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

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

if [ -n "${NAVSIM_DATA_SRC}" ]; then
    if [ ! -d "${NAVSIM_DATA_SRC}" ]; then
        echo "ERROR: NAVSIM_DATA_SRC directory not found: ${NAVSIM_DATA_SRC}" >&2
        exit 1
    fi
    echo "==> Linking NAVSIM dataset from ${NAVSIM_DATA_SRC}"
    for sub in navsim_logs sensor_blobs maps; do
        if [ ! -d "${NAVSIM_DATA_SRC}/${sub}" ]; then
            echo "WARNING: ${NAVSIM_DATA_SRC}/${sub} does not exist, skipping" >&2
            continue
        fi
        ln -sfn "${NAVSIM_DATA_SRC}/${sub}" "${DATASET_DIR}/${sub}"
        echo "    ${DATASET_DIR}/${sub} -> ${NAVSIM_DATA_SRC}/${sub}"
    done
else
    echo "==> NAVSIM_DATA_SRC not set; skipping dataset links"
fi

# ---------------------------------------------------------------------------
# 3. Virtualenv
# ---------------------------------------------------------------------------
VENV_DIR="${NAVSIM_DEVKIT_ROOT}/venv"
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        echo "ERROR: ${PYTHON_BIN} not found; install Python 3.10 with venv support or set PYTHON_BIN" >&2
        exit 1
    fi
    if ! "${PYTHON_BIN}" -c 'import sys; assert sys.version_info[:2] == (3, 10)' >/dev/null 2>&1; then
        echo "ERROR: NAVSIM requires Python 3.10; ${PYTHON_BIN} is not Python 3.10" >&2
        exit 1
    fi
    echo "==> Creating virtualenv at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
    echo "==> Reusing existing virtualenv at ${VENV_DIR}"
fi
if ! "${VENV_DIR}/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 10)' >/dev/null 2>&1; then
    echo "ERROR: existing NAVSIM venv must use Python 3.10: ${VENV_DIR}" >&2
    exit 1
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
pip install --no-deps -e "${LEAD_PROJECT_ROOT}"

echo "==> Installing LEAD dependencies not supplied by NAVSIM"
python - "${NAVSIM_DEVKIT_ROOT}/requirements.txt" <<'PY'
from importlib.metadata import distribution
from packaging.requirements import Requirement
import subprocess
import sys

with open(sys.argv[1]) as requirements_file:
    navsim_requirements = {
        Requirement(line.split("#", 1)[0].strip()).name.lower()
        for line in requirements_file
        if line.split("#", 1)[0].strip()
    }
navsim_runtime_pins = {"numpy", "numba", "torch", "torchaudio", "torchvision"}
lead_requirements = []
for item in distribution("lead").requires or []:
    requirement = Requirement(item)
    if (
        requirement.name.lower() not in navsim_requirements
        and requirement.name.lower() not in navsim_runtime_pins
        and (requirement.marker is None or requirement.marker.evaluate())
    ):
        lead_requirements.append(str(requirement))

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", *lead_requirements, "numba==0.60.0"]
)
PY

# Installing LEAD-only dependencies can upgrade packages shared with NAVSIM.
# Restore NAVSIM's direct pins without resolving or replacing their dependencies.
pip install --no-deps -r "${NAVSIM_DEVKIT_ROOT}/requirements.txt"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
cat <<EOF

==============================================================================
NAVSIM setup complete.

To use it in a new shell, activate the environment:

    source scripts_navsim/navsim_env.sh

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
