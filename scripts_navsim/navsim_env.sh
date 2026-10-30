# NAVSIM environment variables for use with the LEAD repo.
#
# Source this file (do NOT execute it) before building caches, training,
# or running evaluation:
#
#   source scripts/navsim_env.sh
#
# It also activates the NAVSIM virtualenv created by scripts/setup_navsim.sh
# if that venv exists.

# Resolve the LEAD project root from this file's location (scripts/ -> repo root).
_NAVSIM_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export LEAD_PROJECT_ROOT="$(cd "${_NAVSIM_ENV_DIR}/.." && pwd)"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export CARLA_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/CARLA_0915"
export NUPLAN_MAPS_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/navsimv1.1"
export OPENSCENE_DATA_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/dataset"
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${LEAD_PROJECT_ROOT}:${PYTHONPATH:-}"

# Activate the NAVSIM venv by default. Keep the project's CUDA/PyTorch
# environment active for LEAD training with NAVSIM caches by setting
# NAVSIM_ACTIVATE_VENV=0 before sourcing this file.
_NAVSIM_VENV="${NAVSIM_DEVKIT_ROOT}/venv"
if [ "${NAVSIM_ACTIVATE_VENV:-1}" = "0" ]; then
    :
elif [ -f "${_NAVSIM_VENV}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${_NAVSIM_VENV}/bin/activate"
else
    echo "[navsim_env] NAVSIM venv not found at ${_NAVSIM_VENV}."
    echo "[navsim_env] Run scripts/setup_navsim.sh first."
fi
