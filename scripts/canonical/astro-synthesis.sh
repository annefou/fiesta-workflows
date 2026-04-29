#!/usr/bin/env bash
# Canonical reproduction run for the astro-synthesis workflow.
#
# Prepares the Large Scale Structure HEALPix map from FOSCAT_DEMO at
# the same NSIDE the original fiesta-scattering-astro notebook used
# (NSIDE=32, NSTEPS=300), runs the Galaxy workflow against it, and
# prints the result for comparison with the published numbers in the
# fiesta-scattering-astro FORRT nanopub.
#
# Usage:
#     scripts/canonical/astro-synthesis.sh
#
# Apple Silicon: set OMP_NUM_THREADS=1 to dodge the libomp/torch
# segfault. On Linux x86-64 it's a no-op.
#
# Outputs:
#     scripts/canonical/_astro_run/synthesis.npy
#     scripts/canonical/_astro_run/results.json

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${REPO_ROOT}/scripts/canonical/_astro_run"
INPUTS_DIR="${RUN_DIR}/inputs"
OUTPUTS_DIR="${RUN_DIR}/outputs"
WORKFLOW="${REPO_ROOT}/workflows/astro-synthesis/astro-synthesis.gxwf.yml"

mkdir -p "${INPUTS_DIR}" "${OUTPUTS_DIR}"

# Tools must be assembled into a single directory because each tool
# lives on its own feature branch in annefou/galaxy-tools.
FIESTA_TOOLS="${RUN_DIR}/fiesta-tools"
mkdir -p "${FIESTA_TOOLS}"

GALAXY_TOOLS_LOCAL="${GALAXY_TOOLS:-/Users/annef/Documents/ScienceLive/galaxy-tools}"
if [ ! -d "${GALAXY_TOOLS_LOCAL}/.git" ]; then
    echo "ERROR: galaxy-tools clone not found at ${GALAXY_TOOLS_LOCAL}"
    echo "       Set GALAXY_TOOLS=/path/to/your/clone, or clone annefou/galaxy-tools."
    exit 1
fi

for branch in scattering-stacking planktonclas-inference foscat-features foscat-synthesis; do
    git -C "${GALAXY_TOOLS_LOCAL}" archive "${branch}" "tools/${branch}" \
        | tar -x -C "${FIESTA_TOOLS}" --strip-components=1
done

# --- 1. Download the reference LSS map ----------------------------------

LSS_RAW="${INPUTS_DIR}/LSS_map_nside128.npy"
if [ ! -f "${LSS_RAW}" ]; then
    echo "Downloading LSS map (nside=128) from FOSCAT_DEMO..."
    curl -sL -o "${LSS_RAW}" \
        "https://github.com/jmdelouis/FOSCAT_DEMO/raw/main/data/LSS_map_nside128.npy"
    echo "  $(stat -f '%z' "${LSS_RAW}") bytes"
fi

# --- 2. Prepare target_map at canonical NSIDE=32 -----------------------
#
# The original notebook ud_grades the source map to NSIDE=32 and reshapes
# to (1, 12*nside^2). We do the same here.

TARGET_NPY="${INPUTS_DIR}/target.npy"
PYTHON="${PYTHON_BIN:-/Users/annef/Documents/ScienceLive/foscat-venv/bin/python}"

"${PYTHON}" - <<EOF
import numpy as np
import healpy as hp

src = np.load("${LSS_RAW}")
print(f"Source: nside=128, npix={src.shape[0]:,}")

NSIDE = 32
target = hp.ud_grade(src, NSIDE, order_in='NESTED', order_out='NESTED').astype(np.float32)
target = target.reshape(1, 12 * NSIDE * NSIDE)
print(f"Target: nside={NSIDE}, shape={target.shape}, "
      f"mean={target.mean():.4f}, std={target.std():.4f}")
np.save("${TARGET_NPY}", target)
print(f"Wrote ${TARGET_NPY}")
EOF

# --- 3. Build the job spec ---------------------------------------------

JOB_YML="${RUN_DIR}/job.yml"
cat > "${JOB_YML}" <<EOF
target_map:
  class: File
  path: ${TARGET_NPY}
norient: 4
kernelsz: 3
nsteps: 300
seed: 1234
EOF
echo "Wrote ${JOB_YML}"

# --- 4. Run the workflow -----------------------------------------------

cd "${RUN_DIR}"

PLANEMO="${PLANEMO_BIN:-/Users/annef/Documents/ScienceLive/galaxy-tools-dev-env/bin/planemo}"

OMP_NUM_THREADS=1 _CONDA_EXE_SET=1 \
    "${PLANEMO}" run \
        --extra_tools "${FIESTA_TOOLS}" \
        --conda_dependency_resolution \
        --download_outputs \
        --output_directory "${OUTPUTS_DIR}" \
        --output_json "${RUN_DIR}/run-outputs.json" \
        "${WORKFLOW}" "${JOB_YML}"

# --- 5. Pretty-print the results ---------------------------------------

RESULTS_JSON=$(find "${OUTPUTS_DIR}" -name "*.json" -size +50c | head -1)
if [ -n "${RESULTS_JSON}" ]; then
    echo
    echo "=== Canonical run results: ${RESULTS_JSON} ==="
    cat "${RESULTS_JSON}" | "${PYTHON}" -m json.tool
    echo
    echo "Compare scat_improvement_pct to the published number in the"
    echo "fiesta-scattering-astro FORRT nanopub."
else
    echo "No results JSON found in ${OUTPUTS_DIR}/" >&2
    exit 1
fi
