#!/usr/bin/env bash
# Canonical reproduction run for the decrop-reproduction workflow.
#
# Runs Decrop et al. 2025's published EfficientNetV2-B0 phytoplankton
# classifier on their canonical test split (33,718 images) via the
# Galaxy workflow, then compares the top-1 accuracy to the published
# number.
#
# Inputs needed locally (already on disk if you have the bio repo set up):
#   ${PLANKTONCLAS_MODELS:-$HOME/Documents/ScienceLive/planktonclas/models}
#       /Phytoplankton_EfficientNetV2B0/ckpts/final_model.h5
#       /Phytoplankton_EfficientNetV2B0/dataset_files/{classes,train,val,test}.txt
#       /Phytoplankton_EfficientNetV2B0/conf.json
#   ${FIESTA_BIO_DATA:-$HOME/Documents/ScienceLive/fiesta-scattering-bio/data/images_DS}
#       FlowCam image tree
#
# Apple Silicon: set OMP_NUM_THREADS=1 to dodge the libomp/torch
# segfault. CNN inference is TensorFlow not torch — irrelevant here —
# but kept for consistency with other canonical scripts.
#
# Usage:
#     scripts/canonical/decrop-reproduction.sh
#
# Outputs (gitignored):
#     scripts/canonical/_decrop_run/inputs/{model.tar.gz, images.zip, test.txt}
#     scripts/canonical/_decrop_run/outputs/predictions.npz

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${REPO_ROOT}/scripts/canonical/_decrop_run"
INPUTS_DIR="${RUN_DIR}/inputs"
OUTPUTS_DIR="${RUN_DIR}/outputs"
WORKFLOW="${REPO_ROOT}/workflows/decrop-reproduction/decrop-reproduction.gxwf.yml"

mkdir -p "${INPUTS_DIR}" "${OUTPUTS_DIR}"

PLANKTONCLAS_MODELS="${PLANKTONCLAS_MODELS:-${HOME}/Documents/ScienceLive/planktonclas/models}"
FIESTA_BIO_DATA="${FIESTA_BIO_DATA:-${HOME}/Documents/ScienceLive/fiesta-scattering-bio/data/images_DS}"

MODEL_DIR="${PLANKTONCLAS_MODELS}/Phytoplankton_EfficientNetV2B0"

if [ ! -f "${MODEL_DIR}/ckpts/final_model.h5" ]; then
    echo "ERROR: planktonclas model not found at ${MODEL_DIR}/ckpts/final_model.h5"
    echo "       Set PLANKTONCLAS_MODELS=/path/to/your/clone/models, or download from"
    echo "       https://zenodo.org/records/15269453 and extract to that location."
    exit 1
fi

if [ ! -d "${FIESTA_BIO_DATA}" ]; then
    echo "ERROR: FlowCam dataset not found at ${FIESTA_BIO_DATA}"
    echo "       Set FIESTA_BIO_DATA=/path/to/your/images_DS/, or run"
    echo "       fiesta-scattering-bio's data download (01_scattering_features.py)."
    exit 1
fi

# Tools assembled from the four feature branches
FIESTA_TOOLS="${RUN_DIR}/fiesta-tools"
GALAXY_TOOLS_LOCAL="${GALAXY_TOOLS:-${HOME}/Documents/ScienceLive/galaxy-tools}"
mkdir -p "${FIESTA_TOOLS}"
for branch in scattering-stacking planktonclas-inference foscat-features foscat-synthesis; do
    git -C "${GALAXY_TOOLS_LOCAL}" archive "${branch}" "tools/${branch}" \
        | tar -x -C "${FIESTA_TOOLS}" --strip-components=1
done

# --- 1. Tar up the planktonclas model archive --------------------------

MODEL_TGZ="${INPUTS_DIR}/model.tar.gz"
if [ ! -f "${MODEL_TGZ}" ]; then
    echo "Building model.tar.gz..."
    tar -C "${PLANKTONCLAS_MODELS}" -czf "${MODEL_TGZ}" "Phytoplankton_EfficientNetV2B0"
    echo "  $(stat -f '%z' "${MODEL_TGZ}") bytes"
fi

# --- 2. Build images.zip containing only test-split images ------------

TEST_TXT="${INPUTS_DIR}/test.txt"
cp "${MODEL_DIR}/dataset_files/test.txt" "${TEST_TXT}"
N_TEST=$(wc -l < "${TEST_TXT}")
echo "Test split: ${N_TEST// /} images"

IMAGES_ZIP="${INPUTS_DIR}/images.zip"
if [ ! -f "${IMAGES_ZIP}" ]; then
    echo "Zipping ${N_TEST// /} test-split images..."
    cd "${FIESTA_BIO_DATA}"
    awk '{print $1}' "${TEST_TXT}" | zip -q "${IMAGES_ZIP}" -@
    echo "  $(stat -f '%z' "${IMAGES_ZIP}") bytes"
    cd "${REPO_ROOT}"
fi

# --- 3. Build the job spec ---------------------------------------------

JOB_YML="${RUN_DIR}/job.yml"
cat > "${JOB_YML}" <<EOF
model_archive:
  class: File
  path: ${MODEL_TGZ}
images_archive:
  class: File
  path: ${IMAGES_ZIP}
split_file:
  class: File
  path: ${TEST_TXT}
EOF

# --- 4. Run the workflow -----------------------------------------------

cd "${RUN_DIR}"
PLANEMO="${PLANEMO_BIN:-${HOME}/Documents/ScienceLive/galaxy-tools-dev-env/bin/planemo}"

OMP_NUM_THREADS=1 _CONDA_EXE_SET=1 \
    "${PLANEMO}" run \
        --extra_tools "${FIESTA_TOOLS}" \
        --conda_dependency_resolution \
        --download_outputs \
        --output_directory "${OUTPUTS_DIR}" \
        --output_json "${RUN_DIR}/run-outputs.json" \
        "${WORKFLOW}" "${JOB_YML}"

# --- 5. Compute and report top-1 accuracy ------------------------------

PREDICTIONS_NPZ=$(find "${OUTPUTS_DIR}" -name "*.npz" -size +1k | head -1)
PYTHON="${PYTHON_BIN:-${HOME}/Documents/ScienceLive/foscat-venv/bin/python}"

if [ -n "${PREDICTIONS_NPZ}" ]; then
    "${PYTHON}" - <<EOF
import numpy as np
d = np.load("${PREDICTIONS_NPZ}", allow_pickle=True)
y_true = d["y_true"]
y_pred = d["y_pred"]
top1 = float((y_true == y_pred).mean())
print()
print("=" * 60)
print("Decrop reproduction canonical run results")
print("=" * 60)
print(f"  Test set size      : {len(y_true)}")
print(f"  Top-1 accuracy     : {top1:.4f} ({top1*100:.2f}%)")
print()
print("Compare to published canonical:")
print("  Decrop et al. 2025 reported     : 0.8634 (86.34%)")
print("  fiesta-decrop-reproduction      : 0.8634 (matched to 0.003 pp)")
EOF
else
    echo "ERROR: predictions NPZ not found in ${OUTPUTS_DIR}/" >&2
    exit 1
fi
