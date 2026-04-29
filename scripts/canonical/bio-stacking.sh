#!/usr/bin/env bash
# Canonical reproduction run for the bio-stacking workflow.
#
# Reproduces the FIESTA-bio scattering-stacking chain end-to-end via
# Galaxy: 3 foscat-features runs (train_balanced, val, test splits) +
# 2 planktonclas-inference runs (val, test) + 1 scattering-stacking
# meta-classifier. Compares the output to the published numbers in
# the fiesta-scattering-bio FORRT nanopub.
#
# Expected canonical results:
#     CNN top-1 (held-out test):       0.8634
#     Stacked top-1:                    0.8562  (-0.72 pp vs CNN)
#     CNN rare-class recall:            0.4770
#     Stacked rare-class recall:        0.5608  (+8.4 pp headline)
#     Oracle (hard-switch ceiling):     0.6455
#     Classes improved by stacking:     33 of 95
#
# Runtime on Apple Silicon CPU (OMP_NUM_THREADS=1):
#     foscat-features × 3 splits ≈ 20 min
#     planktonclas-inference × 2 splits ≈ 30 min (10-crop TTA)
#     scattering-stacking ≈ 1 min
#     Total: ~50–60 min
#
# Inputs needed locally (same as decrop-reproduction.sh):
#     ${PLANKTONCLAS_MODELS:-...}/Phytoplankton_EfficientNetV2B0/
#     ${FIESTA_BIO_DATA:-...}/images_DS/
#
# Usage:
#     scripts/canonical/bio-stacking.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${REPO_ROOT}/scripts/canonical/_bio_run"
INPUTS_DIR="${RUN_DIR}/inputs"
OUTPUTS_DIR="${RUN_DIR}/outputs"
WORKFLOW="${REPO_ROOT}/workflows/bio-stacking/bio-stacking.gxwf.yml"

mkdir -p "${INPUTS_DIR}" "${OUTPUTS_DIR}"

PLANKTONCLAS_MODELS="${PLANKTONCLAS_MODELS:-${HOME}/Documents/ScienceLive/planktonclas/models}"
FIESTA_BIO_DATA="${FIESTA_BIO_DATA:-${HOME}/Documents/ScienceLive/fiesta-scattering-bio/data/images_DS}"
MODEL_DIR="${PLANKTONCLAS_MODELS}/Phytoplankton_EfficientNetV2B0"

if [ ! -f "${MODEL_DIR}/ckpts/final_model.h5" ] || [ ! -d "${FIESTA_BIO_DATA}" ]; then
    echo "ERROR: planktonclas model or FlowCam dataset missing." >&2
    echo "       Set PLANKTONCLAS_MODELS and FIESTA_BIO_DATA env vars." >&2
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

# --- 1. Build the balanced training split (≤100 per class) ------------

PYTHON="${PYTHON_BIN:-${HOME}/Documents/ScienceLive/foscat-venv/bin/python}"
TRAIN_BALANCED="${INPUTS_DIR}/train_balanced.txt"
VAL_TXT="${INPUTS_DIR}/val.txt"
TEST_TXT="${INPUTS_DIR}/test.txt"
CLASSES_TXT="${INPUTS_DIR}/classes.txt"

cp "${MODEL_DIR}/dataset_files/val.txt" "${VAL_TXT}"
cp "${MODEL_DIR}/dataset_files/test.txt" "${TEST_TXT}"
cp "${MODEL_DIR}/dataset_files/classes.txt" "${CLASSES_TXT}"

if [ ! -f "${TRAIN_BALANCED}" ]; then
    echo "Building balanced training split (≤100 per class, seed=42)..."
    "${PYTHON}" - <<EOF
from collections import defaultdict
import numpy as np

src = "${MODEL_DIR}/dataset_files/train.txt"
N_CLASSES = sum(1 for _ in open("${CLASSES_TXT}"))
MAX_PER_CLASS = 100
SEED = 42

train_by_class = defaultdict(list)
with open(src) as f:
    for line in f:
        rel, lab = line.strip().rsplit(" ", 1)
        train_by_class[int(lab)].append(rel)

rng = np.random.default_rng(SEED)
lines = []
for cls in range(N_CLASSES):
    files = train_by_class.get(cls, [])
    if not files:
        continue
    k = min(MAX_PER_CLASS, len(files))
    chosen = (rng.choice(files, size=k, replace=False) if len(files) > k else files)
    for rel in chosen:
        lines.append(f"{rel} {cls}")
with open("${TRAIN_BALANCED}", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote ${TRAIN_BALANCED}: {len(lines)} entries (vs full train: {sum(len(v) for v in train_by_class.values())})")
EOF
fi

# --- 2. Tar up the planktonclas model archive --------------------------

MODEL_TGZ="${INPUTS_DIR}/model.tar.gz"
if [ ! -f "${MODEL_TGZ}" ]; then
    echo "Building model.tar.gz..."
    tar -C "${PLANKTONCLAS_MODELS}" -czf "${MODEL_TGZ}" "Phytoplankton_EfficientNetV2B0"
fi

# --- 3. Build images.zip containing only train_balanced+val+test images

IMAGES_ZIP="${INPUTS_DIR}/images.zip"
if [ ! -f "${IMAGES_ZIP}" ]; then
    echo "Zipping images for train_balanced + val + test splits..."
    cd "${FIESTA_BIO_DATA}"
    cat "${TRAIN_BALANCED}" "${VAL_TXT}" "${TEST_TXT}" \
        | awk '{print $1}' \
        | sort -u \
        | zip -q "${IMAGES_ZIP}" -@
    echo "  $(stat -f '%z' "${IMAGES_ZIP}") bytes"
    cd "${REPO_ROOT}"
fi

# --- 4. Build the job spec ---------------------------------------------

JOB_YML="${RUN_DIR}/job.yml"
cat > "${JOB_YML}" <<EOF
images_archive:
  class: File
  path: ${IMAGES_ZIP}
train_balanced_split:
  class: File
  path: ${TRAIN_BALANCED}
val_split:
  class: File
  path: ${VAL_TXT}
test_split:
  class: File
  path: ${TEST_TXT}
classes_file:
  class: File
  path: ${CLASSES_TXT}
model_archive:
  class: File
  path: ${MODEL_TGZ}
rare_class_threshold: 200
stacking_C: 1.0
stacking_max_iter: 2000
EOF

# --- 5. Run the workflow -----------------------------------------------

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

# --- 6. Pretty-print the stacking results ------------------------------

RESULTS_JSON=$(find "${OUTPUTS_DIR}" -name "*.json" -size +500c -not -name "run-outputs.json" | head -1)

if [ -n "${RESULTS_JSON}" ]; then
    "${PYTHON}" - <<EOF
import json
with open("${RESULTS_JSON}") as f:
    d = json.load(f)

print()
print("=" * 60)
print("FIESTA-bio canonical reproduction results")
print("=" * 60)
print(f"  CNN alone        — top-1: {d['cnn']['top1']:.4f}  rare-recall: {d['cnn']['rare_recall']:.4f}")
print(f"  Scattering alone — top-1: {d['scattering']['top1']:.4f}  rare-recall: {d['scattering']['rare_recall']:.4f}")
print(f"  50/50 ensemble   — top-1: {d['ens_50_50']['top1']:.4f}  rare-recall: {d['ens_50_50']['rare_recall']:.4f}")
print(f"  Stacked LR       — top-1: {d['stacked_val']['top1']:.4f}  rare-recall: {d['stacked_val']['rare_recall']:.4f}")
print(f"  Oracle ceiling   — top-1: {d['oracle']['top1']:.4f}  rare-recall: {d['oracle']['rare_recall']:.4f}")
print(f"  Classes improved by stacking: {d['n_classes_better_stacked']} / "
      f"{d['n_classes_better_stacked'] + d['n_classes_worse_stacked']}+")
print()
print("Compare to published canonical (FORRT bio nanopub):")
print("  CNN top-1                  : 0.8634")
print("  Stacked top-1              : 0.8562 (-0.72 pp)")
print("  CNN rare recall            : 0.4770")
print("  Stacked rare recall        : 0.5608 (+8.4 pp headline)")
print("  Oracle ceiling rare recall : 0.6455")
print("  Classes improved           : 33 of 95")
EOF
else
    echo "ERROR: stacking results JSON not found in ${OUTPUTS_DIR}/" >&2
    exit 1
fi
