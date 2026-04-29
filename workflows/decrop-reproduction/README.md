# decrop-reproduction workflow

Reproduces the CNN inference step of [Decrop et al. 2025](https://doi.org/10.3389/fmars.2025.1699781):
runs the published EfficientNetV2-B0 phytoplankton classifier on a set
of FlowCam images using the planktonclas pipeline (10-crop test-time
augmentation by default).

## Workflow shape

```
[FlowCam images zip]
       │
       ├─► planktonclas-inference ──► [predictions.npz]
       │
[planktonclas model.tar.gz]
       │
[split file (image_path label)]
```

## Inputs

| Name | Type | Description |
|---|---|---|
| `model_archive` | data | `.tar.gz` of the planktonclas-format model directory (`ckpts/final_model.h5`, `conf.json`, `dataset_files/classes.txt`) |
| `images_archive` | data | `.zip` of FlowCam images, paths relative to archive root |
| `split_file` | data | text file with one `image_path label` per line |

## Output

NPZ with `y_true`, `y_pred`, `full_probs` (float16), `paths`. Schema-compatible
with `cnn_predictions_*.npz` consumed by the [bio-stacking workflow](../bio-stacking/).

## Tier 1 smoke test

Synthetic 6-image / 3-class fixture in `test-data/` (copied from the
per-tool test fixture in `galaxy-tools/tools/planktonclas-inference/test-data/`).
Tiny untrained Keras model; numbers are gibberish but the wiring is exercised.

```bash
planemo test \
    --extra_tools /Users/annef/Documents/ScienceLive/galaxy-tools/tools \
    --no_dependency_resolution \
    decrop-reproduction.gxwf.yml
```

## Tier 2 canonical reproduction

Required inputs for the canonical run:

| Input | Source |
|---|---|
| `model_archive` | [Zenodo record 15269453](https://zenodo.org/records/15269453) — `Phytoplankton_EfficientNetV2B0.tar.gz` (47 MB) |
| `images_archive` | [Zenodo record 10554845](https://zenodo.org/records/10554845) — LifeWatch FlowCam dataset (~650 MB, 7z archive; convert to zip) |
| `split_file` | bundled inside `Phytoplankton_EfficientNetV2B0.tar.gz` at `dataset_files/test.txt` |

**Expected canonical result:** CNN top-1 accuracy on Decrop's test split = **0.8634**
(matches the published result in [Decrop et al. 2025](https://doi.org/10.3389/fmars.2025.1699781)
to 0.003 percentage point — see [fiesta-decrop-reproduction](https://github.com/annefou/fiesta-decrop-reproduction)
for the original verification).

Runtime on Apple Silicon CPU: ~20–30 minutes for the ~33k-image test split with 10-crop TTA.

## Origin

Wraps the inference logic from
[fiesta-decrop-reproduction](https://github.com/annefou/fiesta-decrop-reproduction).
The Galaxy tool in turn wraps `planktonclas.test_utils.predict()`.
