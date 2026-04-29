"""Generate test fixtures for the bio-stacking workflow smoke test.

Produces:
  images.zip            — 9 small synthetic RGB JPEGs across 3 classes
  train.txt             — 3 paths (1 per class), used as the balanced
                          training split
  val.txt               — 3 paths (1 per class)
  test.txt              — 3 paths (1 per class)
  classes.txt           — 3 class names
  model.tar.gz          — minimal planktonclas-format archive with a
                          tiny untrained Keras model (shared shape from
                          the per-tool fixture; copied unchanged from
                          galaxy-tools/tools/planktonclas-inference/test-data/)

Numbers produced by the workflow on this fixture are gibberish — the
model is untrained, the splits are tiny. The fixture only verifies
that the 6-step DAG executes correctly.
"""

import json
import shutil
import tarfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

HERE = Path(__file__).resolve().parent
WORK = HERE / "_build"
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir()

IM_SIZE = 32
N_CLASSES = 3
CLASS_NAMES = ["alpha", "beta", "gamma"]
N_PER_CLASS = 3   # 1 train + 1 val + 1 test per class

SOURCE_SIZES = [
    (32, 32), (96, 72), (24, 40),
    (128, 64), (80, 80), (48, 64),
    (40, 56), (72, 96), (60, 60),
]
assert len(SOURCE_SIZES) == N_CLASSES * N_PER_CLASS

# --- 1. Generate 9 images, named class_<i>_<HxW>.jpg ----------------------

IMG_DIR = WORK / "images"
IMG_DIR.mkdir()
rng = np.random.default_rng(42)

paths_by_class = {c: [] for c in CLASS_NAMES}
src_iter = iter(SOURCE_SIZES)

for cls_idx, cls_name in enumerate(CLASS_NAMES):
    cls_dir = IMG_DIR / cls_name
    cls_dir.mkdir()
    for i in range(N_PER_CLASS):
        base = np.array(
            [
                [80, 40, 200],   # alpha → blueish
                [200, 80, 40],   # beta  → reddish
                [40, 200, 80],   # gamma → greenish
            ][cls_idx],
            dtype=np.int16,
        )
        h, w = next(src_iter)
        noise = rng.integers(-20, 20, size=(h, w, 3))
        arr = np.clip(base[None, None, :] + noise, 0, 255).astype(np.uint8)
        rel_path = f"{cls_name}/{cls_name}_{i}_{h}x{w}.jpg"
        Image.fromarray(arr).save(IMG_DIR / rel_path, "JPEG")
        paths_by_class[cls_name].append(rel_path)

# --- 2. Build splits — 1 image per class per split -----------------------

splits = {"train": [], "val": [], "test": []}
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    splits["train"].append((paths_by_class[cls_name][0], cls_idx))
    splits["val"].append((paths_by_class[cls_name][1], cls_idx))
    splits["test"].append((paths_by_class[cls_name][2], cls_idx))

for name, lines in splits.items():
    with open(HERE / f"{name}.txt", "w") as f:
        for path, lab in lines:
            f.write(f"{path} {lab}\n")
    print(f"Wrote {name}.txt: {len(lines)} entries")

with open(HERE / "classes.txt", "w") as f:
    for name in CLASS_NAMES:
        f.write(name + "\n")
print(f"Wrote classes.txt: {N_CLASSES} classes")

# --- 3. Zip images --------------------------------------------------------

IMAGES_ZIP = HERE / "images.zip"
with zipfile.ZipFile(IMAGES_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for cls_name, rel_paths in paths_by_class.items():
        for rel_path in rel_paths:
            zf.write(IMG_DIR / rel_path, arcname=rel_path)
print(f"Wrote images.zip ({IMAGES_ZIP.stat().st_size} bytes, "
      f"{N_CLASSES * N_PER_CLASS} images)")

# --- 4. Tiny vanilla Keras model in planktonclas format -----------------

inp = keras.Input(shape=(IM_SIZE, IM_SIZE, 3), name="input")
x = layers.Conv2D(8, 3, padding="same", activation="relu")(inp)
x = layers.GlobalAveragePooling2D()(x)
out = layers.Dense(N_CLASSES, activation="softmax")(x)
model = keras.Model(inp, out)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

MODEL_DIR = WORK / "tiny_model"
(MODEL_DIR / "ckpts").mkdir(parents=True)
(MODEL_DIR / "dataset_files").mkdir()
model.save(str(MODEL_DIR / "ckpts" / "final_model.h5"))

conf = {
    "model": {
        "modelname": "tiny_test",
        "image_size": IM_SIZE,
        "num_classes": N_CLASSES,
        "preprocess_mode": "tf",
    },
    "dataset": {
        "mean_RGB": [127.5, 127.5, 127.5],
        "std_RGB": [64.0, 64.0, 64.0],
    },
    "augmentation": {
        "use_augmentation": False,
        "train_mode": None,
        "val_mode": None,
    },
    "general": {"base_directory": ".", "images_directory": "."},
    "testing": {
        "ckpt_name": "final_model.h5",
        "output_directory": None,
        "timestamp": None,
    },
}
with open(MODEL_DIR / "conf.json", "w") as f:
    json.dump(conf, f, indent=2)
with open(MODEL_DIR / "dataset_files" / "classes.txt", "w") as f:
    for name in CLASS_NAMES:
        f.write(name + "\n")

MODEL_TGZ = HERE / "model.tar.gz"
with tarfile.open(MODEL_TGZ, "w:gz") as tf_out:
    tf_out.add(MODEL_DIR, arcname="tiny_model")
print(f"Wrote model.tar.gz ({MODEL_TGZ.stat().st_size} bytes)")

shutil.rmtree(WORK)
print("\nDone.")
