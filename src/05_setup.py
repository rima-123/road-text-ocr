# ===== CELL 14 (9.1): Road system setup =====
import gc
import importlib.util
import os
import random
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow C++ info/warning chatter (errors still show)
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
IN_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/working").is_dir()
IN_COLAB = not IN_KAGGLE and ("COLAB_RELEASE_TAG" in os.environ or
                            importlib.util.find_spec("google.colab") is not None)
for _module, _spec in (("gdown", "gdown>=5.2"), ("rapidfuzz", "rapidfuzz>=3.0")):
    if importlib.util.find_spec(_module) is None:  # optional: the code has fallbacks
        _pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", _spec], capture_output=True, text=True)
        if _pip.returncode:
            print(f"Optional package {_spec} could not be installed (fallbacks are used).")

import tensorflow as tf
import keras
from keras import layers

if keras.backend.backend() != "tensorflow":
    raise RuntimeError("Keras is not using TensorFlow. Restart the kernel and run the install cell first.")

# QUICK_RUN=True shrinks every stage so the whole section finishes in minutes (and uses a tiny synthetic
# stand-in instead of downloading the 17 GB dataset). Its accuracy numbers are meaningless.
QUICK_RUN = False
ROAD_SEED = 1234
keras.utils.set_random_seed(ROAD_SEED)

ROAD_GPUS = tf.config.list_physical_devices("GPU")
for _gpu in ROAD_GPUS:
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass  # The GPU was already initialised by an earlier cell; that is fine.


def _fast_fp16(gpu):
    try:
        capability = tf.config.experimental.get_device_details(gpu).get("compute_capability")
        return bool(capability) and tuple(capability) >= (7, 0)
    except Exception:
        return False


ROAD_MIXED_PRECISION = bool(ROAD_GPUS) and all(_fast_fp16(g) for g in ROAD_GPUS)  # T4/L4/A100 yes, P100 no
if ROAD_GPUS:
    print("GPU:", [g.name for g in ROAD_GPUS], "| mixed precision:", ROAD_MIXED_PRECISION,
          "| training uses the first GPU" if len(ROAD_GPUS) > 1 else "")
else:
    print("WARNING: no GPU found. Kaggle: Settings -> Accelerator -> GPU T4 x2. Colab: Runtime -> Change runtime "
          "type -> T4 GPU.\nFull training on CPU would take many hours; only QUICK_RUN is practical on CPU.")

ROAD_USE_DRIVE = True  # Colab only: models/reports go to Google Drive so a disconnect does not lose training.


def _road_storage_base():
    if IN_KAGGLE:
        return Path("/kaggle/working")  # kept as notebook output when you 'Save Version'
    if IN_COLAB and ROAD_USE_DRIVE:
        try:
            from google.colab import drive
            if not Path("/content/drive/MyDrive").is_dir():
                drive.mount("/content/drive")
            return Path("/content/drive/MyDrive")
        except Exception as exc:  # Mount refused/failed: keep going locally.
            print("Google Drive not mounted, saving locally instead:", exc)
    return Path(".").resolve()


def _scratch_base():
    candidates = [Path("/kaggle/temp"), Path("/tmp")] if IN_KAGGLE else \
        [Path("/content")] if Path("/content").is_dir() else [Path(".").resolve()]
    for base in candidates:
        try:
            base.mkdir(parents=True, exist_ok=True)
            probe = base / ".write_test"
            probe.write_text("ok")
            probe.unlink()
            return base
        except OSError:
            continue
    return Path(".").resolve()


# Quick-run files live in separate *_quick folders so tiny test models are never reused by the real run.
# Big data (dataset, crops, synthetic images) stays on scratch disk: Kaggle keeps only ~20 GB / ~500 files
# of /kaggle/working, so only models, reports and videos are written there.
_ROAD_SUFFIX = "_quick" if QUICK_RUN else ""
_ROAD_SCRATCH = _scratch_base()
ROAD_SAVE_ROOT = _road_storage_base() / f"road_ocr{_ROAD_SUFFIX}"
ROAD_DATA_ROOT = _ROAD_SCRATCH / f"road_data{_ROAD_SUFFIX}"
ROAD_DOWNLOAD_DIR = _ROAD_SCRATCH / "road_downloads"  # shared by quick and full runs (17 GB zip + fonts)
ROAD_PHASE1_DIR = ROAD_SAVE_ROOT / "phase1_synthetic"
ROAD_PHASE2_DIR = ROAD_SAVE_ROOT / "phase2_bstd"
for _d in (ROAD_SAVE_ROOT, ROAD_DATA_ROOT, ROAD_DOWNLOAD_DIR, ROAD_PHASE1_DIR, ROAD_PHASE2_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Kaggle sessions end after 12 h. To continue later, attach this notebook's earlier output with 'Add Input':
# finished models are then loaded, and an interrupted training resumes from its last epoch.
ROAD_RESUME_FROM_INPUT = True
if IN_KAGGLE and ROAD_RESUME_FROM_INPUT and Path("/kaggle/input").is_dir():
    _patterns = (f"*/road_ocr{_ROAD_SUFFIX}/phase*", f"*/*/road_ocr{_ROAD_SUFFIX}/phase*")
    for _src in sorted({p for pat in _patterns for p in Path("/kaggle/input").glob(pat) if p.is_dir()}):
        _dst = ROAD_SAVE_ROOT / _src.name
        if not any(_dst.glob("*.json")) and not any(_dst.glob("*_backup")):
            shutil.copytree(_src, _dst, dirs_exist_ok=True)
            print("Resuming from attached output:", _src)

_free_gb = shutil.disk_usage(ROAD_DATA_ROOT).free / 1024**3
print(f"Environment: {'Kaggle' if IN_KAGGLE else 'Colab' if IN_COLAB else 'local'}")
print(f"Models/reports: {ROAD_SAVE_ROOT}\nData (scratch disk): {ROAD_DATA_ROOT}  free: {_free_gb:.1f} GB`")
print("QUICK_RUN =", QUICK_RUN)
