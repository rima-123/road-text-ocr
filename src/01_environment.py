# ===== CELL 1: Install / environment check =====
import importlib.util
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from importlib import metadata
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")  # this notebook needs Keras 3 on TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")    # hide TensorFlow C++ info/warning chatter
IN_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/working").is_dir()
IN_COLAB = not IN_KAGGLE and ("COLAB_RELEASE_TAG" in os.environ or
                            importlib.util.find_spec("google.colab") is not None)
HAS_NVIDIA_GPU = bool(shutil.which("nvidia-smi")) and \
    subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0
INSTALL_PACKAGES = True  # already-correct packages are skipped, so re-running is quick
TESTED = {"tensorflow": "2.20.0", "keras": "3.13.2"}


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _older_than(version, minimum):
    def parts(v):
        return tuple(int(x) for x in (v.split("+")[0].split(".") + ["0", "0"])[:3] if x.isdigit())
    return version is None or parts(version) < parts(minimum)


def _pip(specs):
    print("pip install", " ".join(specs))
    done = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *specs], capture_output=True, text=True)
    if done.returncode:
        print(done.stdout[-1500:], done.stderr[-1500:])
        raise RuntimeError("pip install failed. On Kaggle: Settings -> Internet -> On, then run this cell again.")


if INSTALL_PACKAGES:
    if IN_COLAB:  # unchanged Colab setup from the original notebook
        _pip(["tensorflow==2.20.0", "keras==3.13.2", "numpy>=1.26,<2.3", "opencv-python-headless==4.10.0.84",
              "Pillow>=10.4,<13", "fonttools>=4.50,<5", "ipywidgets==8.1.7", "ipython>=8,<10",
              "gdown>=5.2", "rapidfuzz>=3.0"])
        from google.colab import output
        output.enable_custom_widget_manager()
        subprocess.run(["apt-get", "-qq", "update"], check=True)
        subprocess.run(["apt-get", "-qq", "install", "-y", "fonts-noto-core", "ffmpeg"], check=True)
    else:  # Kaggle / local: keep the preinstalled stack, add or replace only what differs from the tested set
        loaded = [m for m in ("numpy", "tensorflow", "keras") if m in sys.modules]
        need = []
        if _version("tensorflow") != TESTED["tensorflow"]:
            need.append(f"tensorflow[and-cuda]=={TESTED['tensorflow']}" if HAS_NVIDIA_GPU
                        else f"tensorflow=={TESTED['tensorflow']}")
        if _version("keras") != TESTED["keras"]:
            need.append(f"keras=={TESTED['keras']}")
        if importlib.util.find_spec("cv2") is None:
            need.append("opencv-python-headless==4.10.0.84")
        if _older_than(_version("Pillow") or _version("pillow"), "10.4"):
            need.append("Pillow>=10.4,<13")
        for dist, spec, minimum in (("fonttools", "fonttools>=4.50,<5", "4.50"), ("ipywidgets", "ipywidgets>=8,<9", "8.0"),
                                    ("gdown", "gdown>=5.2", "5.2"), ("rapidfuzz", "rapidfuzz>=3.0", "3.0")):
            if _older_than(_version(dist), minimum):
                need.append(spec)
        if need:
            _pip(need)
            if loaded:
                print("NOTE: packages were changed after", loaded, "had been imported. Use Run -> Restart & "
                      "clear outputs, then run all cells again (this cell will then skip the install).")
        else:
            print("Tested package versions already present.")
        if IN_KAGGLE and shutil.which("apt-get") and hasattr(os, "geteuid") and os.geteuid() == 0:
            subprocess.run(["apt-get", "-qq", "update"], capture_output=True)
            apt = subprocess.run(["apt-get", "-qq", "install", "-y", "fonts-noto-core", "ffmpeg", "libfribidi0",
                                  "libraqm0"], capture_output=True, text=True)
            if apt.returncode:
                print("apt-get could not install everything (continuing; fonts are downloaded below if needed).")
        elif not IN_KAGGLE:
            print("Local Jupyter: install FFmpeg and a Devanagari font separately if they are missing.")

# A Devanagari font is required. If the system has none, fetch Noto (OFL) from github.com/google/fonts.
_SCRATCH = Path("/kaggle/temp") if IN_KAGGLE else Path("/content") if Path("/content").is_dir() else Path(".").resolve()
try:
    _SCRATCH.mkdir(parents=True, exist_ok=True)
except OSError:
    _SCRATCH = Path("/tmp")
DEFAULT_FONT_DIR = None
_font_roots = [Path("/usr/share/fonts"), Path.home() / ".local/share/fonts"]
if not any("devanagari" in p.name.lower() for r in _font_roots if r.is_dir() for p in r.rglob("*.[ot]tf")):
    _fallback = _SCRATCH / "fallback_fonts"
    _fallback.mkdir(parents=True, exist_ok=True)
    for _rel in ("ofl/notosansdevanagari/NotoSansDevanagari[wdth,wght].ttf", "ofl/notosans/NotoSans[wdth,wght].ttf"):
        _out = _fallback / Path(_rel).name
        if not _out.is_file():
            _url = "https://raw.githubusercontent.com/google/fonts/main/" + urllib.parse.quote(_rel)
            with urllib.request.urlopen(_url, timeout=60) as _r:
                _out.write_bytes(_r.read())
    DEFAULT_FONT_DIR = str(_fallback)
    print("System has no Devanagari font; using downloaded Noto fonts in", _fallback)

from PIL import features
if not features.check("raqm"):
    print("WARNING: Pillow cannot shape Hindi text (RAQM/fribidi missing). Hindi rendering will fail. "
          "On Kaggle turn Internet on and rerun this cell (it installs libfribidi0).")
if IN_KAGGLE and not HAS_NVIDIA_GPU:
    print("No GPU detected: Kaggle Settings -> Accelerator -> GPU T4 x2 (then run again).")
print(f"Environment: {'Kaggle' if IN_KAGGLE else 'Colab' if IN_COLAB else 'local'} | "
      f"tensorflow {_version('tensorflow')} | keras {_version('keras')} | GPU: {HAS_NVIDIA_GPU}")
