# ===== A: load a small real OCR dataset from HuggingFace and LOOK at it first =====
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=False)
from datasets import load_dataset
from IPython.display import display

HF_DATASET, HF_CONFIG = "darknight054/indic-mozhi-ocr", "hindi"
HF_DS = load_dataset(HF_DATASET, HF_CONFIG)
print(HF_DS)
_split = "train" if "train" in HF_DS else list(HF_DS)[0]
print("\ncolumns:", HF_DS[_split].column_names)
for _i in range(6):
    _ex = HF_DS[_split][_i]
    _img = next((v for v in _ex.values() if hasattr(v, "size")), None)
    _txt = next((v for k, v in _ex.items() if isinstance(v, str) and len(v) < 60), None)
    print(f"[{_i}] text = {_txt!r} | image = {_img.size if _img else None}")
    if _img:
        display(_img)

# ===== A2: same dataset, correct columns =====
from IPython.display import display

_split = "train"
print("columns:", HF_DS[_split].column_names)
for _i in range(8):
    _ex = HF_DS[_split][_i]
    _img, _txt = _ex["image"], _ex["text"]
    print(f"[{_i}] text = {_txt!r} | size = {_img.size}")
    display(_img)

# ===== B: fine-tune the recognizer on the HF Hindi crops =====
_EX, _EY, _ET, _EL = real_rec_arrays(HF_EVAL, REC_CFG_P2)
print("BEFORE fine-tuning:", {k: round(v, 3) for k, v in
                              recognition_scores(RECOGNIZER_P2, _EX, _ET)["all"].items() if isinstance(v, float)})

HF_OUT = ROAD_SAVE_ROOT / "recognizer_hf"
REC_CFG_HF = RoadRecConfig(batch_size=64, epochs=15, steps_per_epoch=600, lr=3e-4)
RECOGNIZER_HF = train_road_recognizer(REC_CFG_HF, SYN_X, SYN_Y, _EX, _EY, HF_OUT,
                                      real_rows=HF_TRAIN, real_share=0.8,
                                      init_from=ROAD_PHASE2_DIR, tag="hf-finetune")
print("\nAFTER fine-tuning:", {k: round(v, 3) for k, v in
                               recognition_scores(RECOGNIZER_HF, _EX, _ET)["all"].items() if isinstance(v, float)})

SPOTTER_HF = RoadTextSpotter(DETECTOR_P2, RECOGNIZER_HF, LEXICON_P2)
