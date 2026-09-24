# ===== SAVE HELPER: checkpoints ko ek zip me pack karo, phir Save Version -> Quick Save dabao =====
import zipfile as _zipfile

ROAD_BACKUP_ZIP = Path("/kaggle/working") / "road_ocr_checkpoints.zip"
_KEEP = ("detector.weights.h5", "detector.json", "detector_best.weights.h5", "detector_log.csv",
         "recognizer.weights.h5", "recognizer.json", "recognizer_best.weights.h5", "recognizer_log.csv",
         "lexicon.txt", "report_phase1.json", "report_phase2.json")

_found, _total = [], 0
with _zipfile.ZipFile(ROAD_BACKUP_ZIP, "w", _zipfile.ZIP_DEFLATED) as _zf:
    for _phase in sorted(Path("/kaggle/working").glob("road_ocr*/phase*")):
        for _name in _KEEP:
            _f = _phase / _name
            if _f.is_file():
                _zf.write(_f, f"{_phase.parent.name}/{_phase.name}/{_name}")
                _found.append(f"{_phase.name}/{_name}")
                _total += _f.stat().st_size
        for _bk in sorted(_phase.glob("*_backup/*")):   # half-finished training (resumes from last epoch)
            if _bk.is_file():
                _zf.write(_bk, f"{_phase.parent.name}/{_phase.name}/{_bk.parent.name}/{_bk.name}")
                _total += _bk.stat().st_size

print(f"Packed {len(_found)} checkpoint files ({_total / 1024**2:.0f} MB) -> {ROAD_BACKUP_ZIP}")
for _n in _found:
    print("  ", _n)
print(f"\n/kaggle/working: {sum(1 for _ in Path('/kaggle/working').rglob('*') if _.is_file())} files, "
      f"{sum(_.stat().st_size for _ in Path('/kaggle/working').rglob('*') if _.is_file()) / 1024**3:.2f} GB "
      f"(Kaggle limit ~20 GB / ~500 files)")
print("\nAB YEH KIJIYE: Save Version -> Quick Save (code dobara nahi chalega, sirf files save hongi).")
