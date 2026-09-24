# ===== CELL 24 (9.11): Get the dataset =====
BSTD_ZIP_PATH = ""            # empty = automatic: attached Kaggle input (zip or folder), else download with gdown.
                              # Or give a zip / extracted folder path (e.g. on Drive or under /kaggle/input).
BSTD_MAX_SIDE = 1600          # images are stored at most this large (enough for 720p/1080p-like text sizes)
BSTD_MAX_IMAGES = 24 if QUICK_RUN else None   # per split; None = everything
BSTD_STANDIN_FOR_QUICK_RUN = True   # QUICK_RUN without an attached dataset: tiny synthetic stand-in, no download
DELETE_ZIP_AFTER_PREP = IN_KAGGLE   # frees ~17 GB of scratch disk once the images are prepared

BSTD_DIR = ROAD_DATA_ROOT / "bstd"
if bstd_is_prepared(BSTD_DIR, BSTD_MAX_SIDE, BSTD_MAX_IMAGES):
    bstd_files = prepare_bstd(None, BSTD_DIR, max_side=BSTD_MAX_SIDE, max_images=BSTD_MAX_IMAGES)
else:
    bstd_source = find_bstd_source(BSTD_ZIP_PATH)
    if bstd_source is None and QUICK_RUN and BSTD_STANDIN_FOR_QUICK_RUN:
        bstd_source = make_bstd_standin(ROAD_DATA_ROOT / "bstd_standin.zip", ROAD_FONTS)
        print("QUICK_RUN: using a tiny synthetic stand-in in BSTD format (pipeline check only, no download).")
    if bstd_source is None:
        bstd_source = download_bstd_detection(ROAD_DOWNLOAD_DIR)
    print("Dataset source:", bstd_source)
    bstd_files = prepare_bstd(bstd_source, BSTD_DIR, max_side=BSTD_MAX_SIDE, max_images=BSTD_MAX_IMAGES)
    if DELETE_ZIP_AFTER_PREP and Path(bstd_source).parent == ROAD_DOWNLOAD_DIR:
        Path(bstd_source).unlink(missing_ok=True)

BSTD_TRAIN_ALL = load_annotations(bstd_files["train"])
BSTD_TEST = load_annotations(bstd_files["test"])
BSTD_TRAIN, BSTD_VAL = split_train_val(BSTD_TRAIN_ALL, val_fraction=0.05, seed=0)
for _name, _recs in (("BSTD train", BSTD_TRAIN), ("BSTD val (held out from train)", BSTD_VAL), ("BSTD test", BSTD_TEST)):
    dataset_stats(_recs, _name)

CROP_DIR = BSTD_DIR / "crops"
CROPS_TRAIN = read_word_crops(extract_word_crops(BSTD_TRAIN, CROP_DIR, "train"))
CROPS_VAL = read_word_crops(extract_word_crops(BSTD_VAL, CROP_DIR, "val"))
CROPS_TEST = read_word_crops(extract_word_crops(BSTD_TEST, CROP_DIR, "test"))
print("Word crops (Hindi/English/Marathi):",
      {n: dict(Counter(r["lang"] for r in rows)) for n, rows in
       (("train", CROPS_TRAIN), ("val", CROPS_VAL), ("test", CROPS_TEST))})
