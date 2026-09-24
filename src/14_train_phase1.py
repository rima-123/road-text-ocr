# ===== CELL 23 (9.10): Phase 1 — train on synthetic road scenes only =====
Q = QUICK_RUN
PHASE1 = {"scenes": 60 if Q else 4000, "val_scenes": 12 if Q else 250,
          "rec_pool": 600 if Q else 60000, "rec_val": 120 if Q else 2000}
DET_CFG_P1 = DetConfig(train_size=256 if Q else 640, infer_side=640 if Q else 1280, batch_size=2 if Q else 8,
                       epochs=1 if Q else 20, steps_per_epoch=3 if Q else 400, n_val=4 if Q else 96,
                       width=0.5 if Q else 1.0, mixed_precision=ROAD_MIXED_PRECISION)
REC_CFG_P1 = RoadRecConfig(batch_size=16 if Q else 64, epochs=1 if Q else 20, steps_per_epoch=3 if Q else 800)

SYNTH_DIR = ROAD_DATA_ROOT / "synthetic"
SYN_TRAIN = load_annotations(write_synthetic_scenes(SYNTH_DIR, PHASE1["scenes"], ROAD_FONTS, seed=1, split="train"))
SYN_VAL = load_annotations(write_synthetic_scenes(SYNTH_DIR, PHASE1["val_scenes"], ROAD_FONTS, seed=2, split="val"))
dataset_stats(SYN_TRAIN, "synthetic train")
dataset_stats(SYN_VAL, "synthetic val")

DETECTOR_P1 = train_text_detector([SYN_TRAIN], [1.0], SYN_VAL, DET_CFG_P1, ROAD_PHASE1_DIR, tag="phase1-synthetic")

_pool = SYNTH_DIR / f"rec_pool_{PHASE1['rec_pool']}_{REC_CFG_P1.rec_h}x{REC_CFG_P1.rec_w}"
if _pool.with_suffix(".x.npy").exists():
    SYN_X, SYN_Y = np.load(_pool.with_suffix(".x.npy")), np.load(_pool.with_suffix(".y.npy"))
else:
    SYN_X, SYN_Y, _ = make_synthetic_rec_pool(PHASE1["rec_pool"], REC_CFG_P1, ROAD_FONTS, seed=3)
    np.save(_pool.with_suffix(".x.npy"), SYN_X)
    np.save(_pool.with_suffix(".y.npy"), SYN_Y)
SYN_VX, SYN_VY, SYN_VT = make_synthetic_rec_pool(PHASE1["rec_val"], REC_CFG_P1, ROAD_FONTS, seed=4)
RECOGNIZER_P1 = train_road_recognizer(REC_CFG_P1, SYN_X, SYN_Y, SYN_VX, SYN_VY, ROAD_PHASE1_DIR, tag="phase1-synthetic")

LEXICON_P1 = RoadLexicon(ROAD_EN_WORDS + ROAD_EN_PLACES + ROAD_EN_BRANDS + ROAD_HI_WORDS + ROAD_HI_PLACES
                         + list(ENGLISH_WORDS) + list(HINDI_WORDS))
LEXICON_P1.save(ROAD_PHASE1_DIR / "lexicon.txt")
SPOTTER_P1 = RoadTextSpotter(DETECTOR_P1, RECOGNIZER_P1, LEXICON_P1)

print("\nSanity check on SYNTHETIC validation data (easy; this is not the real-world score):")
print({k: {m: round(v, 3) if isinstance(v, float) else v for m, v in d.items()}
       for k, d in recognition_scores(RECOGNIZER_P1, SYN_VX, SYN_VT).items()})
print_spotter_report(evaluate_spotter(SPOTTER_P1, SYN_VAL, care_langs=("hindi", "english", "digits"),
                                      max_images=100, name="Phase 1 on synthetic validation scenes"))
show_spotter_examples(SPOTTER_P1, SYN_VAL, n=2)
