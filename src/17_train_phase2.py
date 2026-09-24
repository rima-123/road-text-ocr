# ===== CELL 26 (9.13): Phase 2 — add the dataset's TRAIN split =====
Q = QUICK_RUN
PHASE2_OUT = ROAD_PHASE2_DIR    # to train longer later: ROAD_SAVE_ROOT / "phase2_more"
PHASE2_INIT = ROAD_PHASE1_DIR   # ...together with PHASE2_INIT = ROAD_PHASE2_DIR (continues from phase-2 weights)
DET_CFG_P2 = DetConfig(train_size=DET_CFG_P1.train_size, infer_side=DET_CFG_P1.infer_side,
                       batch_size=DET_CFG_P1.batch_size, width=DET_CFG_P1.width, epochs=1 if Q else 30,
                       steps_per_epoch=3 if Q else 500, n_val=4 if Q else 96, lr=5e-4,
                       mixed_precision=ROAD_MIXED_PRECISION)
REC_CFG_P2 = RoadRecConfig(batch_size=REC_CFG_P1.batch_size, epochs=1 if Q else 20,
                           steps_per_epoch=3 if Q else 800, lr=5e-4)
SYNTH_SHARE_DET = 0.25   # share of synthetic scenes in detector batches
REAL_SHARE_REC = 0.6     # share of real crops in recognizer batches

DETECTOR_P2 = train_text_detector([BSTD_TRAIN, SYN_TRAIN], [1 - SYNTH_SHARE_DET, SYNTH_SHARE_DET], BSTD_VAL,
                                  DET_CFG_P2, PHASE2_OUT, init_from=PHASE2_INIT, tag="phase2-bstd+synthetic")

REAL_TRAIN_ROWS = usable_crop_rows(CROPS_TRAIN, REC_CFG_P2)
REAL_VAL_ROWS = usable_crop_rows(CROPS_VAL, REC_CFG_P2)[:300 if Q else 3000]
TRAIN_WORDS = [w["text"] for r in BSTD_TRAIN_ALL for w in r["words"]
               if not w["illegible"] and w["lang"] in BSTD_RECOG_LANGS]
print("Real training words added to the synthetic vocabulary:", extend_road_vocab(TRAIN_WORDS))
_pool2 = SYNTH_DIR / f"rec_pool_realvocab_{PHASE1['rec_pool'] // 2}_{REC_CFG_P2.rec_h}x{REC_CFG_P2.rec_w}"
if _pool2.with_suffix(".x.npy").exists():
    SYN2_X, SYN2_Y = np.load(_pool2.with_suffix(".x.npy")), np.load(_pool2.with_suffix(".y.npy"))
else:
    SYN2_X, SYN2_Y, _ = make_synthetic_rec_pool(PHASE1["rec_pool"] // 2, REC_CFG_P2, ROAD_FONTS, seed=6)
    np.save(_pool2.with_suffix(".x.npy"), SYN2_X)
    np.save(_pool2.with_suffix(".y.npy"), SYN2_Y)
VAL_RX, VAL_RY, _, _ = real_rec_arrays(REAL_VAL_ROWS, REC_CFG_P2)
RECOGNIZER_P2 = train_road_recognizer(REC_CFG_P2, np.concatenate([SYN_X, SYN2_X]), np.concatenate([SYN_Y, SYN2_Y]),
                                      VAL_RX if len(VAL_RX) else SYN_VX, VAL_RY if len(VAL_RY) else SYN_VY,
                                      PHASE2_OUT, real_rows=REAL_TRAIN_ROWS, real_share=REAL_SHARE_REC,
                                      init_from=PHASE2_INIT, tag="phase2-bstd+synthetic")

LEXICON_P2 = RoadLexicon(TRAIN_WORDS + ROAD_EN_WORDS + ROAD_EN_PLACES + ROAD_EN_BRANDS + ROAD_HI_WORDS + ROAD_HI_PLACES)
LEXICON_P2.save(PHASE2_OUT / "lexicon.txt")
SPOTTER_P2 = RoadTextSpotter(DETECTOR_P2, RECOGNIZER_P2, LEXICON_P2)
print("Lexicon size:", len(LEXICON_P2), "(train-split words + road word lists only)")
