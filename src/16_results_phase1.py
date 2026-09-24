# ===== CELL 25 (9.12): Phase 1 results on photos and a real-photo road video =====
EVAL_MAX_IMAGES = 12 if QUICK_RUN else None      # None = whole BSTD test split
VIDEO_PHOTOS = 2 if QUICK_RUN else 30
VIDEO_SECONDS_PER_PHOTO = 0.6 if QUICK_RUN else 2.0

P1_IMAGES = evaluate_spotter(SPOTTER_P1, BSTD_TEST, max_images=EVAL_MAX_IMAGES, name="Phase 1 on BSTD test photos")
print_spotter_report(P1_IMAGES)

# Reading only: ground-truth word boxes, so detector mistakes do not count here.
TEST_ROWS = usable_crop_rows(CROPS_TEST, REC_CFG_P1)[:300 if QUICK_RUN else None]
TEST_X, TEST_Y, TEST_T, TEST_L = real_rec_arrays(TEST_ROWS, REC_CFG_P1)
P1_READING = recognition_scores(RECOGNIZER_P1, TEST_X, TEST_T, LEXICON_P1, TEST_L)
print("\nReading accuracy on real test word crops (Phase 1):")
for _g, _v in P1_READING.items():
    print(f"  {_g:<8} n={_v['n']:<6} exact {_v['exact']:.1%}  word acc {_v['word_acc']:.1%}  "
          f"+lexicon {_v['word_acc_lexicon']:.1%}  CER {_v['cer']:.3f}")

VIDEO_DIR = ROAD_SAVE_ROOT / "pseudo_video"
PSEUDO_VIDEO, PSEUDO_GT = make_pseudo_video(pick_video_images(BSTD_TEST, VIDEO_PHOTOS, seed=5), VIDEO_DIR,
                                            seconds_per_image=VIDEO_SECONDS_PER_PHOTO)
P1_VIDEO_RUN = process_road_video(PSEUDO_VIDEO, SPOTTER_P1, ROAD_PHASE1_DIR / "video_runs")
P1_VIDEO = evaluate_video(P1_VIDEO_RUN, PSEUDO_GT)
print_video_report(P1_VIDEO, "Phase 1 on the real-photo road video")
summarize_targets(P1_IMAGES, P1_VIDEO)
save_report({"images": P1_IMAGES, "reading": P1_READING, "video": P1_VIDEO}, ROAD_PHASE1_DIR / "report_phase1.json")
