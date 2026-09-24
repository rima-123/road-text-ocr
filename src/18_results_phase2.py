# ===== CELL 27 (9.14): Phase 2 results vs Phase 1 =====
P2_IMAGES = evaluate_spotter(SPOTTER_P2, BSTD_TEST, max_images=EVAL_MAX_IMAGES, name="Phase 2 on BSTD test photos")
print_spotter_report(P2_IMAGES)
P2_READING = recognition_scores(RECOGNIZER_P2, TEST_X, TEST_T, LEXICON_P2, TEST_L)
P2_VIDEO_RUN = process_road_video(PSEUDO_VIDEO, SPOTTER_P2, PHASE2_OUT / "video_runs")
P2_VIDEO = evaluate_video(P2_VIDEO_RUN, PSEUDO_GT)
print_video_report(P2_VIDEO, "Phase 2 on the real-photo road video")

print(f"\n{'metric':<46}{'Phase 1':>10}{'Phase 2':>10}")
for label, a, b in [
    ("Photos: detection recall", P1_IMAGES["det_recall"], P2_IMAGES["det_recall"]),
    ("Photos: detection precision", P1_IMAGES["det_precision"], P2_IMAGES["det_precision"]),
    ("Photos: found + read (lexicon)", P1_IMAGES["e2e_recall_lexicon"], P2_IMAGES["e2e_recall_lexicon"]),
    ("Photos: found + read, height>=32px",
     P1_IMAGES["groups"].get("height>=32px", {}).get("read_lexicon", 0.0),
     P2_IMAGES["groups"].get("height>=32px", {}).get("read_lexicon", 0.0)),
    ("Crops: Hindi word accuracy (lexicon)", P1_READING.get("hindi", {}).get("word_acc_lexicon", 0.0),
     P2_READING.get("hindi", {}).get("word_acc_lexicon", 0.0)),
    ("Crops: English word accuracy (lexicon)", P1_READING.get("english", {}).get("word_acc_lexicon", 0.0),
     P2_READING.get("english", {}).get("word_acc_lexicon", 0.0)),
    ("Video: instances found", P1_VIDEO["instance_found"], P2_VIDEO["instance_found"]),
    ("Video: instances read (track vote)", P1_VIDEO["instance_read"], P2_VIDEO["instance_read"]),
]:
    print(f"{label:<46}{a:>10.1%}{b:>10.1%}")
summarize_targets(P2_IMAGES, P2_VIDEO)
save_report({"images": P2_IMAGES, "reading": P2_READING, "video": P2_VIDEO}, PHASE2_OUT / "report_phase2.json")
show_spotter_examples(SPOTTER_P2, BSTD_TEST, n=4)
ROAD_VIEWER = frame_viewer(P2_VIDEO_RUN)
