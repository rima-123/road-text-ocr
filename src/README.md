# src/

These files are the notebook's sections as plain scripts, in the same order. They share one namespace
and are meant to be run in sequence (01 → 23), exactly as the notebook runs them; they are not
standalone importable modules. The notebook in `notebooks/` is the entry point.

| File | Section |
|---|---|
| 01_environment.py | package install and environment check |
| 02_charset_fonts.py | charset, fonts, crop preprocessing |
| 03_synth_words_crnn.py | word renderer and the first CRNN (its helpers are reused later) |
| 04_frame_viewer.py | reading frame records, stepping through an annotated video |
| 05_setup.py | paths, GPU, QUICK_RUN switch |
| 06_fonts_vocabulary.py | font list and road vocabulary |
| 07_geometry.py | quadrilaterals, overlap, shrink/expand, augmentation |
| 08_synthetic_scenes.py | synthetic road-scene generator |
| 09_dataset_bstd.py | BSTD download, preparation, word crops |
| 10_detector.py | detector model, targets, loss, training, post-processing |
| 11_recognizer_lexicon.py | recogniser model, CTC loss, training, lexicon |
| 12_spotter_eval.py | end-to-end spotter and metrics |
| 13_video_tracking.py | video processing, tracker, voting, video metrics |
| 14_train_phase1.py | phase 1: synthetic only |
| 15_get_dataset.py | fetch and prepare BSTD |
| 16_results_phase1.py | phase 1 evaluation |
| 17_train_phase2.py | phase 2: BSTD + synthetic |
| 18_results_phase2.py | phase 2 evaluation and comparison |
| 19_finetune_hindi_crops.py | optional recogniser fine-tuning on extra Hindi crops |
| 20_synthetic_dashcam_clip.py | generated dash-camera clip and reading-distance report |
| 21_eval_roadtext1k.py | evaluation on RoadText-1K videos |
| 22_run_own_videos.py | run on your own clips |
| 23_save_checkpoints.py | pack trained models before closing a session |
