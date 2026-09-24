# ===== Score the model on RoadText-1K videos (real dashcam + real ground truth) =====
import gdown

RT_FILE_ID = "1M4SyidKm-g38uaiz-TK088-QfDJeacTY"
RT_JSON = ROAD_DATA_ROOT / "RoadText_1k_train_annotation.json"
if not RT_JSON.is_file():
    gdown.download(id=RT_FILE_ID, output=str(RT_JSON), quiet=False)
print("annotation size:", round(RT_JSON.stat().st_size / 1024**2, 1), "MB")

_ann = json.loads(RT_JSON.read_text(encoding="utf-8"))
VIDEOS_TO_SCORE = [_v for _v in MY_VIDEOS if Path(_v).stem in _ann][:4]
print("videos in annotation:", len(_ann), "| yours with ground truth:",
      [Path(_v).name for _v in VIDEOS_TO_SCORE])

SPOTTER_HF.detector.post.update(box_thresh=0.45, bin_thresh=0.3, infer_side=2000)
_ALL = []
for _v in VIDEOS_TO_SCORE:
    _frames = _ann[Path(_v).stem]
    _gt_path = ROAD_DATA_ROOT / f"rt_gt_{Path(_v).stem}.jsonl"
    with _gt_path.open("w", encoding="utf-8") as _fh:              # their format -> ours
        for _k in range(len(_frames) + 1):
            _lbls = (_frames.get(str(_k + 1)) or {}).get("labels") or []
            _words = []
            for _i, _lb in enumerate(_lbls):
                _b = _lb.get("box2d")
                if not _b:
                    continue
                _poly = [[_b["x1"], _b["y1"]], [_b["x2"], _b["y1"]],
                         [_b["x2"], _b["y2"]], [_b["x1"], _b["y2"]]]
                _txt = _lb.get("ocr") or ""
                _care = bool(_lb.get("category") == "English" and _txt
                             and (_b["y2"] - _b["y1"]) >= 8 and road_clean_label(_txt))
                _words.append({"id": str(_lb.get("id", _i)), "poly": _poly, "text": _txt,
                               "lang": "english", "care": _care})
            _fh.write(json.dumps({"frame_index": _k, "source": Path(_v).stem, "words": _words},
                                 ensure_ascii=False) + "\n")
    _run = process_road_video(_v, SPOTTER_HF, ROAD_SAVE_ROOT / "roadtext_runs", overlay="fused")
    _m = evaluate_video(_run, _gt_path)
    _ALL.append((Path(_v).name, _m))
    print_video_report(_m, f"RoadText-1K: {Path(_v).name}")

print(f"\n{'video':<12}{'det P':>8}{'det R':>8}{'read/frame':>12}{'inst found':>12}{'inst read':>11}")
for _n, _m in _ALL:
    print(f"{_n:<12}{_m['frame_det_precision']:>8.1%}{_m['frame_det_recall']:>8.1%}"
          f"{_m['frame_read_recall']:>12.1%}{_m['instance_found']:>12.1%}{_m['instance_read']:>11.1%}")

# ===== Score on many RoadText-1K videos at once (stable numbers) =====
N_VIDEOS = 30

_files = gdown.download_folder(url=DRIVE_FOLDER, skip_download=True, quiet=True,
                               use_cookies=False, remaining_ok=True) or []
_vids = [f for f in _files if str(f.path).lower().endswith(".mp4")]
_dir = ROAD_DATA_ROOT / "my_dashcam"
SCORE_VIDEOS = []
for _f in _vids:
    if len(SCORE_VIDEOS) >= N_VIDEOS:
        break
    _stem = Path(str(_f.path)).stem
    if _stem not in _ann:
        continue
    _out = _dir / f"{_stem}.mp4"
    if not _out.is_file():
        try:
            gdown.download(id=_f.id, output=str(_out), quiet=True)
        except Exception:
            continue
    if _out.is_file():
        SCORE_VIDEOS.append(str(_out))
print(f"{len(SCORE_VIDEOS)} videos ready\n")

SPOTTER_HF.detector.post.update(box_thresh=0.45, bin_thresh=0.3, infer_side=2000)
_tot = Counter()
for _v in SCORE_VIDEOS:
    _stem = Path(_v).stem
    _frames = _ann[_stem]
    _gt = ROAD_DATA_ROOT / f"rt_gt_{_stem}.jsonl"
    with _gt.open("w", encoding="utf-8") as _fh:
        for _k in range(len(_frames) + 1):
            _words = []
            for _i, _lb in enumerate((_frames.get(str(_k + 1)) or {}).get("labels") or []):
                _b = _lb.get("box2d")
                if not _b:
                    continue
                _txt = _lb.get("ocr") or ""
                _words.append({"id": str(_lb.get("id", _i)),
                               "poly": [[_b["x1"], _b["y1"]], [_b["x2"], _b["y1"]],
                                        [_b["x2"], _b["y2"]], [_b["x1"], _b["y2"]]],
                               "text": _txt, "lang": "english",
                               "care": bool(_lb.get("category") == "English" and _txt
                                            and (_b["y2"] - _b["y1"]) >= 8 and road_clean_label(_txt))})
            _fh.write(json.dumps({"frame_index": _k, "source": _stem, "words": _words}, ensure_ascii=False) + "\n")
    _run = process_road_video(_v, SPOTTER_HF, ROAD_SAVE_ROOT / "roadtext_runs")
    _m = evaluate_video(_run, _gt)
    _n = _m["instances"]
    _tot["inst"] += _n
    _tot["found"] += round(_m["instance_found"] * _n)
    _tot["read"] += round(_m["instance_read"] * _n)
    print(f"{_stem:>4}: {_n:3d} instances | found {_m['instance_found']:5.1%} | read {_m['instance_read']:5.1%}")

print(f"\n=== OVERALL on {len(SCORE_VIDEOS)} real dashcam videos, {_tot['inst']} text instances ===")
print(f"found:            {_tot['found'] / max(_tot['inst'], 1):.1%}")
print(f"found + read:     {_tot['read'] / max(_tot['inst'], 1):.1%}")

# ===== Look at the best-performing videos: annotated video + confident readings =====
from IPython.display import Video, display

BEST_IDS = ["14", "26", "29"]              # sabse achhe teen
MIN_VOTE, MIN_FRAMES = 0.4, 15

for _id in BEST_IDS:
    _v = str(ROAD_DATA_ROOT / "my_dashcam" / f"{_id}.mp4")
    if not Path(_v).is_file():
        print("missing:", _v)
        continue
    print("\n" + "=" * 70 + f"\nvideo {_id}.mp4")
    _run = process_road_video(_v, SPOTTER_HF, ROAD_SAVE_ROOT / "best_runs", overlay="fused")

    _gt = ROAD_DATA_ROOT / f"rt_gt_{_id}.jsonl"                      # ground truth ke shabd
    _truth = sorted({w["text"] for l in open(_gt, encoding="utf-8") if l.strip()
                     for w in json.loads(l)["words"] if w["care"]})
    print("ground truth says:", _truth)

    with open(_run["tracks_csv"], encoding="utf-8-sig", newline="") as _fh:
        _rows = [r for r in csv.DictReader(_fh) if r.get("text")
                 and int(r["frames"]) >= MIN_FRAMES and float(r["vote_share"]) >= MIN_VOTE]
    _rows.sort(key=lambda r: -int(r["frames"]))
    print("model read:")
    for _r in _rows:
        _hit = "OK " if road_match_key(_r["text"]) in {road_match_key(t) for t in _truth} else "   "
        print(f"  {_hit}{_r['text']:<22} vote {float(_r['vote_share']):.0%}  frames {_r['frames']}")

    _mb = Path(_run["output_video"]).stat().st_size / 1024**2
    print(f"annotated video ({_mb:.0f} MB): {_run['output_video']}")
    if _mb <= 60:
        display(Video(_run["output_video"], embed=True, width=900))
