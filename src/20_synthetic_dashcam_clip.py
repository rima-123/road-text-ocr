# ===== Dash-cam clip: camera on the car, mixed signs, 20 s, full report =====
from IPython.display import Video, display

W, H, FPS, SECONDS = 1280, 720, 25, 20
FOCAL, SPEED = 900.0, 9.0          # SPEED = metres per second (normal city/highway speed)
N_SIGNS = 16

_rng = np.random.default_rng(7)
_synth = RoadSceneSynth(ROAD_FONTS)
_bg = _synth.background(_rng, W, H)
_horizon = int(H * 0.45)
_KINDS = (["highway_green"] * 3 + ["highway_blue"] * 2 + ["shop_board"] * 4 + ["yellow_warning"] * 2
          + ["white_board"] * 2 + ["milestone"] + ["number_plate"] * 2)      # mixed road furniture

_signs = []
for _i in range(N_SIGNS):
    _kind = _KINDS[_i % len(_KINDS)]
    _made = _synth.make_sign(_rng, kind=_kind)
    if _made is None:
        continue
    _img, _words, _ = _made
    _plate = _kind == "number_plate"
    _signs.append({"id": _i, "kind": _kind, "img": _img, "words": _words,
                   "X": float(_rng.uniform(-1.2, 1.2)) if _plate else (1 if _rng.random() < .5 else -1) * float(_rng.uniform(4.0, 8.5)),
                   "Y": float(_rng.uniform(0.2, 0.8)) if _plate else float(_rng.uniform(-3.5, -1.2)),
                   "Z": float(18 + _i * (SPEED * SECONDS - 18) / max(1, N_SIGNS - 1) + _rng.uniform(-3, 3)),
                   "w": float(_rng.uniform(0.4, 0.5)) if _plate else float(_rng.uniform(1.5, 4.0))})
print(f"{len(_signs)} signs placed along the road")

_out = ROAD_SAVE_ROOT / "dashcam"; _out.mkdir(parents=True, exist_ok=True)
_raw, _final, _gt_path = _out / "dashcam_raw.mp4", _out / "dashcam.mp4", _out / "dashcam_gt.jsonl"
_writer = cv2.VideoWriter(str(_raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
_lines, _n_frames = [], int(SECONDS * FPS)
for _k in range(_n_frames):
    _t = _k / FPS
    _shake = (_rng.normal(0, 1.2), _rng.normal(0, 1.0))
    _frame = _bg.copy()
    _words_out = []
    for _s in _signs:
        _z = _s["Z"] - SPEED * _t
        if _z < 3.0:
            continue                                              # passed the car
        _target_w = FOCAL * _s["w"] / _z
        if _target_w < 6 or _target_w > 2.2 * W:
            continue
        _cx = W / 2 + FOCAL * _s["X"] / _z + _shake[0]
        _cy = _horizon + FOCAL * _s["Y"] / _z + _shake[1]
        _quads = RoadSceneSynth._paste(_frame, _s["img"], _s["words"],
                                       np.random.default_rng([31, _s["id"]]), _target_w, (_cx, _cy),
                                       persp=0.18, max_rot=2.0)
        if _quads is None:
            continue
        for (_text, _lang, _b, _li), _q in zip(_s["words"], _quads):
            _words_out.append({"id": f"{_s['id']}:{_li}:{_text}", "poly": np.round(_q, 1).tolist(),
                               "text": _text, "lang": _lang, "distance_m": round(_z, 1),
                               "height_px": round(poly_text_height(_q), 1),
                               "care": bool(poly_text_height(_q) >= 8 and _lang in ("hindi", "english", "digits"))})
    if _k % 3 == 0:                                               # light motion blur at normal speed
        _frame = motion_blur(_frame, _rng, max_len=4)
    _frame = np.clip(_frame.astype(np.float32) + _rng.normal(0, 2.5, _frame.shape), 0, 255).astype(np.uint8)
    _writer.write(_frame)
    _lines.append(json.dumps({"frame_index": _k, "source": "dashcam", "words": _words_out}, ensure_ascii=False))
_writer.release()

_enc = _to_h264(_raw, _final, crf=24)
if _enc != _final:
    shutil.move(str(_enc), str(_final))
_cap = cv2.VideoCapture(str(_final)); _dec = 0
while _cap.grab():
    _dec += 1
_cap.release()
_gt_path.write_text("\n".join(_lines[:_dec]) + "\n", encoding="utf-8")
print(f"clip ready: {_dec} frames -> {_final}")
display(Video(str(_final), embed=True, width=800))

SPOTTER_P2.detector.post.update(box_thresh=0.5, bin_thresh=0.35, infer_side=1600)
DASH_RUN = process_road_video(_final, SPOTTER_P2, ROAD_SAVE_ROOT / "dashcam_runs", overlay="fused")
print_video_report(evaluate_video(DASH_RUN, _gt_path), "Model on the dash-cam clip")

# ---- how far away does a sign start being detected?
_gt = {json.loads(l)["frame_index"]: json.loads(l)["words"] for l in open(_gt_path, encoding="utf-8") if l.strip()}
_first = {}
with open(DASH_RUN["frame_records"], encoding="utf-8") as _fh:
    for _line in _fh:
        _rec = json.loads(_line)
        _words = [w for w in _gt.get(_rec["source_frame"], []) if w["care"]]
        _quads = [np.asarray(d["poly"], np.float32) for d in _rec["detections"]]
        _m, _ = match_detections([{**w, "poly": np.asarray(w["poly"], np.float32), "illegible": False} for w in _words],
                                 _quads, [True] * len(_words))
        for _gi, _pi, _ in _m:
            _w, _d = _words[_gi], _rec["detections"][_pi]
            _e = _first.setdefault(_w["id"], {"dist": _w["distance_m"], "px": _w["height_px"], "text": _w["text"],
                                              "read_at": None, "read_px": None})
            if _e["read_at"] is None and road_match_key(_d["text"]) == road_match_key(_w["text"]):
                _e["read_at"], _e["read_px"] = _w["distance_m"], _w["height_px"]
print(f"\n{'sign text':<22}{'first seen':>12}{'first read':>12}{'text height at first read':>28}")
for _e in sorted(_first.values(), key=lambda e: -e["dist"])[:20]:
    _r = f"{_e['read_at']:.0f} m" if _e["read_at"] else "never"
    _p = f"{_e['read_px']:.0f} px" if _e["read_px"] else "-"
    print(f"{_e['text'][:20]:<22}{_e['dist']:>9.0f} m{_r:>12}{_p:>28}")
_read = [e for e in _first.values() if e["read_at"]]
if _read:
    print(f"\nA sign is first read at {np.median([e['read_at'] for e in _read]):.0f} m on average, "
          f"when its text is about {np.median([e['read_px'] for e in _read]):.0f} px tall.")

print("\nOne voted reading per sign:")
top_tracks(DASH_RUN, n=20)
_mb = Path(DASH_RUN["output_video"]).stat().st_size / 1024**2
print(f"\nAnnotated video ({_mb:.0f} MB):", DASH_RUN["output_video"])
if _mb <= 60:
    display(Video(DASH_RUN["output_video"], embed=True, width=900))
DASH_VIEWER = frame_viewer(DASH_RUN)
