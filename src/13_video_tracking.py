# ===== CELL 22 (9.9): Video processing, tracking + voting, pseudo road video, video evaluation =====
_PREFERRED_LABEL_FONTS = ("notosans-regular", "notosansdevanagari-regular", "hind-regular", "poppins-regular",
                          "dejavusans.ttf", "liberationsans-regular", "mukta-regular")


def _label_fonts(font_dir=None):
    fonts = find_fonts(str(font_dir) if font_dir else None, True)
    rank = lambda p: next((i for i, k in enumerate(_PREFERRED_LABEL_FONTS) if k in Path(p).name.lower()), 99)
    return {k: sorted(v, key=rank) for k, v in fonts.items()}


class RoadFrameAnnotator:
    """Draws polygons and labels. overlay='fused' shows the track's voted text, 'raw' shows this frame's text."""
    COLORS = {"accepted": (20, 160, 80), "low_confidence": (215, 125, 0), "unreadable": (200, 40, 40)}

    def __init__(self, font_dir=None, font_px=22, overlay="fused"):
        self.fonts, self.font_px, self.overlay = _label_fonts(font_dir), font_px, overlay

    def _geometry(self, label, max_width):
        try:
            runs, bounds = text_geometry(label, self.fonts, self.font_px)
        except RuntimeError:
            label = "(glyphs missing - see table)"
            runs, bounds = text_geometry(label, self.fonts, self.font_px)
        while len(label) > 4 and bounds[2] - bounds[0] > max_width:
            label = label[:-4] + "..."
            runs, bounds = text_geometry(label, self.fonts, self.font_px)
        return runs, bounds

    def __call__(self, frame, record):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        W, H = image.size
        for d in record["detections"]:
            color = self.COLORS.get(d["status"], (215, 125, 0))
            pts = [(float(x), float(y)) for x, y in d["poly"]]
            draw.line(pts + [pts[0]], fill=color, width=2)
            shown = d.get("fused_text") if self.overlay == "fused" and d.get("fused_text") else d["text"]
            runs, (l, t, r, b) = self._geometry(f"{shown or 'unreadable'} [{d['score']:.2f}]", max(20, W - 16))
            tw, th, pad = r - l, b - t, 3
            x1, y1, x2, y2 = d["box"]
            x = max(0, min(x1, W - tw - 2 * pad))
            y = y1 - th - 2 * pad if y1 >= th + 2 * pad else y2 + 2
            y = max(0, min(y, H - th - 2 * pad))
            draw.rectangle((x, y, x + tw + 2 * pad, y + th + 2 * pad), fill=color)
            draw_text_runs(draw, runs, x + pad - l, y + pad - t, (255, 255, 255))
        footer = f"frame {record.get('frame_index', 0) + 1}  t={record.get('timestamp_sec', 0.0):.2f}s  " \
                 f"{len(record['detections'])} text regions"
        runs, (l, t, r, b) = self._geometry(footer, max(20, W - 16))
        draw.rectangle((0, H - (b - t) - 8, r - l + 8, H), fill=(0, 0, 0))
        draw_text_runs(draw, runs, 4 - l, H - 4 - b, (255, 255, 255))
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class RoadTracker:
    """IoU tracker. Each track keeps score-weighted votes over the texts read in its frames."""

    def __init__(self, iou_thr=0.3, max_gap=8):
        self.iou_thr, self.max_gap = iou_thr, max_gap
        self.tracks, self.next_id = {}, 1

    @staticmethod
    def best_text(track):
        if not track["votes"]:
            return "", 0.0
        total = sum(v["weight"] for v in track["votes"].values())
        best = max(track["votes"].values(), key=lambda v: v["weight"])
        return best["text"], round(best["weight"] / total, 3)

    def update(self, frame_no, detections):
        live = [t for t in self.tracks.values() if frame_no - t["last"] <= self.max_gap]
        pairs = []
        for di, d in enumerate(detections):
            for t in live:
                iou = _box_iou(d["box"], t["box"])
                if iou >= self.iou_thr:
                    pairs.append((iou, di, t["id"]))
        pairs.sort(key=lambda p: -p[0])
        used_d, used_t, assign = set(), set(), {}
        for _, di, tid in pairs:
            if di not in used_d and tid not in used_t:
                used_d.add(di)
                used_t.add(tid)
                assign[di] = tid
        for di, d in enumerate(detections):
            tid = assign.get(di)
            if tid is None:
                tid, self.next_id = self.next_id, self.next_id + 1
                self.tracks[tid] = {"id": tid, "first": frame_no, "last": frame_no, "frames": 0, "box": d["box"], "votes": {}}
            track = self.tracks[tid]
            track["box"], track["last"] = d["box"], frame_no
            track["frames"] += 1
            key = road_match_key(d["text"])
            if key:
                vote = track["votes"].setdefault(key, {"weight": 0.0, "text": d["text"], "best": -1.0, "count": 0})
                vote["weight"] += max(float(d["score"]), 1e-3)
                vote["count"] += 1
                if d["score"] > vote["best"]:
                    vote["best"], vote["text"] = float(d["score"]), d["text"]
            d["track_id"] = tid
            d["fused_text"], d["fused_share"] = self.best_text(track)

    def final_tracks(self):
        rows = []
        for t in self.tracks.values():
            text, share = self.best_text(t)
            rows.append({"track_id": t["id"], "first_frame": t["first"], "last_frame": t["last"], "frames": t["frames"],
                         "text": text, "vote_share": share, "readings": sum(v["count"] for v in t["votes"].values())})
        return rows


def _to_h264(raw_path, final_path, crf=23):
    if shutil.which("ffmpeg"):
        done = subprocess.run(["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", str(raw_path), "-an",
                               "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
                               str(final_path)])
        if done.returncode == 0 and Path(final_path).is_file() and Path(final_path).stat().st_size > 0:
            Path(raw_path).unlink(missing_ok=True)
            return Path(final_path)
    return Path(raw_path)


_ROAD_RECORD_KEYS = ("box", "poly", "text", "raw_text", "score", "det_score", "status", "lexicon_fixed",
                     "track_id", "fused_text", "fused_share")


def process_road_video(video_path, spotter, output_dir, max_frames=None, every_n=1, overlay="fused",
                       preview_every=0):
    """Outputs frames.jsonl + frames.idx + CSVs + annotated video; the summary works with frame_viewer()."""
    source = Path(video_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {source}. Try converting it to H.264 MP4.")
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise ValueError("The video has no decodable frames.")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
        print("Video FPS unknown; assuming 25.")
    H, W = frame.shape[:2]
    size = (W + W % 2, H + H % 2)
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", source.stem)[:50] or "video"
    run_dir = Path(output_dir) / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "annotated_mp4v.mp4"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / every_n, size)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("OpenCV cannot write the annotated video.")
    tracker = RoadTracker(max_gap=max(2, int(round(fps / every_n / 3))))
    annotator = RoadFrameAnnotator(ROAD_FONT_DIR if ROAD_FONT_DIR.is_dir() else None, overlay=overlay)
    started, count, frame_no, n_det, n_acc = time.perf_counter(), 0, 0, 0, 0
    fields = ["frame_index", "source_frame", "timestamp_sec", "track_id", "x1", "y1", "x2", "y2", "text",
              "raw_text", "fused_text", "score", "status"]
    try:
        with (run_dir / "frames.jsonl").open("wb") as records, (run_dir / "frames.idx").open("wb") as index, \
                (run_dir / "detections.csv").open("w", encoding="utf-8-sig", newline="") as table:
            rows = csv.DictWriter(table, fieldnames=fields)
            rows.writeheader()
            while ok and (max_frames is None or count < max_frames):
                if frame.shape[:2] != (H, W):
                    raise ValueError("Frame size changed inside the video.")
                if frame_no % every_n == 0:
                    t0 = time.perf_counter()
                    dets = spotter(frame)
                    tracker.update(frame_no, dets)
                    record = {"frame_index": count, "source_frame": frame_no, "timestamp_sec": round(frame_no / fps, 4),
                              "latency_ms": round(1000 * (time.perf_counter() - t0), 1),
                              "detections": [{k: d.get(k) for k in _ROAD_RECORD_KEYS} for d in dets]}
                    index.write(struct.pack("<Q", records.tell()))
                    records.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
                    for d in dets:
                        rows.writerow({"frame_index": count, "source_frame": frame_no,
                                       "timestamp_sec": record["timestamp_sec"], "track_id": d["track_id"],
                                       "x1": d["box"][0], "y1": d["box"][1], "x2": d["box"][2], "y2": d["box"][3],
                                       "text": d["text"], "raw_text": d["raw_text"], "fused_text": d["fused_text"],
                                       "score": d["score"], "status": d["status"]})
                    n_det += len(dets)
                    n_acc += sum(d["status"] == "accepted" for d in dets)
                    vis = annotator(frame, record)
                    if size != (W, H):
                        vis = cv2.copyMakeBorder(vis, 0, size[1] - H, 0, size[0] - W, cv2.BORDER_REPLICATE)
                    writer.write(vis)
                    count += 1
                    if preview_every and count % preview_every == 0:
                        print(f"  frame {count}: {[d['fused_text'] for d in dets][:6]}")
                ok, frame = cap.read()
                frame_no += 1
    finally:
        writer.release()
        cap.release()
    if count == 0:
        raise ValueError("No frames were processed.")
    video = _to_h264(raw_path, run_dir / "annotated_h264.mp4")
    tracks = tracker.final_tracks()
    with (run_dir / "tracks.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer_t = csv.DictWriter(fh, fieldnames=list(tracks[0]) if tracks else ["track_id"])
        writer_t.writeheader()
        writer_t.writerows(tracks)
    seconds = time.perf_counter() - started
    summary = {"input_video": str(source), "run_dir": str(run_dir), "processed_frames": count,
               "fps": fps / every_n, "frame_records": str(run_dir / "frames.jsonl"),
               "frame_index": str(run_dir / "frames.idx"), "output_video": str(video),
               "detections_csv": str(run_dir / "detections.csv"), "tracks_csv": str(run_dir / "tracks.csv"),
               "tracks": len(tracks), "detections": n_det, "accepted": n_acc, "seconds": round(seconds, 1),
               "processing_fps": round(count / max(seconds, 1e-6), 2)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Processed {count} frames at {summary['processing_fps']} fps -> {run_dir}")
    return summary


def top_tracks(summary, n=15, min_frames=3):
    with open(summary["tracks_csv"], encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("text") and int(r["frames"]) >= min_frames]
    rows.sort(key=lambda r: -int(r["frames"]))
    for r in rows[:n]:
        print(f"track {r['track_id']:>4}  frames {r['first_frame']}-{r['last_frame']} ({r['frames']})  "
              f"vote {float(r['vote_share']):.0%}  {r['text']}")
    return rows


def make_pseudo_video(records, out_dir, name="bstd_test_drive", size=(1280, 720), fps=25, seconds_per_image=2.0,
                      seed=0, care_langs=BSTD_TARGET_LANGS):
    """Camera pan/zoom/shake over real TEST photos + blur + H.264, with exact per-frame ground truth."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path, gt_path = out_dir / f"{name}.mp4", out_dir / f"{name}_gt.jsonl"
    marker = out_dir / f".done_{name}_{len(records)}_{seconds_per_image}_{seed}"
    if marker.exists() and video_path.exists() and gt_path.exists():
        print("Pseudo video already built:", video_path)
        return video_path, gt_path
    W, H = size
    raw_path = out_dir / f"{name}_raw.mp4"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot write the pseudo video.")
    rng = np.random.default_rng(seed)
    frame_rect = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    frames, lines = 0, []
    try:
        for ri, rec in enumerate(records):
            img = cv2.imread(rec["path"], cv2.IMREAD_COLOR)
            if img is None:
                continue
            ih, iw = img.shape[:2]
            n = max(2, int(round(seconds_per_image * fps)))

            def window(zoom):
                ww = min(iw, ih * W / H) / zoom
                wh = ww * H / W
                return rng.uniform(ww / 2, iw - ww / 2), rng.uniform(wh / 2, ih - wh / 2), ww

            (cx0, cy0, w0), (cx1, cy1, w1) = window(rng.uniform(1.0, 1.5)), window(rng.uniform(1.0, 2.0))
            prev = None
            for k in range(n):
                t = k / (n - 1)
                t = t * t * (3 - 2 * t)
                cx = cx0 + (cx1 - cx0) * t + rng.normal(0, 1.2)
                cy = cy0 + (cy1 - cy0) * t + rng.normal(0, 1.2)
                s = W / (w0 + (w1 - w0) * t)
                M = np.array([[s, 0, W / 2 - s * cx], [0, s, H / 2 - s * cy]], np.float32)
                frame = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                if prev is not None:
                    dx, dy = (prev[0] - cx) * s, (prev[1] - cy) * s
                    length = int(min(15, np.hypot(dx, dy)))
                    if length >= 3:
                        kernel = np.zeros((length, length), np.float32)
                        c = (length - 1) / 2
                        ang = math.atan2(dy, dx)
                        cv2.line(kernel, (int(round(c - math.cos(ang) * c)), int(round(c - math.sin(ang) * c))),
                                 (int(round(c + math.cos(ang) * c)), int(round(c + math.sin(ang) * c))), 1.0, 1)
                        frame = cv2.filter2D(frame, -1, kernel / max(kernel.sum(), 1e-6))
                prev = (cx, cy)
                gain = 1.0 + 0.04 * math.sin(frames / 9.0)
                frame = np.clip(frame.astype(np.float32) * gain + rng.normal(0, 2.0, frame.shape), 0, 255).astype(np.uint8)
                writer.write(frame)
                words = []
                for wi, w in enumerate(rec["words"]):
                    p = (w["poly"] * s + np.array([W / 2 - s * cx, H / 2 - s * cy], np.float32)).astype(np.float32)
                    visible = poly_overlap(frame_rect, p)[1]
                    if visible <= 0.02:
                        continue
                    care = (not w["illegible"]) and w["lang"] in care_langs and visible >= 0.95
                    words.append({"id": f"{ri}:{wi}", "poly": np.round(p, 1).tolist(), "text": w["text"],
                                  "lang": w["lang"], "care": bool(care)})
                lines.append(json.dumps({"frame_index": frames, "source": rec["key"], "words": words},
                                        ensure_ascii=False))
                frames += 1
    finally:
        writer.release()
    if frames == 0:
        raise ValueError("No readable images for the pseudo video.")
    final = _to_h264(raw_path, video_path, crf=26)
    if final != video_path:
        shutil.move(str(final), str(video_path))
    cap = cv2.VideoCapture(str(video_path))
    decoded = 0
    while cap.grab():
        decoded += 1
    cap.release()
    if decoded != frames:
        raise RuntimeError(f"Encoded video has {decoded} frames but {frames} were written; ground truth would drift.")
    gt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    marker.write_text("ok")
    print(f"Pseudo road video: {frames} frames from {len(records)} real test photos -> {video_path}")
    return video_path, gt_path


def evaluate_video(summary, gt_path, iou_thr=0.5, found_share=0.5, min_visible=3):
    """Frame-level detection/reading, plus per text instance: found in >=50% of its frames, read by track vote."""
    gt = {}
    with open(gt_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                gt[rec["frame_index"]] = rec["words"]
    track_text = {}
    with open(summary["tracks_csv"], encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            track_text[int(row["track_id"])] = row.get("text", "")
    fc = Counter()
    inst = defaultdict(lambda: {"visible": 0, "found": 0, "tracks": Counter(), "text": "", "lang": ""})
    with open(summary["frame_records"], encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            words = [{**w, "poly": np.asarray(w["poly"], np.float32), "illegible": not w["care"]}
                     for w in gt.get(rec["source_frame"], [])]
            care = [w["care"] for w in words]
            preds = rec["detections"]
            matches, ignored = match_detections(words, [np.asarray(d["poly"], np.float32) for d in preds], care, iou_thr)
            hit = {gi: pi for gi, pi, _ in matches}
            fc["frames"] += 1
            fc["pred"] += len(preds) - len(ignored)
            fc["tp"] += len(matches)
            for gi, w in enumerate(words):
                if not care[gi]:
                    continue
                key = road_match_key(w["text"])
                fc["gt"] += 1
                item = inst[w["id"]]
                item["visible"] += 1
                item["text"], item["lang"] = w["text"], w["lang"]
                if gi in hit:
                    d = preds[hit[gi]]
                    item["found"] += 1
                    item["tracks"][d["track_id"]] += 1
                    fc["read_frame"] += road_match_key(d["text"]) == key
                    fc["read_fused_causal"] += road_match_key(d["fused_text"] or "") == key
    groups = defaultdict(Counter)
    for item in inst.values():
        if item["visible"] < min_visible:
            continue
        found = item["found"] / item["visible"] >= found_share
        read = False
        if item["tracks"]:
            best_track = item["tracks"].most_common(1)[0][0]
            read = road_match_key(track_text.get(best_track, "")) == road_match_key(item["text"])
        for g in ("all", item["lang"]):
            groups[g]["n"] += 1
            groups[g]["found"] += found
            groups[g]["read"] += read
    p = fc["tp"] / max(1, fc["pred"])
    r = fc["tp"] / max(1, fc["gt"])
    a = groups["all"]
    return {"frames": fc["frames"], "frame_det_precision": p, "frame_det_recall": r,
            "frame_det_f1": 2 * p * r / max(1e-9, p + r), "frame_read_recall": fc["read_frame"] / max(1, fc["gt"]),
            "frame_read_recall_fused": fc["read_fused_causal"] / max(1, fc["gt"]), "instances": a["n"],
            "instance_found": a["found"] / max(1, a["n"]), "instance_read": a["read"] / max(1, a["n"]),
            "groups": {g: {"n": c["n"], "found": c["found"] / max(1, c["n"]), "read": c["read"] / max(1, c["n"])}
                       for g, c in sorted(groups.items())}}


def print_video_report(m, title):
    print(f"\n=== {title}: {m['frames']} frames, {m['instances']} text instances ===")
    print(f"Per frame: detection P {m['frame_det_precision']:.1%} R {m['frame_det_recall']:.1%} "
          f"F1 {m['frame_det_f1']:.1%} | read correctly {m['frame_read_recall']:.1%} "
          f"(with running track vote {m['frame_read_recall_fused']:.1%})")
    print(f"Per text instance: found {m['instance_found']:.1%} | read correctly (track vote) {m['instance_read']:.1%}")
    for g, v in m["groups"].items():
        print(f"  {g:<10} n={v['n']:<5} found {v['found']:.1%}  read {v['read']:.1%}")


def download_video(url, dest_dir):
    """Download a video you are allowed to use (e.g. a Wikimedia Commons file URL); convert if OpenCV can't read it."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(urllib.parse.urlparse(url).path).name) or "video.mp4"
    out = dest_dir / name
    if not out.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "road-text-ocr-notebook/1.0 (educational)"})
        part = out.with_name(out.name + ".part")
        with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as fh:
            shutil.copyfileobj(response, fh)
        part.replace(out)
    cap = cv2.VideoCapture(str(out))
    readable = cap.isOpened() and cap.read()[0]
    cap.release()
    if readable:
        return out
    converted = out.with_suffix(".converted.mp4")
    if shutil.which("ffmpeg") and subprocess.run(["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", str(out),
                                                  "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(converted)]).returncode == 0:
        return converted
    raise ValueError(f"OpenCV cannot decode {out}")


def load_road_spotter(phase_dir, min_score=0.5):
    """Rebuild a spotter from a phase folder (e.g. after a Colab restart)."""
    phase_dir = Path(phase_dir)
    lexicon_path = phase_dir / "lexicon.txt"
    words = lexicon_path.read_text(encoding="utf-8").split("\n") if lexicon_path.exists() else []
    return RoadTextSpotter(TextDetector.load(phase_dir), RoadRecognizer.load(phase_dir),
                           RoadLexicon([w for w in words if w]), min_score=min_score)
