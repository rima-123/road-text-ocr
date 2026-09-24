# ===== CELL 21 (9.8): End-to-end spotter + evaluation =====
class RoadTextSpotter:
    """BGR frame -> detections with polygon, box, raw and lexicon-corrected text, score, status."""

    def __init__(self, detector, recognizer, lexicon=None, min_score=0.5, pad_ratio=0.08, use_lexicon=True,
                 max_area_share=0.6, crop_height=96):
        self.detector, self.recognizer, self.lexicon = detector, recognizer, lexicon
        self.min_score, self.pad_ratio, self.use_lexicon = min_score, pad_ratio, use_lexicon
        self.max_area_share, self.crop_height = max_area_share, crop_height

    def __call__(self, frame):
        found = self.detector(frame)
        frame_area = float(frame.shape[0] * frame.shape[1])
        crops, kept = [], []
        for det in found:
            w, h = quad_size(det["quad"])
            if w * h > self.max_area_share * frame_area:  # a 'word' covering most of the frame is noise
                continue
            crop = crop_quad(frame, det["quad"], self.pad_ratio, max_side=1024)
            if crop is None or min(crop.shape[:2]) < 2:
                continue
            if crop.shape[0] > self.crop_height:  # keep memory bounded; the recognizer input is smaller anyway
                new_w = max(2, min(self.crop_height * 16, round(crop.shape[1] * self.crop_height / crop.shape[0])))
                crop = cv2.resize(crop, (new_w, self.crop_height), interpolation=cv2.INTER_AREA)
            crops.append(crop)
            kept.append(det)
        readings = self.recognizer(crops) if crops else []
        out = []
        for det, (raw, score) in zip(kept, readings):
            raw, score = str(raw).strip(), float(score)
            text, fixed = (self.lexicon.correct(raw, score) if (self.lexicon and self.use_lexicon) else (raw, False))
            quad = det["quad"]
            x1, y1 = np.floor(quad.min(axis=0)).astype(int)
            x2, y2 = np.ceil(quad.max(axis=0)).astype(int)
            status = "accepted" if text and score >= self.min_score else "low_confidence" if text else "unreadable"
            out.append({"quad": quad, "poly": np.round(quad, 1).tolist(), "box": [int(x1), int(y1), int(x2), int(y2)],
                        "det_score": round(det["score"], 4), "raw_text": raw, "text": text, "score": round(score, 4),
                        "lexicon_fixed": bool(fixed), "status": status})
        return out


def _size_group(height):
    return "height<16px" if height < 16 else "height16-32px" if height < 32 else "height>=32px"


def evaluate_spotter(spotter, records, care_langs=BSTD_TARGET_LANGS, max_images=None, iou_thr=0.5, name="eval"):
    """Detection P/R/F1 (IoU>=0.5, don't-care aware) and end-to-end 'found AND read correctly'."""
    gt = defaultdict(Counter)      # per group: n, found, read_raw, read_lex
    pr = Counter()                 # prediction-side counts
    cer_values = []
    started, n_img = time.perf_counter(), 0
    for rec in records[:max_images]:
        img = cv2.imread(rec["path"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        n_img += 1
        preds = spotter(img)
        words = rec["words"]
        care = [(not w["illegible"]) and w["lang"] in care_langs for w in words]
        matches, ignored = match_detections(words, [p["quad"] for p in preds], care, iou_thr)
        pred_to_gt = {pi: gi for gi, pi, _ in matches}
        found = {gi: pi for gi, pi, _ in matches}
        for gi, w in enumerate(words):
            if not care[gi]:
                continue
            key = road_match_key(w["text"])
            hit = found.get(gi)
            ok_raw = hit is not None and road_match_key(preds[hit]["raw_text"]) == key
            ok_lex = hit is not None and road_match_key(preds[hit]["text"]) == key
            for g in ("all", w["lang"], _size_group(poly_text_height(w["poly"]))):
                gt[g]["n"] += 1
                gt[g]["found"] += hit is not None
                gt[g]["read_raw"] += ok_raw
                gt[g]["read_lex"] += ok_lex
            if hit is not None:
                cer_values.append(character_error_rate(w["text"], preds[hit]["raw_text"]))
        for pi, p in enumerate(preds):
            if pi in ignored:
                continue
            pr["pred"] += 1
            pr["det_tp"] += pi in pred_to_gt
            if p["status"] == "accepted":
                gkey = road_match_key(words[pred_to_gt[pi]]["text"]) if pi in pred_to_gt else None
                pr["claims_raw"] += 1
                pr["claims_ok_raw"] += gkey is not None and road_match_key(p["raw_text"]) == gkey
                pr["claims_ok_lex"] += gkey is not None and road_match_key(p["text"]) == gkey
    a = gt["all"]
    det_p = pr["det_tp"] / max(1, pr["pred"])
    det_r = a["found"] / max(1, a["n"])
    metrics = {
        "name": name, "images": n_img, "gt_words": a["n"], "predictions": pr["pred"],
        "det_precision": det_p, "det_recall": det_r, "det_f1": 2 * det_p * det_r / max(1e-9, det_p + det_r),
        "e2e_recall_raw": a["read_raw"] / max(1, a["n"]), "e2e_recall_lexicon": a["read_lex"] / max(1, a["n"]),
        "e2e_precision_raw": pr["claims_ok_raw"] / max(1, pr["claims_raw"]),
        "e2e_precision_lexicon": pr["claims_ok_lex"] / max(1, pr["claims_raw"]),
        "read_acc_of_found_raw": a["read_raw"] / max(1, a["found"]),
        "read_acc_of_found_lexicon": a["read_lex"] / max(1, a["found"]),
        "cer_of_found": float(np.mean(cer_values)) if cer_values else None,
        "seconds_per_image": (time.perf_counter() - started) / max(1, n_img),
        "groups": {g: {"n": c["n"], "found": c["found"] / max(1, c["n"]), "read_raw": c["read_raw"] / max(1, c["n"]),
                       "read_lexicon": c["read_lex"] / max(1, c["n"])} for g, c in sorted(gt.items())},
    }
    return metrics


def print_spotter_report(m):
    print(f"\n=== {m['name']}: {m['images']} images, {m['gt_words']} scored words, {m['predictions']} predictions ===")
    print(f"Detection  precision {m['det_precision']:.1%}  recall {m['det_recall']:.1%}  F1 {m['det_f1']:.1%}")
    print(f"Found AND read correctly (recall): {m['e2e_recall_raw']:.1%} raw | {m['e2e_recall_lexicon']:.1%} with lexicon")
    print(f"Accepted readings that are correct (precision): {m['e2e_precision_raw']:.1%} raw | "
          f"{m['e2e_precision_lexicon']:.1%} with lexicon")
    cer = m["cer_of_found"]
    print(f"Reading accuracy on found words: {m['read_acc_of_found_raw']:.1%} raw | "
          f"{m['read_acc_of_found_lexicon']:.1%} with lexicon" + (f" | CER {cer:.3f}" if cer is not None else ""))
    print(f"{'group':<15}{'words':>7}{'found':>9}{'read raw':>10}{'read +lex':>11}")
    for g, v in m["groups"].items():
        print(f"{g:<15}{v['n']:>7}{v['found']:>9.1%}{v['read_raw']:>10.1%}{v['read_lexicon']:>11.1%}")
    print(f"Speed: {m['seconds_per_image']:.2f} s/image")


def target_check(label, value, low=0.80):
    status = "PASS" if value >= low else "NOT YET"
    print(f"  [{status:7}] {label}: {value:.1%} (target 80-90%)")
    return value >= low


def summarize_targets(image_metrics=None, video_metrics=None):
    print("\n=== Target check: 80-90% 'find + read correctly' ===")
    if image_metrics:
        g = image_metrics["groups"]
        target_check("Detection recall, all Hindi+English words", image_metrics["det_recall"])
        target_check("Found+read, all words (lexicon)", image_metrics["e2e_recall_lexicon"])
        for size in ("height>=32px",):
            if size in g:
                target_check(f"Found+read, clearly visible words ({size})", g[size]["read_lexicon"])
        for lang in BSTD_TARGET_LANGS:
            if lang in g:
                target_check(f"Found+read, {lang}", g[lang]["read_lexicon"])
    if video_metrics:
        target_check("Video: text instances found", video_metrics["instance_found"])
        target_check("Video: text instances read correctly (track vote)", video_metrics["instance_read"])


def save_report(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("Report saved:", path)


def show_spotter_examples(spotter, records, n=4, max_width=900):
    """Draw predictions (green = accepted, amber = low confidence) on a few images."""
    from IPython.display import display
    annotator = RoadFrameAnnotator(ROAD_FONT_DIR if ROAD_FONT_DIR.is_dir() else None)
    for rec in records[:n]:
        img = cv2.imread(rec["path"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        dets = spotter(img)
        vis = annotator(img, {"detections": dets, "frame_index": 0, "timestamp_sec": 0.0})
        if vis.shape[1] > max_width:
            vis = cv2.resize(vis, (max_width, round(vis.shape[0] * max_width / vis.shape[1])))
        ok, jpg = cv2.imencode(".jpg", vis)
        if ok:
            display(Image.open(io.BytesIO(jpg.tobytes())))
