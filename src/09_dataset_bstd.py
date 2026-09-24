# ===== CELL 18 (9.5): BSTD dataset — download, prepare, word crops =====
BSTD_DETECTION_FILE_ID = "1S7KUYfB-lQvbu6GtvZxxDPOD5ZZ080M0"
BSTD_TARGET_LANGS = ("hindi", "english")             # languages that are scored
BSTD_RECOG_LANGS = ("hindi", "english", "marathi")   # Marathi is Devanagari too: extra reading practice
BSTD_MANUAL_HELP = f"""
Automatic download did not work (Google Drive often limits very large public files). Do this once:
  1. Open https://drive.google.com/file/d/{BSTD_DETECTION_FILE_ID}/view
  2. Use "Add shortcut to Drive" (uses no quota), or download it and upload the zip to your Drive.
  3. Run the 'Get the dataset' cell again with BSTD_ZIP_PATH set to that file, e.g.
     BSTD_ZIP_PATH = "/content/drive/MyDrive/<name of the zip>.zip"
Kaggle: turn Settings -> Internet ON and rerun; or upload the zip (or its extracted folder) as your own
Kaggle dataset, attach it with 'Add Input' and rerun (it is found automatically).
"""


def _norm_lang(value):
    v = re.sub(r"[^a-z]", "", str(value or "").lower())
    for prefix, name in (("eng", "english"), ("hin", "hindi"), ("mar", "marathi")):
        if v.startswith(prefix):
            return name
    return v or "unknown"


def download_bstd_detection(dest_dir, zip_path=""):
    """Return a local path to the BSTD detection zip (existing file, BSTD_ZIP_PATH, or a gdown download)."""
    if zip_path:
        zp = Path(zip_path).expanduser()
        if zp.is_file() and zipfile.is_zipfile(zp):
            return zp
        raise FileNotFoundError(f"BSTD_ZIP_PATH is not a readable zip file: {zp}")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / "BSTD_detection.zip"
    if target.is_file() and zipfile.is_zipfile(target):
        print("Using the already downloaded", target)
        return target
    free_gb = shutil.disk_usage(dest_dir).free / 1024**3
    if free_gb < 19:
        raise RuntimeError(f"Only {free_gb:.1f} GB free on {dest_dir}; the zip needs ~17 GB plus ~3 GB for the "
                           f"prepared images.\n{BSTD_MANUAL_HELP}")
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("gdown is not installed (pip install gdown)." + BSTD_MANUAL_HELP) from exc
    try:
        gdown.download(id=BSTD_DETECTION_FILE_ID, output=str(target), quiet=False, resume=True)
    except Exception as exc:  # quota page, network error, interrupted transfer
        print("Download error:", exc)
    if not (target.is_file() and zipfile.is_zipfile(target)):
        raise RuntimeError(BSTD_MANUAL_HELP)
    return target


class BstdSource:
    """Read access to the BSTD detection data either as a .zip or as an extracted folder (e.g. Kaggle input)."""

    def __init__(self, path):
        self.path = Path(path)
        if self.path.is_file():
            self.zip = zipfile.ZipFile(self.path)
        elif self.path.is_dir():
            self.zip = None
        else:
            raise FileNotFoundError(self.path)

    def names(self):
        if self.zip is not None:
            return [i.filename for i in self.zip.infolist() if not i.is_dir()]
        return [p.relative_to(self.path).as_posix() for p in self.path.rglob("*") if p.is_file()]

    def size(self, name):
        return self.zip.getinfo(name).file_size if self.zip is not None else (self.path / name).stat().st_size

    def read(self, name):
        return self.zip.read(name) if self.zip is not None else (self.path / name).read_bytes()

    def close(self):
        if self.zip is not None:
            self.zip.close()


def _find_annotation_json(source, names):
    names = [n for n in names if "__MACOSX" not in n]
    hits = [n for n in names if re.search(r"(^|/)BSTD[^/]*\.json$", n, re.I)] or \
        [n for n in names if n.lower().endswith(".json")]
    if not hits:
        raise FileNotFoundError("No BSTD annotation JSON found in the dataset source.")
    return max(hits, key=source.size)


def _apply_exif(img, orientation):
    """Same result as PIL.ImageOps.exif_transpose, independent of the OpenCV version."""
    if orientation == 2:
        return cv2.flip(img, 1)
    if orientation == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(img, 0)
    if orientation == 5:
        return cv2.transpose(img)
    if orientation == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.flip(cv2.transpose(img), -1)
    if orientation == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _decode_for_annotations(data, polys, max_side):
    """Decode (reduced-size when possible) in the orientation that the polygons fit. Returns (img, sx, sy)."""
    raw_w = raw_h = None
    orientation = 1
    try:
        with Image.open(io.BytesIO(data)) as probe:
            raw_w, raw_h = probe.size
            orientation = int(probe.getexif().get(0x0112, 1) or 1)
    except Exception:
        pass
    if orientation not in range(1, 9):
        orientation = 1
    turned = orientation in (5, 6, 7, 8)
    use_exif = True  # the official visualiser (cv2.imread) applies EXIF orientation
    if raw_w and polys and orientation != 1:
        pts = np.concatenate(polys)
        mx, my = float(pts[:, 0].max()), float(pts[:, 1].max())
        ow, oh = (raw_h, raw_w) if turned else (raw_w, raw_h)
        fits_oriented = mx <= ow * 1.02 and my <= oh * 1.02
        fits_raw = mx <= raw_w * 1.02 and my <= raw_h * 1.02
        use_exif = fits_oriented or not fits_raw
    flags = cv2.IMREAD_COLOR
    if raw_w:
        long_side = max(raw_w, raw_h)
        flags = (cv2.IMREAD_REDUCED_COLOR_4 if long_side >= 4 * max_side else
                 cv2.IMREAD_REDUCED_COLOR_2 if long_side >= 2 * max_side else cv2.IMREAD_COLOR)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, flags | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None and flags != cv2.IMREAD_COLOR:
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        return None, 1.0, 1.0
    if use_exif and orientation != 1:
        img = _apply_exif(img, orientation)
    if raw_w:
        ref_w, ref_h = (raw_h, raw_w) if (use_exif and turned) else (raw_w, raw_h)
    else:
        ref_w, ref_h = img.shape[1], img.shape[0]
    sx, sy = img.shape[1] / ref_w, img.shape[0] / ref_h
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        sx, sy = sx * img.shape[1] / w, sy * img.shape[0] / h
    return img, sx, sy


def bstd_is_prepared(out_dir, max_side=1600, max_images=None):
    return (Path(out_dir) / f".done_{max_side}_{max_images}").exists()


def find_bstd_source(explicit=""):
    """Explicit zip/folder > a Kaggle input containing BSTD*.json (extracted or zipped) > None."""
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_dir() or (p.is_file() and zipfile.is_zipfile(p)):
            return p
        raise FileNotFoundError(f"BSTD_ZIP_PATH is neither a folder nor a readable zip: {p}")
    root = Path("/kaggle/input")
    if root.is_dir():
        for js in sorted(root.rglob("*.json")):
            if re.match(r"BSTD.*\.json$", js.name, re.I):
                return root / js.relative_to(root).parts[0]
        for z in sorted(root.rglob("*.zip")):
            try:
                with zipfile.ZipFile(z) as zf:
                    if any(re.search(r"(^|/)BSTD[^/]*\.json$", n, re.I) for n in zf.namelist()):
                        return z
            except zipfile.BadZipFile:
                continue
    return None


def make_bstd_standin(out_zip, fonts, n_train=30, n_test=20, seed=42):
    """Tiny BSTD-format zip drawn by RoadSceneSynth: lets QUICK_RUN check the pipeline without 17 GB."""
    out_zip = Path(out_zip)
    if out_zip.is_file() and zipfile.is_zipfile(out_zip):
        return out_zip
    synth, data, k = RoadSceneSynth(fonts), {}, 0
    part = out_zip.with_name(out_zip.name + ".part")
    with zipfile.ZipFile(part, "w") as zf:
        for split, n, folder in (("train", n_train, "A"), ("test", n_test, "B")):
            for i in range(n):
                img, words = synth.render(np.random.default_rng([seed, k]))
                k += 1
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not ok:
                    continue
                name = f"{folder}/image_{i + 1}.jpg"
                zf.writestr(f"Detection/{name}", buf.tobytes())
                anns = {f"polygon_{j}": {"coordinates": [[int(round(x)), int(round(y))] for x, y in w["poly"]],
                                         "text": "###" if w["ignore"] else w["text"],
                                         "script_language": "Hindi" if w["lang"] == "hindi" else "English"}
                        for j, w in enumerate(words)}
                data[f"{folder}_image_{i + 1}"] = {"annotations": anns, "image_name": name, "split": split,
                                                   "folderName": folder}
        zf.writestr("Detection/BSTD_standin.json", json.dumps(data, ensure_ascii=False))
    part.replace(out_zip)
    return out_zip


def prepare_bstd(source_path, out_dir, max_side=1600, max_images=None, workers=4):
    """Read images straight from the zip or folder (no 17 GB extraction), resize, write JSONL annotations."""
    out_dir = Path(out_dir)
    if bstd_is_prepared(out_dir, max_side, max_images):
        print("BSTD already prepared in", out_dir)
        return {s: out_dir / f"bstd_{s}.jsonl" for s in ("train", "test")}
    started = time.perf_counter()
    zf = BstdSource(source_path)
    try:
        all_names = zf.names()
        json_name = _find_annotation_json(zf, all_names)
        data = json.loads(zf.read(json_name).decode("utf-8"))
        print("Annotations:", json_name, "-", len(data), "images")
        index, by_name = {}, Counter()
        for original in all_names:
            name = original.replace("\\", "/")
            if "__MACOSX" in name or not name.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")):
                continue
            parts = name.split("/")
            index["/".join(parts[-2:]).lower()] = original
            by_name[parts[-1].lower()] += 1
            index.setdefault("name:" + parts[-1].lower(), original)
        jobs, missing, bad_polys = [], 0, 0
        for key, rec in data.items():
            if not isinstance(rec, dict):
                continue
            split = str(rec.get("split", "")).strip().lower()
            split = "test" if split.startswith("test") else "train"
            image_name = str(rec.get("image_name", "") or "").replace("\\", "/")
            folder = str(rec.get("folderName", "") or "")
            file_name = image_name.split("/")[-1]
            candidates = ["/".join(image_name.split("/")[-2:]).lower(), f"{folder}/{file_name}".lower()]
            member = next((index[c] for c in candidates if c in index), None)
            if member is None and by_name[file_name.lower()] == 1:
                member = index.get("name:" + file_name.lower())
            if member is None:
                missing += 1
                continue
            anns = rec.get("annotations") or {}
            words = []
            for ann in (anns.values() if isinstance(anns, dict) else anns):
                if not isinstance(ann, dict):
                    continue
                coords = ann.get("coordinates", ann.get("points", []))
                try:
                    if isinstance(coords, str):
                        coords = json.loads(coords)
                    poly = np.asarray(coords, np.float32).reshape(-1, 2)
                except (ValueError, TypeError):
                    bad_polys += 1
                    continue
                if len(poly) < 3 or not np.isfinite(poly).all():
                    bad_polys += 1
                    continue
                words.append({"poly": poly, "text": str(ann.get("text", "") or ""),
                              "lang": _norm_lang(ann.get("script_language", ann.get("language", "")))})
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(key))[:120]
            jobs.append((member, split, safe, words, {k: rec.get(k) for k in ("environment", "lighting")}))
        jobs.sort(key=lambda j: j[2])
        if max_images:
            by_split = defaultdict(list)
            for job in jobs:
                by_split[job[1]].append(job)
            jobs = [j for s in by_split.values() for j in s[:max_images]]
        print(f"Images to prepare: {len(jobs)} (missing in zip: {missing}, unreadable polygons: {bad_polys})")
        for s in ("train", "test"):
            (out_dir / "images" / s).mkdir(parents=True, exist_ok=True)

        def work(data_bytes, job):
            member, split, safe, words, meta = job
            img, sx, sy = _decode_for_annotations(data_bytes, [w["poly"] for w in words], max_side)
            if img is None:
                return None
            rel = f"images/{split}/{safe}.jpg"
            if not cv2.imwrite(str(out_dir / rel), img, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                return None
            H, W = img.shape[:2]
            out_words = []
            for w in words:
                p = w["poly"] * np.array([sx, sy], np.float32)
                p[:, 0] = np.clip(p[:, 0], 0, W - 1)
                p[:, 1] = np.clip(p[:, 1], 0, H - 1)
                out_words.append({"poly": np.round(p, 1).tolist(), "text": w["text"], "lang": w["lang"]})
            return {"image": rel, "key": safe, "width": W, "height": H, "split": split, "source": "bstd",
                    "words": out_words, **meta}

        results, failed = {"train": [], "test": []}, 0
        pending = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for n, job in enumerate(jobs, 1):
                pending.append(pool.submit(work, zf.read(job[0]), job))
                if len(pending) >= workers * 4 or n == len(jobs):
                    for fut in pending:
                        res = fut.result()
                        if res is None:
                            failed += 1
                        else:
                            results[res["split"]].append(res)
                    pending = []
                if n % 500 == 0:
                    print(f"  {n}/{len(jobs)} images ({time.perf_counter() - started:.0f}s)")
    finally:
        zf.close()
    paths = {}
    for split, recs in results.items():
        paths[split] = out_dir / f"bstd_{split}.jsonl"
        with paths[split].open("w", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out_dir / f".done_{max_side}_{max_images}").write_text("ok")
    print(f"Prepared train={len(results['train'])} test={len(results['test'])} (failed {failed}) "
          f"in {time.perf_counter() - started:.0f}s")
    return paths


def load_annotations(jsonl_path, max_items=None):
    """Records: {'path','key','split','words':[{'poly','text','lang','illegible'}]}."""
    jsonl_path = Path(jsonl_path)
    root, records = jsonl_path.parent, []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            words = []
            for w in rec.get("words", []):
                poly = np.asarray(w["poly"], np.float32).reshape(-1, 2)
                if len(poly) < 3:
                    continue
                text = str(w.get("text", "") or "")
                illegible = bool(w.get("ignore", False)) or not text.strip() or set(text.strip()) <= {"#"}
                words.append({"poly": poly, "text": text, "lang": w.get("lang", "unknown"), "illegible": illegible})
            records.append({"path": str(root / rec["image"]), "key": rec.get("key", rec["image"]),
                            "split": rec.get("split", ""), "source": rec.get("source", ""), "words": words})
            if max_items and len(records) >= max_items:
                break
    return records


def split_train_val(records, val_fraction=0.05, seed=0):
    """Deterministic image-level hold-out from the TRAIN split (the test split stays untouched)."""
    order = sorted(range(len(records)), key=lambda i: records[i]["key"])
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    n_val = max(1, int(round(len(records) * val_fraction))) if len(records) > 1 else 0
    val_ids = set(order[:n_val])
    return [r for i, r in enumerate(records) if i not in val_ids], [records[i] for i in sorted(val_ids)]


def dataset_stats(records, name):
    langs = Counter(w["lang"] for r in records for w in r["words"])
    legible = sum(not w["illegible"] for r in records for w in r["words"])
    top = ", ".join(f"{k}:{v}" for k, v in langs.most_common(6))
    print(f"{name}: {len(records)} images, {sum(langs.values())} words ({legible} legible) | {top}")


def extract_word_crops(records, out_dir, split_name, langs=BSTD_RECOG_LANGS, min_height=8, margin=0.35, max_len=32):
    """Save each usable word with context margin + its quad (so training can jitter it like a detector)."""
    out_dir = Path(out_dir)
    csv_path = out_dir / f"crops_{split_name}.csv"
    marker = out_dir / f".done_crops_{split_name}"
    if marker.exists() and csv_path.exists():
        return csv_path
    (out_dir / split_name).mkdir(parents=True, exist_ok=True)
    rows = []
    for ri, rec in enumerate(records):
        wanted = []
        for wi, w in enumerate(rec["words"]):
            label = None if w["illegible"] or w["lang"] not in langs else road_clean_label(w["text"])
            if label and len(label) <= max_len:
                wanted.append((wi, w, label))
        if not wanted:
            continue
        img = cv2.imread(rec["path"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        for wi, w, label in wanted:
            q = order_quad(w["poly"])
            qw, qh = quad_size(q)
            if min(qw, qh) < min_height or (qh > 1.3 * qw and len(label) >= 3):  # tiny or vertical text
                continue
            m = margin * min(qw, qh)
            x0, y0 = int(max(0, np.floor(q[:, 0].min() - m))), int(max(0, np.floor(q[:, 1].min() - m)))
            x1, y1 = int(min(W, np.ceil(q[:, 0].max() + m))), int(min(H, np.ceil(q[:, 1].max() + m)))
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            name = f"{split_name}/{ri:06d}_{wi:03d}.jpg"
            if not cv2.imwrite(str(out_dir / name), img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95]):
                continue
            quad = (q - np.array([x0, y0], np.float32)).reshape(-1)
            rows.append([name, label, w["lang"], rec["key"]] + [f"{v:.1f}" for v in quad])
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_path", "text", "lang", "source_key", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"])
        writer.writerows(rows)
    marker.write_text("ok")
    print(f"{split_name}: {len(rows)} word crops -> {csv_path}")
    return csv_path


def read_word_crops(csv_path):
    csv_path = Path(csv_path)
    rows = []
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            quad = np.array([float(row[k]) for k in ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4")],
                            np.float32).reshape(4, 2)
            rows.append({"path": str(csv_path.parent / row["image_path"]), "text": row["text"],
                         "lang": row["lang"], "quad": quad})
    return rows


def pick_video_images(records, n, care_langs=BSTD_TARGET_LANGS, seed=0, min_words=2):
    """Test images with several readable Hindi/English words, for the pseudo road video."""
    scored = []
    for r in records:
        care = [w for w in r["words"] if not w["illegible"] and w["lang"] in care_langs]
        if len(care) >= min_words:
            scored.append((np.median([poly_text_height(w["poly"]) for w in care]), r))
    scored.sort(key=lambda t: -t[0])
    pool = [r for _, r in scored[:max(n * 3, n)]]
    rng = np.random.default_rng(seed)
    pick = [pool[i] for i in sorted(rng.permutation(len(pool))[:n])] if pool else []
    return pick
