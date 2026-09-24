# ===== CELL 19 (9.6): Learned text detector (from scratch) =====
ROAD_DET_ARCH = "db_lite_v1"


@dataclass
class DetConfig:
    train_size: int = 640        # square training crops (must be a multiple of 32)
    infer_side: int = 1280       # longer image side used at inference
    batch_size: int = 8
    epochs: int = 30
    steps_per_epoch: int = 500
    lr: float = 1e-3
    width: float = 1.0           # channel multiplier
    shrink_ratio: float = 0.6    # text kernel = polygon shrunk by DB's area/perimeter rule
    min_text_px: float = 6.0     # smaller words are neither text nor background during training
    n_val: int = 96
    mixed_precision: bool = False
    seed: int = 7

    def validate(self):
        if self.train_size % 32 or self.train_size < 128:
            raise ValueError("train_size must be a multiple of 32 and >= 128.")
        if not 0.2 <= self.shrink_ratio < 1:
            raise ValueError("shrink_ratio must be in [0.2, 1).")


def _conv_bn(x, filters, kernel=3, strides=1, act=True):
    x = layers.Conv2D(filters, kernel, strides=strides, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x) if act else x


def _res_block(x, filters, strides=1):
    shortcut = x
    y = _conv_bn(x, filters, 3, strides)
    y = _conv_bn(y, filters, 3, 1, act=False)
    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = _conv_bn(shortcut, filters, 1, strides, act=False)
    return layers.ReLU()(layers.Add()([y, shortcut]))


def build_text_detector(cfg):
    """Residual backbone + FPN -> 2 maps at 1/2 resolution: [text kernel, full text region] logits."""
    c = lambda n: max(8, int(round(n * cfg.width)))
    inp = keras.Input((None, None, 3), name="image_bgr_0_255")
    x = layers.Rescaling(1 / 127.5, offset=-1.0)(inp)
    x = _conv_bn(x, c(32), 3, 2)
    x = _conv_bn(x, c(32))
    c2 = _res_block(_res_block(x, c(48), 2), c(48))                       # 1/4
    c3 = _res_block(_res_block(c2, c(96), 2), c(96))                      # 1/8
    c4 = _res_block(_res_block(_res_block(c3, c(160), 2), c(160)), c(160))  # 1/16
    c5 = _res_block(_res_block(c4, c(256), 2), c(256))                    # 1/32
    f = c(96)
    p5 = _conv_bn(c5, f, 1)
    p4 = layers.Add()([_conv_bn(c4, f, 1), layers.UpSampling2D(2)(p5)])
    p3 = layers.Add()([_conv_bn(c3, f, 1), layers.UpSampling2D(2)(p4)])
    p2 = layers.Add()([_conv_bn(c2, f, 1), layers.UpSampling2D(2)(p3)])
    g = c(48)
    fused = layers.Concatenate()([_conv_bn(p2, g), layers.UpSampling2D(2)(_conv_bn(p3, g)),
                                  layers.UpSampling2D(4)(_conv_bn(p4, g)), layers.UpSampling2D(8)(_conv_bn(p5, g))])
    y = _conv_bn(fused, c(96))
    y = layers.UpSampling2D(2, interpolation="bilinear")(y)              # 1/2
    y = _conv_bn(y, c(48))
    y = layers.Conv2D(2, 1, name="map_logits")(y)
    out = layers.Activation("linear", dtype="float32", name="maps")(y)
    return keras.Model(inp, out, name="road_text_detector_scratch")


def make_det_target(polys, ignore, size_hw, cfg):
    """(H/2, W/2, 3): shrunk text kernel, full text region, loss weight (0 = don't care)."""
    h2, w2 = size_hw[0] // 2, size_hw[1] // 2
    kernel = np.zeros((h2, w2), np.float32)
    region = np.zeros((h2, w2), np.float32)
    weight = np.ones((h2, w2), np.float32)
    for poly, ign in zip(polys, ignore):
        p = np.asarray(poly, np.float32).reshape(-1, 2)
        if len(p) < 3:
            continue
        pts = np.round((p / 2.0 - 0.5) * 4).astype(np.int32)  # map coords, 2 fractional bits
        if ign or poly_text_height(p) < cfg.min_text_px:
            cv2.fillPoly(weight, [pts], 0.0, lineType=cv2.LINE_8, shift=2)
            continue
        hull = _convex(p)
        area = float(cv2.contourArea(hull))
        perimeter = float(cv2.arcLength(hull.reshape(-1, 1, 2), True))
        shrunk = offset_convex(hull, area * (1 - cfg.shrink_ratio ** 2) / max(perimeter, 1e-6))
        if shrunk is None:
            cv2.fillPoly(weight, [pts], 0.0, lineType=cv2.LINE_8, shift=2)
            continue
        cv2.fillPoly(region, [pts], 1.0, lineType=cv2.LINE_8, shift=2)
        spts = np.round((shrunk / 2.0 - 0.5) * 4).astype(np.int32)
        cv2.fillPoly(kernel, [spts], 1.0, lineType=cv2.LINE_8, shift=2)
    return np.stack([kernel, region, weight], axis=-1)


def det_random_view(img, polys, ignore, rng, cfg):
    """Random scale (around the inference scale), text-biased crop, small rotation. Returns view + polygons."""
    S = cfg.train_size
    H0, W0 = img.shape[:2]
    s = cfg.infer_side / max(H0, W0) * float(np.exp(rng.uniform(np.log(0.5), np.log(1.8))))
    s = min(s, 4.0)
    nw, nh = max(1, int(round(W0 * s))), max(1, int(round(H0 * s)))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    P = [np.asarray(p, np.float32).reshape(-1, 2) * np.array([nw / W0, nh / H0], np.float32) for p in polys]
    if nw > S or nh > S:
        if P and rng.random() < 0.85:
            centre = P[int(rng.integers(len(P)))].mean(axis=0)
            x0 = centre[0] - rng.uniform(0.15, 0.85) * S
            y0 = centre[1] - rng.uniform(0.15, 0.85) * S
        else:
            x0, y0 = rng.uniform(0, max(0, nw - S)), rng.uniform(0, max(0, nh - S))
        x0 = float(np.clip(x0, 0, max(0, nw - S)))
        y0 = float(np.clip(y0, 0, max(0, nh - S)))
    else:
        x0 = -rng.uniform(0, S - nw) if nw < S else 0.0
        y0 = -rng.uniform(0, S - nh) if nh < S else 0.0
    angle = float(rng.uniform(-10, 10)) if rng.random() < 0.3 else 0.0
    M = cv2.getRotationMatrix2D((S / 2, S / 2), angle, 1.0)
    M[:, 2] += M[:, :2] @ np.array([-x0, -y0])
    border = tuple(int(v) for v in rng.integers(0, 256, 3))
    view = cv2.warpAffine(im, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    frame = np.array([[0, 0], [S, 0], [S, S], [0, S]], np.float32)
    out_polys, out_ignore = [], []
    for p, ign in zip(P, ignore):
        q = (p @ M[:, :2].T + M[:, 2]).astype(np.float32)
        if q[:, 0].max() < 0 or q[:, 1].max() < 0 or q[:, 0].min() > S or q[:, 1].min() > S:
            continue
        hull = _convex(q)
        area = float(cv2.contourArea(hull))
        if area < 1 or len(hull) < 3:
            continue
        inter, clipped = cv2.intersectConvexConvex(hull, frame)
        if inter <= 0 or clipped is None:
            continue
        if inter / area < 0.999:
            ign = ign or inter / area < 0.7
            q = clipped.reshape(-1, 2).astype(np.float32)
        out_polys.append(q)
        out_ignore.append(bool(ign))
    return view, out_polys, out_ignore


@keras.saving.register_keras_serializable(package="road_ocr", name="DBLiteLoss")
class DBLiteLoss(keras.losses.Loss):
    """Balanced BCE with 3:1 hard-negative mining + Dice, on kernel and region maps."""

    def __init__(self, neg_ratio=3.0, name="db_lite_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.neg_ratio = float(neg_ratio)

    def _balanced_bce(self, logits, gt, weight):
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt, logits=logits)
        pos = gt * weight
        neg = (1.0 - gt) * weight
        n_pos = tf.reduce_sum(pos)
        n_neg = tf.minimum(tf.reduce_sum(neg), tf.maximum(n_pos * self.neg_ratio, 256.0))
        k = tf.cast(n_neg, tf.int32)
        hardest = tf.math.top_k(tf.reshape(bce * neg, [-1]), k=k).values
        return (tf.reduce_sum(bce * pos) + tf.reduce_sum(hardest)) / (n_pos + tf.cast(k, tf.float32) + 1e-6)

    @staticmethod
    def _dice(logits, gt, weight):
        prob = tf.sigmoid(logits)
        inter = tf.reduce_sum(prob * gt * weight)
        union = tf.reduce_sum(prob * weight) + tf.reduce_sum(gt * weight) + 1e-6
        return 1.0 - 2.0 * inter / union

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        kernel_gt, region_gt, weight = y_true[..., 0], y_true[..., 1], y_true[..., 2]
        kernel = self._balanced_bce(y_pred[..., 0], kernel_gt, weight) + self._dice(y_pred[..., 0], kernel_gt, weight)
        region = self._balanced_bce(y_pred[..., 1], region_gt, weight) + self._dice(y_pred[..., 1], region_gt, weight)
        return kernel + 0.5 * region

    def get_config(self):
        return {**super().get_config(), "neg_ratio": self.neg_ratio}


def _det_sample(rec, rng, cfg, augment=True):
    img = cv2.imread(rec["path"], cv2.IMREAD_COLOR)
    if img is None:
        return None
    view, polys, ignore = det_random_view(img, [w["poly"] for w in rec["words"]],
                                          [w["illegible"] for w in rec["words"]], rng, cfg)
    if augment:
        view = photometric_aug(view, rng, 0.8)
    return view, make_det_target(polys, ignore, view.shape[:2], cfg)


def make_det_dataset(sources, weights, cfg, seed=0):
    """Infinite tf.data pipeline; sources = list of record lists, weights = sampling share of each."""
    pairs = [(s, float(w)) for s, w in zip(sources, weights) if s and w > 0]
    if not pairs:
        raise ValueError("No detector training images.")
    sources = [s for s, _ in pairs]
    probs = np.array([w for _, w in pairs], np.float64)
    probs /= probs.sum()
    S, S2 = cfg.train_size, cfg.train_size // 2

    def load(counter):
        counter = int(counter)
        for attempt in range(8):
            rng = np.random.default_rng([seed, counter, attempt])
            src = sources[int(rng.choice(len(sources), p=probs))]
            sample = _det_sample(src[int(rng.integers(len(src)))], rng, cfg)
            if sample is not None:
                return sample
        blank = np.zeros((S2, S2, 3), np.float32)
        blank[..., 2] = 1.0
        return np.zeros((S, S, 3), np.uint8), blank

    ds = tf.data.Dataset.counter().map(lambda c: tf.numpy_function(load, [c], [tf.uint8, tf.float32]),
                                       num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    ds = ds.map(lambda x, y: (tf.cast(tf.ensure_shape(x, (S, S, 3)), tf.float32), tf.ensure_shape(y, (S2, S2, 3))),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(cfg.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def make_det_val_dataset(records, cfg, seed=999):
    xs, ys = [], []
    for i in range(max(cfg.n_val, cfg.batch_size)):
        if not records:
            break
        sample = _det_sample(records[i % len(records)], np.random.default_rng([seed, i]), cfg, augment=False)
        if sample is not None:
            xs.append(sample[0])
            ys.append(sample[1])
    if not xs:
        raise ValueError("No readable validation images for the detector.")
    ds = tf.data.Dataset.from_tensor_slices((np.stack(xs), np.stack(ys)))
    return ds.map(lambda x, y: (tf.cast(x, tf.float32), y)).batch(cfg.batch_size)


def refresh_bn_statistics(model, dataset, n_batches=40):
    """'Precise BN': replace BatchNorm moving statistics by the exact average over n training batches.
    Without this, short trainings keep part of the initial statistics and inference-mode outputs differ
    from training-mode outputs (tested: 0/16 -> 16/16 correct reads on a memorisation check)."""
    bns = [l for l in model.layers if isinstance(l, layers.BatchNormalization)]
    if not bns:
        return
    old = [l.momentum for l in bns]
    try:
        for k, batch in enumerate(dataset.take(n_batches), 1):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            for l in bns:
                l.momentum = (k - 1) / k  # running mean over the k batches seen so far
            model(x, training=True)
    finally:
        for l, m in zip(bns, old):
            l.momentum = m


def _best_logged(log_path, key="val_loss"):
    try:
        with open(log_path, newline="") as fh:
            values = [float(r[key]) for r in csv.DictReader(fh) if r.get(key) not in (None, "", "nan")]
        return min(values) if values else None
    except (OSError, KeyError, ValueError):
        return None


class TextDetector:
    """Callable: BGR frame -> [{'quad': (4,2) float32, 'score': float}] in reading order."""

    def __init__(self, model, cfg, post=None):
        self.model, self.cfg = model, cfg
        r2 = cfg.shrink_ratio ** 2
        self.post = {"bin_thresh": 0.3, "box_thresh": 0.5, "unclip": round((1 - r2) / r2, 2),
                     "min_side": 2.0, "infer_side": cfg.infer_side, "max_boxes": 300}
        if post:
            self.post.update(post)

    def prob_maps(self, frame, infer_side=None):
        side = int(infer_side or self.post["infer_side"])
        H, W = frame.shape[:2]
        scale = min(2.0, side / max(H, W))
        nh, nw = max(32, int(round(H * scale))), max(32, int(round(W * scale)))
        ph, pw = -(-nh // 32) * 32, -(-nw // 32) * 32
        canvas = np.zeros((1, ph, pw, 3), np.float32)
        canvas[0, :nh, :nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        logits = np.asarray(self.model(canvas, training=False), np.float32)[0]
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        return probs[:-(-nh // 2), :-(-nw // 2)], (nw / W, nh / H)

    @staticmethod
    def candidates(kernel, bin_thresh, min_side=2.0, max_boxes=300):
        bitmap = (kernel > bin_thresh).astype(np.uint8)
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for cnt in contours:
            (cx, cy), (w, h), angle = cv2.minAreaRect(cnt)
            w, h = w + 1.0, h + 1.0
            if min(w, h) < min_side:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            mask = np.zeros((bh, bw), np.uint8)
            cv2.fillPoly(mask, [(cnt.reshape(-1, 2) - [x, y]).astype(np.int32)], 1)
            region = kernel[y:y + bh, x:x + bw]
            score = float(region[mask > 0].mean()) if mask.any() else 0.0
            out.append((((cx + 0.5, cy + 0.5), (w, h), angle), score))
        out.sort(key=lambda t: -t[1])
        return out[:max_boxes]

    @staticmethod
    def quads_from_candidates(cands, scales, frame_shape, box_thresh, unclip):
        sx, sy = scales
        H, W = frame_shape[:2]
        out = []
        for rect, score in cands:
            if score < box_thresh:
                continue
            quad = unclip_rect(rect, unclip) * 2.0 / np.array([sx, sy], np.float32)
            quad[:, 0] = np.clip(quad[:, 0], 0, W - 1)
            quad[:, 1] = np.clip(quad[:, 1], 0, H - 1)
            quad = order_quad(quad)
            if min(quad_size(quad)) >= 2:
                out.append({"quad": quad, "score": float(score)})
        if out:
            line = max(8.0, float(np.median([quad_size(d["quad"])[1] for d in out])))
            out.sort(key=lambda d: (round(float(d["quad"][:, 1].mean()) / line), float(d["quad"][:, 0].min())))
        return out

    def boxes_from_maps(self, probs, scales, frame_shape, bin_thresh=None, box_thresh=None, unclip=None):
        p = self.post
        cands = self.candidates(probs[..., 0], p["bin_thresh"] if bin_thresh is None else bin_thresh,
                                p["min_side"], p["max_boxes"])
        return self.quads_from_candidates(cands, scales, frame_shape,
                                          p["box_thresh"] if box_thresh is None else box_thresh,
                                          p["unclip"] if unclip is None else unclip)

    def __call__(self, frame):
        probs, scales = self.prob_maps(frame)
        return self.boxes_from_maps(probs, scales, frame.shape)

    def save(self, out_dir, extra=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(out_dir / "detector.weights.h5"))
        meta = {"type": "road_text_detector", "arch": ROAD_DET_ARCH, "config": asdict(self.cfg), "post": self.post,
                "initialization": "random", "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), **(extra or {})}
        (out_dir / "detector.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, out_dir):
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / "detector.json").read_text(encoding="utf-8"))
        if meta.get("initialization") != "random":
            raise ValueError("This detector was not trained from random initialisation; refusing to load it.")
        if meta.get("arch") != ROAD_DET_ARCH:
            raise ValueError(f"Detector in {out_dir} has an older architecture; train it again (retrain=True).")
        cfg = DetConfig(**{k: v for k, v in meta["config"].items() if k in DetConfig.__dataclass_fields__})
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("float32")
        try:
            model = build_text_detector(cfg)
        finally:
            keras.mixed_precision.set_global_policy(previous)
        model.load_weights(str(out_dir / "detector.weights.h5"))
        return cls(model, cfg, meta.get("post"))


def match_detections(words, pred_quads, care, iou_thr=0.5):
    """One-to-one matches of 'care' words; unmatched predictions lying on don't-care words are ignored."""
    care_idx = [i for i, c in enumerate(care) if c]
    dont_idx = [i for i, c in enumerate(care) if not c]
    raw = match_polys([words[i]["poly"] for i in care_idx], list(pred_quads), iou_thr)
    matches = [(care_idx[g], p, iou) for g, p, iou in raw]
    matched = {p for _, p, _ in matches}
    ignored = set()
    for pi, quad in enumerate(pred_quads):
        if pi in matched:
            continue
        box = poly_bbox(quad)
        for gi in dont_idx:
            gb = poly_bbox(words[gi]["poly"])
            if gb[0] > box[2] or box[0] > gb[2] or gb[1] > box[3] or box[1] > gb[3]:
                continue
            if poly_overlap(words[gi]["poly"], quad)[1] > 0.5:
                ignored.add(pi)
                break
    return matches, ignored


def det_care(word, langs=("hindi", "english", "digits")):
    return (not word["illegible"]) and word["lang"] in langs


def calibrate_detector(detector, records, care_fn=det_care, max_images=100, grid=None):
    """Pick bin/box thresholds and unclip ratio with the best F1 on held-out (never test) images."""
    grid = grid or {"bin_thresh": [0.25, 0.35, 0.45], "box_thresh": [0.45, 0.55, 0.65, 0.75],
                    "unclip": [0.9, 1.2, 1.5, 1.8, 2.1, 2.5]}
    cache = []
    for rec in records[:max_images]:
        img = cv2.imread(rec["path"], cv2.IMREAD_COLOR)
        if img is not None:
            probs, scales = detector.prob_maps(img)
            cache.append((probs[..., 0], scales, img.shape, rec["words"], [care_fn(w) for w in rec["words"]]))
    if not cache:
        print("Calibration skipped: no images.")
        return detector.post
    best = None
    for bt in grid["bin_thresh"]:
        cands = [TextDetector.candidates(k, bt, detector.post["min_side"], detector.post["max_boxes"])
                 for k, *_ in cache]
        for ur in grid["unclip"]:
            for st in grid["box_thresh"]:
                tp = fp = fn = 0
                for cand, (_, scales, shape, words, care) in zip(cands, cache):
                    preds = TextDetector.quads_from_candidates(cand, scales, shape, st, ur)
                    matches, ignored = match_detections(words, [d["quad"] for d in preds], care)
                    tp += len(matches)
                    fp += len(preds) - len(matches) - len(ignored)
                    fn += sum(care) - len(matches)
                f1 = 2 * tp / max(1, 2 * tp + fp + fn)
                key = (round(f1, 4), bt, st)  # on ties prefer stricter thresholds (fewer spurious boxes)
                if best is None or key > best[0]:
                    best = (key, {"bin_thresh": bt, "box_thresh": st, "unclip": ur}, tp, fp, fn)
    (f1, _, _), params, tp, fp, fn = best
    detector.post.update(params)
    print(f"Calibrated on {len(cache)} held-out images: {params} -> F1 {f1:.3f} "
          f"(P {tp / max(1, tp + fp):.3f}, R {tp / max(1, tp + fn):.3f})")
    return detector.post


def train_text_detector(sources, weights, val_records, cfg, out_dir, init_from=None, retrain=False, tag=""):
    """Train (or resume) a detector in out_dir. A finished detector is loaded instead unless retrain=True."""
    cfg.validate()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backup, best_path, log_path = out_dir / "detector_backup", out_dir / "detector_best.weights.h5", out_dir / "detector_log.csv"
    if (out_dir / "detector.json").exists() and not retrain:
        try:
            detector = TextDetector.load(out_dir)
            print("A trained detector already exists in", out_dir, "- loaded it (retrain=True trains again).")
            return detector
        except ValueError as exc:
            print(exc, "-> training a new one.")
            retrain = True
    if retrain:
        shutil.rmtree(backup, ignore_errors=True)
        for p in (best_path, log_path, out_dir / "detector.json"):
            p.unlink(missing_ok=True)
    keras.utils.set_random_seed(cfg.seed)
    previous = keras.mixed_precision.global_policy().name
    keras.mixed_precision.set_global_policy("mixed_float16" if cfg.mixed_precision else "float32")
    try:
        model = build_text_detector(cfg)
        if init_from and (Path(init_from) / "detector.json").exists():
            try:
                source = TextDetector.load(init_from)
                if source.cfg.width == cfg.width:
                    model.set_weights(source.model.get_weights())
                    print("Initialised from our own earlier checkpoint:", init_from)
                else:
                    print("init_from has a different width; starting from random weights.")
            except ValueError as exc:
                print("Not using", init_from, "for initialisation:", exc)
        model.compile(optimizer=keras.optimizers.Adam(cfg.lr, clipnorm=5.0), loss=DBLiteLoss(), jit_compile=False)
    finally:
        keras.mixed_precision.set_global_policy(previous)
    print(f"Detector {tag}: {model.count_params():,} parameters, all randomly initialised"
          + (" (then our phase-1 weights)" if init_from else ""))
    callbacks = [
        keras.callbacks.BackupAndRestore(str(backup)),
        keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_loss", save_best_only=True,
                                        save_weights_only=True, initial_value_threshold=_best_logged(log_path)),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8),
        keras.callbacks.CSVLogger(str(log_path), append=True),
    ]
    started = time.perf_counter()
    model.fit(make_det_dataset(sources, weights, cfg, seed=cfg.seed), steps_per_epoch=cfg.steps_per_epoch,
              epochs=cfg.epochs, validation_data=make_det_val_dataset(val_records, cfg), callbacks=callbacks, verbose=2)
    if best_path.exists():
        model.load_weights(str(best_path))
    refresh_bn_statistics(model, make_det_dataset(sources, weights, cfg, seed=cfg.seed + 1))
    print(f"Detector training finished in {(time.perf_counter() - started) / 60:.1f} min")
    previous = keras.mixed_precision.global_policy().name
    keras.mixed_precision.set_global_policy("float32")
    try:
        inference_model = build_text_detector(cfg)
    finally:
        keras.mixed_precision.set_global_policy(previous)
    inference_model.set_weights(model.get_weights())
    detector = TextDetector(inference_model, cfg)
    calibrate_detector(detector, val_records)
    detector.save(out_dir, {"tag": tag, "init_from": str(init_from) if init_from else None})
    del model
    gc.collect()
    return detector
