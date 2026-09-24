# ===== CELL 20 (9.7): Road recognizer (CRNN + CTC) and lexicon =====
ROAD_REC_ARCH = "road_crnn_v2"


@dataclass
class RoadRecConfig:
    rec_h: int = 48
    rec_w: int = 320
    max_len: int = 32
    batch_size: int = 64
    epochs: int = 20
    steps_per_epoch: int = 1000
    lr: float = 1e-3
    visual_order: bool = True
    hindi: bool = True
    seed: int = 11

    def validate(self):
        if self.rec_h % 16 or self.rec_w % 4:
            raise ValueError("rec_h must be a multiple of 16 and rec_w a multiple of 4.")
        if self.rec_w // 4 < 2 * self.max_len - 1:
            raise ValueError("rec_w/4 must be >= 2*max_len-1 for CTC.")


def build_road_recognizer(cfg):
    inp = keras.Input((cfg.rec_h, cfg.rec_w, 1), name="image")

    def cbr(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        return layers.ReLU()(x)

    x = layers.MaxPooling2D((2, 2))(cbr(inp, 32))   # few channels at full resolution keeps memory low
    x = layers.MaxPooling2D((2, 2))(cbr(x, 64))
    x = layers.MaxPooling2D((2, 1))(cbr(cbr(x, 128), 128))
    x = layers.MaxPooling2D((2, 1))(cbr(cbr(x, 256), 256))
    x = layers.Conv2D(320, (cfg.rec_h // 16, 1), padding="valid", use_bias=False)(x)  # collapse height to 1
    x = layers.ReLU()(layers.BatchNormalization()(x))
    x = layers.Reshape((cfg.rec_w // 4, 320))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(len(CHARS) + 1, dtype="float32", name="logits")(x)
    return keras.Model(inp, out, name="road_crnn_scratch")


@keras.saving.register_keras_serializable(package="road_ocr", name="RoadCTCLoss")
class RoadCTCLoss(keras.losses.Loss):
    """CTC with sparse labels (0 = padding/blank). Same value as make_ctc_loss, but uses TF's CTC kernel
    instead of the dense while-loop, so it is faster and does not clash with section 4's traced loss."""

    def __init__(self, name="road_ctc_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        labels = tf.sparse.from_dense(tf.cast(y_true, tf.int32))
        logits = tf.cast(y_pred, tf.float32)
        steps = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1])
        return tf.nn.ctc_loss(labels=labels, logits=logits, label_length=None, logit_length=steps,
                              logits_time_major=False, blank_index=BLANK)


def _encode_padded(text, cfg):
    ids = encode_label(text, cfg)
    return np.array(ids + [BLANK] * (cfg.max_len - len(ids)), np.int32)


def _to_uint8(x):
    return np.clip(np.round(x[..., 0] * 255.0), 0, 255).astype(np.uint8)


def make_synthetic_rec_pool(n, cfg, fonts, seed=0, plain_share=0.12):
    """n preprocessed sign-style crops (uint8, rec_h x rec_w) + padded labels."""
    synth = RoadSceneSynth(fonts)
    old_cfg = OCRConfig(max_len=cfg.max_len, hindi=cfg.hindi, visual_order=cfg.visual_order)
    X = np.zeros((n, cfg.rec_h, cfg.rec_w), np.uint8)
    Y = np.zeros((n, cfg.max_len), np.int32)
    texts, i, k = [], 0, 0
    started = time.perf_counter()
    while i < n:
        rng = np.random.default_rng([seed, k])
        k += 1
        if rng.random() < plain_share:  # the notebook's original flat renderer, for extra font variety
            text = random_text(rng, old_cfg)
            try:
                samples = [(render_text_image(text, rng, fonts), text)]
            except RuntimeError:
                samples = []
        else:
            samples = synth.word_crops(rng)
        for crop, text in samples:
            label = road_clean_label(text)
            if label is None or len(label) > cfg.max_len or crop is None or min(crop.shape[:2]) < 2:
                continue
            X[i] = _to_uint8(preprocess_crop(crop, cfg))
            Y[i] = _encode_padded(label, cfg)
            texts.append(label)
            i += 1
            if i >= n:
                break
        if k % 5000 == 0:
            print(f"  synthetic crops {i}/{n} ({i / (time.perf_counter() - started):.0f}/s)")
    print(f"Synthetic recognizer pool: {n} crops in {time.perf_counter() - started:.0f}s")
    return X, Y, texts


def usable_crop_rows(rows, cfg):
    out = []
    for r in rows:
        label = road_clean_label(r["text"])
        if label and len(label) <= cfg.max_len:
            out.append({**r, "text": label})
    return out


def _real_crop(row, rng, augment):
    img = cv2.imread(row["path"], cv2.IMREAD_COLOR)
    if img is None:
        return None
    if augment:
        crop = crop_quad(img, jitter_quad(row["quad"], rng, 0.08), float(rng.uniform(0.0, 0.15)))
        if crop is not None and rng.random() < 0.7:
            crop = augment_crop(crop, rng)
    else:
        crop = crop_quad(img, row["quad"], 0.08)
    return crop


def real_rec_arrays(rows, cfg, seed=0):
    """Un-augmented real crops -> (X uint8, Y labels, texts, langs)."""
    X, Y, texts, langs = [], [], [], []
    for r in rows:
        crop = _real_crop(r, np.random.default_rng(seed), augment=False)
        if crop is None or min(crop.shape[:2]) < 2:
            continue
        X.append(_to_uint8(preprocess_crop(crop, cfg)))
        Y.append(_encode_padded(r["text"], cfg))
        texts.append(r["text"])
        langs.append(r["lang"])
    if not X:
        return np.zeros((0, cfg.rec_h, cfg.rec_w), np.uint8), np.zeros((0, cfg.max_len), np.int32), [], []
    return np.stack(X), np.stack(Y), texts, langs


def make_rec_dataset(cfg, synth_X, synth_Y, real_rows=(), real_share=0.6, seed=0):
    H, W, L = cfg.rec_h, cfg.rec_w, cfg.max_len
    parts, weights = [], []
    if len(synth_X):
        n = len(synth_X)

        def get_synth(i):
            i = int(i)
            return synth_X[i], synth_Y[i]

        ds = tf.data.Dataset.range(n).shuffle(min(n, 200000), seed=seed, reshuffle_each_iteration=True).repeat()
        parts.append(ds.map(lambda i: tf.numpy_function(get_synth, [i], [tf.uint8, tf.int32]),
                            num_parallel_calls=tf.data.AUTOTUNE))
        weights.append(1.0 - real_share if real_rows else 1.0)
    if real_rows:
        rows = list(real_rows)
        labels = np.stack([_encode_padded(r["text"], cfg) for r in rows])

        def get_real(counter):
            counter = int(counter)
            for attempt in range(5):
                rng = np.random.default_rng([seed, counter, attempt])
                j = int(rng.integers(len(rows)))
                crop = _real_crop(rows[j], rng, augment=True)
                if crop is not None and min(crop.shape[:2]) >= 2:
                    return _to_uint8(preprocess_crop(crop, cfg)), labels[j]
            return np.zeros((H, W), np.uint8), np.zeros((L,), np.int32)

        parts.append(tf.data.Dataset.counter().map(lambda c: tf.numpy_function(get_real, [c], [tf.uint8, tf.int32]),
                                                   num_parallel_calls=tf.data.AUTOTUNE, deterministic=False))
        weights.append(real_share if len(synth_X) else 1.0)
    if not parts:
        raise ValueError("No recognizer training data.")
    ds = parts[0] if len(parts) == 1 else tf.data.Dataset.sample_from_datasets(parts, weights=weights, seed=seed)
    ds = ds.map(lambda x, y: (tf.cast(tf.ensure_shape(x, (H, W)), tf.float32)[..., None] / 255.0,
                              tf.ensure_shape(y, (L,))), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(cfg.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


class RoadRecognizer:
    """Same call contract as ScratchRecognizer: list of BGR crops -> [(text, score)]."""

    def __init__(self, model, cfg, batch_size=32):
        self.model, self.cfg, self.batch_size = model, cfg, batch_size

    def __call__(self, crops):
        results = []
        for start in range(0, len(crops), self.batch_size):
            batch = []
            for crop in crops[start:start + self.batch_size]:
                ok = crop is not None and getattr(crop, "size", 0) and min(crop.shape[:2]) >= 2
                batch.append(preprocess_crop(crop, self.cfg) if ok else
                             np.zeros((self.cfg.rec_h, self.cfg.rec_w, 1), np.float32))
            logits = np.asarray(self.model(np.stack(batch), training=False))
            results.extend(decode_logits(logits, self.cfg.visual_order))
        return results

    def save(self, out_dir, extra=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(out_dir / "recognizer.weights.h5"))
        meta = {"type": "road_recognizer", "arch": ROAD_REC_ARCH, "config": asdict(self.cfg), "chars": CHARS,
                "preprocess_version": PREPROCESS_VERSION, "initialization": "random",
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), **(extra or {})}
        (out_dir / "recognizer.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, out_dir):
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / "recognizer.json").read_text(encoding="utf-8"))
        if meta.get("initialization") != "random":
            raise ValueError("This recognizer was not trained from random initialisation; refusing to load it.")
        if meta.get("arch") != ROAD_REC_ARCH:
            raise ValueError(f"Recognizer in {out_dir} has an older architecture; train it again (retrain=True).")
        if meta.get("chars") != CHARS or meta.get("preprocess_version") != PREPROCESS_VERSION:
            raise ValueError("Recognizer checkpoint does not match this notebook's charset/preprocessing.")
        cfg = RoadRecConfig(**{k: v for k, v in meta["config"].items() if k in RoadRecConfig.__dataclass_fields__})
        model = build_road_recognizer(cfg)
        model.load_weights(str(out_dir / "recognizer.weights.h5"))
        return cls(model, cfg)


def recognition_scores(recognizer, X, texts, lexicon=None, langs=None):
    """Word accuracy (exact and case/punctuation-insensitive), CER; optionally per language."""
    if not len(X):
        return {}
    preds = []
    for start in range(0, len(X), 64):
        batch = X[start:start + 64].astype(np.float32)[..., None] / 255.0
        preds.extend(decode_logits(np.asarray(recognizer.model(batch, training=False)), recognizer.cfg.visual_order))
    groups = defaultdict(lambda: Counter())
    cer = defaultdict(list)
    for i, ((text, score), truth) in enumerate(zip(preds, texts)):
        fixed = lexicon.correct(text, score)[0] if lexicon else text
        for g in ("all", langs[i] if langs else None):
            if g is None:
                continue
            groups[g]["n"] += 1
            groups[g]["exact"] += text == truth
            groups[g]["match"] += road_match_key(text) == road_match_key(truth)
            groups[g]["match_lex"] += road_match_key(fixed) == road_match_key(truth)
            cer[g].append(character_error_rate(truth, text))
    return {g: {"n": c["n"], "exact": c["exact"] / c["n"], "word_acc": c["match"] / c["n"],
                "word_acc_lexicon": c["match_lex"] / c["n"], "cer": float(np.mean(cer[g]))} for g, c in groups.items()}


def train_road_recognizer(cfg, synth_X, synth_Y, val_X, val_Y, out_dir, real_rows=(), real_share=0.6,
                          init_from=None, retrain=False, tag=""):
    cfg.validate()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backup, best_path, log_path = (out_dir / "recognizer_backup", out_dir / "recognizer_best.weights.h5",
                                   out_dir / "recognizer_log.csv")
    if (out_dir / "recognizer.json").exists() and not retrain:
        try:
            recognizer = RoadRecognizer.load(out_dir)
            print("A trained recognizer already exists in", out_dir, "- loaded it (retrain=True trains again).")
            return recognizer
        except ValueError as exc:
            print(exc, "-> training a new one.")
            retrain = True
    if retrain:
        shutil.rmtree(backup, ignore_errors=True)
        for p in (best_path, log_path, out_dir / "recognizer.json"):
            p.unlink(missing_ok=True)
    keras.utils.set_random_seed(cfg.seed)
    model = build_road_recognizer(cfg)
    if init_from and (Path(init_from) / "recognizer.json").exists():
        try:
            source = RoadRecognizer.load(init_from)
            if (source.cfg.rec_h, source.cfg.rec_w, source.cfg.max_len) == (cfg.rec_h, cfg.rec_w, cfg.max_len):
                model.set_weights(source.model.get_weights())
                print("Initialised from our own earlier checkpoint:", init_from)
        except ValueError as exc:
            print("Not using", init_from, "for initialisation:", exc)
    model.compile(optimizer=keras.optimizers.Adam(cfg.lr, clipnorm=5.0), loss=RoadCTCLoss(), jit_compile=False)
    print(f"Recognizer {tag}: {model.count_params():,} parameters, random init"
          + (" (then our phase-1 weights)" if init_from else ""))
    val_ds = tf.data.Dataset.from_tensor_slices((val_X, val_Y)).map(
        lambda x, y: (tf.cast(x, tf.float32)[..., None] / 255.0, y)).batch(cfg.batch_size)
    callbacks = [
        keras.callbacks.BackupAndRestore(str(backup)),
        keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_loss", save_best_only=True,
                                        save_weights_only=True, initial_value_threshold=_best_logged(log_path)),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6),
        keras.callbacks.CSVLogger(str(log_path), append=True),
    ]
    started = time.perf_counter()
    model.fit(make_rec_dataset(cfg, synth_X, synth_Y, real_rows, real_share, seed=cfg.seed),
              steps_per_epoch=cfg.steps_per_epoch, epochs=cfg.epochs, validation_data=val_ds,
              callbacks=callbacks, verbose=2)
    if best_path.exists():
        model.load_weights(str(best_path))
    refresh_bn_statistics(model, make_rec_dataset(cfg, synth_X, synth_Y, real_rows, real_share, seed=cfg.seed + 1))
    print(f"Recognizer training finished in {(time.perf_counter() - started) / 60:.1f} min")
    recognizer = RoadRecognizer(model, cfg)
    recognizer.save(out_dir, {"tag": tag, "init_from": str(init_from) if init_from else None,
                              "real_crops": len(real_rows), "synthetic_crops": int(len(synth_X))})
    return recognizer


class RoadLexicon:
    """Nearest-word correction for LOW-confidence readings, from a plain word list (e.g. training labels)."""

    def __init__(self, words, max_score=0.9):
        self.max_score = max_score
        forms, counts = {}, Counter()
        for word in words:
            label = road_clean_label(word)
            if not label:
                continue
            for token in [label] + label.split():
                key = road_match_key(token)
                if len(key) >= 3 and not any(ch.isdigit() for ch in key):
                    counts[key] += 1
                    forms.setdefault(key, token)
        self.forms = forms
        self.keys = sorted(forms, key=lambda k: -counts[k])
        self._by_len = defaultdict(list)
        for key in self.keys:
            self._by_len[len(key)].append(key)
        try:
            from rapidfuzz import process
            from rapidfuzz.distance import Levenshtein
            self._extract = lambda q, limit: process.extractOne(q, self.keys, scorer=Levenshtein.distance,
                                                                score_cutoff=limit)
        except ImportError:
            self._extract = None

    def __len__(self):
        return len(self.keys)

    def save(self, path):
        Path(path).write_text("\n".join(self.forms[k] for k in self.keys), encoding="utf-8")

    def _nearest(self, key, limit):
        if self._extract is not None:
            hit = self._extract(key, limit)
            return hit[0] if hit else None
        best, best_d = None, limit + 1
        for n in range(len(key) - limit, len(key) + limit + 1):
            for cand in self._by_len.get(n, ()):
                d = character_error_rate(key, cand) * max(1, len(key))
                if d < best_d:
                    best, best_d = cand, d
        return best if best_d <= limit else None

    def correct(self, text, score):
        """(text, changed). Keeps confident or in-vocabulary readings and anything with digits."""
        key = road_match_key(text)
        if (not self.keys or score >= self.max_score or len(key) < 3 or key in self.forms
                or any(ch.isdigit() for ch in key)):
            return text, False
        limit = 1 if len(key) <= 5 else 2 if len(key) <= 10 else 3
        best = self._nearest(key, limit)
        return (self.forms[best], True) if best else (text, False)
