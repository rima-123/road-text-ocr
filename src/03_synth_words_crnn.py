# ===== CELL 4: Synthetic data, CRNN + CTC, training function =====
ENGLISH_WORDS = """STOP GO START EXIT SPEED SCHOOL DANGER PARKING LIMIT AHEAD ROAD CLOSED
RAILWAY CROSSING NO ENTRY HEAVY VEHICLES SLOW TURN LEFT RIGHT ZONE HOSPITAL BUS
STATION AIRPORT CITY CENTRE HIGHWAY TOLL PLAZA BRIDGE NARROW WORK DIVERSION DETOUR
ONE WAY KEEP CLEAR HORN PROHIBITED PEDESTRIAN MAXIMUM CAUTION WARNING EMERGENCY
POLICE FUEL HOTEL RESTAURANT MARKET GATE SAFETY FIRST ROOM OFFICE ACCESS BAGGAGE
COFFEE STREET GREEN YELLOW FOLLOW CROSS TUNNEL WEEKEND SMALL DEPARTMENT ENGINEERING""".split()
HINDI_WORDS = """विभाग संगणक विज्ञान अभियांत्रिकी कोलकाता पटना जयपुर लखनऊ मुंबई दिल्ली
चेन्नई गोरखपुर प्रयागराज वाराणसी कानपुर विश्वविद्यालय महाविद्यालय कार्यालय
अस्पताल पुलिस थाना बाजार सड़क मार्ग नगर ग्राम जिला राज्य भारत रेलवे स्टेशन हवाई
अड्डा प्रवेश निकास खतरा सावधान धीरे चलें आगे विद्यालय छात्रावास पुस्तकालय प्रयोगशाला
केंद्र शाखा कक्ष भवन द्वार उत्तर दक्षिण पूर्व पश्चिम गति सीमा वाहन प्रतीक्षा कृपया
धन्यवाद स्वागत शौचालय सूचना निषेध पार्किंग आपातकाल चिकित्सा मुख्य करें""".split()


def random_text(rng, config):
    for _ in range(100):
        if config.hindi and rng.random() < 0.45:
            text = " ".join(str(x) for x in rng.choice(HINDI_WORDS, size=int(rng.integers(1, 4))))
        elif rng.random() < 0.18:
            n = int(rng.integers(2, min(18, config.max_len) + 1))
            letters = list(rng.choice(list(string.ascii_letters + string.digits + "-./"), size=n))
            if n > 2 and rng.random() < 0.5:
                j = int(rng.integers(n - 1))
                letters[j + 1] = letters[j]  # CTC learns doubled letters, e.g. SCHOOL.
            text = "".join(letters)
        else:
            text = " ".join(str(x) for x in rng.choice(ENGLISH_WORDS, size=int(rng.integers(1, 5))))
            style = rng.random()
            text = text.lower() if style < 0.15 else text.title() if style < 0.4 else text
        if len(text) <= config.max_len:
            return text
    return "GO"


def augment_crop(bgr, rng):
    """Mild camera-like variation. This does not substitute for real training crops."""
    h, w = bgr.shape[:2]
    out = bgr.copy()
    if rng.random() < 0.45:
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        jitter = rng.uniform(-1, 1, (4, 2)).astype(np.float32) * [min(w * .02, h * .12), h * .09]
        dst = (src + jitter).astype(np.float32)
        out = cv2.warpPerspective(out, cv2.getPerspectiveTransform(src, dst), (w, h),
                                  borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.5:
        gradient = np.linspace(rng.uniform(.55, 1.1), rng.uniform(.65, 1.15), w)[None, :, None]
        out = np.clip(out.astype(np.float32) * gradient, 0, 255).astype(np.uint8)
    if rng.random() < .25:
        kernel = np.zeros((3, 3), np.float32)
        kernel[1, :] = 1 / 3
        out = cv2.filter2D(out, -1, kernel)
    elif rng.random() < .3:
        out = cv2.GaussianBlur(out, (3, 3), float(rng.uniform(.3, 1.0)))
    if rng.random() < .3:
        scale = float(rng.uniform(.6, .9))
        out = cv2.resize(cv2.resize(out, (max(2, round(w * scale)), max(2, round(h * scale)))), (w, h))
    out = np.clip(out.astype(np.float32) + rng.normal(0, rng.uniform(0, 5), out.shape), 0, 255).astype(np.uint8)
    if rng.random() < .25:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(45, 95))])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


def render_text_image(text, rng, fonts, augment=True, backgrounds=()):
    runs, (l, t, r, b) = text_geometry(text, fonts, int(rng.integers(26, 65)), rng)
    pad = int(rng.integers(3, 10))
    w, h = max(1, r - l) + 2 * pad, max(1, b - t) + 2 * pad
    light = bool(rng.random() > .25)
    bg = tuple(int(x) for x in rng.integers(190, 256, 3)) if light else tuple(int(x) for x in rng.integers(0, 60, 3))
    fg = tuple(int(x) for x in rng.integers(0, 65, 3)) if light else tuple(int(x) for x in rng.integers(195, 256, 3))
    image = Image.new("RGB", (w, h), bg)
    if backgrounds and rng.random() < .35:
        with Image.open(backgrounds[int(rng.integers(len(backgrounds)))]) as raw:
            texture = raw.convert("RGB").resize((w, h))
        image = Image.blend(image, texture, .25)  # Keep the target text legible.
    draw_text_runs(ImageDraw.Draw(image), runs, pad - l, pad - t, fg)
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return augment_crop(bgr, rng) if augment and rng.random() < .65 else bgr


def read_labelled_crops(csv_path, config):
    """Optional real crop CSV: image_path,text. Paths are relative to the CSV."""
    if not csv_path:
        return []
    path, records = Path(csv_path).resolve(), []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not {"image_path", "text"}.issubset(reader.fieldnames or []):
            raise ValueError("Real-crop CSV requires image_path,text columns.")
        for row in reader:
            text = row["text"].strip()
            encode_label(text, config)
            image_path = (path.parent / row["image_path"]).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            records.append((image_path, text))
    return records


def make_dataset(n, seed, config, fonts, real_crops=(), backgrounds=()):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, config.rec_h, config.rec_w, 1), dtype=np.uint8)
    y = np.zeros((n, config.max_len), dtype=np.int32)
    texts = []
    for i in range(n):
        if real_crops and rng.random() < .4:
            path, text = real_crops[int(rng.integers(len(real_crops)))]
            crop = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if crop is None:
                raise ValueError(f"Cannot decode training crop: {path}")
            crop = augment_crop(crop, rng)
        else:
            text = random_text(rng, config)
            crop = render_text_image(text, rng, fonts, backgrounds=backgrounds)
        encoded = encode_label(text, config)
        x[i] = np.rint(preprocess_crop(crop, config) * 255).astype(np.uint8)
        y[i, :len(encoded)] = encoded
        texts.append(text)
    return x, y, texts


def build_crnn(config):
    """Same CNN -> BiLSTM -> CTC design as v3, with a larger configurable width."""
    import keras
    from keras import layers
    config.validate()
    inp = keras.Input(shape=(config.rec_h, config.rec_w, 1), name="image")
    x = inp
    for filters, pool in [(32, (2, 2)), (64, (2, 2)), (96, (2, 1)), (128, (2, 1))]:
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(pool)(x)
    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((config.rec_w // 4, -1))(x)
    x = layers.Dense(96, activation="relu")(x)
    x = layers.Dropout(.25)(x)
    x = layers.Bidirectional(layers.LSTM(96, return_sequences=True))(x)
    out = layers.Dense(len(CHARS) + 1, name="logits")(x)
    return keras.Model(inp, out, name="crnn_ctc_from_scratch")


def make_ctc_loss():
    import keras
    from keras import ops

    @keras.saving.register_keras_serializable(package="scratch_ocr")
    class CTCLoss(keras.losses.Loss):
        def call(self, y_true, y_pred):
            target = ops.cast(y_true, "int32")
            lengths = ops.sum(ops.cast(target > 0, "int32"), axis=-1)
            output_lengths = ops.ones_like(lengths) * ops.shape(y_pred)[1]
            return ops.ctc_loss(target, y_pred, lengths, output_lengths, mask_index=BLANK)
    return CTCLoss()


def decode_logits(logits, visual_order=True):
    """Greedy CTC: collapse path repeats FIRST, then remove blanks.

    Score is mean per-character peak posterior, excluding blank runs. It is a
    heuristic, not a calibrated probability of a correctly transcribed line.
    """
    logits = np.asarray(logits, np.float32)
    if logits.ndim != 3 or logits.shape[-1] != len(CHARS) + 1 or not np.isfinite(logits).all():
        raise ValueError("Invalid logits or model/charset mismatch.")
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)
    results = []
    for sequence in probs:
        ids = sequence.argmax(axis=-1)
        chars, scores, start = [], [], 0
        while start < len(ids):
            end = start + 1
            while end < len(ids) and ids[end] == ids[start]:
                end += 1
            token = int(ids[start])
            if token != BLANK:
                chars.append(CHARS[token - 1])
                scores.append(float(sequence[start:end, token].max()))
            start = end
        text = "".join(chars)
        results.append((to_logical(text) if visual_order else text, float(np.mean(scores)) if scores else 0.0))
    return results


def character_error_rate(reference, predicted):
    previous = list(range(len(predicted) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(predicted, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1] / max(1, len(reference))


class ScratchRecognizer:
    def __init__(self, model, config):
        self.model, self.config = model, config
        if tuple(model.input_shape[1:]) != (config.rec_h, config.rec_w, 1):
            raise ValueError("Checkpoint input dimensions do not match its metadata.")
        if model.output_shape[-1] != len(CHARS) + 1:
            raise ValueError("Checkpoint output charset does not match this version.")

    def __call__(self, crops):
        if not crops:
            return []
        results = []
        for i in range(0, len(crops), self.config.batch_size):
            x = np.stack([preprocess_crop(c, self.config) for c in crops[i:i + self.config.batch_size]])
            logits = np.asarray(self.model(x, training=False))
            results.extend(decode_logits(logits, self.config.visual_order))
        return results

    @classmethod
    def load(cls, checkpoint):
        import keras
        path = Path(checkpoint)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if (meta.get("chars") != CHARS or meta.get("preprocess_version") != PREPROCESS_VERSION
                or meta.get("initialization") != "random" or meta.get("trained_epochs", 0) < 1):
            raise ValueError("Use a checkpoint + JSON produced by this version's own training cell.")
        config = OCRConfig(**meta["config"])
        config.validate()
        return cls(keras.models.load_model(path, compile=False), config)


def train_from_scratch(config=None, output_dir="ocr_models", font_dir=None,
                       real_train_csv=None, real_val_csv=None, backgrounds_dir=None):
    import tensorflow as tf
    import keras
    config = config or OCRConfig()
    config.validate()
    keras.utils.set_random_seed(config.seed)
    fonts = find_fonts(font_dir, config.hindi)
    real_train = read_labelled_crops(real_train_csv, config)
    real_val = read_labelled_crops(real_val_csv, config)
    if {p for p, _ in real_train} & {p for p, _ in real_val}:
        raise ValueError("Use disjoint real training and validation crop files.")
    backgrounds = sorted(p for p in Path(backgrounds_dir).rglob("*")
                         if p.suffix.lower() in (".png", ".jpg", ".jpeg")) if backgrounds_dir else []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Generating labelled crops; no video labels or downloaded weights are used.")
    x, y, _ = make_dataset(config.n_train, config.seed + 1, config, fonts, real_train, backgrounds)
    vx, vy, vt = make_dataset(config.n_val, config.seed + 2, config, fonts, real_val)
    def dataset(images, labels, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        if shuffle:
            ds = ds.shuffle(min(len(images), 2048), seed=config.seed)
        ds = ds.batch(config.batch_size).map(lambda a, b: (tf.cast(a, tf.float32) / 255., b))
        return ds.prefetch(1)
    model = build_crnn(config)  # Random initializers; never load a base model here.
    model.compile(optimizer=keras.optimizers.Adam(2e-3, clipnorm=5.0),
                  loss=make_ctc_loss(), jit_compile=False)
    history = model.fit(dataset(x, y, True), validation_data=dataset(vx, vy),
                        epochs=config.epochs, verbose=2,
                        callbacks=[keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=.5, patience=3),
                                   keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)])
    checkpoint = out_dir / "crnn_from_scratch.keras"
    model.save(checkpoint)
    metadata = {"config": asdict(config), "chars": CHARS, "preprocess_version": PREPROCESS_VERSION,
                "initialization": "random", "trained_epochs": len(history.history["loss"])}
    checkpoint.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    predicted = []
    for i in range(0, len(vx), config.batch_size):
        predicted.extend(t for t, _ in decode_logits(np.asarray(model(vx[i:i + config.batch_size].astype(np.float32) / 255., training=False)), config.visual_order))
    metrics = {"validation_samples": len(vt), "exact_match": float(np.mean([a == b for a, b in zip(vt, predicted)])),
               "mean_cer": float(np.mean([character_error_rate(a, b) for a, b in zip(vt, predicted)])),
               "scope": "Synthetic crops, plus independently supplied validation crops if configured; not a real-video benchmark."}
    (out_dir / "validation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("Saved", checkpoint, "and", checkpoint.with_suffix(".json"))
    return ScratchRecognizer(model, config), checkpoint
