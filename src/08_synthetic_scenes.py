# ===== CELL 17 (9.4): Synthetic road scenes + sign-style word crops =====
SIGN_KINDS = {  # kind: sampling weight
    "highway_green": 0.14, "highway_blue": 0.08, "white_board": 0.10, "yellow_warning": 0.06,
    "brown_tourist": 0.03, "black_board": 0.05, "red_board": 0.05, "shop_board": 0.20, "banner": 0.10,
    "wall_paint": 0.08, "milestone": 0.03, "number_plate": 0.04, "plain_text": 0.04,
}
_KIND_NAMES = list(SIGN_KINDS)
_KIND_P = np.array([SIGN_KINDS[k] for k in _KIND_NAMES], np.float64) / sum(SIGN_KINDS.values())
_FIXED_STYLE = {  # kind: (background RGB, text RGB, border RGB or None)
    "highway_green": ((0, 106, 58), (255, 255, 255), (255, 255, 255)),
    "highway_blue": ((0, 72, 152), (255, 255, 255), (255, 255, 255)),
    "yellow_warning": ((250, 196, 0), (15, 15, 15), (15, 15, 15)),
    "brown_tourist": ((112, 62, 26), (255, 255, 255), (255, 255, 255)),
    "black_board": ((18, 18, 18), (255, 205, 0), None),
    "red_board": ((190, 22, 26), (255, 255, 255), (255, 255, 255)),
    "milestone": ((245, 245, 240), (10, 10, 10), None),
    "number_plate": ((245, 245, 245), (10, 10, 10), (10, 10, 10)),
}


def _jitter_rgb(rng, rgb, amount=18):
    return tuple(int(np.clip(v + rng.integers(-amount, amount + 1), 0, 255)) for v in rgb)


def _luma(rgb):
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def _contrast_rgb(rng, bg, low_contrast=False):
    dark_bg = _luma(bg) < 128
    for _ in range(20):
        if dark_bg:
            fg = tuple(int(v) for v in rng.integers(170, 256, 3))
        else:
            fg = tuple(int(v) for v in rng.integers(0, 110, 3))
        if rng.random() < 0.3:  # saturated text colour
            fg = tuple(int(v) for v in (rng.permutation([int(rng.integers(150, 256)), int(rng.integers(0, 80)),
                                                          int(rng.integers(0, 256))])))
        gap = abs(_luma(fg) - _luma(bg))
        if gap >= (35 if low_contrast else 85):
            return fg
    return (255, 255, 255) if dark_bg else (0, 0, 0)


def _box_overlap(a, b):
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(smaller, 1e-6)


@lru_cache(maxsize=512)
def _road_font(path, size):
    return ImageFont.truetype(str(path), int(size))


def _text_runs(text, latin_path, deva_path, size, fonts):
    """Script-split runs [(segment, font, advance)], ink bounds and total advance; None if no font fits."""
    runs, advance = [], 0.0
    left = top = right = bottom = 0.0
    for segment in re.findall(r"[\u0900-\u097f]+|[^\u0900-\u097f]+", text):
        is_deva = bool(re.match(r"[\u0900-\u097f]", segment))
        path = deva_path if is_deva else latin_path
        if not font_supports(path, segment):
            pool = fonts["deva"] if is_deva else fonts["latin"]
            usable = [p for p in pool if font_supports(p, segment)]
            if not usable:
                return None
            path = usable[0]
        font = _road_font(path, size)
        l, t, r, b = font.getbbox(segment, anchor="ls")
        left, top = min(left, advance + l), min(top, t)
        right, bottom = max(right, advance + r), max(bottom, b)
        runs.append((segment, font, advance))
        advance += font.getlength(segment)
    if not runs or right - left < 1 or bottom - top < 1:
        return None
    return runs, (left, top, right, bottom), advance


def _draw_runs(draw, runs, x, baseline, fill, stroke=0, stroke_fill=None):
    for segment, font, advance in runs:
        draw.text((x + advance, baseline), segment, font=font, fill=fill, anchor="ls",
                  stroke_width=int(stroke), stroke_fill=stroke_fill)


def _noise_texture(rng, w, h, base_rgb, amount=25.0):
    """Smooth random texture as an RGB float array."""
    low = rng.normal(0, 1, (max(1, h // 24 + 1), max(1, w // 24 + 1), 3)).astype(np.float32)
    low = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC) * amount
    fine = rng.normal(0, amount * 0.25, (h, w, 3)).astype(np.float32)
    return np.clip(np.asarray(base_rgb, np.float32)[None, None, :] + low + fine, 0, 255)


class RoadSceneSynth:
    """Procedural roadside scenes with word-level quads. Everything is drawn; no images are downloaded."""

    def __init__(self, fonts, scene_sizes=((960, 540), (1024, 576), (1280, 720), (800, 600))):
        self.fonts = fonts
        self.scene_sizes = scene_sizes

    # ------------------------------------------------------------------ sign content
    def _lines(self, rng, kind):
        def words(lang, lo, hi):
            return [(road_word(rng, lang), lang) for _ in range(int(rng.integers(lo, hi + 1)))]

        def distance(hindi):
            km = int(rng.integers(1, 400))
            if hindi:
                num = _to_deva_digits(str(km)) if rng.random() < 0.4 else str(km)
                return [(num, "digits"), (str(rng.choice(["कि.मी.", "किमी"])), "hindi")]
            return [(f"{km}", "digits"), (str(rng.choice(["km", "KM", "Km"])), "english")]

        if kind in ("highway_green", "highway_blue", "brown_tourist"):
            lines = []
            for _ in range(int(rng.integers(1, 4))):
                hindi = rng.random() < 0.5
                line = [(str(rng.choice(ROAD_HI_PLACES if hindi else ROAD_EN_PLACES)), "hindi" if hindi else "english")]
                if not hindi and rng.random() < 0.3:
                    line[0] = (line[0][0].title(), "english")
                if rng.random() < 0.7:
                    line += distance(hindi)
                lines.append(line)
            return lines
        if kind == "milestone":
            hindi = rng.random() < 0.5
            return [[(str(rng.choice(ROAD_HI_PLACES if hindi else ROAD_EN_PLACES)), "hindi" if hindi else "english")],
                    [(str(int(rng.integers(0, 300))), "digits")]]
        if kind == "number_plate":
            state = str(rng.choice(["UP", "DL", "HR", "MP", "RJ", "BR", "UK", "PB"]))
            plate = [state, f"{int(rng.integers(1, 99)):02d}", "".join(rng.choice(list(string.ascii_uppercase), 2)),
                     f"{int(rng.integers(1, 9999)):04d}"]
            if rng.random() < 0.4:
                plate = ["".join(plate)]
            return [[(p, "english") for p in plate]]
        if kind == "shop_board":
            lang = "hindi" if rng.random() < 0.5 else "english"
            lines = [words(lang, 1, 3)]
            if rng.random() < 0.6:
                lang2 = "hindi" if rng.random() < 0.5 else "english"
                second = words(lang2, 1, 3)
                if rng.random() < 0.5:
                    second.append((random_number_text(rng), "digits"))
                lines.append(second)
            return lines
        if kind in ("yellow_warning", "white_board", "black_board", "red_board", "banner", "wall_paint"):
            lines = []
            for _ in range(int(rng.integers(1, 4))):
                lang = "hindi" if rng.random() < 0.5 else "english"
                line = words(lang, 1, 3)
                if rng.random() < 0.15:
                    line.append((random_number_text(rng, hindi=lang == "hindi"), "digits"))
                lines.append(line)
            return lines
        lang = "hindi" if rng.random() < 0.5 else ("english" if rng.random() < 0.8 else "digits")
        return [words(lang, 1, 4)]

    def _style(self, rng, kind):
        if kind in _FIXED_STYLE:
            bg, fg, border = _FIXED_STYLE[kind]
            bg = _jitter_rgb(rng, bg)
            if kind == "number_plate" and rng.random() < 0.3:
                bg = _jitter_rgb(rng, (250, 205, 0))
            return bg, _jitter_rgb(rng, fg, 12), border
        if kind == "white_board":
            bg = _jitter_rgb(rng, (240, 240, 236), 12)
            fg = tuple(int(v) for v in [(10, 10, 10), (20, 40, 150), (170, 20, 20)][int(rng.integers(3))])
            border = [None, (190, 20, 20), (20, 20, 20)][int(rng.integers(3))]
            return bg, fg, border
        bg = tuple(int(v) for v in rng.integers(0, 256, 3))
        if rng.random() < 0.5:  # vivid shop colours
            bg = tuple(int(v) for v in rng.permutation([int(rng.integers(160, 256)), int(rng.integers(0, 90)),
                                                        int(rng.integers(0, 256))]))
        fg = _contrast_rgb(rng, bg, low_contrast=rng.random() < 0.08)
        border = None if rng.random() < 0.6 else _contrast_rgb(rng, bg)
        return bg, fg, border

    def make_sign(self, rng, kind=None):
        """RGBA sign image and its words [(text, lang, [x1,y1,x2,y2], line_index)]."""
        kind = kind or str(rng.choice(_KIND_NAMES, p=_KIND_P))
        latin = str(rng.choice(self.fonts["latin"]))
        deva = str(rng.choice(self.fonts["deva"]))
        base = int(rng.integers(15, 37)) * 2  # quantised sizes keep the font cache small
        effects = kind in ("banner", "shop_board", "plain_text") and rng.random() < 0.4
        stroke = int(max(1, base // 18)) if effects and rng.random() < 0.6 else 0
        laid = []
        for li, line in enumerate(self._lines(rng, kind)):
            size = base if li == 0 else max(14, int(base * rng.uniform(0.5, 1.0)) // 2 * 2)
            placed = []
            for text, lang in line:
                if road_clean_label(text) is None or len(text) > 30:
                    continue
                geometry = _text_runs(text, latin, deva, size, self.fonts)
                if geometry is not None:
                    placed.append((text, lang) + geometry)
            if placed:
                laid.append((size, placed))
        if not laid:
            return None
        bg, fg, border = self._style(rng, kind)
        pad_x = int(base * rng.uniform(0.3, 1.0)) + stroke
        pad_y = int(base * rng.uniform(0.2, 0.7)) + stroke
        arrow = kind in ("highway_green", "highway_blue", "brown_tourist") and rng.random() < 0.5
        arrow_w = int(base * 1.6) if arrow else 0
        arrow_left = bool(rng.random() < 0.5)
        align = "left" if kind in ("highway_green", "highway_blue", "shop_board") and rng.random() < 0.5 else "center"
        rows = []
        for size, placed in laid:
            space = size * rng.uniform(0.25, 0.45)
            width = sum(g[1][2] - g[1][0] for _, _, *g in placed) + space * (len(placed) - 1)
            top = min(g[1][1] for _, _, *g in placed)
            bottom = max(g[1][3] for _, _, *g in placed)
            rows.append((size, placed, space, width, top, bottom))
        inner_w = max(r[3] for r in rows)
        gap = base * rng.uniform(0.15, 0.5)
        inner_h = sum(r[5] - r[4] for r in rows) + gap * (len(rows) - 1)
        W = int(math.ceil(inner_w + 2 * pad_x + arrow_w))
        H = int(math.ceil(inner_h + 2 * pad_y))
        if W > 4000 or H > 2000:
            return None
        image = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        radius = int(min(W, H) * rng.uniform(0, 0.15)) if kind not in ("wall_paint", "plain_text") else 0
        if kind in ("banner", "wall_paint", "shop_board", "plain_text") and rng.random() < 0.6:
            texture = _noise_texture(rng, W, H, bg, amount=float(rng.uniform(6, 28)))
            if kind == "banner" and rng.random() < 0.5:
                ramp = np.linspace(rng.uniform(0.7, 1.0), rng.uniform(1.0, 1.3), W, dtype=np.float32)[None, :, None]
                texture = np.clip(texture * ramp, 0, 255)
            mask = Image.new("L", (W, H), 0)
            ImageDraw.Draw(mask).rounded_rectangle((0, 0, W - 1, H - 1), radius=radius, fill=255)
            image.paste(Image.fromarray(texture.astype(np.uint8), "RGB"), (0, 0), mask)
        else:
            draw.rounded_rectangle((0, 0, W - 1, H - 1), radius=radius, fill=bg + (255,))
        if kind == "milestone":
            cap = _jitter_rgb(rng, [(250, 190, 0), (0, 120, 60), (0, 80, 160)][int(rng.integers(3))])
            draw.rectangle((0, 0, W - 1, int(H * 0.28)), fill=cap + (255,))
        if border is not None and kind not in ("wall_paint", "plain_text"):
            bw = max(2, int(base * rng.uniform(0.06, 0.14)))
            inset = int(bw * rng.uniform(0.5, 2.0))
            draw.rounded_rectangle((inset, inset, W - 1 - inset, H - 1 - inset), radius=max(0, radius - inset),
                                   outline=_jitter_rgb(rng, border, 10) + (255,), width=bw)
        if arrow:  # arrows are sign graphics, not text: good hard negatives
            ax = pad_x * 0.5 if arrow_left else W - pad_x * 0.5 - arrow_w
            ay, s = H / 2, arrow_w * 0.4
            direction = float(rng.choice([0, 90, 180, 270]))
            shape = np.array([[-s, -s * 0.25], [0, -s * 0.25], [0, -s * 0.7], [s, 0], [0, s * 0.7], [0, s * 0.25],
                              [-s, s * 0.25]], np.float32)
            rot = np.radians(direction)
            shape = shape @ np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]], np.float32).T
            draw.polygon([(float(ax + arrow_w * 0.4 + x), float(ay + y)) for x, y in shape], fill=fg + (255,))
        shadow = effects and rng.random() < 0.5
        words, y = [], pad_y
        region_x0 = pad_x + (arrow_w if arrow and arrow_left else 0)
        region_w = W - 2 * pad_x - arrow_w
        for li, (size, placed, space, width, top, bottom) in enumerate(rows):
            x = region_x0 if align == "left" else region_x0 + (region_w - width) / 2
            baseline = y - top
            for text, lang, runs, (l, t, r, b), advance in placed:
                pen = x - l
                if shadow:
                    off = max(1, size // 20)
                    _draw_runs(draw, runs, pen + off, baseline + off, (0, 0, 0, 160))
                _draw_runs(draw, runs, pen, baseline, fg + (255,), stroke,
                           _contrast_rgb(rng, fg) + (255,) if stroke else None)
                margin = 0.06 * size + stroke
                words.append((text, lang, [x - margin, baseline + t - margin, x + (r - l) + margin,
                                           baseline + b + margin], li))
                x += (r - l) + space
            y += (bottom - top) + gap
        return image, words, kind

    # ------------------------------------------------------------------ compositing
    @staticmethod
    def _paste(canvas, sign, words, rng, target_w, centre, persp=0.35, max_rot=6.0):
        """Alpha-blend a perspective-warped sign into canvas (in place). Returns word quads or None."""
        sw, sh = sign.size
        boxes = np.array([w[2] for w in words], np.float32).reshape(-1, 4)
        if target_w / sw < 0.9:  # area-downscale first: avoids aliasing in warpPerspective
            nw = max(2, int(round(target_w)))
            nh = max(2, int(round(sh * nw / sw)))
            boxes = boxes * np.array([nw / sw, nh / sh, nw / sw, nh / sh], np.float32)
            sign = sign.resize((nw, nh), Image.Resampling.BOX)
            sw, sh = nw, nh
        scale = target_w / sw
        tw, th = sw * scale, sh * scale
        yaw = rng.uniform(-persp, persp) if rng.random() < 0.6 else 0.0
        pitch = rng.uniform(-persp, persp) * 0.4 if rng.random() < 0.3 else 0.0
        xs = np.array([-tw / 2, tw / 2, tw / 2, -tw / 2]) * (1 - abs(yaw) * 0.3)
        ys = np.array([-th / 2, -th / 2, th / 2, th / 2])
        ys = ys * (1 - yaw * np.array([-1, 1, 1, -1]) * 0.5)
        xs = xs * (1 - pitch * np.array([-1, -1, 1, 1]) * 0.5)
        angle = np.radians(rng.uniform(-max_rot, max_rot)) if rng.random() < 0.5 else 0.0
        dst = np.stack([xs * np.cos(angle) - ys * np.sin(angle), xs * np.sin(angle) + ys * np.cos(angle)], 1)
        dst = (dst + np.asarray(centre, np.float64)[None, :]).astype(np.float32)
        x0, y0 = int(np.floor(dst[:, 0].min())), int(np.floor(dst[:, 1].min()))
        x1, y1 = int(np.ceil(dst[:, 0].max())) + 1, int(np.ceil(dst[:, 1].max())) + 1
        H, W = canvas.shape[:2]
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H or x1 - x0 < 2 or y1 - y0 < 2:
            return None
        src = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
        patch = cv2.warpPerspective(np.asarray(sign), shift @ matrix, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        alpha = patch[..., 3:4].astype(np.float32) / 255.0
        roi = canvas[y0:y1, x0:x1].astype(np.float32)
        canvas[y0:y1, x0:x1] = np.clip(roi * (1 - alpha) + patch[..., 2::-1].astype(np.float32) * alpha,
                                       0, 255).astype(np.uint8)
        corners = np.stack([boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]]], 1)
        quads = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 4, 2)
        return [order_quad(q) for q in quads]

    def background(self, rng, W, H):
        """Sky, buildings, road with lane marks, trees, poles, vehicles: text-free clutter."""
        img = np.zeros((H, W, 3), np.float32)
        horizon = int(H * rng.uniform(0.25, 0.6))
        night = rng.random() < 0.12
        sky_top = np.array(_jitter_rgb(rng, (230, 190, 120), 40)[::-1], np.float32)  # BGR
        sky_low = np.array(_jitter_rgb(rng, (240, 225, 200), 30)[::-1], np.float32)
        t = np.linspace(0, 1, max(1, horizon), dtype=np.float32)[:, None, None]
        img[:horizon] = sky_top * (1 - t) + sky_low * t
        ground = np.array(_jitter_rgb(rng, [(120, 130, 140), (90, 120, 150), (100, 140, 110)][int(rng.integers(3))], 25),
                          np.float32)
        img[horizon:] = ground
        for _ in range(int(rng.integers(3, 12))):  # buildings / walls
            bw, bh = int(rng.uniform(0.05, 0.35) * W), int(rng.uniform(0.1, 0.55) * H)
            bx = int(rng.integers(-bw // 2, W))
            top = max(0, horizon + int(rng.integers(-10, 30)) - bh)
            colour = np.array(rng.integers(40, 230, 3), np.float32)
            img[top:horizon + int(rng.integers(0, 40)), max(0, bx):bx + bw] = colour
            if rng.random() < 0.6:  # windows grid
                wcol = colour * rng.uniform(0.3, 0.7)
                step = int(rng.integers(12, 40))
                for yy in range(top + step // 2, horizon - step // 2, step):
                    for xx in range(max(0, bx) + step // 3, min(W, bx + bw) - step // 3, step):
                        cv2.rectangle(img, (xx, yy), (xx + step // 2, yy + step // 2), wcol.tolist(), -1)
            elif rng.random() < 0.5:  # rolling shutter stripes
                y_s = max(top, horizon - int(bh * 0.4))
                for yy in range(y_s, horizon, int(rng.integers(3, 7))):
                    cv2.line(img, (max(0, bx), yy), (min(W - 1, bx + bw), yy), (colour * 0.7).tolist(), 1)
        vx = W * rng.uniform(0.3, 0.7)  # road in perspective with lane markings
        road = np.array([[vx - W * 0.03, horizon], [vx + W * 0.03, horizon], [W * 1.3, H], [-W * 0.3, H]], np.int32)
        cv2.fillPoly(img, [road], _jitter_rgb(rng, (70, 70, 72), 15))
        lane_col = (255, 255, 255) if rng.random() < 0.7 else (0, 200, 255)
        for lane in rng.uniform(-0.25, 0.25, int(rng.integers(1, 3))):
            for k in range(12):
                a, b = (k + 0.2) / 12, (k + 0.6) / 12
                p = [(vx + (lane * W * 3) * s * s, horizon + (H - horizon) * s * s) for s in (a, b)]
                width = max(1, int(1 + 10 * b * b))
                cv2.line(img, (int(p[0][0]), int(p[0][1])), (int(p[1][0]), int(p[1][1])), lane_col, width)
        if rng.random() < 0.25:  # zebra crossing
            yz = int(horizon + (H - horizon) * rng.uniform(0.5, 0.9))
            for xz in range(0, W, int(rng.integers(20, 45))):
                cv2.rectangle(img, (xz, yz), (xz + 12, yz + int(rng.integers(8, 25))), (235, 235, 235), -1)
        for _ in range(int(rng.integers(0, 6))):  # trees
            cx, cy = int(rng.integers(0, W)), int(horizon - rng.uniform(0, 0.25) * H)
            r = int(rng.uniform(0.03, 0.1) * W)
            cv2.rectangle(img, (cx - r // 8, cy), (cx + r // 8, min(H - 1, cy + r * 2)), (30, 50, 70), -1)
            for _ in range(8):
                cv2.circle(img, (int(cx + rng.normal(0, r * 0.5)), int(cy - r * 0.5 + rng.normal(0, r * 0.4))),
                           int(r * rng.uniform(0.3, 0.6)), _jitter_rgb(rng, (40, 110, 50), 30), -1)
        for _ in range(int(rng.integers(0, 5))):  # poles and wires
            px = int(rng.integers(0, W))
            cv2.line(img, (px, int(rng.integers(0, horizon))), (px, H), (90, 90, 95), int(rng.integers(2, 7)))
        for _ in range(int(rng.integers(0, 4))):
            y_w = int(rng.integers(0, horizon))
            cv2.line(img, (0, y_w), (W, y_w + int(rng.integers(-30, 30))), (40, 40, 40), 1)
        for _ in range(int(rng.integers(0, 4))):  # vehicle bodies
            vw, vh = int(rng.uniform(0.08, 0.25) * W), int(rng.uniform(0.06, 0.18) * H)
            vx0, vy0 = int(rng.integers(0, max(1, W - vw))), int(rng.integers(horizon, max(horizon + 1, H - vh)))
            body = _jitter_rgb(rng, tuple(int(v) for v in rng.integers(20, 240, 3)), 5)
            cv2.rectangle(img, (vx0, vy0), (vx0 + vw, vy0 + vh), body, -1)
            cv2.rectangle(img, (vx0 + vw // 8, vy0 + vh // 8), (vx0 + vw * 7 // 8, vy0 + vh // 2), (60, 50, 40), -1)
            for wx in (vx0 + vw // 5, vx0 + vw * 4 // 5):
                cv2.circle(img, (wx, vy0 + vh), max(2, vh // 4), (15, 15, 15), -1)
        for _ in range(int(rng.integers(0, 8))):  # random clutter strokes (text-like hard negatives)
            p1 = (int(rng.integers(0, W)), int(rng.integers(0, H)))
            p2 = (p1[0] + int(rng.integers(-60, 60)), p1[1] + int(rng.integers(-20, 20)))
            cv2.line(img, p1, p2, _jitter_rgb(rng, (128, 128, 128), 120), int(rng.integers(1, 4)))
        img += _noise_texture(rng, W, H, (0, 0, 0), amount=float(rng.uniform(4, 16)))
        if night:
            img *= rng.uniform(0.25, 0.5)
        return np.clip(img, 0, 255).astype(np.uint8)

    def render(self, rng, size=None, max_signs=7):
        """One scene: (BGR image, [{'poly','text','lang','ignore'}])."""
        W, H = size or self.scene_sizes[int(rng.integers(len(self.scene_sizes)))]
        canvas = self.background(rng, W, H)
        occupied, words_out = [], []
        for _ in range(int(rng.integers(1, max_signs + 1))):
            made = self.make_sign(rng)
            if made is None:
                continue
            sign, words, kind = made
            heights = [w[2][3] - w[2][1] for w in words]
            text_h = float(np.median(heights))
            want_h = float(np.exp(rng.uniform(np.log(6), np.log(75))))
            target_w = sign.size[0] * want_h / max(text_h, 1)
            target_w = min(target_w, W * 0.9, sign.size[0] * (H * 0.8) / sign.size[1])
            target_h = sign.size[1] * target_w / sign.size[0]
            if target_w < 8:
                continue
            for _attempt in range(12):
                cx = rng.uniform(target_w * 0.6 + 2, max(target_w * 0.6 + 3, W - target_w * 0.6 - 2))
                cy = rng.uniform(target_h * 0.6 + 2, max(target_h * 0.6 + 3, H - target_h * 0.6 - 2))
                box = (cx - target_w * 0.6, cy - target_h * 0.6, cx + target_w * 0.6, cy + target_h * 0.6)
                if any(_box_overlap(box, other) > 0.05 for other in occupied):
                    continue
                if kind in ("highway_green", "highway_blue", "brown_tourist", "yellow_warning", "milestone") \
                        and rng.random() < 0.8:
                    pole_w = max(2, int(target_w * 0.03))
                    for px in ((cx - target_w * 0.3, cx + target_w * 0.3) if target_w > 120 else (cx,)):
                        cv2.rectangle(canvas, (int(px - pole_w), int(cy)), (int(px + pole_w), H - 1), (110, 110, 115), -1)
                quads = self._paste(canvas, sign, words, rng, target_w, (cx, cy))
                if quads is None:
                    continue
                occupied.append(box)
                for (text, lang, _, _), quad in zip(words, quads):
                    words_out.append({"poly": np.round(quad, 2).tolist(), "text": text, "lang": lang,
                                      "ignore": bool(poly_text_height(quad) < 7)})
                break
        if rng.random() < 0.2:  # haze
            fog = np.full_like(canvas, int(rng.integers(150, 230)))
            amount = float(rng.uniform(0.1, 0.35))
            canvas = cv2.addWeighted(canvas, 1.0 - amount, fog, amount, 0)
        if rng.random() < 0.15:  # rain streaks
            for _ in range(int(rng.integers(50, 300))):
                x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
                cv2.line(canvas, (x, y), (x + int(rng.integers(-4, 4)), y + int(rng.integers(8, 20))), (200, 200, 200), 1)
        return canvas, words_out

    def texture_patch(self, rng, w, h):
        if rng.random() < 0.5:
            full = self.background(rng, max(64, w * 2), max(64, h * 2))
            x0 = int(rng.integers(0, full.shape[1] - w + 1))
            y0 = int(rng.integers(0, full.shape[0] - h + 1))
            return full[y0:y0 + h, x0:x0 + w].copy()
        base = tuple(int(v) for v in rng.integers(0, 256, 3))
        return _noise_texture(rng, w, h, base, float(rng.uniform(5, 40))).astype(np.uint8)

    def word_crops(self, rng, max_crops=8):
        """Recognizer samples [(BGR crop, text)] cut out with detector-like quad jitter."""
        made = self.make_sign(rng)
        if made is None:
            return []
        sign, words, _ = made
        text_h = float(np.median([w[2][3] - w[2][1] for w in words]))
        want_h = float(np.exp(rng.uniform(np.log(10), np.log(64))))
        target_w = sign.size[0] * want_h / max(text_h, 1)
        target_h = sign.size[1] * target_w / sign.size[0]
        if target_w > 3000 or target_h > 1500 or target_w < 6:
            return []
        pw = int(target_w * rng.uniform(1.15, 1.5)) + 16
        ph = int(target_h * rng.uniform(1.2, 1.6)) + 16
        patch = self.texture_patch(rng, pw, ph)
        quads = self._paste(patch, sign, words, rng, target_w, (pw / 2, ph / 2), persp=0.3, max_rot=5)
        if quads is None:
            return []
        patch = photometric_aug(patch, rng, 0.8)
        samples = []
        items = list(zip(words, quads))
        for (text, _, _, _), quad in items:
            q = jitter_quad(quad, rng, 0.1) if rng.random() < 0.85 else quad
            crop = crop_quad(patch, q, pad_ratio=float(rng.uniform(0.0, 0.2)))
            if crop is not None and min(crop.shape[:2]) >= 4:
                samples.append((crop, text))
        for (w1, q1), (w2, q2) in zip(items, items[1:]):  # some two-word crops from the same line
            if w1[3] == w2[3] and rng.random() < 0.25:
                a, b = order_quad(q1), order_quad(q2)
                crop = crop_quad(patch, np.array([a[0], b[1], b[2], a[3]], np.float32),
                                 pad_ratio=float(rng.uniform(0.0, 0.15)))
                if crop is not None and min(crop.shape[:2]) >= 4:
                    samples.append((crop, f"{w1[0]} {w2[0]}"))
        rng.shuffle(samples)
        return samples[:max_crops]


def write_synthetic_scenes(out_dir, n, fonts, seed=0, split="train", quality=92):
    """Render n scenes to disk in the same JSONL format used for the real dataset."""
    out_dir = Path(out_dir)
    img_dir = out_dir / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_path = out_dir / f"synthetic_{split}.jsonl"
    done_marker = out_dir / f".done_{split}_{n}_{seed}"
    if done_marker.exists() and ann_path.exists():
        print("Synthetic scenes already present:", ann_path)
        return ann_path
    synth = RoadSceneSynth(fonts)
    started = time.perf_counter()
    with ann_path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            rng = np.random.default_rng([seed, i])
            image, words = synth.render(rng)
            rel = f"images/{split}/scene_{seed}_{i:06d}.jpg"
            if not cv2.imwrite(str(out_dir / rel), image, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                raise OSError(f"Cannot write {out_dir / rel}")
            fh.write(json.dumps({"image": rel, "width": image.shape[1], "height": image.shape[0], "split": split,
                                 "source": "synthetic", "words": words}, ensure_ascii=False) + "\n")
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{n} scenes ({(i + 1) / (time.perf_counter() - started):.1f}/s)")
    done_marker.write_text("ok")
    print(f"Wrote {n} synthetic scenes to {ann_path} in {time.perf_counter() - started:.0f}s")
    return ann_path
