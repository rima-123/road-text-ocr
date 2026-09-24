# ===== CELL 16 (9.3): Geometry helpers =====
def order_quad(points):
    """Return 4 points as top-left, top-right, bottom-right, bottom-left (for text up to ~45 deg)."""
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    if len(pts) != 4:
        pts = cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.float32)
    pts = pts[np.argsort(pts[:, 0], kind="stable")]
    left, right = pts[:2], pts[2:]
    tl, bl = left[np.argsort(left[:, 1], kind="stable")]
    tr, br = right[np.argsort(right[:, 1], kind="stable")]
    return np.array([tl, tr, br, bl], np.float32)


def quad_size(quad):
    q = np.asarray(quad, np.float32).reshape(4, 2)
    width = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))
    height = max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))
    return float(width), float(height)


def poly_text_height(poly):
    """Short side of the minimum-area rectangle: the text height for horizontal-ish words."""
    pts = np.asarray(poly, np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    return float(min(cv2.minAreaRect(pts)[1]))


def expand_quad(quad, pad_x, pad_y):
    """Grow an ordered quad along its own axes by pad_x / pad_y pixels on every side."""
    q = order_quad(quad).astype(np.float64)
    ux = (q[1] - q[0]) + (q[2] - q[3])
    uy = (q[3] - q[0]) + (q[2] - q[1])
    ux /= max(np.linalg.norm(ux), 1e-6)
    uy /= max(np.linalg.norm(uy), 1e-6)
    sx = np.array([-1, 1, 1, -1], np.float64)[:, None]
    sy = np.array([-1, -1, 1, 1], np.float64)[:, None]
    return (q + sx * pad_x * ux + sy * pad_y * uy).astype(np.float32)


def crop_quad(image, quad, pad_ratio=0.12, max_side=2048):
    """Perspective-rectify one (rotated/skewed) word quad into an upright crop; None if degenerate."""
    q = order_quad(quad)
    width, height = quad_size(q)
    if width < 2 or height < 2:
        return None
    pad = pad_ratio * min(width, height)
    q = expand_quad(q, pad, pad)
    width, height = quad_size(q)
    out_w = int(np.clip(round(width), 2, max_side))
    out_h = int(np.clip(round(height), 2, max_side))
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(image, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def jitter_quad(quad, rng, amount=0.12):
    """Random detector-like imprecision: corner noise plus random expansion/shrink."""
    q = order_quad(quad)
    _, height = quad_size(q)
    height = max(height, 2.0)
    noise = rng.normal(0, amount * height * 0.4, (4, 2)).astype(np.float32)
    noise[:, 0] *= 1.3
    q = q + noise
    # Mostly grow: a crop that cuts into glyphs would no longer match its label.
    return expand_quad(q, rng.uniform(0.02, 0.3) * height, rng.uniform(-0.02, 0.15) * height)


def _convex(poly):
    pts = np.asarray(poly, np.float32).reshape(-1, 2)
    return cv2.convexHull(pts).reshape(-1, 2).astype(np.float32)


def poly_bbox(poly):
    pts = np.asarray(poly, np.float32).reshape(-1, 2)
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def poly_overlap(a, b):
    """(IoU, intersection / area(b)) for two polygons, using their convex hulls."""
    ca, cb = _convex(a), _convex(b)
    area_a, area_b = float(cv2.contourArea(ca)), float(cv2.contourArea(cb))
    if area_a <= 0 or area_b <= 0 or len(ca) < 3 or len(cb) < 3:
        return 0.0, 0.0
    inter, _ = cv2.intersectConvexConvex(ca, cb)
    inter = float(max(0.0, inter))
    union = area_a + area_b - inter
    return (inter / union if union > 0 else 0.0), inter / area_b


def match_polys(gt_polys, pred_polys, iou_thr=0.5):
    """Greedy one-to-one matching by IoU. Returns [(gt_index, pred_index, iou)]."""
    if not gt_polys or not pred_polys:
        return []
    gboxes = [poly_bbox(p) for p in gt_polys]
    pboxes = [poly_bbox(p) for p in pred_polys]
    pairs = []
    for gi, gb in enumerate(gboxes):
        for pi, pb in enumerate(pboxes):
            if gb[0] > pb[2] or pb[0] > gb[2] or gb[1] > pb[3] or pb[1] > gb[3]:
                continue
            iou, _ = poly_overlap(gt_polys[gi], pred_polys[pi])
            if iou >= iou_thr:
                pairs.append((iou, gi, pi))
    pairs.sort(key=lambda t: -t[0])
    used_g, used_p, out = set(), set(), []
    for iou, gi, pi in pairs:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            out.append((gi, pi, iou))
    return out


def offset_convex(poly, dist):
    """Move every edge of a convex polygon inward by `dist` pixels. None if the polygon collapses."""
    p = _convex(poly).astype(np.float64)
    if len(p) < 3:
        return None
    centre = p.mean(axis=0)
    lines = []
    for i in range(len(p)):
        a, b = p[i], p[(i + 1) % len(p)]
        t = b - a
        length = float(np.hypot(t[0], t[1]))
        if length < 1e-6:
            continue
        t /= length
        normal = np.array([-t[1], t[0]])
        if np.dot(centre - a, normal) < 0:
            normal = -normal
        lines.append((a + dist * normal, t))
    if len(lines) < 3:
        return None
    out = []
    for i in range(len(lines)):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-9:
            out.append(p2)
            continue
        s = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
        out.append(p1 + s * d1)
    out = np.asarray(out, np.float32)
    before = cv2.contourArea(p.astype(np.float32).reshape(-1, 1, 2), oriented=True)
    after = cv2.contourArea(out.reshape(-1, 1, 2), oriented=True)
    if abs(after) < 1e-3 or np.sign(after) != np.sign(before):
        return None
    if dist > 0:
        hull = p.astype(np.float32).reshape(-1, 1, 2)
        if any(cv2.pointPolygonTest(hull, (float(x), float(y)), False) < 0 for x, y in out):
            return None
    return out


def unclip_rect(rect, ratio):
    """DB-style expansion of a (centre, size, angle) rectangle by area*ratio/perimeter."""
    (cx, cy), (w, h), angle = rect
    dist = (w * h) * ratio / max(2.0 * (w + h), 1e-6)
    return cv2.boxPoints(((cx, cy), (w + 2 * dist, h + 2 * dist), angle)).astype(np.float32)


def motion_blur(image, rng, max_len=9):
    length = int(rng.integers(3, max_len + 1))
    kernel = np.zeros((length, length), np.float32)
    angle = float(rng.uniform(0, 180))
    centre = (length - 1) / 2
    dx, dy = np.cos(np.radians(angle)) * centre, np.sin(np.radians(angle)) * centre
    cv2.line(kernel, (int(round(centre - dx)), int(round(centre - dy))),
             (int(round(centre + dx)), int(round(centre + dy))), 1.0, 1)
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(image, -1, kernel)


def photometric_aug(image, rng, strength=1.0):
    """Camera-like colour/blur/noise/compression changes for whole images (BGR uint8)."""
    out = image.astype(np.float32)
    if rng.random() < 0.8 * strength:
        out = out * rng.uniform(0.6, 1.35) + rng.uniform(-35, 35)
    if rng.random() < 0.35 * strength:
        out = out * rng.uniform(0.8, 1.2, 3).astype(np.float32)[None, None, :]
    if rng.random() < 0.25 * strength:
        gray = out.mean(axis=2, keepdims=True)
        out = gray + (out - gray) * rng.uniform(0.2, 1.5)
    if rng.random() < 0.3 * strength:  # uneven light / shadow band
        h, w = out.shape[:2]
        gx = np.linspace(rng.uniform(0.5, 1.2), rng.uniform(0.5, 1.2), w, dtype=np.float32)[None, :, None]
        gy = np.linspace(rng.uniform(0.7, 1.1), rng.uniform(0.7, 1.1), h, dtype=np.float32)[:, None, None]
        out = out * gx * gy
    out = np.clip(out, 0, 255).astype(np.uint8)
    choice = rng.random()
    if choice < 0.15 * strength:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), float(rng.uniform(0.5, 1.6)))
    elif choice < 0.3 * strength:
        out = motion_blur(out, rng)
    if rng.random() < 0.15 * strength:
        h, w = out.shape[:2]
        f = float(rng.uniform(0.45, 0.85))
        small = cv2.resize(out, (max(2, int(w * f)), max(2, int(h * f))), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.3 * strength:
        noise = rng.normal(0, rng.uniform(2, 10), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.3 * strength:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 90))])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out
