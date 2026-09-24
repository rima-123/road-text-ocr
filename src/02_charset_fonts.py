# ===== CELL 2: Imports =====
"""Hindi/English video OCR, trained from scratch; no downloaded model weights.

Derived from video_text_recognition_v3.ipynb. Run --help for training and video
commands. OpenCV proposes text regions; a randomly initialized CRNN learns to
read them. A saved checkpoint must come from this project's own training run.
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import math
import os
import re
import shutil
import string
import struct
import subprocess
import tempfile
import time
import unicodedata
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

# ===== CELL 3: Charset, fonts, crop preprocessing =====
LATIN_CHARS = " " + string.punctuation + string.digits + string.ascii_uppercase + string.ascii_lowercase
DEVA_CHARS = "".join(chr(i) for i in range(0x0900, 0x0980)
                     if unicodedata.category(chr(i)) != "Cn")
CHARS = LATIN_CHARS + DEVA_CHARS
BLANK = 0
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}
DEVA_CONS = "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळऴऱऩक़ख़ग़ज़ड़ढ़फ़य़"
VIRAMA, NUKTA, PRE_BASE_MATRA = "्", "़", "ि"
PREPROCESS_VERSION = "colour_border_polarity_v1"


@dataclass
class OCRConfig:
    rec_h: int = 32
    rec_w: int = 384
    max_len: int = 48
    n_train: int = 9000
    n_val: int = 600
    epochs: int = 20
    batch_size: int = 32
    seed: int = 42
    hindi: bool = True
    visual_order: bool = True

    def validate(self):
        if self.rec_h != 32 or self.rec_w < 64 or self.rec_w % 4:
            raise ValueError("Use rec_h=32 and rec_w >= 64 divisible by 4.")
        if self.max_len < 1 or 2 * self.max_len - 1 > self.rec_w // 4:
            raise ValueError("CTC needs rec_w/4 >= 2*max_len-1 for repeated characters.")
        if min(self.n_train, self.n_val, self.epochs, self.batch_size) < 1:
            raise ValueError("Training counts, epochs, and batch_size must be positive.")


def _base_end(text, i):
    j = i + 1
    if j < len(text) and text[j] == NUKTA:
        j += 1
    while j + 1 < len(text) and text[j] == VIRAMA and text[j + 1] in DEVA_CONS:
        j += 2
        if j < len(text) and text[j] == NUKTA:
            j += 1
    return j


def to_visual(text):
    """Retain the original notebook's pre-base-i convention, including nukta.

    This is a limited CTC label convention, not a general Indic shaping engine.
    Pillow/RAQM performs the actual glyph shaping when images are rendered.
    """
    out, i = [], 0
    while i < len(text):
        if text[i] in DEVA_CONS:
            j = _base_end(text, i)
            k = j
            while k < len(text) and unicodedata.category(text[k]).startswith("M"):
                k += 1
            marks = text[j:k]
            out.append(PRE_BASE_MATRA * marks.count(PRE_BASE_MATRA)
                       + text[i:j] + marks.replace(PRE_BASE_MATRA, ""))
            i = k
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def to_logical(text):
    out, i = [], 0
    while i < len(text):
        if text[i] == PRE_BASE_MATRA and i + 1 < len(text) and text[i + 1] in DEVA_CONS:
            j = _base_end(text, i + 1)
            out.append(text[i + 1:j] + PRE_BASE_MATRA)
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def encode_label(text, config):
    unknown = set(text) - set(CHARS)
    if unknown:
        raise ValueError(f"Unsupported characters in training label: {unknown!r}")
    if len(text) > config.max_len:
        raise ValueError(f"Label exceeds max_len={config.max_len}: {text!r}")
    text = to_visual(text) if config.visual_order else text
    return [CHAR_TO_IDX[c] for c in text]


@lru_cache(maxsize=256)
def _font_characters(path):
    from fontTools.ttLib import TTFont
    with TTFont(path, lazy=True) as font:
        return frozenset(font.getBestCmap() or {})


def font_supports(path, text):
    return all(c.isspace() or ord(c) in _font_characters(str(path)) for c in text)


def find_fonts(font_dir=None, hindi=True):
    roots = [Path(font_dir)] if font_dir else []
    roots += [Path("/usr/share/fonts"), Path.home() / ".local/share/fonts",
              Path("/Library/Fonts"), Path("C:/Windows/Fonts")]
    paths = sorted({str(p) for root in roots if root.is_dir()
                    for p in root.rglob("*") if p.suffix.lower() in (".ttf", ".otf")})
    latin, deva = [], []
    for p in paths:
        try:
            if font_supports(p, "ABCabc0123456789"):
                latin.append(p)
            if font_supports(p, "विभागसंगणकविज्ञानअड्डाहिंदी"):
                deva.append(p)
        except Exception:
            continue  # An unreadable font is not a usable training font.
    if not latin:
        raise RuntimeError("Install a Latin TrueType/OpenType font or set FONT_DIR/--font-dir.")
    if hindi and (not deva or not features.check("raqm")):
        raise RuntimeError("Hindi needs a Devanagari font and Pillow RAQM. In Colab run "
                           "the setup cell (fonts-noto-core); locally install a Noto Sans "
                           "Devanagari font and a Pillow build with RAQM.")
    return {"latin": latin, "deva": deva}


@lru_cache(maxsize=256)
def _font(path, size):
    return ImageFont.truetype(str(path), int(size))


def _font_for(text, fonts, rng=None):
    pool = fonts["deva"] if any(c in DEVA_CHARS for c in text) else fonts["latin"]
    usable = [p for p in pool if font_supports(p, text)]
    if not usable:
        raise RuntimeError(f"No installed font covers this text: {text!r}. Set FONT_DIR.")
    return usable[int(rng.integers(len(usable)))] if rng is not None else usable[0]


def text_geometry(text, fonts, size, rng=None):
    """Lay out Latin/Devanagari runs on a shared baseline with script fonts.

    A Devanagari-only font need not also contain Latin UI labels or digits.
    Keep complete Indic runs together so RAQM can shape conjuncts and matras.
    """
    runs, advance = [], 0.0
    left = top = right = bottom = 0.0
    for segment in re.findall(r"[\u0900-\u097f]+|[^\u0900-\u097f]+", text):
        font = _font(_font_for(segment, fonts, rng), size)
        l, t, r, b = font.getbbox(segment, anchor="ls")
        left, top = min(left, advance + l), min(top, t)
        right, bottom = max(right, advance + r), max(bottom, b)
        runs.append((segment, font, advance))
        advance += font.getlength(segment)
    return runs, (math.floor(left), math.floor(top), math.ceil(max(right, advance)), math.ceil(bottom))


def draw_text_runs(draw, runs, x, baseline, fill):
    for segment, font, advance in runs:
        draw.text((x + advance, baseline), segment, font=font, fill=fill, anchor="ls")


def colour_gray(bgr):
    """Choose gray/channel with strong Otsu separation for colored sign lettering."""
    if bgr.ndim == 2:
        return bgr
    channels = [cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), *cv2.split(bgr),
                cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]]
    best, best_score = channels[0], -1.0
    for channel in channels:
        hist = np.bincount(channel.ravel(), minlength=256).astype(np.float64)
        prob = hist / max(1, hist.sum())
        weight = prob.cumsum()
        mean = (prob * np.arange(256)).cumsum()
        score = float(np.max((mean[-1] * weight - mean) ** 2 /
                             (weight * (1 - weight) + 1e-12)))
        if score > best_score:
            best, best_score = channel, score
    return best


def preprocess_crop(bgr, config):
    """Same transformation in training/inference; letterbox without stretching."""
    if bgr is None or not bgr.size:
        raise ValueError("Empty text crop.")
    gray = colour_gray(bgr)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    if float(np.median(border)) > float(np.mean(gray)):
        gray = 255 - gray
    low, high = np.percentile(gray, [2, 98])
    if high - low > 5:
        gray = np.clip((gray.astype(np.float32) - low) * 255 / (high - low), 0, 255).astype(np.uint8)
    h, w = gray.shape
    scale = min(config.rec_h / h, config.rec_w / w)
    rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(gray, (rw, rh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((config.rec_h, config.rec_w), np.uint8)
    top = (config.rec_h - rh) // 2
    canvas[top:top + rh, :rw] = resized
    return (canvas.astype(np.float32) / 255.0)[..., None]
