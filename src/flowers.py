"""Procedural flower illustrations (no image assets needed) rendered with Pillow.

render_flower(gesture_key, size) -> PIL.Image  (cached). Drawn at 2x and down-sampled for smooth edges.
"""
from __future__ import annotations

import math
import random
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFilter

import config

SS = 2  # supersampling factor
GREEN, DGREEN = (76, 153, 84), (46, 112, 62)


def _mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


_PROFILES = {
    "oval": lambda t: math.sin(math.pi * t),
    "round": lambda t: math.sin(math.pi * t) ** 0.55,
    "pointed": lambda t: math.sin(math.pi * t ** 0.8) ** 1.1,
    "tear": lambda t: math.sin(math.pi * t ** 0.65) ** 0.9,
}


def _petal(d, cx, cy, ang, length, width, color, shape="oval", outline=None, shine=0.35):
    prof, n = _PROFILES[shape], 26
    side = [(t * length, prof(t) * width / 2) for t in (i / n for i in range(n + 1))]
    pts = side + [(x, -y) for x, y in reversed(side)]
    ca, sa = math.cos(ang), math.sin(ang)
    poly = [(cx + x * ca - y * sa, cy + x * sa + y * ca) for x, y in pts]
    d.polygon(poly, fill=color, outline=outline or _mix(color, (0, 0, 0), 0.25))
    if shine > 0:  # lighter inner highlight
        inner = [(cx + (x * 0.8 + length * 0.05) * ca - y * 0.5 * sa,
                  cy + (x * 0.8 + length * 0.05) * sa + y * 0.5 * ca) for x, y in pts]
        d.polygon(inner, fill=_mix(color, (255, 255, 255), shine))


def _ring(d, cx, cy, count, length, width, color, shape="oval", offset=0.0, start=0.0, shine=0.35):
    for k in range(count):
        a = offset + 2 * math.pi * k / count
        _petal(d, cx + math.cos(a) * start, cy + math.sin(a) * start, a, length, width, color, shape, shine=shine)


def _disc(d, cx, cy, r, color, outline=None):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=outline)


def _stem(d, S, cx, cy, lean=0.0, leaves=True, width=None):
    width = width or S * 0.022
    bottom = (cx + lean, S * 0.97)
    ctrl = (cx + lean * 0.2 - S * 0.04, (cy + bottom[1]) / 2)
    pts = [((1 - t) ** 2 * cx + 2 * (1 - t) * t * ctrl[0] + t * t * bottom[0],
            (1 - t) ** 2 * cy + 2 * (1 - t) * t * ctrl[1] + t * t * bottom[1]) for t in (i / 40 for i in range(41))]
    d.line(pts, fill=DGREEN, width=int(width), joint="curve")
    if leaves:
        for frac, side in ((0.55, -1), (0.72, 1)):
            x, y = pts[int(frac * 40)]
            ang = math.radians(-35 if side > 0 else 215)
            _petal(d, x, y, ang, S * 0.2, S * 0.09, GREEN, "pointed", shine=0.2)
    return pts


def _background(S, gradient, seed):
    top, bottom = gradient
    img = Image.new("RGB", (S, S))
    px = ImageDraw.Draw(img)
    for y in range(S):
        px.line([(0, y), (S, y)], fill=_mix(top, bottom, y / S))
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    g, rnd = ImageDraw.Draw(glow), random.Random(seed)
    for _ in range(14):  # soft bokeh
        r, x, y = rnd.randint(S // 25, S // 9), rnd.randint(0, S), rnd.randint(0, S)
        g.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, rnd.randint(30, 80)))
    glow = glow.filter(ImageFilter.GaussianBlur(S * 0.012))
    return Image.alpha_composite(img.convert("RGBA"), glow)


# ------------------------------------------------------------- flower heads
def _rose(d, S, cx, cy, R):
    cols = [(214, 40, 70), (200, 30, 62), (185, 22, 55), (165, 15, 48), (140, 10, 40)]
    specs = [(7, 1.0, 0.95), (6, 0.82, 0.8), (5, 0.64, 0.64), (4, 0.46, 0.5), (3, 0.28, 0.34)]
    for i, (n, L, W) in enumerate(specs):
        _ring(d, cx, cy, n, R * L, R * W, cols[i], "round", offset=i * 2.399, shine=0.18)
    for k in range(3):  # spiral heart
        _disc(d, cx, cy, R * (0.12 - k * 0.03), cols[min(4, 2 + k)], outline=(100, 0, 25))


def _sunflower(d, S, cx, cy, R):
    _ring(d, cx, cy, 26, R * 1.25, R * 0.30, (245, 183, 20), "pointed", start=R * 0.45, shine=0.2)
    _ring(d, cx, cy, 26, R * 1.1, R * 0.28, (255, 210, 50), "pointed", offset=0.12, start=R * 0.45, shine=0.2)
    _disc(d, cx, cy, R * 0.62, (92, 56, 24), outline=(60, 35, 12))
    for i in range(260):  # Fibonacci seed pattern
        r, a = R * 0.58 * math.sqrt(i / 260), i * 2.39996
        _disc(d, cx + r * math.cos(a), cy + r * math.sin(a), R * 0.035,
              (140, 92, 40) if i % 2 else (58, 34, 14))


def _tulip(d, S, cx, cy, R):
    bx, by, up = cx, cy + R * 0.9, -math.pi / 2
    c = [(214, 54, 110), (232, 76, 128), (250, 120, 160)]
    _petal(d, bx, by, up - 0.22, R * 1.85, R * 1.1, c[0], "tear")
    _petal(d, bx, by, up + 0.22, R * 1.85, R * 1.1, c[0], "tear")
    _petal(d, bx, by, up, R * 1.95, R * 1.15, c[1], "tear")
    _petal(d, bx, by, up - 0.08, R * 1.5, R * 0.6, c[2], "tear", shine=0.2)


def _lily(d, S, cx, cy, R):
    for i in range(6):
        a = i * math.pi / 3 - math.pi / 2 + (0.0 if i % 2 == 0 else 0.0)
        col = (255, 240, 244) if i % 2 == 0 else (250, 214, 226)
        _petal(d, cx, cy, a, R * 1.45, R * 0.55, col, "pointed", shine=0.2)
        for k in range(5):  # speckles
            t = 0.25 + 0.12 * k
            _disc(d, cx + math.cos(a) * R * t * 1.4 + math.sin(a) * R * 0.04 * (k % 2 * 2 - 1),
                  cy + math.sin(a) * R * t * 1.4 - math.cos(a) * R * 0.04 * (k % 2 * 2 - 1), R * 0.022, (200, 70, 110))
    for i in range(6):
        a = i * math.pi / 3 + 0.3
        ex, ey = cx + math.cos(a) * R * 0.75, cy + math.sin(a) * R * 0.75
        d.line([(cx, cy), (ex, ey)], fill=(170, 200, 120), width=int(R * 0.035))
        _disc(d, ex, ey, R * 0.06, (190, 90, 30))
    _disc(d, cx, cy, R * 0.07, (120, 170, 80))


def _orchid(d, S, cx, cy, R):
    mag, lt = (176, 56, 160), (222, 150, 214)
    _petal(d, cx, cy, -math.pi / 2, R * 1.2, R * 0.55, lt, "pointed")
    _petal(d, cx, cy, math.pi / 2 + 0.7, R * 1.2, R * 0.55, lt, "pointed")
    _petal(d, cx, cy, math.pi / 2 - 0.7, R * 1.2, R * 0.55, lt, "pointed")
    _petal(d, cx, cy, math.pi + 0.22, R * 1.5, R * 1.35, (236, 176, 228), "round", shine=0.25)
    _petal(d, cx, cy, -0.22, R * 1.5, R * 1.35, (236, 176, 228), "round", shine=0.25)
    _petal(d, cx, cy + R * 0.1, math.pi / 2, R * 0.95, R * 0.62, mag, "tear", shine=0.2)
    _disc(d, cx, cy, R * 0.16, (255, 214, 90), outline=(180, 120, 20))
    _disc(d, cx, cy + R * 0.02, R * 0.07, (255, 255, 255))


def _daisy(d, S, cx, cy, R):
    _ring(d, cx, cy, 24, R * 1.35, R * 0.26, (250, 248, 250), "round", start=R * 0.3, shine=0.0)
    _ring(d, cx, cy, 24, R * 1.25, R * 0.24, (255, 255, 255), "round", offset=0.13, start=R * 0.3, shine=0.0)
    _disc(d, cx, cy, R * 0.45, (255, 200, 30), outline=(200, 140, 10))
    for i in range(70):
        r, a = R * 0.4 * math.sqrt(i / 70), i * 2.39996
        _disc(d, cx + r * math.cos(a), cy + r * math.sin(a), R * 0.025, (230, 150, 20))


def _lavender(d, S, cx, cy, R):  # draws its own stems
    cols = [(150, 100, 210), (170, 120, 225), (190, 150, 235), (125, 80, 190)]
    rnd = random.Random(7)
    for off, lean, top in ((-S * 0.13, -S * 0.07, cy - R * 0.1), (0, 0, cy - R * 0.55), (S * 0.13, S * 0.07, cy - R * 0.2)):
        x0 = cx + off
        pts = _stem(d, S, x0, top, lean=lean * 0.4, leaves=False, width=S * 0.014)
        n = 24
        for i in range(n):
            t = i / (n - 1)
            y = top + t * R * 1.55
            sz = R * (0.10 + 0.06 * math.sin(math.pi * min(1, t * 1.2)))
            for s in (-1, 1):
                _petal(d, x0 + s * sz * 0.4, y, -math.pi / 2 + s * 0.6, sz * 2.0, sz * 1.0,
                       rnd.choice(cols), "tear", shine=0.25)
        _petal(d, x0, top - R * 0.02, -math.pi / 2, R * 0.28, R * 0.12, cols[2], "tear", shine=0.2)


def _lotus(d, S, cx, cy, R):
    d.ellipse([cx - R * 2.0, cy + R * 1.0, cx + R * 2.0, cy + R * 1.9], fill=(30, 110, 90), outline=(15, 70, 55))
    d.ellipse([cx - R * 1.5, cy + R * 1.15, cx + R * 1.5, cy + R * 1.75], fill=(50, 140, 105))
    bx, by, up = cx, cy + R * 1.2, -math.pi / 2
    for spread, L, W, col in ((0.5, 1.9, 0.75, (240, 130, 175)), (0.28, 1.8, 0.7, (248, 160, 195)), (0.0, 1.7, 0.7, (252, 185, 212))):
        for k in (-2, -1, 1, 2) if spread else (0,):
            _petal(d, bx, by, up + k * spread * 0.55, R * L, R * W, col, "pointed", shine=0.3)
    for k in (-1, 1):
        _petal(d, bx, by, up + k * 0.16, R * 1.55, R * 0.6, (255, 215, 230), "pointed", shine=0.25)
    _disc(d, bx, by - R * 0.2, R * 0.13, (255, 205, 60), outline=(200, 140, 20))


_HEADS = {"rose": (_rose, 0.26, 0.36, True), "sunflower": (_sunflower, 0.2, 0.36, True),
          "tulip": (_tulip, 0.22, 0.38, True), "lily": (_lily, 0.24, 0.36, True),
          "orchid": (_orchid, 0.22, 0.38, True), "daisy": (_daisy, 0.23, 0.36, True),
          "lavender": (_lavender, 0.2, 0.34, False), "lotus": (_lotus, 0.2, 0.5, False)}


@lru_cache(maxsize=32)
def render_flower(gesture_key: str, size: int = 520) -> Image.Image:
    """Render the flower mapped to `gesture_key` (e.g. 'open_palm' -> rose)."""
    info = config.FLOWERS[gesture_key]
    fn, rfrac, yfrac, stem = _HEADS[info["key"]]
    S = size * SS
    img = _background(S, info["gradient"], seed=hash(info["key"]) % 1000)
    d = ImageDraw.Draw(img)
    cx, cy, R = S / 2, S * yfrac, S * rfrac
    if info["key"] == "lotus":
        d.rectangle([0, S * 0.66, S, S], fill=(110, 185, 200))
        for y in range(int(S * 0.66), S, max(4, S // 40)):
            d.line([(0, y), (S, y)], fill=(150, 210, 222), width=1)
    if stem:
        _stem(d, S, cx, cy + R * 0.2, lean=S * 0.03)
    fn(d, S, cx, cy, R)
    return img.convert("RGB").resize((size, size), Image.LANCZOS)


def placeholder(size: int = 520) -> Image.Image:
    """Neutral 'waiting' image shown before the first gesture is confirmed."""
    img = _background(size * SS, ((30, 34, 48), (18, 20, 30)), seed=3)
    d = ImageDraw.Draw(img)
    S = size * SS
    for k in range(8):
        _petal(d, S / 2, S / 2, k * math.pi / 4, S * 0.2, S * 0.09, (70, 76, 100), "round", shine=0.1)
    _disc(d, S / 2, S / 2, S * 0.04, (110, 118, 150))
    return img.convert("RGB").resize((size, size), Image.LANCZOS)
