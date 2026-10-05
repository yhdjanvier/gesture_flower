"""Draws the whole UI INSIDE the video frame (so it is also visible in browser full-screen):
top bar (prediction + confidence), bottom 2-second progress bar and, once a gesture is confirmed,
a flower panel (flower picture, name, meaning, description) on the right side.
Pure OpenCV/NumPy - no Streamlit needed.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

import config
from src.flowers import render_flower

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY, GREEN, BLUE = (255, 255, 255), (190, 196, 215), (110, 230, 140), (255, 170, 80)


@lru_cache(maxsize=64)
def _flower_bgr(gesture: str, size: int) -> np.ndarray:
    rgb = np.array(render_flower(gesture, 520).convert("RGB"))
    return cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (size, size), interpolation=cv2.INTER_AREA)


def _text(img, txt, org, scale, color=WHITE, thick=1):
    x, y = org
    cv2.putText(img, txt, (x, y), FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)   # outline
    cv2.putText(img, txt, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def _wrap(txt, scale, max_w):
    lines, cur = [], ""
    for word in txt.split():
        test = (cur + " " + word).strip()
        if cv2.getTextSize(test, FONT, scale, 1)[0][0] > max_w and cur:
            lines.append(cur)
            cur = word
        else:
            cur = test
    return lines + ([cur] if cur else [])


def _shade(img, x0, y0, x1, y1, alpha=0.62, color=(28, 22, 22)):
    roi = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, color), alpha, 0)


def draw_overlay(img: np.ndarray, s: dict, display: dict, confirm_age: float = 9.9) -> None:
    """Draw in place. `s` = StabilityState dict; `confirm_age` = seconds since the flower was confirmed."""
    h, w = img.shape[:2]
    k = h / 480.0
    bar_h, prog_h = int(44 * k), int(30 * k)

    _shade(img, 0, 0, w, bar_h, 0.6)                                   # top bar
    if s["hand_present"] and s["predicted"]:
        _text(img, f"{display.get(s['predicted'], s['predicted'])}  {s['confidence'] * 100:.0f}%",
              (int(14 * k), int(30 * k)), 0.9 * k, WHITE, 2)
    else:
        _text(img, "Show your hand", (int(14 * k), int(30 * k)), 0.9 * k, GREY, 2)

    _shade(img, 0, h - prog_h, w, h, 0.65)                            # bottom progress bar
    if s["candidate"]:
        done = s["progress"] >= 1.0
        cv2.rectangle(img, (0, h - prog_h), (int(w * s["progress"]), h), GREEN if done else BLUE, -1)
        msg = ("Confirmed: " if done else f"Hold steady {s['remaining']:.1f}s: ") + display.get(s["candidate"], "")
    else:
        msg = "Hold a gesture steady for 2 seconds"
    _text(img, msg, (int(14 * k), h - int(9 * k)), 0.6 * k, WHITE, 1)

    g = s.get("confirmed")
    if not g:                                                          # no gesture -> no flower
        return
    f = config.FLOWERS[g]
    pw = int(w * 0.33)
    x0, y0 = w - pw - int(14 * k), bar_h + int(10 * k)
    x1, y1 = w - int(14 * k), h - prog_h - int(10 * k)
    p = min(1.0, confirm_age / 0.6)                                    # bloom-in animation
    ease = 1 - (1 - p) ** 3
    _shade(img, x0, y0, x1, y1, 0.68 * max(p, 0.25))
    pad = int(12 * k)
    size = max(40, min(pw - 2 * pad, int((y1 - y0) * 0.55)))
    cur = max(20, int(size * (0.65 + 0.35 * ease)))
    ix, iy = x0 + (pw - cur) // 2, y0 + pad + (size - cur) // 2
    pic = _flower_bgr(g, cur)
    region = img[iy:iy + cur, ix:ix + cur]
    img[iy:iy + cur, ix:ix + cur] = cv2.addWeighted(region, 1 - ease, pic, ease, 0)
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 1, cv2.LINE_AA)

    y = y0 + pad + size + int(30 * k)
    _text(img, f["name"], (x0 + pad, y), 1.0 * k, WHITE, 2)
    y += int(22 * k)
    _text(img, f["latin"], (x0 + pad, y), 0.45 * k, GREY, 1)
    for line in _wrap(f["description"], 0.5 * k, pw - 2 * pad):
        y += int(22 * k)
        if y > y1 - int(40 * k):
            break
        _text(img, line, (x0 + pad, y), 0.5 * k, WHITE, 1)
    _text(img, f"Meaning: {f['meaning']}", (x0 + pad, y1 - int(26 * k)), 0.48 * k, GREEN, 1)
    _text(img, f"Gesture: {display.get(g, g)}", (x0 + pad, y1 - int(8 * k)), 0.45 * k, GREY, 1)
