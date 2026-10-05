"""Application logic: turn noisy per-frame predictions into a confirmed gesture.

A gesture is CONFIRMED only after the same gesture has been predicted with
smoothed confidence >= min_confidence for `hold_seconds` (default 2 s).
 * probabilities are averaged over `smooth_window` frames (removes flicker)
 * a different confident gesture restarts the timer immediately
 * short dropouts (hand lost / low confidence) up to `grace_seconds` are tolerated
 * longer dropouts cancel the pending candidate
 * the confirmed gesture only changes when ANOTHER gesture is confirmed,
   or is cleared when no hand has been seen for `release_seconds` (flower disappears)
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class StabilityState:
    hand_present: bool = False
    predicted: Optional[str] = None      # smoothed arg-max (may be low confidence)
    confidence: float = 0.0
    candidate: Optional[str] = None      # gesture currently being timed
    progress: float = 0.0                # 0..1
    remaining: float = 0.0               # seconds left
    confirmed: Optional[str] = None
    confirm_count: int = 0               # increments on every new confirmation

    def as_dict(self) -> dict:
        return asdict(self)


class StabilityTracker:
    def __init__(self, labels: Sequence[str], hold_seconds: float = 2.0, min_confidence: float = 0.80,
                 grace_seconds: float = 0.35, smooth_window: int = 5, release_seconds: float = 0.8):
        self.labels = list(labels)
        self.hold_seconds = float(hold_seconds)
        self.min_confidence = float(min_confidence)
        self.grace_seconds = float(grace_seconds)
        self.release_seconds = float(release_seconds)
        self._probs = deque(maxlen=max(1, int(smooth_window)))
        self.reset()

    def reset(self) -> None:
        self._probs.clear()
        self._candidate = None
        self._start = 0.0
        self._last_good = 0.0
        self._last_seen = -1e9
        self._confirmed = None
        self._count = 0

    def update(self, probs: Optional[np.ndarray], now: Optional[float] = None) -> StabilityState:
        now = time.monotonic() if now is None else now
        label, conf = None, 0.0
        if probs is None:
            if now - self._last_seen > self.grace_seconds:
                self._probs.clear()
        else:
            self._last_seen = now
            self._probs.append(np.asarray(probs, dtype=np.float64))
            avg = np.mean(self._probs, axis=0)
            idx = int(np.argmax(avg))
            label, conf = self.labels[idx], float(avg[idx])

        valid = label is not None and conf >= self.min_confidence
        if valid:
            if label == self._candidate:
                self._last_good = now
            else:                                  # new gesture -> restart timer
                self._candidate, self._start, self._last_good = label, now, now
        elif self._candidate is not None and now - self._last_good > self.grace_seconds:
            self._candidate = None                 # lost it for too long -> cancel

        if probs is None and now - self._last_seen > self.release_seconds:
            self._confirmed = None                 # hand gone -> flower disappears
            self._candidate = None

        progress, remaining = 0.0, 0.0
        if self._candidate is not None:
            elapsed = now - self._start
            progress = min(1.0, elapsed / self.hold_seconds)
            remaining = max(0.0, self.hold_seconds - elapsed)
            if progress >= 1.0 and self._candidate != self._confirmed:
                self._confirmed = self._candidate
                self._count += 1

        return StabilityState(hand_present=probs is not None, predicted=label, confidence=conf,
                              candidate=self._candidate, progress=progress, remaining=remaining,
                              confirmed=self._confirmed, confirm_count=self._count)
