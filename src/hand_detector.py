"""MediaPipe hand detection that works with BOTH MediaPipe APIs.

* mediapipe <= ~0.10.2x : legacy `mp.solutions.hands`
* newer mediapipe       : Tasks `HandLandmarker` (needs `hand_landmarker.task`, auto-downloaded once
                          into models/ on first run)
Both give the same 21 landmarks, so features/model/app are identical for either backend.
"""
from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np

import config
from src.features import landmarks_to_array

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
TASK_PATH = config.MODEL_DIR / "hand_landmarker.task"

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


@dataclass
class HandResult:
    raw: np.ndarray        # (63,) pixel-like landmarks
    is_left: bool          # handedness as seen by the USER (frame must be mirrored first)
    proto: object          # object with `.landmark` (normalised x,y) used for drawing
    score: float


def _ensure_task_model() -> str:
    if not TASK_PATH.exists():
        TASK_PATH.parent.mkdir(exist_ok=True)
        try:
            print("[hand_detector] downloading hand_landmarker.task (~8 MB, one time)...")
            urllib.request.urlretrieve(MODEL_URL, TASK_PATH)
        except Exception as exc:
            TASK_PATH.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download the MediaPipe hand model. Download it manually from\n"
                f"{MODEL_URL}\nand save it as {TASK_PATH}") from exc
    return str(TASK_PATH)


class HandDetector:
    def __init__(self):
        import mediapipe as mp
        self._mp = mp
        self._legacy = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
        self._last_ts = 0
        if self._legacy:
            self._hands = mp.solutions.hands.Hands(static_image_mode=False, **config.MP_HANDS)
        else:
            from mediapipe.tasks.python import BaseOptions, vision
            m = config.MP_HANDS
            opts = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=_ensure_task_model()),
                running_mode=vision.RunningMode.VIDEO, num_hands=m["max_num_hands"],
                min_hand_detection_confidence=m["min_detection_confidence"],
                min_hand_presence_confidence=m["min_detection_confidence"],
                min_tracking_confidence=m["min_tracking_confidence"])
            self._hands = vision.HandLandmarker.create_from_options(opts)

    def detect(self, rgb: np.ndarray) -> Optional[HandResult]:
        """`rgb` must already be mirrored (selfie view) so handedness labels are correct."""
        h, w = rgb.shape[:2]
        if self._legacy:
            rgb.flags.writeable = False
            res = self._hands.process(rgb)
            rgb.flags.writeable = True
            if not res.multi_hand_landmarks:
                return None
            proto = res.multi_hand_landmarks[0]
            cls = res.multi_handedness[0].classification[0]
            return HandResult(landmarks_to_array(proto, w, h), cls.label == "Left", proto, float(cls.score))
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)   # must strictly increase
        self._last_ts = ts
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._hands.detect_for_video(image, ts)
        if not res.hand_landmarks:
            return None
        proto = SimpleNamespace(landmark=res.hand_landmarks[0])
        cat = res.handedness[0][0]
        return HandResult(landmarks_to_array(proto, w, h), cat.category_name == "Left", proto, float(cat.score))

    def draw(self, bgr: np.ndarray, result: HandResult) -> None:
        h, w = bgr.shape[:2]
        pts = [(int(l.x * w), int(l.y * h)) for l in result.proto.landmark]
        for a, b in HAND_CONNECTIONS:
            cv2.line(bgr, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            cv2.circle(bgr, p, 5 if i in (4, 8, 12, 16, 20) else 3, (90, 90, 255), -1, cv2.LINE_AA)

    def close(self) -> None:
        self._hands.close()
