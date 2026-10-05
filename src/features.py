"""Feature engineering: MediaPipe hand landmarks -> fixed-length numeric vector.

This module is the ONLY place where landmarks are turned into features. It is
imported by preprocessing (training) AND by live inference, which guarantees
that both use exactly the same maths (no training/serving skew).

Raw landmark format (63 numbers per hand): x0,y0,z0,...,x20,y20,z20 in
*pixel-like* units (x*width, y*height, z*width), so that x and y share one scale.
"""
from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21
RAW_DIM = NUM_LANDMARKS * 3
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
TIP_IDS = (4, 8, 12, 16, 20)
PIP_IDS = (3, 6, 10, 14, 18)
PALM_IDS = (5, 9, 13, 17)          # MCP knuckles of index..pinky
# (a, b, c): angle at joint b between bones b->a and b->c, 3 per finger
ANGLE_TRIPLETS = (
    (0, 1, 2), (1, 2, 3), (2, 3, 4),          # thumb
    (0, 5, 6), (5, 6, 7), (6, 7, 8),          # index
    (0, 9, 10), (9, 10, 11), (10, 11, 12),    # middle
    (0, 13, 14), (13, 14, 15), (14, 15, 16),  # ring
    (0, 17, 18), (17, 18, 19), (18, 19, 20),  # pinky
)
_EPS = 1e-8

RAW_COLUMNS = (
    ["label", "handedness", "session", "block"]
    + [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in "xyz"]
)


def _build_feature_names() -> tuple:
    names = []
    for i in range(1, NUM_LANDMARKS):                       # 60
        names += [f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"]
    names += [f"tip_wrist_dist_{f}" for f in FINGER_NAMES]  # 5
    for a in range(5):                                      # 10
        for b in range(a + 1, 5):
            names.append(f"tip_dist_{FINGER_NAMES[a]}_{FINGER_NAMES[b]}")
    for f in FINGER_NAMES:                                  # 15
        for j in range(3):
            names.append(f"joint_angle_{f}_{j}")
    names += ["palm_dir_x", "palm_dir_y", "palm_dir_z"]     # 3
    names += ["thumb_dir_x", "thumb_dir_y", "thumb_dir_z"]  # 3
    names += [f"extension_ratio_{f}" for f in FINGER_NAMES]  # 5
    return tuple(names)


FEATURE_NAMES = _build_feature_names()
FEATURE_DIM = len(FEATURE_NAMES)  # 101


def landmarks_to_array(hand_landmarks, width: int, height: int) -> np.ndarray:
    """Convert a MediaPipe hand (object with `.landmark`) to a raw (63,) vector."""
    pts = np.array([[lm.x * width, lm.y * height, lm.z * width] for lm in hand_landmarks.landmark],
                   dtype=np.float64)
    return pts.reshape(-1)


def normalize_landmarks(raw, is_left: bool) -> np.ndarray:
    """Translation-, scale- and handedness-normalised landmarks, shape (21, 3).

    1. Translate so the wrist (landmark 0) is the origin   -> position invariance
    2. Mirror x for LEFT hands                              -> one model for both hands
    3. Divide by mean wrist->knuckle distance (5,9,13,17)   -> distance/size invariance
    Rotation is deliberately NOT removed: orientation separates thumbs-up / thumbs-down.
    """
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.size != RAW_DIM:
        raise ValueError(f"Expected {RAW_DIM} raw values, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Landmarks contain NaN/inf")
    pts = arr.reshape(NUM_LANDMARKS, 3).copy()
    pts -= pts[0]
    if is_left:
        pts[:, 0] *= -1.0
    scale = float(np.mean(np.linalg.norm(pts[list(PALM_IDS)], axis=1)))
    if scale < 1e-6:
        raise ValueError("Degenerate hand (zero size)")
    return pts / scale


def _joint_angle(a, b, c) -> float:
    v1, v2 = a - b, c - b
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + _EPS))
    return float(np.arccos(np.clip(cos, -1.0, 1.0)) / np.pi)  # 0..1


def extract_features(raw, is_left: bool) -> np.ndarray:
    """Raw landmarks -> float32 feature vector of length FEATURE_DIM (101)."""
    p = normalize_landmarks(raw, is_left)
    tips = p[list(TIP_IDS)]
    tip_norms = np.linalg.norm(tips, axis=1)
    pair = [np.linalg.norm(tips[a] - tips[b]) for a in range(5) for b in range(a + 1, 5)]
    angles = [_joint_angle(p[a], p[b], p[c]) for a, b, c in ANGLE_TRIPLETS]
    palm_dir = p[9] / (np.linalg.norm(p[9]) + _EPS)
    thumb_vec = p[4] - p[2]
    thumb_dir = thumb_vec / (np.linalg.norm(thumb_vec) + _EPS)
    ratio = np.clip(tip_norms / (np.linalg.norm(p[list(PIP_IDS)], axis=1) + _EPS), 0.0, 5.0)
    out = np.concatenate([p[1:].reshape(-1), tip_norms, pair, angles, palm_dir, thumb_dir, ratio])
    out = out.astype(np.float32)
    if out.shape[0] != FEATURE_DIM:  # defensive: should never happen
        raise RuntimeError(f"Feature length {out.shape[0]} != {FEATURE_DIM}")
    return out


def extract_features_batch(raw_matrix: np.ndarray, left_flags) -> np.ndarray:
    return np.stack([extract_features(r, bool(l)) for r, l in zip(raw_matrix, left_flags)]).astype(np.float32)


def augment_landmarks(raw, rng: np.random.Generator, max_rotation_deg: float = 15.0,
                      noise_std: float = 0.02) -> np.ndarray:
    """Training-time augmentation on RAW landmarks: random in-plane rotation + Gaussian jitter."""
    pts = np.asarray(raw, dtype=np.float64).reshape(NUM_LANDMARKS, 3).copy()
    wrist = pts[0].copy()
    pts -= wrist
    scale = float(np.mean(np.linalg.norm(pts[list(PALM_IDS)], axis=1)))
    ang = np.deg2rad(rng.uniform(-max_rotation_deg, max_rotation_deg))
    c, s = np.cos(ang), np.sin(ang)
    x, y = pts[:, 0].copy(), pts[:, 1].copy()
    pts[:, 0], pts[:, 1] = c * x - s * y, s * x + c * y
    pts += rng.normal(0.0, noise_std * scale, pts.shape)
    return (pts + wrist).reshape(-1)
