"""Dependency-light tests (no TensorFlow / webcam needed). Run: python -m pytest -q  or  python tests/test_core.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

import config
from src.features import (FEATURE_DIM, RAW_COLUMNS, augment_landmarks, extract_features, normalize_landmarks)
from src.preprocess import split_by_blocks
from src.stability import StabilityTracker


def fake_hand(seed=0):
    r = np.random.default_rng(seed)
    pts = r.normal(size=(21, 3)) * 20
    pts[0] = [100, 200, 0]
    pts[[5, 9, 13, 17]] = pts[0] + np.array([[-20, -60, 0], [0, -65, 0], [18, -60, 0], [35, -50, 0]])
    return pts.reshape(-1)


def test_feature_shape_and_finite():
    f = extract_features(fake_hand(), False)
    assert f.shape == (FEATURE_DIM,) and np.isfinite(f).all() and FEATURE_DIM == 101


def test_translation_scale_invariance():
    raw = fake_hand(1)
    moved = (raw.reshape(21, 3) * 2.5 + np.array([40, -30, 0])).reshape(-1)
    assert np.allclose(extract_features(raw, False), extract_features(moved, False), atol=1e-4)


def test_left_hand_mirrors_to_right():
    raw = fake_hand(2).reshape(21, 3)
    mirrored = raw.copy(); mirrored[:, 0] = -mirrored[:, 0] + 300
    assert np.allclose(extract_features(raw.reshape(-1), False), extract_features(mirrored.reshape(-1), True), atol=1e-4)


def test_orientation_is_kept():  # thumbs up vs down must differ
    up = fake_hand(3).reshape(21, 3); down = up.copy(); down[:, 1] = 2 * down[0, 1] - down[:, 1]
    assert not np.allclose(extract_features(up.reshape(-1), False), extract_features(down.reshape(-1), False))


def test_augmentation_keeps_shape():
    out = augment_landmarks(fake_hand(), np.random.default_rng(0))
    assert out.shape == (63,) and normalize_landmarks(out, False).shape == (21, 3)


def test_split_has_no_block_leakage():
    rows = []
    for g in config.GESTURES:
        for i in range(300):
            rows.append([g, "Right", "s1", i // config.BLOCK_SIZE] + list(fake_hand(i % 7)))
    df = split_by_blocks(pd.DataFrame(rows, columns=RAW_COLUMNS))
    assert df.groupby("group")["split"].nunique().max() == 1
    assert set(df["split"]) == {"train", "val", "test"}


def _p(i, n=8, c=0.95):
    p = np.full(n, (1 - c) / (n - 1)); p[i] = c; return p


def test_confirms_after_two_seconds():
    t = StabilityTracker(config.GESTURES, 2.0, 0.8, 0.35, 5)
    s = None
    for k in range(0, 41):                       # 0..2.0 s at 20 fps
        s = t.update(_p(1), now=k * 0.05)
    assert s.confirmed is None or s.confirmed == config.GESTURES[1]
    s = t.update(_p(1), now=2.1)
    assert s.confirmed == config.GESTURES[1] and s.confirm_count == 1


def test_not_confirmed_before_two_seconds_and_reset_on_change():
    t = StabilityTracker(config.GESTURES, 2.0, 0.8, 0.35, 1)
    for k in range(30):
        s = t.update(_p(0), now=k * 0.05)        # 1.5 s
    assert s.confirmed is None and 0.7 < s.progress < 0.8
    s = t.update(_p(2), now=1.55)                # gesture change -> timer restarts
    assert s.candidate == config.GESTURES[2] and s.progress == 0.0


def test_single_noisy_frame_does_not_change_flower():
    t = StabilityTracker(config.GESTURES, 2.0, 0.8, 0.35, 5)
    for k in range(0, 45):
        s = t.update(_p(3), now=k * 0.05)
    assert s.confirmed == config.GESTURES[3]
    s = t.update(_p(5), now=2.25)                # one wrong frame
    for k in range(1, 20):
        s = t.update(_p(3), now=2.25 + k * 0.05)
    assert s.confirmed == config.GESTURES[3] and s.confirm_count == 1


def test_long_hand_loss_cancels_timer():
    t = StabilityTracker(config.GESTURES, 2.0, 0.8, 0.35, 5)
    for k in range(10):
        t.update(_p(4), now=k * 0.05)
    s = t.update(None, now=1.5)
    assert s.candidate is None and s.progress == 0.0


def test_flower_disappears_when_hand_leaves():
    t = StabilityTracker(config.GESTURES, 2.0, 0.8, 0.35, 5, release_seconds=0.8)
    for k in range(45):
        s = t.update(_p(2), now=k * 0.05)
    assert s.confirmed == config.GESTURES[2]
    s = t.update(None, now=2.4)                  # brief loss -> still shown
    assert s.confirmed == config.GESTURES[2]
    s = t.update(None, now=3.3)                  # gone > 0.8 s -> cleared
    assert s.confirmed is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
