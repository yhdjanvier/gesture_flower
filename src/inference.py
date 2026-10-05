"""Load the trained FNN + saved preprocessing and classify one hand."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

import config
from src.features import FEATURE_DIM, extract_features


class ArtifactsMissing(RuntimeError):
    """Raised when the model / scaler / labels have not been produced yet."""


def artifacts_ready() -> list:
    """Return the list of missing required files (empty list == ready)."""
    needed = [config.MODEL_PATH, config.SCALER_PATH, config.LABELS_PATH]
    return [str(p.relative_to(config.ROOT)) for p in needed if not p.exists()]


class GestureClassifier:
    def __init__(self):
        missing = artifacts_ready()
        if missing:
            raise ArtifactsMissing("Missing: " + ", ".join(missing)
                                   + ". Run `python -m src.preprocess` then `python -m src.train`.")
        import tensorflow as tf
        self.model = tf.keras.models.load_model(config.MODEL_PATH, compile=False)
        scaler = json.loads(config.SCALER_PATH.read_text())
        self.mean = np.asarray(scaler["mean"], dtype=np.float32)
        self.scale = np.asarray(scaler["scale"], dtype=np.float32)
        labels = json.loads(config.LABELS_PATH.read_text())
        self.labels = [labels[str(i)] for i in range(len(labels))]
        if self.mean.shape[0] != FEATURE_DIM or self.model.input_shape[-1] != FEATURE_DIM:
            raise RuntimeError("Feature dimension mismatch between code, scaler and model. Re-run preprocess + train.")
        if self.model.output_shape[-1] != len(self.labels):
            raise RuntimeError("Number of classes in model and label map differ. Re-run train.")
        self.predict_proba(np.zeros(63) + np.arange(63), False)  # warm-up

    def predict_proba(self, raw: np.ndarray, is_left: bool) -> np.ndarray:
        x = (extract_features(raw, is_left) - self.mean) / self.scale   # SAME as training
        return self.model(x[None, :], training=False).numpy()[0]

    def predict(self, raw: np.ndarray, is_left: bool):
        p = self.predict_proba(raw, is_left)
        i = int(np.argmax(p))
        return self.labels[i], float(p[i]), p


@lru_cache(maxsize=1)
def get_classifier() -> GestureClassifier:
    return GestureClassifier()
